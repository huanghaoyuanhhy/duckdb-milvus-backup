//! Page-level decoding: DATA_PAGE V1/V2, PLAIN and dictionary encodings,
//! definition/repetition levels, and the per-column-chunk cursor the scan
//! pulls rows from.
//!
//! Layout facts verified against real Milvus binlogs (parquet-go 12.0.1 and
//! arrow C++ 17): DATA_PAGE V1 compresses levels+values together, with each
//! level block prefixed by a 4-byte LE length; DATA_PAGE_V2 stores levels
//! uncompressed (repetition levels first) and only compresses the values.

use duckdb_neo::{Error, Result};

use super::codec::decompress;
use super::footer::{Codec, ColumnChunkMeta, PhysicalType, SchemaLeaf};
use super::rle::{decode_hybrid, level_bit_width};
use super::thrift::{ThriftReader, T_I32, T_STRUCT};

const PAGE_TYPE_DATA: i32 = 0;
const PAGE_TYPE_DICTIONARY: i32 = 2;
const PAGE_TYPE_DATA_V2: i32 = 3;

const ENC_PLAIN: i32 = 0;
const ENC_PLAIN_DICTIONARY: i32 = 2;
const ENC_RLE_DICTIONARY: i32 = 8;

/// Non-null values of one page. BYTE_ARRAY and FIXED_LEN_BYTE_ARRAY share the
/// `Bytes` form (flat buffer + n+1 offsets) so downstream code stays uniform.
#[derive(Debug, PartialEq)]
pub(crate) enum Values {
    Bool(Vec<bool>),
    Int32(Vec<i32>),
    Int64(Vec<i64>),
    Float(Vec<f32>),
    Double(Vec<f64>),
    Bytes { data: Vec<u8>, offsets: Vec<u32> },
}

impl Values {
    pub(crate) fn len(&self) -> usize {
        match self {
            Values::Bool(v) => v.len(),
            Values::Int32(v) => v.len(),
            Values::Int64(v) => v.len(),
            Values::Float(v) => v.len(),
            Values::Double(v) => v.len(),
            Values::Bytes { offsets, .. } => offsets.len().saturating_sub(1),
        }
    }

    fn take(&self, indices: &[u64]) -> Result<Values> {
        macro_rules! gather {
            ($values:expr) => {
                indices
                    .iter()
                    .map(|&index| {
                        $values
                            .get(index as usize)
                            .copied()
                            .ok_or_else(|| Error::invalid_input("dictionary index out of range"))
                    })
                    .collect::<Result<Vec<_>>>()
            };
        }
        match self {
            Values::Bool(values) => gather!(values).map(Values::Bool),
            Values::Int32(values) => gather!(values).map(Values::Int32),
            Values::Int64(values) => gather!(values).map(Values::Int64),
            Values::Float(values) => gather!(values).map(Values::Float),
            Values::Double(values) => gather!(values).map(Values::Double),
            Values::Bytes { data, offsets } => {
                let mut out_data = Vec::with_capacity(data.len());
                let mut out_offsets = Vec::with_capacity(indices.len() + 1);
                out_offsets.push(0u32);
                for &index in indices {
                    let index = index as usize;
                    let start = *offsets
                        .get(index)
                        .ok_or_else(|| Error::invalid_input("dictionary index out of range"))?
                        as usize;
                    let end = *offsets
                        .get(index + 1)
                        .ok_or_else(|| Error::invalid_input("dictionary index out of range"))?
                        as usize;
                    out_data.extend_from_slice(&data[start..end]);
                    out_offsets.push(out_data.len() as u32);
                }
                Ok(Values::Bytes {
                    data: out_data,
                    offsets: out_offsets,
                })
            }
        }
    }
}

/// One decoded data page. `num_values` counts levels (including nulls);
/// `values` holds only the rows whose definition level equals the leaf max.
#[derive(Debug)]
pub(crate) struct Page {
    pub(crate) num_values: u32,
    pub(crate) values: Values,
    pub(crate) def_levels: Option<Vec<u16>>,
    pub(crate) rep_levels: Option<Vec<u16>>,
}

struct PageHeader {
    page_type: i32,
    uncompressed_size: usize,
    compressed_size: usize,
    /// DATA_PAGE: (num_values, value encoding).
    data: Option<(i32, i32)>,
    /// DICTIONARY_PAGE: (num_values, encoding).
    dictionary: Option<(i32, i32)>,
    data_v2: Option<DataPageHeaderV2>,
}

struct DataPageHeaderV2 {
    num_values: i32,
    encoding: i32,
    def_len: usize,
    rep_len: usize,
    is_compressed: bool,
}

fn parse_page_header(data: &[u8]) -> Result<(PageHeader, usize)> {
    let mut reader = ThriftReader::new(data);
    reader.enter_struct();

    let mut header = PageHeader {
        page_type: -1,
        uncompressed_size: 0,
        compressed_size: 0,
        data: None,
        dictionary: None,
        data_v2: None,
    };
    while let Ok((field_id, field_type)) = reader.field_header() {
        if field_type == super::thrift::T_STOP {
            break;
        }
        match (field_id, field_type) {
            (1, T_I32) => header.page_type = reader.zigzag()? as i32,
            (2, T_I32) => header.uncompressed_size = reader.zigzag()? as usize,
            (3, T_I32) => header.compressed_size = reader.zigzag()? as usize,
            (5, T_STRUCT) => header.data = parse_kv_page(&mut reader)?,
            (7, T_STRUCT) => header.dictionary = parse_kv_page(&mut reader)?,
            (8, T_STRUCT) => {
                // DataPageHeaderV2: 1 num_values, 4 encoding, 5 def length,
                // 6 rep length, 7 is_compressed (bool lives in the header).
                reader.enter_struct();
                let mut v2 = DataPageHeaderV2 {
                    num_values: 0,
                    encoding: 0,
                    def_len: 0,
                    rep_len: 0,
                    is_compressed: true,
                };
                while let Ok((nested_id, nested_type)) = reader.field_header() {
                    if nested_type == super::thrift::T_STOP {
                        break;
                    }
                    match (nested_id, nested_type) {
                        (1, T_I32) => v2.num_values = reader.zigzag()? as i32,
                        (4, T_I32) => v2.encoding = reader.zigzag()? as i32,
                        (5, T_I32) => v2.def_len = reader.zigzag()? as usize,
                        (6, T_I32) => v2.rep_len = reader.zigzag()? as usize,
                        (7, super::thrift::T_BOOL_TRUE) => v2.is_compressed = true,
                        (7, super::thrift::T_BOOL_FALSE) => v2.is_compressed = false,
                        _ => reader.skip(nested_type)?,
                    }
                }
                header.data_v2 = Some(v2);
            }
            _ => reader.skip(field_type)?,
        }
    }
    Ok((header, reader.consumed()))
}

/// Read the two leading i32 fields (`num_values`, `encoding`) shared by
/// DataPageHeader and DictionaryPageHeader.
fn parse_kv_page(reader: &mut ThriftReader<'_>) -> Result<Option<(i32, i32)>> {
    reader.enter_struct();
    let mut num_values: Option<i32> = None;
    let mut encoding: Option<i32> = None;
    while let Ok((nested_id, nested_type)) = reader.field_header() {
        if nested_type == super::thrift::T_STOP {
            return Ok(num_values.zip(encoding));
        }
        match (nested_id, nested_type) {
            (1, T_I32) => num_values = Some(reader.zigzag()? as i32),
            (2, T_I32) => encoding = Some(reader.zigzag()? as i32),
            _ => reader.skip(nested_type)?,
        }
    }
    Err(Error::invalid_input("page header struct not terminated"))
}

/// Sequential reader over one column chunk of one row group. Owns a clone of
/// the file view, so it can live in per-thread scan state.
pub(crate) struct ChunkReader {
    view: super::FileView,
    leaf: SchemaLeaf,
    meta: ColumnChunkMeta,
    dict: Option<Values>,
    next_offset: u64,
    emitted: u64,
}

impl ChunkReader {
    pub(crate) fn open(
        view: super::FileView,
        leaf: SchemaLeaf,
        meta: ColumnChunkMeta,
    ) -> Result<Self> {
        if meta.data_page_offset < 0 {
            return Err(Error::invalid_input("column chunk has no data page offset"));
        }
        let dict = match meta.dictionary_page_offset {
            Some(offset) if offset >= 0 => Some(read_dictionary_page(
                &view,
                offset as u64,
                meta.codec,
                leaf.physical,
            )?),
            _ => None,
        };
        let next_offset = meta.data_page_offset as u64;
        Ok(ChunkReader {
            view,
            leaf,
            meta,
            dict,
            next_offset,
            emitted: 0,
        })
    }

    pub(crate) fn next_page(&mut self) -> Result<Option<Page>> {
        if self.emitted >= self.meta.num_values {
            return Ok(None);
        }
        let (header, header_len) = read_header_at(&self.view, self.next_offset)?;
        match header.page_type {
            PAGE_TYPE_DATA => {
                let (num_values, encoding) = header
                    .data
                    .ok_or_else(|| Error::invalid_input("DATA_PAGE without header"))?;
                let compressed = self.view.read_vec(
                    self.next_offset + header_len as u64,
                    header.compressed_size,
                )?;
                let page = decompress(
                    codec_code(self.meta.codec),
                    &compressed,
                    header.uncompressed_size,
                )
                .and_then(|page| decode_data_page_v1(&page, &self.leaf, num_values as usize, encoding, self.dict.as_ref()))?;
                self.emitted += page.num_values as u64;
                self.next_offset += (header_len + header.compressed_size) as u64;
                Ok(Some(page))
            }
            PAGE_TYPE_DATA_V2 => {
                let v2 = header
                    .data_v2
                    .ok_or_else(|| Error::invalid_input("DATA_PAGE_V2 without header"))?;
                let body = self.view.read_vec(
                    self.next_offset + header_len as u64,
                    header.compressed_size,
                )?;
                let page =
                    decode_data_page_v2(&body, &self.leaf, &v2, self.meta.codec, header.uncompressed_size, self.dict.as_ref())?;
                self.emitted += page.num_values as u64;
                self.next_offset += (header_len + header.compressed_size) as u64;
                Ok(Some(page))
            }
            other => Err(Error::invalid_input(format!(
                "unexpected parquet page type {other} inside a column chunk"
            ))),
        }
    }
}

/// Map the footer codec enum onto the wire code the decompressor takes.
fn codec_code(codec: Codec) -> i32 {
    match codec {
        Codec::Uncompressed => super::codec::CODEC_UNCOMPRESSED,
        Codec::Snappy => super::codec::CODEC_SNAPPY,
        Codec::Zstd => super::codec::CODEC_ZSTD,
        Codec::Other(code) => code,
    }
}

/// Page headers are a few dozen bytes but their exact size is unknown before
/// parsing; grow the read window until thrift parsing succeeds.
fn read_header_at(view: &super::FileView, offset: u64) -> Result<(PageHeader, usize)> {
    if offset >= view.size() {
        return Err(Error::invalid_input(format!(
            "page offset {offset} beyond end of parquet region ({} bytes)",
            view.size()
        )));
    }
    let remaining = (view.size() - offset) as usize;
    let mut window = 128usize.min(remaining);
    loop {
        let bytes = view.read_vec(offset, window)?;
        if let Ok((header, consumed)) = parse_page_header(&bytes) {
            return Ok((header, consumed));
        }
        if window >= remaining {
            return Err(Error::invalid_input(
                "failed to parse parquet page header (page may be corrupt)",
            ));
        }
        window = (window * 4).min(remaining).min(1 << 20);
    }
}

fn read_dictionary_page(
    view: &super::FileView,
    offset: u64,
    codec: Codec,
    physical: PhysicalType,
) -> Result<Values> {
    let (header, header_len) = read_header_at(view, offset)?;
    if header.page_type != PAGE_TYPE_DICTIONARY {
        return Err(Error::invalid_input(format!(
            "expected dictionary page, found page type {}",
            header.page_type
        )));
    }
    let (num_values, _encoding) = header
        .dictionary
        .ok_or_else(|| Error::invalid_input("DICTIONARY_PAGE without header"))?;
    let compressed = view.read_vec(offset + header_len as u64, header.compressed_size)?;
    let body = decompress(codec_code(codec), &compressed, header.uncompressed_size)?;
    decode_plain_values(&body, num_values as usize, physical)
}

fn decode_data_page_v1(
    page: &[u8],
    leaf: &SchemaLeaf,
    num_values: usize,
    encoding: i32,
    dict: Option<&Values>,
) -> Result<Page> {
    let mut pos = 0usize;
    let rep_levels = if leaf.max_rep > 0 {
        let (levels, consumed) = read_level_block(page, pos, level_bit_width(leaf.max_rep), num_values)?;
        pos = consumed;
        Some(levels)
    } else {
        None
    };
    let def_levels = if leaf.max_def > 0 {
        let (levels, consumed) = read_level_block(page, pos, level_bit_width(leaf.max_def), num_values)?;
        pos = consumed;
        Some(levels)
    } else {
        None
    };
    let non_null = def_levels
        .as_ref()
        .map(|levels| levels.iter().filter(|&&level| level == leaf.max_def).count())
        .unwrap_or(num_values);
    let values = decode_values(&page[pos..], encoding, leaf, non_null, dict)?;
    Ok(Page {
        num_values: num_values as u32,
        values,
        def_levels,
        rep_levels,
    })
}

/// One V1 level block: 4-byte LE length, then an RLE hybrid stream.
fn read_level_block(
    page: &[u8],
    pos: usize,
    bit_width: u8,
    num_values: usize,
) -> Result<(Vec<u16>, usize)> {
    let len_bytes = page
        .get(pos..pos + 4)
        .ok_or_else(|| Error::invalid_input("level block length truncated"))?;
    let len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
    let levels_bytes = page
        .get(pos + 4..pos + 4 + len)
        .ok_or_else(|| Error::invalid_input("level block truncated"))?;
    let levels = decode_hybrid(levels_bytes, bit_width, num_values)?
        .into_iter()
        .map(|value| value as u16)
        .collect();
    Ok((levels, pos + 4 + len))
}

fn decode_data_page_v2(
    body: &[u8],
    leaf: &SchemaLeaf,
    v2: &DataPageHeaderV2,
    codec: Codec,
    uncompressed_size: usize,
    dict: Option<&Values>,
) -> Result<Page> {
    let num_values = v2.num_values as usize;
    // V2 order: repetition levels, definition levels, encoded values.
    let mut cursor = 0usize;
    let rep_levels = if v2.rep_len > 0 {
        let levels = decode_hybrid(
            body.get(cursor..cursor + v2.rep_len)
                .ok_or_else(|| Error::invalid_input("V2 repetition levels truncated"))?,
            level_bit_width(leaf.max_rep),
            num_values,
        )?
        .into_iter()
        .map(|value| value as u16)
        .collect();
        cursor += v2.rep_len;
        Some(levels)
    } else {
        None
    };
    let def_levels = if v2.def_len > 0 {
        let levels: Vec<u16> = decode_hybrid(
            body.get(cursor..cursor + v2.def_len)
                .ok_or_else(|| Error::invalid_input("V2 definition levels truncated"))?,
            level_bit_width(leaf.max_def),
            num_values,
        )?
        .into_iter()
        .map(|value| value as u16)
        .collect();
        cursor += v2.def_len;
        Some(levels)
    } else {
        None
    };

    let non_null = def_levels
        .as_ref()
        .map(|levels| levels.iter().filter(|&&level| level == leaf.max_def).count())
        .unwrap_or(num_values);
    let values_len = uncompressed_size
        .checked_sub(cursor)
        .ok_or_else(|| Error::invalid_input("negative V2 values size"))?;
    let values_data = if v2.is_compressed {
        let compressed = body
            .get(cursor..)
            .ok_or_else(|| Error::invalid_input("V2 values truncated"))?;
        decompress(codec_code(codec), compressed, values_len)?
    } else {
        body.get(cursor..cursor + values_len)
            .ok_or_else(|| Error::invalid_input("V2 values truncated"))?
            .to_vec()
    };
    let values = decode_values(&values_data, v2.encoding, leaf, non_null, dict)?;
    Ok(Page {
        num_values: num_values as u32,
        values,
        def_levels,
        rep_levels,
    })
}

fn decode_values(
    data: &[u8],
    encoding: i32,
    leaf: &SchemaLeaf,
    count: usize,
    dict: Option<&Values>,
) -> Result<Values> {
    match encoding {
        ENC_PLAIN => decode_plain_values(data, count, leaf.physical),
        ENC_PLAIN_DICTIONARY | ENC_RLE_DICTIONARY => {
            let dict = dict.ok_or_else(|| {
                Error::invalid_input("dictionary-encoded page without a dictionary")
            })?;
            let bit_width = *data
                .first()
                .ok_or_else(|| Error::invalid_input("dictionary page has no bit width"))?;
            let indices = decode_hybrid(&data[1..], bit_width, count)?;
            dict.take(&indices)
        }
        other => Err(Error::invalid_input(format!(
            "unsupported parquet value encoding {other} (only PLAIN and dictionary \
             encodings are produced by Milvus)"
        ))),
    }
}

/// PLAIN-encoded values of one physical type; `count` non-null values.
pub(crate) fn decode_plain_values(
    data: &[u8],
    count: usize,
    physical: PhysicalType,
) -> Result<Values> {
    let mut cursor = 0usize;
    macro_rules! fixed {
        ($size:expr, $read:expr, $variant:expr) => {{
            let mut out = Vec::with_capacity(count);
            for _ in 0..count {
                let bytes = data
                    .get(cursor..cursor + $size)
                    .ok_or_else(|| Error::invalid_input("PLAIN values truncated"))?;
                out.push($read(bytes));
                cursor += $size;
            }
            $variant(out)
        }};
    }
    Ok(match physical {
        PhysicalType::Boolean => {
            let mut out = Vec::with_capacity(count);
            for index in 0..count {
                let byte = *data
                    .get(index / 8)
                    .ok_or_else(|| Error::invalid_input("PLAIN booleans truncated"))?;
                out.push(byte & (1 << (index % 8)) != 0);
            }
            Values::Bool(out)
        }
        PhysicalType::Int32 => fixed!(4, le_i32, Values::Int32),
        PhysicalType::Int64 => fixed!(8, le_i64, Values::Int64),
        PhysicalType::Float => fixed!(
            4,
            |b: &[u8]| f32::from_le_bytes(b.try_into().unwrap()),
            Values::Float
        ),
        PhysicalType::Double => fixed!(
            8,
            |b: &[u8]| f64::from_le_bytes(b.try_into().unwrap()),
            Values::Double
        ),
        PhysicalType::ByteArray => {
            let mut offsets = Vec::with_capacity(count + 1);
            offsets.push(0u32);
            let mut out = Vec::with_capacity(data.len());
            for _ in 0..count {
                let len_bytes = data
                    .get(cursor..cursor + 4)
                    .ok_or_else(|| Error::invalid_input("PLAIN byte array length truncated"))?;
                let len = u32::from_le_bytes(len_bytes.try_into().unwrap()) as usize;
                let bytes = data
                    .get(cursor + 4..cursor + 4 + len)
                    .ok_or_else(|| Error::invalid_input("PLAIN byte array truncated"))?;
                out.extend_from_slice(bytes);
                cursor += 4 + len;
                offsets.push(out.len() as u32);
            }
            Values::Bytes {
                data: out,
                offsets,
            }
        }
        PhysicalType::FixedLenByteArray(len) => {
            let len = len.max(0) as usize;
            if data.len() < count * len {
                return Err(Error::invalid_input(format!(
                    "PLAIN fixed byte array truncated: {} bytes for {}x{len}",
                    data.len(),
                    count
                )));
            }
            let offsets = (0..=count).map(|index| (index * len) as u32).collect();
            Values::Bytes {
                data: data[..count * len].to_vec(),
                offsets,
            }
        }
    })
}

/// Fully-decoded column chunk: concatenated page values plus raw level
/// arrays (row = definition-level position).
#[derive(Debug)]
pub(crate) struct ColumnData {
    pub(crate) values: Values,
    pub(crate) def_levels: Option<Vec<u16>>,
    pub(crate) rep_levels: Option<Vec<u16>>,
    pub(crate) max_def: u16,
}

impl ColumnData {
    pub(crate) fn rows(&self) -> usize {
        self.def_levels
            .as_ref()
            .map(|levels| levels.len())
            .unwrap_or_else(|| self.values.len())
    }

    /// Whether the row at `index` carries a value (def level == max).
    pub(crate) fn is_valid(&self, index: usize) -> bool {
        match &self.def_levels {
            None => true,
            Some(levels) => levels.get(index).copied().unwrap_or(0) == self.max_def,
        }
    }
}

fn merge_values(target: &mut Values, source: Values) -> Result<()> {
    match (target, source) {
        (Values::Bool(a), Values::Bool(b)) => a.extend(b),
        (Values::Int32(a), Values::Int32(b)) => a.extend(b),
        (Values::Int64(a), Values::Int64(b)) => a.extend(b),
        (Values::Float(a), Values::Float(b)) => a.extend(b),
        (Values::Double(a), Values::Double(b)) => a.extend(b),
        (
            Values::Bytes { data, offsets },
            Values::Bytes {
                data: source_data,
                offsets: source_offsets,
            },
        ) => {
            let base = data.len() as u32;
            data.extend(source_data);
            offsets.extend(source_offsets[1..].iter().map(|offset| offset + base));
        }
        (target, source) => {
            return Err(Error::invalid_input(format!(
                "page value variant mismatch inside one column chunk: {target:?} vs {source:?}"
            )))
        }
    }
    Ok(())
}

/// Decode every page of the chunk into one [`ColumnData`].
pub(crate) fn read_chunk(reader: &mut ChunkReader) -> Result<ColumnData> {
    let max_def = reader.leaf.max_def;
    let mut values: Option<Values> = None;
    let mut def_levels: Option<Vec<u16>> = None;
    let mut rep_levels: Option<Vec<u16>> = None;
    while let Some(page) = reader.next_page()? {
        merge_values(values.get_or_insert_with(|| match &page.values {
            Values::Bool(_) => Values::Bool(Vec::new()),
            Values::Int32(_) => Values::Int32(Vec::new()),
            Values::Int64(_) => Values::Int64(Vec::new()),
            Values::Float(_) => Values::Float(Vec::new()),
            Values::Double(_) => Values::Double(Vec::new()),
            Values::Bytes { .. } => Values::Bytes {
                data: Vec::new(),
                offsets: vec![0],
            },
        }), page.values)?;
        if let Some(levels) = page.def_levels {
            def_levels.get_or_insert_default().extend(levels);
        }
        if let Some(levels) = page.rep_levels {
            rep_levels.get_or_insert_default().extend(levels);
        }
    }
    Ok(ColumnData {
        values: values.ok_or_else(|| Error::invalid_input("column chunk has no pages"))?,
        def_levels,
        rep_levels,
        max_def,
    })
}

fn le_i32(bytes: &[u8]) -> i32 {
    i32::from_le_bytes(bytes.try_into().unwrap())
}

fn le_i64(bytes: &[u8]) -> i64 {
    i64::from_le_bytes(bytes.try_into().unwrap())
}
