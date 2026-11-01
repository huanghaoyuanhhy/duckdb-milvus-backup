//! Parquet file footer (`FileMetaData`) parsing into the small subset of
//! structures the binlog reader needs: leaf schema with def/rep levels and
//! field ids, plus row-group column chunk locations.
//!
//! Thrift field ids follow the canonical `parquet.thrift`.

use duckdb_neo::{Error, Result};

use super::thrift::{T_BINARY, T_I32, T_I64, T_LIST, T_STRUCT, ThriftReader};

/// parquet `Type` enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PhysicalType {
    Boolean,
    Int32,
    Int64,
    Float,
    Double,
    ByteArray,
    FixedLenByteArray(i32),
}

impl PhysicalType {
    fn from_code(code: i32, type_length: Option<i32>) -> Result<Self> {
        Ok(match code {
            0 => PhysicalType::Boolean,
            1 => PhysicalType::Int32,
            2 => PhysicalType::Int64,
            3 => {
                return Err(Error::invalid_input(
                    "INT96 parquet columns are not produced by Milvus and are unsupported",
                ))
            }
            4 => PhysicalType::Float,
            5 => PhysicalType::Double,
            6 => PhysicalType::ByteArray,
            7 => PhysicalType::FixedLenByteArray(
                type_length.ok_or_else(|| Error::invalid_input("FLBA column missing type_length"))?,
            ),
            other => {
                return Err(Error::invalid_input(format!(
                    "unknown parquet physical type {other}"
                )))
            }
        })
    }
}

/// The logical annotations that change how a value is interpreted. Only what
/// Milvus actually writes; everything else keeps the physical interpretation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum LogicalKind {
    #[default]
    None,
    /// UTF8 string (VARCHAR).
    String,
    /// Signed/unsigned integer of the given width, stored in INT32/INT64.
    Int { bits: u8, signed: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Repetition {
    Required,
    Optional,
    Repeated,
}

/// One schema leaf (column). Nested group columns keep their parent field id
/// so storage-v2 column-group files and LIST columns can be matched back to
/// Milvus fields.
#[derive(Debug, Clone)]
pub(crate) struct SchemaLeaf {
    /// Dotted path from the root, e.g. `item` inside a LIST group; kept for
    /// diagnostics and tests.
    #[allow(dead_code)]
    pub(crate) path: Vec<String>,
    pub(crate) physical: PhysicalType,
    /// Logical annotation (STRING/INT width); the scan layer maps types from
    /// the Milvus schema, so this is informational.
    #[allow(dead_code)]
    pub(crate) logical: LogicalKind,
    pub(crate) max_def: u16,
    pub(crate) max_rep: u16,
    pub(crate) field_id: Option<i64>,
    /// For LIST leaves: whether the LIST group itself is OPTIONAL (decides
    /// whether definition level 0 means "null list").
    pub(crate) list_group_optional: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Codec {
    Uncompressed,
    Snappy,
    Zstd,
    Other(i32),
}

impl Codec {
    fn from_code(code: i32) -> Codec {
        match code {
            super::codec::CODEC_UNCOMPRESSED => Codec::Uncompressed,
            super::codec::CODEC_SNAPPY => Codec::Snappy,
            super::codec::CODEC_ZSTD => Codec::Zstd,
            other => Codec::Other(other),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct ColumnChunkMeta {
    pub(crate) codec: Codec,
    /// Total values in this chunk, including nulls.
    pub(crate) num_values: u64,
    pub(crate) data_page_offset: i64,
    pub(crate) dictionary_page_offset: Option<i64>,
    /// Part of the parsed metadata; page walking uses the page headers.
    #[allow(dead_code)]
    pub(crate) total_compressed_size: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct RowGroupMeta {
    /// From the footer; row counts are also derived from decoded pages.
    #[allow(dead_code)]
    pub(crate) num_rows: u64,
    /// One entry per schema leaf, same order.
    pub(crate) columns: Vec<ColumnChunkMeta>,
}

#[derive(Debug)]
pub(crate) struct ParquetMeta {
    pub(crate) leaves: Vec<SchemaLeaf>,
    pub(crate) row_groups: Vec<RowGroupMeta>,
    /// Writer signature (e.g. "parquet-go version 12.0.1"); used by tests to
    /// pin the expected writer generation.
    #[allow(dead_code)]
    pub(crate) created_by: Option<String>,
    /// Footer key-value metadata, in file order. milvus-storage packed files
    /// carry their field-id mapping here (`group_field_id_list`).
    pub(crate) key_value: Vec<(String, String)>,
}

/// Parse milvus-storage's `group_field_id_list` footer entry: `;`-separated
/// groups, one per parquet leaf column in schema order, each a `,`-separated
/// list of Milvus field ids (outer arrow field first for nested types). See
/// milvus-storage `cpp/src/common/metadata.cpp` (GroupFieldIDList).
pub(crate) fn group_field_ids(key_value: &[(String, String)], leaf_count: usize) -> Option<Vec<i64>> {
    let value = key_value
        .iter()
        .find(|(key, _)| key == "group_field_id_list")
        .map(|(_, value)| value)?;
    let mut ids = Vec::with_capacity(leaf_count);
    for group in value.split(';') {
        let first = group.split(',').next()?;
        ids.push(first.parse::<i64>().ok()?);
    }
    (ids.len() == leaf_count).then_some(ids)
}

/// Read and parse the footer of a parquet file (or of an embedded payload
/// region; `view` bounds the region). The reader anchors at the region end.
pub(crate) fn read_footer(view: &super::FileView) -> Result<ParquetMeta> {
    let size = view.size();
    if size < 12 {
        return Err(Error::invalid_input(format!(
            "parquet region too small for a footer: {size} bytes"
        )));
    }
    let tail = view.read_vec(size - 8, 8)?;
    if &tail[4..8] != b"PAR1" {
        return Err(Error::invalid_input(
            "parquet region does not end with the PAR1 footer magic",
        ));
    }
    let footer_len = u32::from_le_bytes(tail[..4].try_into().unwrap()) as u64;
    let footer_start = size - 8 - footer_len;
    if footer_len == 0 || footer_start < 4 {
        return Err(Error::invalid_input(format!(
            "invalid parquet footer length {footer_len}"
        )));
    }
    let footer = view.read_vec(footer_start, footer_len as usize)?;
    parse_file_metadata(&footer)
}

fn parse_file_metadata(data: &[u8]) -> Result<ParquetMeta> {
    // First pass: schema (field 2) + created_by (field 6); everything else is
    // skipped and row groups are parsed in a focused second pass.
    let mut reader = ThriftReader::new(data);
    reader.enter_struct();

    let mut raw_schema: Vec<RawSchemaElement> = Vec::new();
    let mut created_by: Option<String> = None;
    let mut key_value: Vec<(String, String)> = Vec::new();

    while let Ok((field_id, field_type)) = reader.field_header() {
        if field_type == super::thrift::T_STOP {
            break;
        }
        match (field_id, field_type) {
            (2, T_LIST) => parse_element_list(&mut reader, &mut raw_schema)?,
            (5, T_LIST) => parse_key_value_metadata(&mut reader, &mut key_value)?,
            (6, T_BINARY) => created_by = Some(String::from_utf8_lossy(reader.binary()?).into_owned()),
            _ => reader.skip(field_type)?,
        }
    }

    let mut leaves: Vec<SchemaLeaf> = Vec::new();
    build_leaves(raw_schema, &mut leaves)?;

    let mut row_groups: Vec<RowGroupMeta> = Vec::new();
    let mut reader = ThriftReader::new(data);
    reader.enter_struct();
    while let Ok((field_id, field_type)) = reader.field_header() {
        if field_type == super::thrift::T_STOP {
            break;
        }
        if field_id == 4 {
            parse_row_groups(&mut reader, &mut row_groups)?;
        } else {
            reader.skip(field_type)?;
        }
    }

    if row_groups.is_empty() {
        return Err(Error::invalid_input("parquet footer has no row groups"));
    }
    for group in &row_groups {
        if group.columns.len() != leaves.len() {
            return Err(Error::invalid_input(format!(
                "row group has {} column chunks but schema has {} leaves",
                group.columns.len(),
                leaves.len()
            )));
        }
    }
    Ok(ParquetMeta {
        leaves,
        row_groups,
        created_by,
        key_value,
    })
}

/// parquet `KeyValue` list: structs of `1: string key, 2: string value`.
fn parse_key_value_metadata(
    reader: &mut ThriftReader<'_>,
    out: &mut Vec<(String, String)>,
) -> Result<()> {
    let (element_type, size) = reader.list_header()?;
    if element_type != T_STRUCT {
        return Err(Error::invalid_input("key-value metadata is not a struct list"));
    }
    for _ in 0..size {
        reader.enter_struct();
        let mut key: Option<String> = None;
        let mut value = String::new();
        while let Ok((field_id, field_type)) = reader.field_header() {
            if field_type == super::thrift::T_STOP {
                break;
            }
            match (field_id, field_type) {
                (1, T_BINARY) => key = Some(String::from_utf8_lossy(reader.binary()?).into_owned()),
                (2, T_BINARY) => value = String::from_utf8_lossy(reader.binary()?).into_owned(),
                _ => reader.skip(field_type)?,
            }
        }
        if let Some(key) = key {
            out.push((key, value));
        }
    }
    Ok(())
}

fn parse_element_list(
    reader: &mut ThriftReader<'_>,
    out: &mut Vec<RawSchemaElement>,
) -> Result<()> {
    let (element_type, size) = reader.list_header()?;
    if element_type != T_STRUCT {
        return Err(Error::invalid_input("schema list is not a struct list"));
    }
    for _ in 0..size {
        out.push(parse_schema_element(reader)?);
    }
    Ok(())
}

#[derive(Default)]
struct RawSchemaElement {
    type_code: Option<i32>,
    type_length: Option<i32>,
    repetition: Option<i32>,
    name: String,
    num_children: i32,
    converted_type: Option<i32>,
    field_id: Option<i32>,
}

fn parse_schema_element(reader: &mut ThriftReader<'_>) -> Result<RawSchemaElement> {
    let mut element = RawSchemaElement::default();
    reader.enter_struct();
    while let Ok((field_id, field_type)) = reader.field_header() {
        if field_type == super::thrift::T_STOP {
            return Ok(element);
        }
        match (field_id, field_type) {
            (1, T_I32) => element.type_code = Some(reader.zigzag()? as i32),
            (2, T_I32) => element.type_length = Some(reader.zigzag()? as i32),
            (3, T_I32) => element.repetition = Some(reader.zigzag()? as i32),
            (4, T_BINARY) => element.name = String::from_utf8_lossy(reader.binary()?).into_owned(),
            (5, T_I32) => element.num_children = reader.zigzag()? as i32,
            (6, T_I32) => element.converted_type = Some(reader.zigzag()? as i32),
            (9, T_I32) => element.field_id = Some(reader.zigzag()? as i32),
            _ => reader.skip(field_type)?,
        }
    }
    Err(Error::invalid_input("schema element not terminated"))
}

/// Walk the preorder element list into leaves, accumulating def/rep levels
/// and carrying ancestor field ids down (storage-v2 sets field_id on the top
/// group; parquet LIST nests `element` under a `list` group). The first
/// element is the schema root; its name is not part of leaf paths.
fn build_leaves(raw: Vec<RawSchemaElement>, leaves: &mut Vec<SchemaLeaf>) -> Result<()> {
    #[derive(Clone)]
    struct Frame {
        remaining_children: i32,
        def: u16,
        rep: u16,
        field_id: Option<i64>,
        pushed_name: bool,
        repetition: Repetition,
    }

    let mut stack: Vec<Frame> = Vec::new();
    let mut path: Vec<String> = Vec::new();

    if raw.is_empty() {
        return Err(Error::invalid_input("empty parquet schema"));
    }

    let mut index = 0usize;
    loop {
        let element = raw
            .get(index)
            .ok_or_else(|| Error::invalid_input("schema tree truncated"))?;

        let is_root = stack.is_empty() && index == 0;
        let pushed_name = !is_root;
        if pushed_name {
            path.push(element.name.clone());
        }

        let repetition = match element.repetition {
            None | Some(0) => Repetition::Required,
            Some(1) => Repetition::Optional,
            Some(2) => Repetition::Repeated,
            Some(other) => {
                return Err(Error::invalid_input(format!(
                    "unknown repetition type {other}"
                )))
            }
        };
        // The schema root's repetition is meaningless (writers mark it
        // REPEATED by convention); readers must not count it into levels.
        let repetition = if is_root { Repetition::Required } else { repetition };
        let def = stack.last().map(|f| f.def).unwrap_or(0)
            + u16::from(repetition != Repetition::Required);
        let rep = stack.last().map(|f| f.rep).unwrap_or(0)
            + u16::from(repetition == Repetition::Repeated);
        let field_id = element
            .field_id
            .map(i64::from)
            .or_else(|| stack.last().and_then(|f| f.field_id));

        if element.num_children > 0 {
            stack.push(Frame {
                remaining_children: element.num_children,
                def,
                rep,
                field_id,
                pushed_name,
                repetition,
            });
            index += 1;
            continue;
        }

        // Leaf. For LIST columns the LIST group sits two levels up
        // (group -> repeated `list` -> element).
        let type_code = element.type_code.ok_or_else(|| {
            Error::invalid_input(format!("leaf {} has no physical type", element.name))
        })?;
        let list_group_optional = rep > 0
            && stack
                .len()
                .checked_sub(2)
                .and_then(|position| stack.get(position))
                .is_some_and(|frame| frame.repetition == Repetition::Optional);
        let logical = logical_kind(element.converted_type);
        leaves.push(SchemaLeaf {
            path: path.clone(),
            physical: PhysicalType::from_code(type_code, element.type_length)?,
            logical,
            max_def: def,
            max_rep: rep,
            field_id,
            list_group_optional,
        });
        if pushed_name {
            path.pop();
        }

        // Unwind finished groups. `index` must end at the next unconsumed
        // element: exactly one advance for this whole unwind, no matter how
        // many exhausted groups pop (a flat schema pops none, a LIST column
        // pops two).
        while let Some(frame) = stack.last_mut() {
            frame.remaining_children -= 1;
            if frame.remaining_children > 0 {
                break;
            }
            if frame.pushed_name {
                path.pop();
            }
            stack.pop();
        }
        index += 1;
        if stack.is_empty() {
            // The schema is one tree rooted at the first element; once the
            // root unwinds, everything is consumed.
            break;
        }
    }
    Ok(())
}

fn logical_kind(converted_type: Option<i32>) -> LogicalKind {
    match converted_type {
        Some(0) | Some(4) | Some(19) => LogicalKind::String, // UTF8 / ENUM / JSON
        Some(15) => LogicalKind::Int { bits: 8, signed: true },
        Some(16) => LogicalKind::Int { bits: 16, signed: true },
        Some(17) => LogicalKind::Int { bits: 32, signed: true },
        Some(18) => LogicalKind::Int { bits: 64, signed: true },
        Some(11) => LogicalKind::Int { bits: 8, signed: false },
        Some(12) => LogicalKind::Int { bits: 16, signed: false },
        Some(13) => LogicalKind::Int { bits: 32, signed: false },
        Some(14) => LogicalKind::Int { bits: 64, signed: false },
        _ => LogicalKind::None,
    }
}

fn parse_row_groups(reader: &mut ThriftReader<'_>, out: &mut Vec<RowGroupMeta>) -> Result<()> {
    let (element_type, size) = reader.list_header()?;
    if element_type != T_STRUCT {
        return Err(Error::invalid_input("row_groups is not a struct list"));
    }
    for _ in 0..size {
        out.push(parse_row_group(reader)?);
    }
    Ok(())
}

fn parse_row_group(reader: &mut ThriftReader<'_>) -> Result<RowGroupMeta> {
    let mut columns = Vec::new();
    let mut num_rows: Option<u64> = None;
    reader.enter_struct();
    while let Ok((field_id, field_type)) = reader.field_header() {
        if field_type == super::thrift::T_STOP {
            return Ok(RowGroupMeta {
                num_rows: num_rows.unwrap_or(0),
                columns,
            });
        }
        match (field_id, field_type) {
            (1, T_LIST) => {
                let (element_type, size) = reader.list_header()?;
                if element_type != T_STRUCT {
                    return Err(Error::invalid_input("columns is not a struct list"));
                }
                for _ in 0..size {
                    columns.push(parse_column_chunk(reader)?);
                }
            }
            (3, T_I64) => num_rows = Some(reader.zigzag()? as u64),
            _ => reader.skip(field_type)?,
        }
    }
    Err(Error::invalid_input("row group not terminated"))
}

fn parse_column_chunk(reader: &mut ThriftReader<'_>) -> Result<ColumnChunkMeta> {
    let mut meta: Option<ColumnChunkMeta> = None;
    reader.enter_struct();
    while let Ok((field_id, field_type)) = reader.field_header() {
        if field_type == super::thrift::T_STOP {
            return meta.ok_or_else(|| Error::invalid_input("column chunk has no meta_data"));
        }
        if field_id == 3 {
            meta = Some(parse_column_meta(reader)?);
        } else {
            reader.skip(field_type)?;
        }
    }
    Err(Error::invalid_input("column chunk not terminated"))
}

fn parse_column_meta(reader: &mut ThriftReader<'_>) -> Result<ColumnChunkMeta> {
    let mut codec = Codec::Uncompressed;
    let mut num_values: i64 = 0;
    let mut data_page_offset: i64 = 0;
    let mut dictionary_page_offset: Option<i64> = None;
    let mut total_compressed_size: i64 = 0;
    reader.enter_struct();
    while let Ok((field_id, field_type)) = reader.field_header() {
        if field_type == super::thrift::T_STOP {
            return Ok(ColumnChunkMeta {
                codec,
                num_values: num_values as u64,
                data_page_offset,
                dictionary_page_offset,
                total_compressed_size,
            });
        }
        match (field_id, field_type) {
            (4, T_I32) => codec = Codec::from_code(reader.zigzag()? as i32),
            (5, T_I64) => num_values = reader.zigzag()?,
            (7, T_I64) => total_compressed_size = reader.zigzag()?,
            (9, T_I64) => data_page_offset = reader.zigzag()?,
            (11, T_I64) => dictionary_page_offset = Some(reader.zigzag()?),
            _ => reader.skip(field_type)?,
        }
    }
    Err(Error::invalid_input("column meta not terminated"))
}
