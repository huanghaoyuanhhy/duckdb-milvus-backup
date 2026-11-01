//! Minimal Avro Object Container File (OCF) reader for snapshot manifests.
//!
//! Snapshot-format backups embed two OCF payloads: one avro record per
//! segment (`snapshots/<coll>/manifests/<snap>/<seg>.avro`, written by
//! milvus-backup, format version >= 6 carries the full writer schema) and the
//! milvus-storage dataset manifest (`{base}/_metadata/manifest-<ver>.avro`,
//! written by milvus-storage's C++ `DataFileWriter`). Both are single-record
//! containers, but this reader decodes any block count generically against
//! the embedded writer schema — no compiled-in schema matrix, so future field
//! additions decode without changes (unknown fields are ignored by the
//! extraction layer).
//!
//! Only the codecs the real writers emit are supported: `null` (both the Go
//! ocf encoder default and milvus-storage) and `snappy`. Values decode into a
//! small dynamic tree; record fields resolve by name at extraction time,
//! which is the avro schema-resolution contract for added fields.

use std::collections::HashMap;

use duckdb_neo::{Error, Result};

pub(crate) const OCF_MAGIC: &[u8; 4] = b"Obj\x01";

/// A decoded avro value. Unions resolve to whichever branch held the value;
/// the variant carries the interpretation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AvroValue {
    Null,
    Bool(bool),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    Bytes(Vec<u8>),
    String(String),
    Array(Vec<AvroValue>),
    /// Decoded as insertion-ordered pairs; avro maps are unordered in
    /// principle but the writers we read emit small maps.
    Map(Vec<(String, AvroValue)>),
    /// Enum symbol index within the writer schema's symbol list.
    Enum(i32),
}

impl AvroValue {
    pub(crate) fn as_long(&self) -> Option<i64> {
        match self {
            AvroValue::Int(value) => Some(*value as i64),
            AvroValue::Long(value) => Some(*value),
            _ => None,
        }
    }

    pub(crate) fn as_str(&self) -> Option<&str> {
        match self {
            AvroValue::String(value) => Some(value),
            _ => None,
        }
    }

    pub(crate) fn as_array(&self) -> Option<&[AvroValue]> {
        match self {
            AvroValue::Array(values) => Some(values),
            _ => None,
        }
    }

    /// Field of a record value by name.
    pub(crate) fn field(&self, name: &str) -> Option<&AvroValue> {
        match self {
            AvroValue::Map(_) => self
                .as_map()?
                .iter()
                .find(|(key, _)| key == name)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    fn as_map(&self) -> Option<&[(String, AvroValue)]> {
        match self {
            AvroValue::Map(pairs) => Some(pairs),
            _ => None,
        }
    }
}

/// Decode ONE record from a raw (non-OCF) avro binary payload using the
/// given writer schema JSON. Snapshot format versions < 6 store their segment
/// manifests this way: no container, no embedded schema — the schema comes
/// from the snapshot metadata's format_version.
pub(crate) fn decode_raw_record(data: &[u8], schema_json: &str) -> Result<AvroValue> {
    let schema: serde_json::Value = serde_json::from_str(schema_json)
        .map_err(|error| Error::invalid_input(format!("invalid avro schema JSON: {error}")))?;
    let mut names = HashMap::new();
    let schema = parse_schema(&schema, &mut names)?;
    let mut cursor = Cursor { data, pos: 0 };
    read_value(&mut cursor, &schema)
}

// --- schema ------------------------------------------------------------------

#[derive(Debug, Clone)]
enum AvroSchema {
    Null,
    Boolean,
    Int,
    Long,
    Float,
    Double,
    Bytes,
    String,
    Array(Box<AvroSchema>),
    Map(Box<AvroSchema>),
    Enum,
    Fixed,
    /// Union of branch schemas; branch 0 index precedes the value on the wire.
    Union(Vec<AvroSchema>),
    Record(Vec<(String, AvroSchema)>),
}

/// Parse the JSON schema embedded in an OCF header. Named types collect into
/// `names` so later string references (e.g. `"type": "AvroMsgPosition"`)
/// resolve.
fn parse_schema(value: &serde_json::Value, names: &mut HashMap<String, AvroSchema>) -> Result<AvroSchema> {
    if let Some(text) = value.as_str() {
        return named_or_primitive(text, names);
    }
    if let Some(branches) = value.as_array() {
        return Ok(AvroSchema::Union(
            branches
                .iter()
                .map(|branch| parse_schema(branch, names))
                .collect::<Result<Vec<_>>>()?,
        ));
    }
    let object = value.as_object().ok_or_else(|| {
        Error::invalid_input(format!("unsupported avro schema shape: {value}"))
    })?;
    let type_name = object
        .get("type")
        .and_then(|t| t.as_str())
        .ok_or_else(|| Error::invalid_input("avro schema object without a type"))?;
    match type_name {
        "record" => {
            let name = object
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or_default()
                .to_string();
            let mut fields = Vec::new();
            if let Some(list) = object.get("fields").and_then(|f| f.as_array()) {
                for field in list {
                    let field_name = field
                        .get("name")
                        .and_then(|n| n.as_str())
                        .unwrap_or_default()
                        .to_string();
                    let field_type = parse_schema(
                        field.get("type").unwrap_or(&serde_json::Value::Null),
                        names,
                    )?;
                    fields.push((field_name, field_type));
                }
            }
            if !name.is_empty() {
                names.insert(name, AvroSchema::Record(fields.clone()));
            }
            Ok(AvroSchema::Record(fields))
        }
        "array" => {
            let items = parse_schema(
                object.get("items").unwrap_or(&serde_json::Value::Null),
                names,
            )?;
            Ok(AvroSchema::Array(Box::new(items)))
        }
        "map" => {
            let values = parse_schema(
                object.get("values").unwrap_or(&serde_json::Value::Null),
                names,
            )?;
            Ok(AvroSchema::Map(Box::new(values)))
        }
        "enum" => Ok(AvroSchema::Enum),
        "fixed" => Ok(AvroSchema::Fixed),
        primitive => named_or_primitive(primitive, names),
    }
}

fn named_or_primitive(text: &str, names: &HashMap<String, AvroSchema>) -> Result<AvroSchema> {
    match text {
        "null" => Ok(AvroSchema::Null),
        "boolean" => Ok(AvroSchema::Boolean),
        "int" => Ok(AvroSchema::Int),
        "long" => Ok(AvroSchema::Long),
        "float" => Ok(AvroSchema::Float),
        "double" => Ok(AvroSchema::Double),
        "bytes" => Ok(AvroSchema::Bytes),
        "string" => Ok(AvroSchema::String),
        // Named references resolve eagerly (cloned); manifest schemas are
        // small and non-recursive, so this cannot loop.
        name => names
            .get(name)
            .cloned()
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "avro schema references unknown named type {name:?}"
                ))
            }),
    }
}

// --- OCF container -----------------------------------------------------------

/// Decode every record of an OCF payload against its embedded writer schema.
pub(crate) fn read_ocf_records(data: &[u8]) -> Result<Vec<AvroValue>> {
    if data.len() < 4 || &data[..4] != OCF_MAGIC {
        return Err(Error::invalid_input(
            "avro payload is not an object container file (missing Obj\x01 magic)",
        ));
    }
    let mut cursor = Cursor { data, pos: 4 };

    // Header metadata: map<string, bytes> under the fixed header schema.
    let mut schema_json: Option<Vec<u8>> = None;
    let mut codec = "null".to_string();
    let header_schema = AvroSchema::Map(Box::new(AvroSchema::Bytes));
    let metadata = read_value(&mut cursor, &header_schema)?;
    if let AvroValue::Map(pairs) = metadata {
        for (key, value) in pairs {
            let AvroValue::Bytes(bytes) = value else {
                continue;
            };
            match key.as_str() {
                "avro.schema" => schema_json = Some(bytes),
                "avro.codec" => codec = String::from_utf8_lossy(&bytes).into_owned(),
                _ => {}
            }
        }
    }
    let schema_json = schema_json.ok_or_else(|| {
        Error::invalid_input("avro container header lacks the avro.schema metadata key")
    })?;
    let schema: serde_json::Value = serde_json::from_slice(&schema_json)
        .map_err(|error| Error::invalid_input(format!("invalid avro schema JSON: {error}")))?;
    let mut names = HashMap::new();
    let schema = parse_schema(&schema, &mut names)?;
    let codec = match codec.as_str() {
        "null" => Codec::Null,
        "snappy" => Codec::Snappy,
        other => {
            return Err(Error::invalid_input(format!(
                "unsupported avro container codec {other:?} (readers emit null or snappy)"
            )));
        }
    };

    // 16-byte sync marker closes the header.
    let sync_start = cursor.pos;
    if cursor.data.len() < sync_start + 16 {
        return Err(Error::invalid_input("avro header truncated before sync marker"));
    }
    let sync = &cursor.data[sync_start..sync_start + 16];
    cursor.pos = sync_start + 16;

    let mut records = Vec::new();
    while cursor.pos < cursor.data.len() {
        let count = cursor.read_long()?;
        if count < 0 {
            return Err(Error::invalid_input(
                "avro block count is negative (skip blocks are not used by snapshot manifests)",
            ));
        }
        let block_size = cursor.read_long()?;
        if block_size < 0 {
            return Err(Error::invalid_input("negative avro block size"));
        }
        let block_size = block_size as usize;
        let block_start = cursor.pos;
        if block_start + block_size + 16 > cursor.data.len() {
            return Err(Error::invalid_input("avro block exceeds the container size"));
        }
        let block = &cursor.data[block_start..block_start + block_size];
        cursor.pos = block_start + block_size;
        let stored_sync = &cursor.data[cursor.pos..cursor.pos + 16];
        if stored_sync != sync {
            return Err(Error::invalid_input("avro block sync marker mismatch"));
        }
        cursor.pos += 16;

        let decoded = codec.decode(block)?;
        let mut block_cursor = Cursor { data: &decoded, pos: 0 };
        for _ in 0..count {
            records.push(read_value(&mut block_cursor, &schema)?);
        }
    }
    Ok(records)
}

#[derive(Clone, Copy)]
enum Codec {
    Null,
    Snappy,
}

impl Codec {
    fn decode(&self, block: &[u8]) -> Result<Vec<u8>> {
        match self {
            Codec::Null => Ok(block.to_vec()),
            Codec::Snappy => {
                // Avro snappy blocks end with a 4-byte big-endian CRC32 of
                // the uncompressed payload.
                let payload = block
                    .get(..block.len().saturating_sub(4))
                    .ok_or_else(|| Error::invalid_input("avro snappy block too short"))?;
                let mut decoder = snap::raw::Decoder::new();
                let out = decoder
                    .decompress_vec(payload)
                    .map_err(|error| Error::invalid_input(format!("avro snappy block: {error}")))?;
                let expected = u32::from_be_bytes(
                    block[block.len() - 4..]
                        .try_into()
                        .expect("4 bytes checked above"),
                );
                if crc32(&out) != expected {
                    return Err(Error::invalid_input("avro snappy block CRC32 mismatch"));
                }
                Ok(out)
            }
        }
    }
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// --- value decoding ----------------------------------------------------------

struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Cursor<'_> {
    fn byte(&mut self) -> Result<u8> {
        let byte = self
            .data
            .get(self.pos)
            .copied()
            .ok_or_else(|| Error::invalid_input("avro payload truncated"))?;
        self.pos += 1;
        Ok(byte)
    }

    /// Zigzag variable-length integer (avro int/long encoding).
    fn varlong(&mut self) -> Result<i64> {
        let mut value: u64 = 0;
        let mut shift = 0u32;
        loop {
            let byte = self.byte()?;
            if shift >= 64 {
                return Err(Error::invalid_input("avro varint overflow"));
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        Ok(((value >> 1) as i64) ^ -((value & 1) as i64))
    }

    fn read_long(&mut self) -> Result<i64> {
        self.varlong()
    }

    fn read_bytes(&mut self) -> Result<&[u8]> {
        let len = self.varlong()?;
        if len < 0 {
            return Err(Error::invalid_input(format!(
                "negative avro byte length {len} at offset {}",
                self.pos
            )));
        }
        let end = self.pos + len as usize;
        let bytes = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| Error::invalid_input("avro payload truncated"))?;
        self.pos = end;
        Ok(bytes)
    }
}

fn read_value(cursor: &mut Cursor<'_>, schema: &AvroSchema) -> Result<AvroValue> {
    Ok(match schema {
        AvroSchema::Null => AvroValue::Null,
        AvroSchema::Boolean => AvroValue::Bool(cursor.byte()? != 0),
        AvroSchema::Int => AvroValue::Int(cursor.varlong()? as i32),
        AvroSchema::Long => AvroValue::Long(cursor.varlong()?),
        AvroSchema::Float => {
            // Fixed 4-byte little-endian, not length-prefixed.
            let mut bytes = [0u8; 4];
            for slot in &mut bytes {
                *slot = cursor.byte()?;
            }
            AvroValue::Float(f32::from_le_bytes(bytes))
        }
        AvroSchema::Double => {
            let mut bytes = [0u8; 8];
            for slot in &mut bytes {
                *slot = cursor.byte()?;
            }
            AvroValue::Double(f64::from_le_bytes(bytes))
        }
        AvroSchema::Bytes => AvroValue::Bytes(cursor.read_bytes()?.to_vec()),
        AvroSchema::String => {
            let bytes = cursor.read_bytes()?;
            AvroValue::String(String::from_utf8_lossy(bytes).into_owned())
        }
        AvroSchema::Array(items) => {
            let mut values = Vec::new();
            loop {
                let mut count = cursor.varlong()?;
                if count == 0 {
                    break;
                }
                if count < 0 {
                    // Negative count: a byte-size marker precedes the items,
                    // which still follow with |count| entries. The Go and C++
                    // manifest writers both use this single-item block form.
                    let _size = cursor.varlong()?;
                    count = -count;
                }
                for _ in 0..count {
                    values.push(read_value(cursor, items)?);
                }
            }
            AvroValue::Array(values)
        }
        AvroSchema::Map(values) => {
            let mut pairs = Vec::new();
            loop {
                let mut count = cursor.varlong()?;
                if count == 0 {
                    break;
                }
                if count < 0 {
                    let _size = cursor.varlong()?;
                    count = -count;
                }
                for _ in 0..count {
                    let key = match read_value(cursor, &AvroSchema::String)? {
                        AvroValue::String(key) => key,
                        _ => return Err(Error::invalid_input("avro map key is not a string")),
                    };
                    pairs.push((key, read_value(cursor, values)?));
                }
            }
            AvroValue::Map(pairs)
        }
        AvroSchema::Enum => AvroValue::Enum(cursor.varlong()? as i32),
        AvroSchema::Fixed => AvroValue::Bytes(cursor.read_bytes()?.to_vec()),
        AvroSchema::Union(branches) => {
            let index = cursor.varlong()?;
            if index < 0 {
                return Err(Error::invalid_input(format!(
                    "negative avro union branch index {index}"
                )));
            }
            let branch = branches.get(index as usize).ok_or_else(|| {
                Error::invalid_input(format!("avro union branch {index} out of range"))
            })?;
            read_value(cursor, branch)?
        }
        AvroSchema::Record(fields) => {
            let mut pairs = Vec::with_capacity(fields.len());
            for (name, field_schema) in fields {
                pairs.push((name.clone(), read_value(cursor, field_schema)?));
            }
            AvroValue::Map(pairs)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-built OCF for `{"type":"record","name":"R","fields":[...]}` with a
    /// null codec: header, one block, one record.
    fn ocf(schema_json: &str, record_body: &[u8]) -> Vec<u8> {
        ocf_with_codec(schema_json, record_body, None)
    }

    fn ocf_with_codec(schema_json: &str, record_body: &[u8], codec: Option<&str>) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(OCF_MAGIC);
        // metadata map: one entry (avro.schema) + optional codec, then count 0
        let write_bytes = |out: &mut Vec<u8>, bytes: &[u8]| {
            let len = bytes.len() as i64;
            let zigzag = ((len << 1) ^ (len >> 63)) as u64;
            varint(out, zigzag);
            out.extend_from_slice(bytes);
        };
        let pairs = 1 + usize::from(codec.is_some());
        varint(&mut out, ((pairs as i64) << 1) as u64); // zigzag of the pair count
        write_bytes(&mut out, b"avro.schema");
        write_bytes(&mut out, schema_json.as_bytes());
        if let Some(codec) = codec {
            write_bytes(&mut out, b"avro.codec");
            write_bytes(&mut out, codec.as_bytes());
        }
        out.push(0); // map terminator
        out.extend_from_slice(&[0u8; 16]); // sync
        // Block: count 1, size, body, sync. The snappy block is the snap
        // (length-prefixed raw snappy) payload plus the 4-byte CRC32 that
        // avro appends.
        let block = match codec {
            Some("snappy") => {
                let mut encoder = snap::raw::Encoder::new();
                let compressed = encoder.compress_vec(record_body).unwrap();
                compressed
                    .into_iter()
                    .chain(crc32(record_body).to_be_bytes())
                    .collect::<Vec<u8>>()
            }
            _ => record_body.to_vec(),
        };
        out.push(2);
        write_bytes(&mut out, &block);
        out.extend_from_slice(&[0u8; 16]);
        out
    }

    fn varint(out: &mut Vec<u8>, mut value: u64) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if value == 0 {
                break;
            }
        }
    }

    #[test]
    fn decodes_flat_record_fields() {
        // fields: id (long), name (string), payload (bytes). Lengths are
        // zigzag varints, so "ok" (2 bytes) encodes as 0x04.
        let schema = r#"{"type":"record","name":"R","fields":[
            {"name":"id","type":"long"},
            {"name":"name","type":"string"},
            {"name":"payload","type":"bytes"}]}"#;
        let body: &[u8] = &[
            0x2a, // id: 42 -> zigzag 21
            0x04, b'o', b'k', // "ok"
            0x04, 0x01, 0x02, // bytes [1,2]
        ];
        let records = read_ocf_records(&ocf(schema, body)).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].field("id").and_then(AvroValue::as_long), Some(21));
        assert_eq!(records[0].field("name").and_then(AvroValue::as_str), Some("ok"));
        assert_eq!(
            records[0].field("payload"),
            Some(&AvroValue::Bytes(vec![1, 2]))
        );
    }

    #[test]
    fn decodes_nested_arrays_maps_and_unions() {
        // binlogs: array of records {field_id: long, entries: array of long},
        // flags: map<string, long>, maybe: union [null, long]
        let schema = r#"{"type":"record","name":"R","fields":[
            {"name":"binlogs","type":{"type":"array","items":{
                "type":"record","name":"FB","fields":[
                    {"name":"field_id","type":"long"},
                    {"name":"entries","type":{"type":"array","items":"long"}}]}}},
            {"name":"flags","type":{"type":"map","values":"long"}},
            {"name":"maybe","type":["null","long"]}]}"#;
        let body: &[u8] = &[
            0x02, // binlogs array: one element
            0x14, // field_id: 20 -> 10
            0x04, 0x1e, 0x2c, // entries: 2 elements [15, 22]
            0x00, // entries end
            0x00, // binlogs end
            0x02, 0x02, b'k', 0x14, 0x00, // map {"k": 10}, end
            0x02, 0x2a, // union branch 1 (long): 21
        ];
        let records = read_ocf_records(&ocf(schema, body)).unwrap();
        let record = &records[0];
        let binlogs = record.field("binlogs").and_then(AvroValue::as_array).unwrap();
        assert_eq!(binlogs.len(), 1);
        assert_eq!(
            binlogs[0].field("field_id").and_then(AvroValue::as_long),
            Some(10)
        );
        let entries = binlogs[0].field("entries").and_then(AvroValue::as_array).unwrap();
        assert_eq!(
            entries.iter().map(AvroValue::as_long).collect::<Vec<_>>(),
            vec![Some(15), Some(22)]
        );
        assert_eq!(
            record.field("flags").and_then(AvroValue::as_map).unwrap(),
            &[("k".to_string(), AvroValue::Long(10))]
        );
        assert_eq!(record.field("maybe").and_then(AvroValue::as_long), Some(21));
    }

    #[test]
    fn resolves_named_type_references() {
        // pos referenced by name after its definition.
        let schema = r#"{"type":"record","name":"R","fields":[
            {"name":"pos","type":{"type":"record","name":"P","fields":[
                {"name":"ts","type":"long"}]}},
            {"name":"pos2","type":"P"}]}"#;
        let body: &[u8] = &[
            0x14, // pos.ts = 10
            0x2a, // pos2.ts = 21
        ];
        let records = read_ocf_records(&ocf(schema, body)).unwrap();
        assert_eq!(
            records[0].field("pos2").unwrap().field("ts").and_then(AvroValue::as_long),
            Some(21)
        );
    }

    #[test]
    fn snappy_blocks_decode_with_crc_check() {
        let schema = r#"{"type":"record","name":"R","fields":[{"name":"id","type":"long"}]}"#;
        let container = ocf_with_codec(
            schema,
            &[0x2a],
            Some("snappy"),
        );
        let records = read_ocf_records(&container).unwrap();
        assert_eq!(records[0].field("id").and_then(AvroValue::as_long), Some(21));
    }

    #[test]
    fn rejects_bad_magic_and_unknown_codec() {
        let err = read_ocf_records(b"NOPE0000").unwrap_err();
        assert!(err.to_string().contains("not an object container"));

        let container = ocf_with_codec(
            r#"{"type":"record","name":"R","fields":[{"name":"id","type":"long"}]}"#,
            &[0x2a],
            Some("deflate"),
        );
        let err = read_ocf_records(&container).unwrap_err();
        assert!(err.to_string().contains("unsupported avro container codec"));
    }
}
