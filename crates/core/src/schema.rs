//! Milvus collection-schema parsing and the Milvus -> DuckDB type mapping
//! for `milvus_backup_scan`.
//!
//! Field JSON mixes camelCase (`fieldID`, `autoID`) with snake_case; both are
//! accepted. Type mapping decisions (agreed design):
//! * FloatVector -> FLOAT[dim] (duckdb-neo fixed-size ARRAY);
//! * SparseFloatVector -> STRUCT(indices UINTEGER[], values FLOAT[]); the
//!   on-disk blob is consecutive little-endian `<u32 index><f32 value>` pairs;
//! * native Array -> LIST(element) when the schema carries element_type;
//!   legacy protobuf-backed arrays degrade to BLOB;
//! * JSON/String/VarChar -> VARCHAR (lossless; `::JSON` / `->>` still work);
//! * Binary/Float16/BFloat16/Int8 vectors and Geometry -> BLOB passthrough;
//! * anything else is a bind-time error rather than a guessed type.

use duckdb_neo::{Error, Result};
use serde::Deserialize;

/// One parsed schema field of a collection.
#[derive(Debug, Clone)]
pub(crate) struct MilvusField {
    pub(crate) field_id: i64,
    pub(crate) name: String,
    pub(crate) data_type: i64,
    /// Kept for future column-visibility decisions; DuckDB table columns are
    /// always nullable-capable.
    #[allow(dead_code)]
    pub(crate) nullable: bool,
    pub(crate) dim: Option<i64>,
    pub(crate) element_type: Option<i64>,
    pub(crate) is_primary_key: bool,
}

/// The DuckDB shape a Milvus field maps to; scan turns this into a
/// `LogicalType` through the connection link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TargetKind {
    Boolean,
    TinyInt,
    SmallInt,
    Int,
    BigInt,
    Float,
    Double,
    VarChar,
    /// FLOAT[dim].
    FloatArray(usize),
    /// STRUCT(indices UINTEGER[], values FLOAT[]).
    SparseStruct,
    /// LIST of an element kind (native Array fields).
    List(Box<TargetKind>),
    Blob,
}

/// Milvus `DataType` codes we know how to read.
pub(crate) const T_BOOL: i64 = 1;
pub(crate) const T_INT8: i64 = 2;
pub(crate) const T_INT16: i64 = 3;
pub(crate) const T_INT32: i64 = 4;
pub(crate) const T_INT64: i64 = 5;
pub(crate) const T_FLOAT: i64 = 10;
pub(crate) const T_DOUBLE: i64 = 11;
pub(crate) const T_STRING: i64 = 20;
pub(crate) const T_VARCHAR: i64 = 21;
pub(crate) const T_ARRAY: i64 = 22;
pub(crate) const T_JSON: i64 = 23;
pub(crate) const T_GEOMETRY: i64 = 24;
pub(crate) const T_TEXT: i64 = 25;
pub(crate) const T_BINARY_VECTOR: i64 = 100;
pub(crate) const T_FLOAT_VECTOR: i64 = 101;
pub(crate) const T_FLOAT16_VECTOR: i64 = 102;
pub(crate) const T_BFLOAT16_VECTOR: i64 = 103;
pub(crate) const T_SPARSE_FLOAT_VECTOR: i64 = 104;
pub(crate) const T_INT8_VECTOR: i64 = 105;

/// Parse `schema.fields` entries.
pub(crate) fn parse_fields(raw: Option<Vec<RawField>>) -> Result<Vec<MilvusField>> {
    raw.unwrap_or_default()
        .into_iter()
        .map(|field| {
            let field_id = field
                .field_id
                .ok_or_else(|| Error::invalid_input("schema field without fieldID"))?;
            let name = field
                .name
                .ok_or_else(|| Error::invalid_input(format!("field {field_id} without name")))?;
            let data_type = field
                .data_type
                .ok_or_else(|| Error::invalid_input(format!("field {name} without data_type")))?;
            let dim = field
                .type_params
                .unwrap_or_default()
                .into_iter()
                .filter(|kv| kv.key.as_deref() == Some("dim"))
                .find_map(|kv| kv.value.and_then(|value| value.parse::<i64>().ok()));
            Ok(MilvusField {
                field_id,
                name,
                data_type,
                nullable: field.nullable.unwrap_or(false),
                dim,
                element_type: field.element_type,
                is_primary_key: field.is_primary_key.unwrap_or(false),
            })
        })
        .collect()
}

/// Map a Milvus data type (and its dim/element type) to the DuckDB target.
pub(crate) fn target_kind(field: &MilvusField) -> Result<TargetKind> {
    target_kind_of_code(field.data_type, field.dim, field.element_type, &field.name)
}

pub(crate) fn target_kind_of_code(
    code: i64,
    dim: Option<i64>,
    element_type: Option<i64>,
    field_name: &str,
) -> Result<TargetKind> {
    match code {
        T_BOOL => Ok(TargetKind::Boolean),
        T_INT8 => Ok(TargetKind::TinyInt),
        T_INT16 => Ok(TargetKind::SmallInt),
        T_INT32 => Ok(TargetKind::Int),
        T_INT64 => Ok(TargetKind::BigInt),
        T_FLOAT => Ok(TargetKind::Float),
        T_DOUBLE => Ok(TargetKind::Double),
        T_STRING | T_VARCHAR | T_JSON | T_TEXT => Ok(TargetKind::VarChar),
        T_GEOMETRY | T_BINARY_VECTOR | T_FLOAT16_VECTOR | T_BFLOAT16_VECTOR | T_INT8_VECTOR => {
            // No native DuckDB counterpart; the raw bytes are lossless and
            // user-decodable (float16 = 2*dim LE bytes, binary = (dim+7)/8).
            Ok(TargetKind::Blob)
        }
        T_FLOAT_VECTOR => {
            let dim = dim.ok_or_else(|| {
                Error::invalid_input(format!(
                    "field {field_name} is a FloatVector but the schema has no dim type param"
                ))
            })?;
            if dim <= 0 {
                return Err(Error::invalid_input(format!(
                    "field {field_name} has invalid vector dim {dim}"
                )));
            }
            Ok(TargetKind::FloatArray(dim as usize))
        }
        T_SPARSE_FLOAT_VECTOR => Ok(TargetKind::SparseStruct),
        T_ARRAY => match element_type {
            Some(element) => {
                // Element types are scalar or vector codes; dim is not
                // carried per element in the schema, so vector elements
                // degrade to BLOB.
                let element_kind = target_kind_of_code(element, None, None, field_name)?;
                Ok(TargetKind::List(Box::new(match element_kind {
                    TargetKind::FloatArray(_) => TargetKind::Blob,
                    other => other,
                })))
            }
            // Pre-2.5 arrays store a protobuf ScalarField blob.
            None => Ok(TargetKind::Blob),
        },
        other => Err(Error::invalid_input(format!(
            "field {field_name} has Milvus DataType {other}, which this build cannot \
             decode yet (unknown on-disk layout)"
        ))),
    }
}

// --- raw JSON shapes -------------------------------------------------------

#[derive(Deserialize, Clone)]
pub(crate) struct RawField {
    pub(crate) name: Option<String>,
    #[serde(alias = "fieldID", alias = "field_id")]
    pub(crate) field_id: Option<i64>,
    pub(crate) is_primary_key: Option<bool>,
    pub(crate) data_type: Option<i64>,
    pub(crate) element_type: Option<i64>,
    pub(crate) nullable: Option<bool>,
    pub(crate) type_params: Option<Vec<KeyValuePair>>,
}

#[derive(Deserialize, Clone)]
pub(crate) struct KeyValuePair {
    pub(crate) key: Option<String>,
    pub(crate) value: Option<String>,
}
