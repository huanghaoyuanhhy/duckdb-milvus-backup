//! Delta-log decoding into tombstones.
//!
//! Milvus suppresses an inserted row (pk, insert_ts) when a delete tombstone
//! (pk, delete_ts) exists with the same pk and `delete_ts > insert_ts`
//! (strict; verified against milvus segcore `DeletedRecord`). Because TSOs
//! are monotonic per pk, keeping the maximum delete ts per pk is sufficient.
//!
//! Three on-disk deltalog shapes exist across writer generations (all
//! verified against real backups / the milvus writer sources):
//! * storage-v1 envelope, single UTF8 column of `{"pk":..,"ts":..,"pkType":..}`
//!   JSON rows (column name drifts: `val` / `delta` / field-id string);
//! * storage-v1 envelope with extras `version == "MULTI_FIELD"`: 2-column
//!   {pk, ts} parquet;
//! * storage-v2: bare 2-column {pk, ts} parquet (pk located positionally, as
//!   milvus-backup's own l0compact reader does).

use std::collections::HashMap;

use duckdb_neo::{Error, Result};
use serde::Deserialize;

use crate::binlog::{self, BinlogKind};
use crate::parquet::{self, ChunkReader, ColumnData, FileView, PhysicalType, Values};

/// A primary key value.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) enum Pk {
    Int(i64),
    Str(String),
}

/// pk -> maximum delete timestamp.
#[derive(Debug, Default)]
pub(crate) struct Tombstones {
    by_pk: HashMap<Pk, u64>,
}

impl Tombstones {
    pub(crate) fn is_empty(&self) -> bool {
        self.by_pk.is_empty()
    }

    /// milvus semantics: a row dies iff a tombstone for its pk carries a
    /// strictly larger timestamp.
    pub(crate) fn is_deleted(&self, pk: &Pk, insert_ts: u64) -> bool {
        self.by_pk
            .get(pk)
            .is_some_and(|&delete_ts| delete_ts > insert_ts)
    }

    pub(crate) fn record(&mut self, pk: Pk, delete_ts: u64) {
        self.by_pk
            .entry(pk)
            .and_modify(|current| {
                if delete_ts > *current {
                    *current = delete_ts;
                }
            })
            .or_insert(delete_ts);
    }

    /// Decode one deltalog file (any of the three shapes) into entries.
    pub(crate) fn read_file(view: &FileView) -> Result<Vec<(Pk, u64)>> {
        match binlog::sniff(view)? {
            BinlogKind::BareParquet => read_pk_ts_parquet(view),
            BinlogKind::Envelope => {
                let envelope = binlog::parse_envelope(view)?;
                let mut entries = Vec::new();
                for index in 0..envelope.events.len() {
                    let payload = envelope.event_payload(view, index)?;
                    if envelope.multi_field {
                        entries.extend(read_pk_ts_parquet(&payload)?);
                    } else {
                        entries.extend(read_json_variant(&payload)?);
                    }
                }
                Ok(entries)
            }
        }
    }
}

#[derive(Deserialize)]
struct DeleteLogJson {
    pk: serde_json::Value,
    ts: u64,
}

/// JSON deltalog variant: one UTF8 string per row.
fn read_json_variant(payload: &FileView) -> Result<Vec<(Pk, u64)>> {
    let meta = parquet::read_footer(payload)?;
    let string_leaf = meta
        .leaves
        .iter()
        .position(|leaf| matches!(leaf.physical, PhysicalType::ByteArray))
        .ok_or_else(|| Error::invalid_input("deltalog JSON payload has no string column"))?;
    let view = payload.slice(0, payload.size())?;
    let mut entries = Vec::new();
    for group in &meta.row_groups {
        let mut reader = ChunkReader::open(
            view.clone(),
            meta.leaves[string_leaf].clone(),
            group.columns[string_leaf].clone(),
        )?;
        let data = parquet::read_chunk(&mut reader)?;
        entries.extend(json_rows_to_entries(&data)?);
    }
    Ok(entries)
}

fn json_rows_to_entries(data: &ColumnData) -> Result<Vec<(Pk, u64)>> {
    let Values::Bytes { data: bytes, offsets } = &data.values else {
        return Err(Error::invalid_input(
            "deltalog JSON payload column is not byte array",
        ));
    };
    let mut entries = Vec::with_capacity(data.rows());
    let mut value_index = 0usize;
    for row in 0..data.rows() {
        if !data.is_valid(row) {
            return Err(Error::invalid_input(
                "deltalog JSON payload contains a NULL row",
            ));
        }
        let start = offsets[value_index] as usize;
        let end = offsets[value_index + 1] as usize;
        value_index += 1;
        let text = std::str::from_utf8(&bytes[start..end])
            .map_err(|error| Error::invalid_input(format!("deltalog row is not UTF-8: {error}")))?;
        let row: DeleteLogJson = serde_json::from_str(text)
            .map_err(|error| Error::invalid_input(format!("bad deltalog JSON row: {error}")))?;
        entries.push((json_pk(row.pk)?, row.ts));
    }
    Ok(entries)
}

/// The JSON `pk` shape decides the key type: number => Int64, string => VarChar.
fn json_pk(pk: serde_json::Value) -> Result<Pk> {
    match pk {
        serde_json::Value::Number(number) => number
            .as_i64()
            .map(Pk::Int)
            .ok_or_else(|| Error::invalid_input("deltalog pk number does not fit i64")),
        serde_json::Value::String(text) => Ok(Pk::Str(text)),
        other => Err(Error::invalid_input(format!(
            "unsupported deltalog pk JSON value: {other}"
        ))),
    }
}

/// {pk, ts} two-column parquet (storage-v2 deltalog and MULTI_FIELD variant).
/// Columns are located positionally like milvus-backup's own reader.
fn read_pk_ts_parquet(payload: &FileView) -> Result<Vec<(Pk, u64)>> {
    let meta = parquet::read_footer(payload)?;
    if meta.leaves.len() < 2 {
        return Err(Error::invalid_input(format!(
            "deltalog {{pk,ts}} payload has {} columns, expected 2",
            meta.leaves.len()
        )));
    }
    let view = payload.slice(0, payload.size())?;
    let mut entries = Vec::new();
    for group in &meta.row_groups {
        let mut pk_reader = ChunkReader::open(view.clone(), meta.leaves[0].clone(), group.columns[0].clone())?;
        let pk_data = parquet::read_chunk(&mut pk_reader)?;
        let mut ts_reader = ChunkReader::open(view.clone(), meta.leaves[1].clone(), group.columns[1].clone())?;
        let ts_data = parquet::read_chunk(&mut ts_reader)?;

        let pks = pk_column(&pk_data)?;
        let tss = ts_column(&ts_data)?;
        if pks.len() != tss.len() {
            return Err(Error::invalid_input(format!(
                "deltalog pk/ts row count mismatch: {} vs {}",
                pks.len(),
                tss.len()
            )));
        }
        for (index, pk) in pks.into_iter().enumerate() {
            if !pk_data.is_valid(index) || !ts_data.is_valid(index) {
                return Err(Error::invalid_input("deltalog contains a NULL pk/ts row"));
            }
            entries.push((pk, tss[index]));
        }
    }
    Ok(entries)
}

fn pk_column(data: &ColumnData) -> Result<Vec<Pk>> {
    match &data.values {
        Values::Int64(values) => Ok(values.iter().map(|&value| Pk::Int(value)).collect()),
        Values::Bytes { data: bytes, offsets } => {
            let mut pks = Vec::with_capacity(offsets.len().saturating_sub(1));
            for window in offsets.windows(2) {
                let text = std::str::from_utf8(&bytes[window[0] as usize..window[1] as usize])
                    .map_err(|error| {
                        Error::invalid_input(format!("deltalog string pk is not UTF-8: {error}"))
                    })?;
                pks.push(Pk::Str(text.to_string()));
            }
            Ok(pks)
        }
        other => Err(Error::invalid_input(format!(
            "unsupported deltalog pk column type {other:?} (Int64 or UTF8 expected)"
        ))),
    }
}

fn ts_column(data: &ColumnData) -> Result<Vec<u64>> {
    match &data.values {
        Values::Int64(values) => Ok(values.iter().map(|&value| value as u64).collect()),
        other => Err(Error::invalid_input(format!(
            "unsupported deltalog ts column type {other:?} (Int64 expected)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::parquet::{BlockSource, FileView};

    fn l0_delta_view() -> FileView {
        let path = format!(
            "{}/testdata/l0_delta_json.envelope",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = Arc::new(std::fs::read(path).expect("testdata"));
        let len = bytes.len() as u64;
        FileView::new(Arc::new(BlockSource::Mem(bytes)), 0, len)
    }

    #[test]
    fn reads_the_json_variant_strictly_after_insert() {
        let entries = Tombstones::read_file(&l0_delta_view()).unwrap();
        assert_eq!(entries.len(), 9, "the real L0 delta deletes pk 0..8");
        let mut tombstones = Tombstones::default();
        for (pk, ts) in entries {
            tombstones.record(pk, ts);
        }
        let delete_ts = 456769326808039425u64;
        // Same ts does NOT delete (strict >), matching milvus segcore.
        assert!(!tombstones.is_deleted(&Pk::Int(0), delete_ts));
        assert!(tombstones.is_deleted(&Pk::Int(0), delete_ts - 1));
        assert!(!tombstones.is_deleted(&Pk::Int(9), delete_ts));
        assert!(!tombstones.is_deleted(&Pk::Str("0".to_string()), 0));
    }

    #[test]
    fn records_keep_the_maximum_delete_ts() {
        let mut tombstones = Tombstones::default();
        tombstones.record(Pk::Int(1), 50);
        tombstones.record(Pk::Int(1), 70);
        tombstones.record(Pk::Int(1), 60);
        assert!(tombstones.is_deleted(&Pk::Int(1), 69));
        assert!(!tombstones.is_deleted(&Pk::Int(1), 70));
    }
}
