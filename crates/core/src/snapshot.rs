//! Snapshot-format backups (`"format": "snapshot"`).
//!
//! Instead of `binlogs/` + level meta files, a snapshot backup carries an
//! exported Milvus snapshot bundle under `bundle/exports/<uuid>/`:
//!
//! ```text
//! <backup>/bundle/exports/<uuid>/
//!   snapshots/<collection_id>/metadata/<snapshot_id>.json   <- entry point
//!   snapshots/<collection_id>/manifests/<snapshot_id>/<segment_id>.avro
//!   files/<source-relative-path>                            <- data files
//! ```
//!
//! The metadata JSON is a protojson `datapb.SnapshotMetadata` (snake_case
//! proto names, optional fields omitted). Its `manifest_list` names one avro
//! record per segment (OCF with the writer schema embedded, snapshot format
//! version >= 6); each record mirrors `datapb.SegmentDescription`:
//! `binlog_files` carry direct object keys for storage-v1/v2 segments while
//! storage-v3 segments drop them and reference a milvus-storage manifest via
//! `storagev2_manifest_list` (`{"ver":N,"base_path":"..."}`), whose
//! `_metadata/manifest-<ver>.avro` lists the packed parquet data files.
//! milvus-backup rewrites every path reference to point into the bundle, so
//! keys are absolute from the *writer's* storage root; they are relocated by
//! locating the `bundle/exports/` anchor inside the key (mirroring
//! `snapshotbundle.DeriveBundleRoot`).
//!
//! Data files are byte-verbatim copies of what the Milvus server wrote, so
//! the parquet/envelope readers from the binlog format apply unchanged; the
//! only new decoding is the avro manifests themselves and milvus-storage's
//! `group_field_id_list` footer metadata (see `parquet::footer`).

use std::collections::HashMap;
use std::sync::LazyLock;

use serde::Deserialize;

use duckdb_neo::{Error, Result, file::FileSystem};

use crate::avro::{self, AvroValue};
use crate::meta::{parse_json, read_required};
use crate::scan::{BinlogRef, FieldBinlog, SegmentFile, SegmentTask};

/// `CollectionBackupInfo.snapshot_backup` (milvus-backup `core/proto/backup.proto`).
#[derive(Deserialize)]
pub(crate) struct SnapshotBackupInfo {
    /// Bundle metadata JSON, relative to the backup directory.
    metadata_path: Option<String>,
    #[allow(dead_code)]
    total_files: Option<i64>,
    #[allow(dead_code)]
    total_bytes: Option<i64>,
}

/// protojson serializes int64 as strings; numbers appear in hand-written or
/// older files, so both shapes are accepted.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ProtoInt(pub i64);

impl<'de> Deserialize<'de> for ProtoInt {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl serde::de::Visitor<'_> for Visitor {
            type Value = ProtoInt;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("an integer or a decimal-string integer")
            }
            fn visit_i64<E: serde::de::Error>(
                self,
                value: i64,
            ) -> std::result::Result<ProtoInt, E> {
                Ok(ProtoInt(value))
            }
            fn visit_u64<E: serde::de::Error>(
                self,
                value: u64,
            ) -> std::result::Result<ProtoInt, E> {
                Ok(ProtoInt(value as i64))
            }
            fn visit_str<E: serde::de::Error>(
                self,
                value: &str,
            ) -> std::result::Result<ProtoInt, E> {
                value
                    .parse::<i64>()
                    .map(ProtoInt)
                    .map_err(serde::de::Error::custom)
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

/// protojson `datapb.SnapshotMetadata` (UseProtoNames, EmitUnpopulated=false).
#[derive(Deserialize)]
pub(crate) struct SnapshotMetadata {
    /// Snapshot format version the writer used: >= 6 emits OCF manifests,
    /// < 6 raw avro binary decoded against a static version schema.
    format_version: Option<i32>,
    #[serde(default)]
    manifest_list: Vec<String>,
    #[serde(default)]
    storagev2_manifest_list: Vec<StorageManifestRef>,
    /// Integrity declaration of the manifest set; read data comes from the
    /// manifests themselves.
    #[allow(dead_code)]
    #[serde(default)]
    segment_ids: Option<Vec<ProtoInt>>,
    collection: Option<CollectionDescription>,
}

#[derive(Deserialize)]
struct StorageManifestRef {
    segment_id: Option<ProtoInt>,
    manifest: Option<String>,
}

/// protojson `datapb.CollectionDescription` — only the parts used for
/// metadata counts.
#[derive(Deserialize)]
struct CollectionDescription {
    /// Partition name -> id; absent when the writer emitted none.
    #[serde(default)]
    partitions: Option<std::collections::BTreeMap<String, ProtoInt>>,
}

/// Cheap per-collection counts from the bundle metadata JSON alone (no avro
/// manifest reads, keeping `milvus_backup_collections` a metadata-only call).
pub(crate) struct BundleCounts {
    /// Size of the collection description's partition map; NULL when the
    /// writer omitted it.
    pub(crate) partition_count: Option<i64>,
    /// Number of segment manifests. Includes L0 (delete-only) segments —
    /// without reading the manifests they cannot be told apart.
    pub(crate) segment_count: i64,
    /// Partition map keys (deterministic order).
    pub(crate) partition_names: Option<Vec<String>>,
}

/// Read one collection's bundle metadata and count what it declares.
pub(crate) fn count_bundle(
    fs: &FileSystem<'_>,
    dir: &str,
    snapshot_backup: &SnapshotBackupInfo,
) -> Result<BundleCounts> {
    let bundle = load_bundle(fs, dir, snapshot_backup)?;
    let partitions = bundle
        .metadata
        .collection
        .as_ref()
        .and_then(|collection| collection.partitions.as_ref());
    Ok(BundleCounts {
        partition_count: partitions.map(|map| map.len() as i64),
        segment_count: bundle.metadata.manifest_list.len() as i64,
        partition_names: partitions.map(|map| map.keys().cloned().collect()),
    })
}

/// One collection's loaded bundle: the parsed metadata plus the export
/// namespace it lives in.
pub(crate) struct LoadedBundle {
    metadata: SnapshotMetadata,
}

/// The pieces `milvus_backup_scan` derives from a bundle instead of the
/// binlog-format level files.
pub(crate) struct SnapshotPlan {
    pub(crate) tasks: Vec<SegmentTask>,
    pub(crate) delta_sources: Vec<Vec<String>>,
    pub(crate) estimated_rows: u64,
}

/// datapb.SegmentLevel values that matter here.
const SEGMENT_LEVEL_L0: i64 = 1;
/// Storage versions whose insert files are milvus-storage manifests rather
/// than datapb binlog entries.
const STORAGE_V3: i64 = 3;

/// Read the snapshot metadata JSON a collection entry points at.
pub(crate) fn load_bundle(
    fs: &FileSystem<'_>,
    dir: &str,
    snapshot_backup: &SnapshotBackupInfo,
) -> Result<LoadedBundle> {
    let metadata_path = snapshot_backup.metadata_path.as_deref().ok_or_else(|| {
        Error::invalid_input(
            "snapshot-format collection records no snapshot bundle metadata_path; \
             the backup was written by an older milvus-backup",
        )
    })?;
    let path = format!("{dir}/{metadata_path}");
    let bytes = read_required(fs, &path)?;
    let metadata: SnapshotMetadata = parse_json(&bytes, &path)?;
    Ok(LoadedBundle { metadata })
}

/// Relocate a bundle object key (absolute from the writer's storage root,
/// possibly URI-formed) to a path RELATIVE to the backup directory by finding
/// the `bundle/exports/` anchor, the same anchor `DeriveBundleRoot` uses in
/// reverse. Callers join it with the backup dir like every other candidate
/// path.
pub(crate) fn resolve_bundle_key(key: &str) -> Result<String> {
    let normalized = normalize_key(key);
    let marker = "bundle/exports/";
    let anchor = normalized.rfind(marker).ok_or_else(|| {
        Error::invalid_input(format!(
            "bundle key {key:?} contains no {marker:?} anchor; cannot relocate it under \
             the backup directory"
        ))
    })?;
    Ok(normalized[anchor..].to_string())
}

/// Strip a URI transport prefix (`s3://bucket/`, `minio://host/bucket/`),
/// keeping the object key. milvus-backup writes bare keys, so this only
/// matters for hand-edited or source-shaped references.
fn normalize_key(key: &str) -> &str {
    if let Some(scheme_end) = key.find("://") {
        let rest = &key[scheme_end + 3..];
        // Skip bucket (URI-host form) or host/bucket (endpoint form): both
        // leave the object key after the first '/'.
        return match rest.find('/') {
            Some(index) => &rest[index + 1..],
            None => rest,
        };
    }
    key
}

/// Build the scan plan: one task per data segment, tombstone sources from
/// every deltalog, row estimate from the manifests.
pub(crate) fn build_plan(
    fs: &FileSystem<'_>,
    dir: &str,
    bundle: &LoadedBundle,
) -> Result<SnapshotPlan> {
    // segment_id -> milvus-storage manifest reference (storage v2/v3).
    let storage_manifests: HashMap<i64, &str> = bundle
        .metadata
        .storagev2_manifest_list
        .iter()
        .filter_map(|reference| {
            Some((reference.segment_id.map(|id| id.0)?, reference.manifest.as_deref()?))
        })
        .collect();

    let mut tasks = Vec::new();
    let mut delta_sources: Vec<Vec<String>> = Vec::new();
    let mut estimated_rows = 0u64;

    for manifest_path in &bundle.metadata.manifest_list {
        let resolved_manifest = format!("{dir}/{}", resolve_bundle_key(manifest_path)?);
        let record = read_segment_manifest(
            fs,
            &resolved_manifest,
            bundle.metadata.format_version.unwrap_or(0),
        )?;
        let segment = SegmentManifest::extract(&record, &resolved_manifest)?;

        // Delta logs are datapb binlog-format files in every storage version.
        for field_log in &segment.deltalogs {
            for entry in &field_log.binlogs {
                let Some(log_path) = &entry.log_path else {
                    continue;
                };
                let resolved = resolve_bundle_key(log_path)?;
                if !delta_sources
                    .iter()
                    .any(|existing| existing.first() == Some(&resolved))
                {
                    delta_sources.push(vec![resolved]);
                }
            }
        }

        // Storage v3 drops insert binlogs from the manifest record; its data
        // lives behind the milvus-storage reference instead. L0 segments and
        // v1/v2 segments without insert logs are delete-only tombstone
        // sources.
        let is_v3 = segment.storage_version >= STORAGE_V3;
        let is_l0 = segment.segment_level == SEGMENT_LEVEL_L0;
        if is_l0 || (!is_v3 && segment.binlogs.is_empty()) {
            continue;
        }

        let files = if is_v3 {
            let reference = storage_manifests.get(&segment.segment_id).copied().ok_or_else(|| {
                Error::invalid_input(format!(
                    "storage-v3 segment {} has no milvus-storage manifest reference in the \
                     snapshot metadata",
                    segment.segment_id
                ))
            })?;
            milvus_storage_files(fs, dir, reference)?
        } else {
            segment
                .binlogs
                .iter()
                .flat_map(|field_log| {
                    field_log.binlogs.iter().filter_map(|entry| {
                        let log_path = entry.log_path.as_deref()?;
                        let resolved = resolve_bundle_key(log_path).ok()?;
                        Some(SegmentFile {
                            candidates: vec![resolved],
                            declared: field_log.field_id,
                        })
                    })
                })
                .collect()
        };

        estimated_rows += segment.num_of_rows.max(0) as u64;
        tasks.push(SegmentTask {
            segment_id: segment.segment_id,
            files,
        });
    }

    tasks.sort_by_key(|task| task.segment_id);
    Ok(SnapshotPlan {
        tasks,
        delta_sources,
        estimated_rows,
    })
}

/// Read one segment manifest: OCF (snapshot format >= 6, schema embedded) or
/// raw avro binary (format < 6, decoded against the static version schema),
/// always the single-record shape milvus-backup's writer produces.
fn read_segment_manifest(
    fs: &FileSystem<'_>,
    path: &str,
    format_version: i32,
) -> Result<AvroValue> {
    let bytes = read_required(fs, path)?;
    let record = if bytes.starts_with(avro::OCF_MAGIC) {
        let mut records = avro::read_ocf_records(&bytes).map_err(|error| {
            Error::invalid_input(format!("failed to read avro manifest {path}: {error}"))
        })?;
        match records.len() {
            1 => records.remove(0),
            0 => {
                return Err(Error::invalid_input(format!(
                    "snapshot segment manifest {path} holds no records"
                )))
            }
            count => {
                return Err(Error::invalid_input(format!(
                    "snapshot segment manifest {path} holds {count} records; expected one"
                )))
            }
        }
    } else {
        let schema = segment_manifest_schema(format_version).ok_or_else(|| {
            Error::invalid_input(format!(
                "snapshot format version {format_version} has no raw-binary manifest schema \
                 here; OCF manifests (format >= 6) must carry the Obj\\x01 magic"
            ))
        })?;
        avro::decode_raw_record(&bytes, schema).map_err(|error| {
            Error::invalid_input(format!(
                "failed to decode raw avro manifest {path} (snapshot format \
                 {format_version}): {error}"
            ))
        })?
    };
    Ok(record)
}

/// The avro writer schema of a segment manifest for raw-binary snapshot
/// formats, mirroring milvus-backup `internal/snapshotbundle/manifest.go`:
/// versions only ever append default-bearing fields, and the writer encodes
/// with exactly the schema below (same field order).
fn segment_manifest_schema(version: i32) -> Option<&'static str> {
    match version {
        0 | 1 => Some(&*MANIFEST_SCHEMA_V1),
        2 => Some(MANIFEST_SCHEMA_V2),
        3 => Some(&*MANIFEST_SCHEMA_V3),
        4 => Some(&*MANIFEST_SCHEMA_V4),
        5 => Some(&*MANIFEST_SCHEMA_V5),
        _ => None,
    }
}

static MANIFEST_SCHEMA_V1: LazyLock<String> = LazyLock::new(|| {
    MANIFEST_SCHEMA_V2.replace(
        "\n        {\"name\": \"index_store_path_version\", \"type\": \"int\", \"default\": 0}",
        "",
    )
});

static MANIFEST_SCHEMA_V3: LazyLock<String> = LazyLock::new(|| {
    MANIFEST_SCHEMA_V2.replace(
        "{\"name\": \"is_sorted\", \"type\": \"boolean\"},",
        "{\"name\": \"is_sorted\", \"type\": \"boolean\"},\n        {\"name\": \
         \"commit_timestamp\", \"type\": \"long\", \"default\": 0},",
    )
});

static MANIFEST_SCHEMA_V4: LazyLock<String> = LazyLock::new(|| {
    MANIFEST_SCHEMA_V3.replace(
        "{\"name\": \"field_id\", \"type\": \"long\"},\n        {\"name\": \"binlogs\",",
        "{\"name\": \"field_id\", \"type\": \"long\"},\n        {\"name\": \
         \"child_fields\", \"type\": {\"type\": \"array\", \"items\": \"long\"}, \
         \"default\": []},\n        {\"name\": \"format\", \"type\": \"string\", \
         \"default\": \"\"},\n        {\"name\": \"binlogs\",",
    )
});

static MANIFEST_SCHEMA_V5: LazyLock<String> = LazyLock::new(|| {
    MANIFEST_SCHEMA_V4.replace(
        "{\"name\": \"commit_timestamp\", \"type\": \"long\", \"default\": 0},",
        "{\"name\": \"commit_timestamp\", \"type\": \"long\", \"default\": 0},\n        \
         {\"name\": \"manifest_has_index\", \"type\": \"boolean\", \"default\": false},",
    )
});

/// milvus-backup `AvroSchemaV2` — the base of every raw manifest version.
const MANIFEST_SCHEMA_V2: &str = r#"{
  "type": "record",
  "name": "ManifestEntry",
  "fields": [
    {"name": "segment_id", "type": "long"},
    {"name": "partition_id", "type": "long"},
    {"name": "segment_level", "type": "long"},
    {"name": "channel_name", "type": "string"},
    {"name": "num_of_rows", "type": "long"},
    {"name": "start_position", "type": {"type": "record", "name": "AvroMsgPosition", "fields": [
      {"name": "channel_name", "type": "string"},
      {"name": "msg_id", "type": "bytes"},
      {"name": "msg_group", "type": "string"},
      {"name": "timestamp", "type": "long"}
    ]}},
    {"name": "dml_position", "type": "AvroMsgPosition"},
    {"name": "storage_version", "type": "long"},
    {"name": "is_sorted", "type": "boolean"},
    {"name": "binlog_files", "type": {"type": "array", "items": {
      "type": "record", "name": "AvroFieldBinlog", "fields": [
        {"name": "field_id", "type": "long"},
        {"name": "binlogs", "type": {"type": "array", "items": {
          "type": "record", "name": "AvroBinlog", "fields": [
            {"name": "entries_num", "type": "long"},
            {"name": "timestamp_from", "type": "long"},
            {"name": "timestamp_to", "type": "long"},
            {"name": "log_path", "type": "string"},
            {"name": "log_size", "type": "long"},
            {"name": "log_id", "type": "long"},
            {"name": "memory_size", "type": "long"}
          ]
        }}}
      ]
    }}},
    {"name": "deltalog_files", "type": {"type": "array", "items": "AvroFieldBinlog"}},
    {"name": "statslog_files", "type": {"type": "array", "items": "AvroFieldBinlog"}},
    {"name": "bm25_statslog_files", "type": {"type": "array", "items": "AvroFieldBinlog"}},
    {"name": "index_files", "type": {"type": "array", "items": {
      "type": "record", "name": "AvroIndexFilePathInfo", "fields": [
        {"name": "segment_id", "type": "long"},
        {"name": "field_id", "type": "long"},
        {"name": "index_id", "type": "long"},
        {"name": "build_id", "type": "long"},
        {"name": "index_name", "type": "string"},
        {"name": "index_params", "type": {"type": "array", "items": {
          "type": "record", "name": "AvroKeyValuePair", "fields": [
            {"name": "key", "type": "string"},
            {"name": "value", "type": "string"}
          ]
        }}},
        {"name": "index_file_paths", "type": {"type": "array", "items": "string"}},
        {"name": "serialized_size", "type": "long"},
        {"name": "index_version", "type": "long"},
        {"name": "num_rows", "type": "long"},
        {"name": "current_index_version", "type": "int"},
        {"name": "mem_size", "type": "long"},
        {"name": "current_scalar_index_version", "type": "int", "default": 0},
        {"name": "index_store_path_version", "type": "int", "default": 0}
      ]
    }}},
    {"name": "text_index_files", "type": {"type": "array", "items": {
      "type": "record", "name": "AvroTextIndexEntry", "fields": [
        {"name": "field_id", "type": "long"},
        {"name": "stats", "type": {"type": "record", "name": "AvroTextIndexStats", "fields": [
          {"name": "field_id", "type": "long"},
          {"name": "version", "type": "long"},
          {"name": "files", "type": {"type": "array", "items": "string"}},
          {"name": "log_size", "type": "long"},
          {"name": "memory_size", "type": "long"},
          {"name": "build_id", "type": "long"}
        ]}}
      ]
    }}},
    {"name": "json_key_index_files", "type": {"type": "array", "items": {
      "type": "record", "name": "AvroJSONKeyIndexEntry", "fields": [
        {"name": "field_id", "type": "long"},
        {"name": "stats", "type": {"type": "record", "name": "AvroJSONKeyStats", "fields": [
          {"name": "field_id", "type": "long"},
          {"name": "version", "type": "long"},
          {"name": "files", "type": {"type": "array", "items": "string"}},
          {"name": "log_size", "type": "long"},
          {"name": "memory_size", "type": "long"},
          {"name": "build_id", "type": "long"},
          {"name": "json_key_stats_data_format", "type": "long"}
        ]}}
      ]
    }}}
  ]
}"#;

/// The fields of the per-segment avro manifest record that the scan needs.
struct SegmentManifest {
    segment_id: i64,
    #[allow(dead_code)]
    partition_id: i64,
    segment_level: i64,
    num_of_rows: i64,
    storage_version: i64,
    binlogs: Vec<FieldBinlog>,
    deltalogs: Vec<FieldBinlog>,
}

impl SegmentManifest {
    fn extract(record: &AvroValue, path: &str) -> Result<SegmentManifest> {
        let long = |name: &str, default: i64| -> Result<i64> {
            Ok(record.field(name).and_then(AvroValue::as_long).unwrap_or(default))
        };
        Ok(SegmentManifest {
            segment_id: long("segment_id", -1)?,
            partition_id: long("partition_id", -1)?,
            segment_level: long("segment_level", 0)?,
            num_of_rows: long("num_of_rows", 0)?,
            storage_version: long("storage_version", 0)?,
            binlogs: extract_field_binlogs(record.field("binlog_files"), path, "binlog_files")?,
            deltalogs: extract_field_binlogs(
                record.field("deltalog_files"),
                path,
                "deltalog_files",
            )?,
        })
    }
}

fn extract_field_binlogs(
    value: Option<&AvroValue>,
    path: &str,
    name: &str,
) -> Result<Vec<FieldBinlog>> {
    let Some(values) = value.and_then(AvroValue::as_array) else {
        return Ok(Vec::new());
    };
    values
        .iter()
        .map(|entry| {
            let field_id = entry.field("field_id").and_then(AvroValue::as_long);
            let mut binlogs = Vec::new();
            if let Some(list) = entry.field("binlogs").and_then(AvroValue::as_array) {
                for binlog in list {
                    binlogs.push(BinlogRef {
                        log_path: binlog
                            .field("log_path")
                            .and_then(AvroValue::as_str)
                            .map(str::to_string),
                    });
                }
            }
            if binlogs.is_empty() {
                return Err(Error::invalid_input(format!(
                    "snapshot manifest {path} has a {name} entry with no binlogs"
                )));
            }
            Ok(FieldBinlog {
                field_id,
                binlogs,
            })
        })
        .collect()
}

/// Enumerate a storage-v3 segment's packed parquet files from its
/// milvus-storage manifest: `{base}/_metadata/manifest-<ver>.avro` lists
/// column groups whose file paths are relative to `{base}/_data/`.
fn milvus_storage_files(
    fs: &FileSystem<'_>,
    dir: &str,
    reference: &str,
) -> Result<Vec<SegmentFile>> {
    let (version, base_path) = parse_manifest_reference(reference)?;
    let base = format!("{dir}/{}", resolve_bundle_key(&base_path)?);
    let manifest_path = format!("{base}/_metadata/manifest-{version}.avro");
    let bytes = read_required(fs, &manifest_path)?;
    let records = avro::read_ocf_records(&bytes).map_err(|error| {
        Error::invalid_input(format!(
            "failed to read milvus-storage manifest {manifest_path} (legacy pre-OCF \
             manifests are not supported): {error}"
        ))
    })?;
    if records.len() != 1 {
        return Err(Error::invalid_input(format!(
            "milvus-storage manifest {manifest_path} holds {} records; expected one",
            records.len()
        )));
    }
    let record = &records[0];

    let Some(groups) = record.field("column_groups").and_then(AvroValue::as_array) else {
        return Err(Error::invalid_input(format!(
            "milvus-storage manifest {manifest_path} has no column_groups"
        )));
    };
    let mut files = Vec::new();
    for group in groups {
        let format = group
            .field("format")
            .and_then(AvroValue::as_str)
            .unwrap_or_default();
        if format != "parquet" {
            return Err(Error::invalid_input(format!(
                "milvus-storage manifest {manifest_path} has a column group of format \
                 {format:?}; only parquet column groups can be read"
            )));
        }
        let Some(list) = group.field("files").and_then(AvroValue::as_array) else {
            continue;
        };
        for file in list {
            let Some(path) = file.field("path").and_then(AvroValue::as_str) else {
                return Err(Error::invalid_input(format!(
                    "milvus-storage manifest {manifest_path} has a column group file without \
                     a path"
                )));
            };
            // Stored relative to `{base}/_data/`; absolute (URI) paths are
            // kept whole by the writer for external tables.
            let key = if path.contains("://") || path.starts_with('/') {
                path.to_string()
            } else {
                format!("{base}/_data/{path}")
            };
            let resolved = resolve_bundle_key(&key)?;
            files.push(SegmentFile {
                candidates: vec![resolved],
                declared: None,
            });
        }
    }
    Ok(files)
}

/// Parse the `{"ver":N,"base_path":"..."}` manifest reference string.
fn parse_manifest_reference(reference: &str) -> Result<(i64, String)> {    #[derive(Deserialize)]
    struct ManifestPath {
        #[serde(alias = "ver")]
        version: Option<i64>,
        #[serde(alias = "base_path")]
        base_path: Option<String>,
    }
    let parsed: ManifestPath = parse_json(reference.as_bytes(), "manifest reference")?;
    let version = parsed.version.unwrap_or(1);
    let base_path = parsed.base_path.ok_or_else(|| {
        Error::invalid_input(format!(
            "milvus-storage manifest reference {reference:?} has no base_path"
        ))
    })?;
    Ok((version, base_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Decodes one real manifest captured from a Milvus 3.0.0 snapshot
    /// backup (raw avro, format version 4): the L0 segment holding the
    /// collection's five delete tombstones.
    #[test]
    fn decodes_real_v4_l0_manifest() {
        let bytes = include_bytes!("../../../test/fixtures/avro/l0_segment_manifest_v4.avro");
        let schema = segment_manifest_schema(4).unwrap();
        let record = avro::decode_raw_record(bytes, schema).unwrap();
        assert_eq!(
            record.field("segment_id").and_then(AvroValue::as_long),
            Some(470162243053621852)
        );
        assert_eq!(
            record.field("segment_level").and_then(AvroValue::as_long),
            Some(1)
        );
        assert_eq!(
            record.field("num_of_rows").and_then(AvroValue::as_long),
            Some(0)
        );
        let deltalogs = record
            .field("deltalog_files")
            .and_then(AvroValue::as_array)
            .unwrap();
        assert_eq!(deltalogs.len(), 1);
        assert_eq!(
            deltalogs[0].field("field_id").and_then(AvroValue::as_long),
            Some(100)
        );
        let binlogs = deltalogs[0]
            .field("binlogs")
            .and_then(AvroValue::as_array)
            .unwrap();
        assert_eq!(binlogs.len(), 1);
        assert!(binlogs[0]
            .field("log_path")
            .and_then(AvroValue::as_str)
            .unwrap()
            .contains("delta_log"));
    }
}
