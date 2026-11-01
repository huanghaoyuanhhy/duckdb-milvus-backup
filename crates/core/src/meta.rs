//! Shared plumbing for reading milvus-backup meta JSON through DuckDB's
//! `FileSystem` (local paths and `s3://` via httpfs alike).
//!
//! Backup layout: `{root}/{name}/meta/{backup,collection,partition,segment}_meta.json`,
//! each level file wrapping its entries as `{"infos": [...]}`. The per-level
//! files cover old, modern and snapshot writer formats uniformly, so
//! `full_meta.json` is only read for the one thing newer writers record
//! exclusively there: collection-level L0 segments (see [`read_full_meta_l0`]).
//! Only meta JSON is ever read, never binlog data.

use std::collections::HashMap;

use serde::Deserialize;

use duckdb_neo::{
    Error, Result,
    bind_arguments::BindArgument,
    file::{FileBuilder, FileSystem},
};

/// Wrapper layout of the per-level meta files.
#[derive(Deserialize)]
pub(crate) struct LevelFile<T> {
    pub(crate) infos: Option<Vec<T>>,
}

/// Read a whole file, erroring when it does not exist.
pub(crate) fn read_required(fs: &FileSystem, path: &str) -> Result<Vec<u8>> {
    read_optional(fs, path)?.ok_or_else(|| Error::invalid_input(format!("file not found: {path}")))
}

/// Read a whole file, returning `None` when it does not exist.
pub(crate) fn read_optional(fs: &FileSystem, path: &str) -> Result<Option<Vec<u8>>> {
    if !fs.exists(path)? {
        return Ok(None);
    }
    let file = FileBuilder::new(fs, path)?.read().open()?;
    let size = file.size()? as usize;
    Ok(Some(file.read_at(0, size)?))
}

/// Deserialize the `infos` entries of one level file; `None` when absent.
pub(crate) fn read_infos<T: serde::de::DeserializeOwned>(
    fs: &FileSystem,
    dir: &str,
    level: &str,
) -> Result<Option<Vec<T>>> {
    let path = format!("{dir}/meta/{level}.json");
    read_optional(fs, &path)?
        .map(|bytes| {
            parse_json(&bytes, &path).map(|level: LevelFile<T>| level.infos.unwrap_or_default())
        })
        .transpose()
}

pub(crate) fn parse_json<T: serde::de::DeserializeOwned>(bytes: &[u8], path: &str) -> Result<T> {
    serde_json::from_slice(bytes)
        .map_err(|error| Error::invalid_input(format!("failed to parse {path}: {error}")))
}

/// Collection-level L0 (delete-only) segments per collection id, taken from
/// `full_meta.json`. Milvus 2.6-era writers record L0 segments only in the
/// full tree and leave `l0_segments` out of `collection_meta.json`, so the
/// level files alone under-report delete data. Returns an empty map when the
/// file is absent (old backups) or carries no L0.
pub(crate) fn read_full_meta_l0(
    fs: &FileSystem,
    dir: &str,
) -> Result<HashMap<i64, Vec<serde_json::Value>>> {
    let path = format!("{dir}/meta/full_meta.json");
    let Some(bytes) = read_optional(fs, &path)? else {
        return Ok(HashMap::new());
    };
    let value: serde_json::Value = parse_json(&bytes, &path)?;
    let mut out: HashMap<i64, Vec<serde_json::Value>> = HashMap::new();
    let Some(collections) = value.get("collection_backups").and_then(|v| v.as_array()) else {
        return Ok(out);
    };
    for collection in collections {
        let Some(id) = collection.get("collection_id").and_then(|v| v.as_i64()) else {
            continue;
        };
        let Some(l0) = collection.get("l0_segments").and_then(|v| v.as_array()) else {
            continue;
        };
        if !l0.is_empty() {
            out.entry(id).or_default().extend(l0.iter().cloned());
        }
    }
    Ok(out)
}

/// Path of a backup directory inside a backup root, rejecting names that
/// could escape the root (`..`, path separators).
pub(crate) fn backup_dir(root: &str, backup_name: &str) -> Result<String> {
    if backup_name.is_empty()
        || backup_name == "."
        || backup_name == ".."
        || backup_name.contains('/')
        || backup_name.contains('\\')
    {
        return Err(Error::invalid_parameter(format!(
            "invalid backup name: {backup_name:?}"
        )));
    }
    Ok(format!("{root}/{backup_name}"))
}

/// Fold a constant string argument (table function params must be constants:
/// bind runs once per call site, a non-constant arg would re-read meta per row).
pub(crate) fn constant_string(
    arguments: &[BindArgument],
    index: usize,
    name: &str,
) -> Result<String> {
    let argument = arguments
        .get(index)
        .ok_or_else(|| Error::invalid_parameter(format!("{name} is required")))?;
    let value = argument.value.as_ref().ok_or_else(|| {
        Error::invalid_input(format!(
            "{name} must be a constant string, e.g. milvus_backups('/mnt/backups')"
        ))
    })?;
    value
        .get::<String>()?
        .ok_or_else(|| Error::invalid_input(format!("{name} must not be NULL")))
}

/// Normalize a backup root: a trailing slash must not produce `//` in paths;
/// keep "/" itself usable.
pub(crate) fn normalize_root(root: String) -> String {
    let trimmed = root.trim_end_matches('/');
    if trimmed.is_empty() {
        "/".to_string()
    } else {
        trimmed.to_string()
    }
}
