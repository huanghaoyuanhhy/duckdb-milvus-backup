//! Shared plumbing for reading milvus-backup meta JSON through DuckDB's
//! `FileSystem` (local paths and `s3://` via httpfs alike).
//!
//! Backup layout: `{root}/{name}/meta/{backup,collection,partition,segment}_meta.json`,
//! each level file wrapping its entries as `{"infos": [...]}`. `full_meta.json`
//! is deliberately never read: the per-level files cover old, modern and
//! snapshot writer formats uniformly. Only meta JSON is ever read, never
//! binlog data.

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
