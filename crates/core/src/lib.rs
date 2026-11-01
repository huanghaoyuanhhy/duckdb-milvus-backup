mod backups;
mod binlog;
mod collections;
mod meta;
mod parquet;
mod paths;
mod scan;
mod schema;
mod tombstone;
mod version;

use duckdb_neo::{Result, extension::Extension};

pub fn register(extension: &Extension) -> Result<()> {
    backups::register(extension)?;
    collections::register(extension)?;
    scan::register(extension)?;
    version::register(extension)?;

    Ok(())
}
