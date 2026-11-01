mod backups;
mod collections;
mod meta;
mod version;

use duckdb_neo::{Result, extension::Extension};

pub fn register(extension: &Extension) -> Result<()> {
    backups::register(extension)?;
    collections::register(extension)?;
    version::register(extension)?;

    Ok(())
}
