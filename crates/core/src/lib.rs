mod backups;
mod version;

use duckdb_neo::{Result, extension::Extension};

pub fn register(extension: &Extension) -> Result<()> {
    backups::register(extension)?;
    version::register(extension)?;

    Ok(())
}
