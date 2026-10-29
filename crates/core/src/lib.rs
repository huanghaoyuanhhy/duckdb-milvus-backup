mod version;

use duckdb_neo::{Result, extension::Extension};

pub fn register(extension: &Extension) -> Result<()> {
    version::register(extension)?;

    Ok(())
}
