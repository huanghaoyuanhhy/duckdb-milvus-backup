use duckdb_loadable_macros::duckdb_entrypoint_c_api_v2;
use duckdb_neo::{Result, extension::Extension};

#[duckdb_entrypoint_c_api_v2(extension_name = "milvus_backup", api_version = "2.0.0")]
pub fn entrypoint(extension: &Extension) -> Result<()> {
    milvus_backup_core::register(extension)
}
