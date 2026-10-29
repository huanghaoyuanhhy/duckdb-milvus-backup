use duckdb_neo::{
    DuckDBType, Result,
    connection::Context,
    extension::Extension,
    scalar::{ScalarCallbacks, ScalarFunctionBuilder, ScalarInitCallbacks},
    signature::{FunctionDocumentationBuilder, SignatureBuilder},
    vector::{Unknown, Vector},
};

pub(crate) struct VersionScalar;

impl ScalarInitCallbacks for VersionScalar {
    fn init(
        &self,
        _bind_data: Option<&Self::BindData>,
        _context: &Context,
    ) -> Result<Self::InitData> {
        Ok(concat!("milvus_backup ", env!("CARGO_PKG_VERSION")).to_string())
    }
}

impl ScalarCallbacks for VersionScalar {
    type BindData = ();
    type InitData = String;

    fn exec(
        &self,
        _bind_data: Option<&Self::BindData>,
        init_data: Option<&mut Self::InitData>,
        _context: &Context,
        _vectors: &mut duckdb_neo::data_chunk::VectorCollection,
        output: Vector<'_, Unknown>,
    ) -> Result<()> {
        // A zero-parameter scalar is evaluated on a one-row chunk.
        let mut output = output.cast::<String>()?;
        output.set_size(1)?;
        let mut writer = output.writer()?;
        writer.set(0, Some(init_data.unwrap().as_str()))?;
        Ok(())
    }
}

pub(crate) fn register(extension: &Extension) -> Result<()> {
    ScalarFunctionBuilder::new(
        "milvus_backup_version",
        SignatureBuilder::new()
            .return_type(String::logical_type(extension)?)
            .documentation(
                FunctionDocumentationBuilder::new()
                    .add_category("example")
                    .add_example("version", "milvus_backup_version()")
                    .description("Return the milvus_backup extension version."),
            ),
        VersionScalar,
    )
    .with_init()
    .register(extension)?;

    Ok(())
}
