//! `milvus_backups(root_path)` — one row per backup under a backup root.
//!
//! Backup layout: `{root}/{name}/meta/{backup,collection,partition,segment}_meta.json`,
//! each level file wrapping its entries as `{"infos": [...]}`. `backup_meta.json`
//! carries no collections (the writer strips the tree into the level files), and
//! snapshot-format backups (`format: "snapshot"`) stop after
//! `collection_meta.json` — partition/segment files do not exist for them.
//! All reads go through DuckDB's FileSystem, so `s3://` roots work wherever
//! httpfs is loaded; only meta JSON is ever read, never binlogs.

use std::sync::atomic::{AtomicUsize, Ordering};

use serde::Deserialize;

use duckdb_neo::{
    DuckDBType, Error, Result,
    bind_arguments::BindArgument,
    connection::Context,
    data_chunk::DataChunkRef,
    extension::Extension,
    file::{FileBuilder, FileSystem},
    signature::{FunctionDocumentationBuilder, FunctionParameter, SignatureBuilder},
    table_function::{
        BindFunctionHandle, ExecColumnInfo, InitColumnData, OrderPreservation,
        TableFunctionBuilder, TableFunctionCallbacks,
    },
    types::{List, TimestampTzValue},
};

pub(crate) struct BackupsFunction;

pub(crate) struct BindData {
    /// Backup directories, sorted for deterministic output across runs.
    backup_dirs: Vec<String>,
}

pub(crate) struct GlobalState {
    next_backup: AtomicUsize,
}

// One backup = one row, so a claim is just an index; state lives long enough
// in the atomics and no per-thread cursor is needed.
type LocalState = ();

const META_SUFFIX: &str = "/meta/backup_meta.json";

impl TableFunctionCallbacks for BackupsFunction {
    type BindData = BindData;
    type GlobalState = GlobalState;
    type LocalState = LocalState;

    fn bind(
        &self,
        context: &Context,
        arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        let root = constant_string(&arguments, "root_path")?;
        // A trailing slash must not produce `//` in the glob pattern; keep "/" itself usable.
        let root = root.trim_end_matches('/');
        let root = if root.is_empty() { "/" } else { root };

        let fs = FileSystem::new(context)?;
        // The v2 C API offers no directory listing, only glob — this is the
        // narrowest possible pattern. Listing stays proportional to the backup
        // count, not the total file count, because httpfs lists dense S3 trees
        // hierarchically (delimiter per level, descending only matching dirs).
        let mut backup_dirs: Vec<String> = fs
            .glob(&format!("{root}/*{META_SUFFIX}"))?
            .into_iter()
            // The glob pattern guarantees the suffix; strip defensively anyway.
            .filter_map(|meta_path| meta_path.strip_suffix(META_SUFFIX).map(str::to_string))
            .collect();
        backup_dirs.sort();

        bind_handle.set_cardinality(backup_dirs.len(), true)?;
        // Threads claim backups from a shared counter, so arrival order varies;
        // users who need a stable order write ORDER BY.
        bind_handle.set_order(OrderPreservation::None)?;

        bind_handle.add_result_column("path", &String::logical_type(context)?)?;
        bind_handle.add_result_column("name", &String::logical_type(context)?)?;
        bind_handle.add_result_column("id", &String::logical_type(context)?)?;
        bind_handle.add_result_column("milvus_version", &String::logical_type(context)?)?;
        bind_handle.add_result_column("format", &String::logical_type(context)?)?;
        bind_handle.add_result_column("created_at", &TimestampTzValue::logical_type(context)?)?;
        bind_handle.add_result_column("backup_timestamp", &u64::logical_type(context)?)?;
        bind_handle.add_result_column("size_bytes", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("databases", &Vec::<String>::logical_type(context)?)?;
        bind_handle
            .add_result_column("collection_names", &Vec::<String>::logical_type(context)?)?;
        bind_handle.add_result_column("collection_count", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("partition_count", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("segment_count", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("num_rows_inserted", &i64::logical_type(context)?)?;
        bind_handle.add_result_column("has_delete_logs", &bool::logical_type(context)?)?;
        bind_handle.add_result_column("has_rbac", &bool::logical_type(context)?)?;

        Ok(BindData { backup_dirs })
    }

    fn init_global_state(
        &self,
        bind_data: &Self::BindData,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        let backup_count = bind_data.backup_dirs.len();
        Ok((
            Some(GlobalState {
                next_backup: AtomicUsize::new(0),
            }),
            (backup_count > 0).then_some(backup_count),
        ))
    }

    fn exec(
        &self,
        bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        _local_state: Option<&mut Self::LocalState>,
        context: &Context,
        mut output: DataChunkRef<'_>,
        _column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        let global_state = global_state
            .expect("global state is always created by init_global_state")
            .next_backup
            .fetch_add(1, Ordering::SeqCst);

        if global_state >= bind_data.backup_dirs.len() {
            output.get_vector_at_mut::<String>(0)?.set_size(0)?;
            return Ok(());
        }

        let dir = bind_data.backup_dirs[global_state].as_str();
        let fs = FileSystem::new(context)?;
        let row = BackupRow::read(&fs, dir)?;

        macro_rules! write_column {
            ($index:expr, $ty:ty, $value:expr) => {{
                let mut vector = output.get_vector_at_mut::<$ty>($index)?;
                vector.set_size(1)?;
                vector.writer()?.set(0, $value)?;
            }};
        }

        write_column!(0, String, Some(dir));
        write_column!(1, String, row.name.as_deref());
        write_column!(2, String, row.id.as_deref());
        write_column!(3, String, row.milvus_version.as_deref());
        write_column!(4, String, Some(row.format.as_str()));
        write_column!(5, TimestampTzValue, row.created_at);
        write_column!(6, u64, row.backup_timestamp);
        write_column!(7, i64, row.size_bytes);
        write_column!(
            8,
            List<String>,
            row.databases.as_ref().map(|names| names
                .iter()
                .map(|name| Some(name.as_str()))
                .collect::<Vec<Option<&str>>>())
        );
        write_column!(
            9,
            List<String>,
            row.collection_names.as_ref().map(|names| names
                .iter()
                .map(|name| Some(name.as_str()))
                .collect::<Vec<Option<&str>>>())
        );
        write_column!(10, i64, row.collection_count);
        write_column!(11, i64, row.partition_count);
        write_column!(12, i64, row.segment_count);
        write_column!(13, i64, row.num_rows_inserted);
        write_column!(14, bool, row.has_delete_logs);
        write_column!(15, bool, row.has_rbac);

        Ok(())
    }
}

/// One materialized `milvus_backups` row; `None` means the source field is
/// absent, not zero — old backups lack most of these fields legitimately.
struct BackupRow {
    name: Option<String>,
    id: Option<String>,
    milvus_version: Option<String>,
    format: String,
    created_at: Option<TimestampTzValue>,
    backup_timestamp: Option<u64>,
    size_bytes: Option<i64>,
    databases: Option<Vec<String>>,
    collection_names: Option<Vec<String>>,
    collection_count: Option<i64>,
    partition_count: Option<i64>,
    segment_count: Option<i64>,
    num_rows_inserted: Option<i64>,
    has_delete_logs: Option<bool>,
    has_rbac: Option<bool>,
}

impl BackupRow {
    fn read(fs: &FileSystem, dir: &str) -> Result<BackupRow> {
        let meta_path = format!("{dir}/meta/backup_meta.json");
        let meta: BackupMeta = parse_json(&read_required(fs, &meta_path)?, &meta_path)?;

        let colls: Option<Vec<CollectionBackup>> = read_infos(fs, dir, "collection_meta")?;
        let partitions: Option<Vec<serde_json::Value>> = read_infos(fs, dir, "partition_meta")?;
        let segments: Option<Vec<SegmentBackup>> = read_infos(fs, dir, "segment_meta")?;

        // The proto documents empty/absent format as the binlog format; the
        // snapshot writer always sets the field.
        let is_snapshot = meta.format.as_deref() == Some("snapshot");
        let format = match meta.format.as_deref() {
            Some(f) if !f.is_empty() => f.to_string(),
            _ => "binlog".to_string(),
        };

        // start_time is ms (older writer); snapshot-era backups only carry the
        // per-collection physical timestamp in seconds.
        let created_at = meta
            .start_time
            .and_then(|ms| ms.checked_mul(1_000))
            .map(TimestampTzValue)
            .or_else(|| {
                colls
                    .as_ref()?
                    .iter()
                    .filter_map(|coll| coll.backup_physical_timestamp)
                    .max()
                    .and_then(|s| i64::try_from(s).ok())
                    .and_then(|s| s.checked_mul(1_000_000))
                    .map(TimestampTzValue)
            });

        let backup_timestamp = meta.backup_timestamp.or_else(|| {
            colls
                .as_ref()?
                .iter()
                .filter_map(|coll| coll.backup_timestamp)
                .max()
        });

        // database_backups is absent in older backups; fall back to the distinct
        // collection-level db_name values, which those backups do carry.
        let databases = meta
            .database_backups
            .map(|dbs| {
                dbs.iter()
                    .filter_map(|db| db.db_name.clone())
                    .collect::<Vec<_>>()
            })
            .or_else(|| {
                colls.as_ref().map(|colls| {
                    let mut names = Vec::new();
                    for name in colls.iter().filter_map(|coll| coll.db_name.clone()) {
                        if !names.contains(&name) {
                            names.push(name);
                        }
                    }
                    names
                })
            });

        // Names are qualified `db.collection` so same-named collections in
        // different databases stay distinguishable; db_name is absent in old
        // backups, where the bare collection name is all that exists.
        let collection_names = colls.as_ref().map(|colls| {
            colls.iter()
                .filter_map(|coll| match (&coll.db_name, &coll.collection_name) {
                    (Some(db), Some(name)) if !db.is_empty() => Some(format!("{db}.{name}")),
                    (_, Some(name)) => Some(name.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
        });

        let collection_count = colls.as_ref().map(|colls| colls.len() as i64);

        // Snapshot backups provably have no partition/segment level; a missing
        // level file in a binlog backup is unread state, reported as NULL.
        let partition_count = partitions
            .as_ref()
            .map(|parts| parts.len() as i64)
            .or_else(|| is_snapshot.then_some(0));

        let segment_count = segments
            .as_ref()
            .map(|segs| segs.len() as i64)
            .or_else(|| is_snapshot.then_some(0));

        // Rows written into segments, not the final row count: deletes live in
        // delta logs (and l0 segments) and are not subtracted here.
        let num_rows_inserted = segments.as_ref().map(|segs| {
            segs.iter()
                .map(|segment| segment.num_of_rows.unwrap_or(0))
                .sum::<i64>()
        });

        // Empty segments still carry `"deltalogs": [{}]` stubs, so deletes are
        // detected by non-empty `binlogs` inside the stubs; l0 segments are pure
        // delta data and count by their existence.
        let has_delete_logs = match segments.as_ref() {
            Some(segments) => {
                let has_l0 = colls.as_ref().is_some_and(|colls| {
                    colls
                        .iter()
                        .any(|coll| coll.l0_segments.as_ref().is_some_and(|l0| !l0.is_empty()))
                });
                Some(
                    has_l0
                        || segments.iter().any(|segment| {
                            segment
                                .deltalogs
                                .iter()
                                .any(|delta| !delta.binlogs.is_empty())
                        }),
                )
            }
            None if is_snapshot => Some(false),
            None => None,
        };

        // Absent rbac_meta (older backups) is unknown, not false.
        let has_rbac = meta.rbac_meta.map(|rbac| {
            !(rbac.users.is_empty()
                && rbac.roles.is_empty()
                && rbac.grants.is_empty()
                && rbac.privilege_groups.is_empty())
        });

        Ok(BackupRow {
            name: meta.name,
            id: meta.id,
            milvus_version: meta.milvus_version,
            format,
            created_at,
            backup_timestamp,
            size_bytes: meta.size,
            databases,
            collection_names,
            collection_count,
            partition_count,
            segment_count,
            num_rows_inserted,
            has_delete_logs,
            has_rbac,
        })
    }
}

#[derive(Deserialize)]
struct BackupMeta {
    id: Option<String>,
    name: Option<String>,
    format: Option<String>,
    backup_timestamp: Option<u64>,
    size: Option<i64>,
    milvus_version: Option<String>,
    /// Milliseconds since epoch; deprecated in the proto but written by older tools.
    start_time: Option<i64>,
    rbac_meta: Option<RbacMeta>,
    database_backups: Option<Vec<DatabaseBackup>>,
}

#[derive(Deserialize)]
struct DatabaseBackup {
    db_name: Option<String>,
}

#[derive(Deserialize)]
struct RbacMeta {
    #[serde(default)]
    users: Vec<serde_json::Value>,
    #[serde(default)]
    roles: Vec<serde_json::Value>,
    #[serde(default)]
    grants: Vec<serde_json::Value>,
    #[serde(default)]
    privilege_groups: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct LevelFile<T> {
    infos: Option<Vec<T>>,
}

#[derive(Deserialize)]
struct CollectionBackup {
    collection_name: Option<String>,
    db_name: Option<String>,
    /// Hybrid logical TSO.
    backup_timestamp: Option<u64>,
    /// Seconds since epoch.
    backup_physical_timestamp: Option<u64>,
    /// L0 delta segments live at collection level, not under partitions.
    l0_segments: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize)]
struct SegmentBackup {
    num_of_rows: Option<i64>,
    #[serde(default)]
    deltalogs: Vec<DeltaLog>,
}

#[derive(Deserialize)]
struct DeltaLog {
    #[serde(default)]
    binlogs: Vec<serde_json::Value>,
}

fn read_required(fs: &FileSystem, path: &str) -> Result<Vec<u8>> {
    read_optional(fs, path)?.ok_or_else(|| Error::invalid_input(format!("file not found: {path}")))
}

fn read_optional(fs: &FileSystem, path: &str) -> Result<Option<Vec<u8>>> {
    if !fs.exists(path)? {
        return Ok(None);
    }
    let file = FileBuilder::new(fs, path)?.read().open()?;
    let size = file.size()? as usize;
    Ok(Some(file.read_at(0, size)?))
}

fn read_infos<T: serde::de::DeserializeOwned>(
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

fn parse_json<T: serde::de::DeserializeOwned>(bytes: &[u8], path: &str) -> Result<T> {
    serde_json::from_slice(bytes)
        .map_err(|error| Error::invalid_input(format!("failed to parse {path}: {error}")))
}

fn constant_string(arguments: &[BindArgument], name: &str) -> Result<String> {
    let argument = arguments
        .first()
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

pub(crate) fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "milvus_backups",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("root_path"))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .add_category("milvus")
                    .description(
                        "List the Milvus backups stored under a backup root path (local \
                         directory or object storage such as s3:// via httpfs). One row per \
                         backup; reads only the meta JSON files, never binlog data.",
                    )
                    .add_example(
                        "list backups",
                        "SELECT * FROM milvus_backups('/mnt/milvus-backup')",
                    ),
            ),
        BackupsFunction,
    )
    .register(extension)?;

    Ok(())
}
