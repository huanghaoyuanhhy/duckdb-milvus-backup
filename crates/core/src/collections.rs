//! `milvus_backup_collections(root_path, backup_name)` — one row per
//! collection stored inside a single Milvus backup.
//!
//! Two overloads share one row layout: the two-key form locates the backup as
//! `{root_path}/{backup_name}`, the single-argument form takes the backup
//! directory itself and looks directly under it. Neither globs. Rows come
//! from `collection_meta.json`; partition/segment aggregates come from
//! `partition_meta.json`/`segment_meta.json` attributed by collection id
//! (falling back to the owning partition when a segment omits it). Old
//! backups without ids on those entries cannot attribute anything, so the
//! affected columns stay NULL rather than claiming zero — the same
//! "unknown is NULL" rule as `milvus_backups`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde::Deserialize;

use duckdb_neo::{
    DuckDBType, Error, Result,
    bind_arguments::BindArgument,
    connection::{Context, FFILink},
    data_chunk::DataChunkRef,
    extension::Extension,
    file::FileSystem,
    logical_type::LogicalType,
    signature::{FunctionDocumentationBuilder, FunctionParameter, SignatureBuilder},
    table_function::{
        BindFunctionHandle, ExecColumnInfo, InitColumnData, OrderPreservation,
        TableFunctionBuilder, TableFunctionCallbacks,
    },
    types::{List, Struct, StructSchema, StructValue, StructWrite},
};

use crate::meta::{
    backup_dir, constant_string, normalize_root, parse_json, read_infos, read_required,
};

pub(crate) struct CollectionsFunction;

/// Single-argument overload: the path is the backup directory itself.
pub(crate) struct CollectionsAtPathFunction;

pub(crate) struct BindData {
    /// The `milvus_backups` join key: the addressed backup name for the
    /// two-key form, the name recorded in backup_meta.json for the
    /// single-argument form (NULL when the meta omits it).
    backup_name: Option<String>,
    /// Fully materialized rows. Parsing happens at bind so each exec call is
    /// a pure emission of one already-parsed row instead of re-reading the
    /// same meta files per row.
    rows: Vec<CollectionRow>,
}

pub(crate) struct GlobalState {
    next_collection: AtomicUsize,
}

// One collection = one row, claimed from the shared counter like backups.
type LocalState = ();

/// DuckDB STRUCT schema of a `fields` entry. Field order must match the
/// write order in `exec`.
struct FieldStruct;

impl StructSchema for FieldStruct {
    fn fields<C: FFILink + ?Sized>(link: &C) -> Result<Vec<(&'static str, LogicalType)>> {
        Ok(vec![
            ("name", String::logical_type(link)?),
            ("field_id", i64::logical_type(link)?),
            ("data_type", String::logical_type(link)?),
            ("is_primary_key", bool::logical_type(link)?),
            ("element_type", String::logical_type(link)?),
            ("dim", i64::logical_type(link)?),
            ("auto_id", bool::logical_type(link)?),
            ("is_dynamic", bool::logical_type(link)?),
        ])
    }
}

/// DuckDB STRUCT schema of an `indexes` entry.
struct IndexStruct;

impl StructSchema for IndexStruct {
    fn fields<C: FFILink + ?Sized>(link: &C) -> Result<Vec<(&'static str, LogicalType)>> {
        Ok(vec![
            ("index_name", String::logical_type(link)?),
            ("index_type", String::logical_type(link)?),
            ("field_name", String::logical_type(link)?),
        ])
    }
}

impl TableFunctionCallbacks for CollectionsFunction {
    type BindData = BindData;
    type GlobalState = GlobalState;
    type LocalState = LocalState;

    fn bind(
        &self,
        context: &Context,
        arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        let root = normalize_root(constant_string(&arguments, 0, "root_path")?);
        let backup_name = constant_string(&arguments, 1, "backup_name")?;
        let dir = backup_dir(&root, &backup_name)?;
        bind_backup_dir(context, &dir, Some(backup_name), bind_handle)
    }

    fn init_global_state(
        &self,
        bind_data: &Self::BindData,
        context: &Context,
        column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        init_global(bind_data, context, column_data)
    }

    fn exec(
        &self,
        bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        local_state: Option<&mut Self::LocalState>,
        context: &Context,
        output: DataChunkRef<'_>,
        column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        exec_rows(
            bind_data,
            global_state,
            local_state,
            context,
            output,
            column_info,
        )
    }
}

impl TableFunctionCallbacks for CollectionsAtPathFunction {
    type BindData = BindData;
    type GlobalState = GlobalState;
    type LocalState = LocalState;

    fn bind(
        &self,
        context: &Context,
        arguments: Vec<BindArgument>,
        bind_handle: BindFunctionHandle<'_>,
    ) -> Result<Self::BindData> {
        let path = normalize_root(constant_string(&arguments, 0, "path")?);
        bind_backup_dir(context, &path, None, bind_handle)
    }

    fn init_global_state(
        &self,
        bind_data: &Self::BindData,
        context: &Context,
        column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        init_global(bind_data, context, column_data)
    }

    fn exec(
        &self,
        bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        local_state: Option<&mut Self::LocalState>,
        context: &Context,
        output: DataChunkRef<'_>,
        column_info: ExecColumnInfo<'_>,
    ) -> Result<()> {
        exec_rows(
            bind_data,
            global_state,
            local_state,
            context,
            output,
            column_info,
        )
    }
}

/// Shared bind tail for both overloads: everything after the backup
/// directory is known. `provided_name` is the two-key form's argument; the
/// single-argument form passes `None` and falls back to the name recorded in
/// backup_meta.json.
fn bind_backup_dir(
    context: &Context,
    dir: &str,
    provided_name: Option<String>,
    bind_handle: BindFunctionHandle<'_>,
) -> Result<BindData> {
    let fs = FileSystem::new(context)?;

    // backup_meta.json is the writer's "this is a backup" marker
    // (milvus-backup's own Exist() checks it); read it for a precise
    // not-found error and the backup-level fallbacks.
    let backup_meta_path = format!("{dir}/meta/backup_meta.json");
    if !fs.exists(&backup_meta_path)? {
        return Err(Error::invalid_input(format!(
            "backup not found: {dir}; expected a backup directory containing \
             meta/backup_meta.json, or use milvus_backup_collections(root_path, backup_name)"
        )));
    }
    let backup_meta: BackupMeta =
        parse_json(&read_required(&fs, &backup_meta_path)?, &backup_meta_path)?;

    // collection_meta.json is written by every format (snapshot included),
    // so its absence means a corrupt backup, not an empty one.
    let colls: Vec<CollectionInfo> = read_infos(&fs, dir, "collection_meta")?.ok_or_else(|| {
        Error::invalid_input(format!("file not found: {dir}/meta/collection_meta.json"))
    })?;
    let partitions = read_infos(&fs, dir, "partition_meta")?;
    let segments = read_infos(&fs, dir, "segment_meta")?;

    let rows = build_rows(&backup_meta, colls, partitions, segments);

    bind_handle.set_cardinality(rows.len(), true)?;
    // Threads claim collections from a shared counter, so arrival order
    // varies; users who need a stable order write ORDER BY.
    bind_handle.set_order(OrderPreservation::None)?;

    bind_handle.add_result_column("backup_name", &String::logical_type(context)?)?;
    bind_handle.add_result_column("db_name", &String::logical_type(context)?)?;
    bind_handle.add_result_column("collection_name", &String::logical_type(context)?)?;
    bind_handle.add_result_column("collection_id", &i64::logical_type(context)?)?;
    bind_handle.add_result_column("backup_timestamp", &u64::logical_type(context)?)?;
    bind_handle.add_result_column("backup_physical_timestamp", &u64::logical_type(context)?)?;
    bind_handle.add_result_column("shards_num", &i64::logical_type(context)?)?;
    bind_handle.add_result_column("consistency_level", &String::logical_type(context)?)?;
    bind_handle.add_result_column("size_bytes", &i64::logical_type(context)?)?;
    bind_handle.add_result_column("partition_count", &i64::logical_type(context)?)?;
    bind_handle.add_result_column("segment_count", &i64::logical_type(context)?)?;
    bind_handle.add_result_column("num_rows_inserted", &i64::logical_type(context)?)?;
    bind_handle.add_result_column("has_delete_logs", &bool::logical_type(context)?)?;
    bind_handle.add_result_column("has_index", &bool::logical_type(context)?)?;
    bind_handle.add_result_column("aliases", &Vec::<String>::logical_type(context)?)?;
    bind_handle.add_result_column("partition_names", &Vec::<String>::logical_type(context)?)?;
    bind_handle.add_result_column(
        "indexes",
        &Vec::<StructValue<IndexStruct>>::logical_type(context)?,
    )?;
    bind_handle.add_result_column(
        "fields",
        &Vec::<StructValue<FieldStruct>>::logical_type(context)?,
    )?;

    let backup_name = provided_name.or(backup_meta.name);
    Ok(BindData { backup_name, rows })
}

/// Shared init for both overloads.
fn init_global(
    bind_data: &BindData,
    _context: &Context,
    _column_data: InitColumnData<'_>,
) -> Result<(Option<GlobalState>, Option<usize>)> {
    let collection_count = bind_data.rows.len();
    Ok((
        Some(GlobalState {
            next_collection: AtomicUsize::new(0),
        }),
        (collection_count > 0).then_some(collection_count),
    ))
}

/// Shared exec for both overloads: emit the claimed row.
fn exec_rows(
    bind_data: &BindData,
    global_state: Option<&GlobalState>,
    _local_state: Option<&mut LocalState>,
    _context: &Context,
    mut output: DataChunkRef<'_>,
    _column_info: ExecColumnInfo<'_>,
) -> Result<()> {
    let claimed = global_state
        .expect("global state is always created by init_global_state")
        .next_collection
        .fetch_add(1, Ordering::SeqCst);

    if claimed >= bind_data.rows.len() {
        output.get_vector_at_mut::<String>(0)?.set_size(0)?;
        return Ok(());
    }

    let row = &bind_data.rows[claimed];

    macro_rules! write_column {
        ($index:expr, $ty:ty, $value:expr) => {{
            let mut vector = output.get_vector_at_mut::<$ty>($index)?;
            vector.set_size(1)?;
            vector.writer()?.set(0, $value)?;
        }};
    }

    write_column!(0, String, bind_data.backup_name.as_deref());
    write_column!(1, String, row.db_name.as_deref());
    write_column!(2, String, row.collection_name.as_deref());
    write_column!(3, i64, row.collection_id);
    write_column!(4, u64, row.backup_timestamp);
    write_column!(5, u64, row.backup_physical_timestamp);
    write_column!(6, i64, row.shards_num);
    write_column!(7, String, row.consistency_level.as_deref());
    write_column!(8, i64, row.size_bytes);
    write_column!(9, i64, row.partition_count);
    write_column!(10, i64, row.segment_count);
    write_column!(11, i64, row.num_rows_inserted);
    write_column!(12, bool, row.has_delete_logs);
    write_column!(13, bool, row.has_index);

    // LIST columns: None = the level data itself is unknown; [] would
    // mean "provably empty".
    let mut write_string_list = |index: usize, values: Option<&Vec<String>>| -> Result<()> {
        let mut vector = output.get_vector_at_mut::<List<String>>(index)?;
        vector.set_size(1)?;
        vector.writer()?.set(
            0,
            values.map(|names| {
                names
                    .iter()
                    .map(|name| Some(name.as_str()))
                    .collect::<Vec<Option<&str>>>()
            }),
        )?;
        Ok(())
    };
    write_string_list(14, row.aliases.as_ref())?;
    write_string_list(15, row.partition_names.as_ref())?;

    // LIST(STRUCT) columns: bind declared the struct field names via the
    // StructSchema impls, so writes only need the values in that order.
    let indexes = row.indexes.as_ref().map(|indexes| {
        indexes
            .iter()
            .map(|index| {
                Some(
                    StructWrite::default()
                        .field::<String>(index.index_name.as_deref())
                        .field::<String>(index.index_type.as_deref())
                        .field::<String>(index.field_name.as_deref()),
                )
            })
            .collect::<Vec<Option<StructWrite>>>()
    });
    let mut vector = output.get_vector_at_mut::<List<Struct>>(16)?;
    vector.set_size(1)?;
    vector.writer()?.set(0, indexes)?;

    let fields = row.fields.as_ref().map(|fields| {
        fields
            .iter()
            .map(|field| {
                Some(
                    StructWrite::default()
                        .field::<String>(field.name.as_deref())
                        .field::<i64>(field.field_id)
                        .field::<String>(field.data_type.as_deref())
                        .field::<bool>(field.is_primary_key)
                        .field::<String>(field.element_type.as_deref())
                        .field::<i64>(field.dim)
                        .field::<bool>(field.auto_id)
                        .field::<bool>(field.is_dynamic),
                )
            })
            .collect::<Vec<Option<StructWrite>>>()
    });
    let mut vector = output.get_vector_at_mut::<List<Struct>>(17)?;
    vector.set_size(1)?;
    vector.writer()?.set(0, fields)?;

    Ok(())
}

/// One materialized `milvus_backup_collections` row; `None` means the source
/// field is absent or unknowable, not zero.
struct CollectionRow {
    db_name: Option<String>,
    collection_name: Option<String>,
    collection_id: Option<i64>,
    backup_timestamp: Option<u64>,
    backup_physical_timestamp: Option<u64>,
    shards_num: Option<i64>,
    consistency_level: Option<String>,
    size_bytes: Option<i64>,
    partition_count: Option<i64>,
    segment_count: Option<i64>,
    num_rows_inserted: Option<i64>,
    has_delete_logs: Option<bool>,
    has_index: Option<bool>,
    aliases: Option<Vec<String>>,
    partition_names: Option<Vec<String>>,
    indexes: Option<Vec<IndexEntry>>,
    fields: Option<Vec<FieldEntry>>,
}

struct FieldEntry {
    name: Option<String>,
    field_id: Option<i64>,
    data_type: Option<String>,
    is_primary_key: Option<bool>,
    element_type: Option<String>,
    dim: Option<i64>,
    auto_id: Option<bool>,
    is_dynamic: Option<bool>,
}

struct IndexEntry {
    index_name: Option<String>,
    index_type: Option<String>,
    field_name: Option<String>,
}

fn build_rows(
    backup_meta: &BackupMeta,
    colls: Vec<CollectionInfo>,
    partitions: Option<Vec<PartitionInfo>>,
    segments: Option<Vec<SegmentInfo>>,
) -> Vec<CollectionRow> {
    // The proto documents empty/absent format as the binlog format.
    let is_snapshot = backup_meta.format.as_deref() == Some("snapshot");

    // partition id -> collection id, for segments that omit their own
    // collection id.
    let partition_owner: HashMap<i64, i64> = partitions
        .iter()
        .flatten()
        .filter_map(|partition| Some((partition.partition_id?, partition.collection_id?)))
        .collect();
    let segment_owner = |segment: &SegmentInfo| -> Option<i64> {
        segment
            .collection_id
            .or_else(|| partition_owner.get(&segment.partition_id?).copied())
    };

    colls
        .into_iter()
        .map(|coll| {
            let collection_id = coll.collection_id;

            // Entries attributable to this collection. Without an id nothing
            // is attributable, and those counts stay NULL.
            let owned_partitions: Vec<&PartitionInfo> = match (&partitions, collection_id) {
                (Some(parts), Some(id)) => parts
                    .iter()
                    .filter(|partition| partition.collection_id == Some(id))
                    .collect(),
                _ => Vec::new(),
            };
            let owned_segments: Vec<&SegmentInfo> = match (&segments, collection_id) {
                (Some(segs), Some(id)) => segs
                    .iter()
                    .filter(|segment| segment_owner(segment) == Some(id))
                    .collect(),
                _ => Vec::new(),
            };
            // An entry that resolves to no collection at all could still
            // belong to this one (missing ids in old backups), so only a
            // fully-attributed level file proves exact counts or "no deletes
            // here".
            let partitions_unattributed = partitions.as_ref().is_some_and(|parts| {
                parts
                    .iter()
                    .any(|partition| partition.collection_id.is_none())
            });
            let segments_unattributed = segments
                .as_ref()
                .is_some_and(|segs| segs.iter().any(|segment| segment_owner(segment).is_none()));
            let deletes_unattributed = match collection_id {
                Some(_) => segments_unattributed,
                // Without an id nothing proves a segment is not ours.
                None => segments.is_some(),
            };

            // Provable absence (snapshot format, no level file) reports zero;
            // everything else that cannot be attributed reports NULL.
            let partition_count = match (partitions.as_ref(), collection_id) {
                (_, None) => None,
                (None, Some(_)) if is_snapshot => Some(0),
                (None, Some(_)) => None,
                (Some(_), Some(_)) if partitions_unattributed => None,
                (Some(_), Some(_)) => Some(owned_partitions.len() as i64),
            };
            let segment_count = match (segments.as_ref(), collection_id) {
                (_, None) => None,
                (None, Some(_)) if is_snapshot => Some(0),
                (None, Some(_)) => None,
                (Some(_), Some(_)) if segments_unattributed => None,
                (Some(_), Some(_)) => Some(owned_segments.len() as i64),
            };

            let num_rows_inserted = match (&segments, collection_id) {
                // An incomplete attribution would make the sum a lower bound,
                // not the truth.
                (Some(_), Some(_)) if !segments_unattributed => Some(
                    owned_segments
                        .iter()
                        .map(|segment| segment.num_of_rows.unwrap_or(0))
                        .sum(),
                ),
                _ => None,
            };

            let partition_names = match (&partitions, collection_id) {
                (_, None) => None,
                (None, Some(_)) if is_snapshot => Some(Vec::new()),
                (None, Some(_)) => None,
                (Some(_), Some(_)) if partitions_unattributed => None,
                (Some(_), Some(_)) => Some(
                    owned_partitions
                        .iter()
                        .filter_map(|partition| partition.partition_name.clone())
                        .collect(),
                ),
            };

            // L0 segments are pure delta data at collection level, so their
            // existence alone proves deletes; everything else needs the
            // segment file attributed.
            let l0_has_data = coll.l0_segments.as_ref().is_some_and(|l0| !l0.is_empty());
            let has_delete_logs = if l0_has_data {
                Some(true)
            } else if segments.is_none() {
                // Snapshot backups provably have no segment level; a missing
                // level file in a binlog backup is unread state.
                if is_snapshot { Some(false) } else { None }
            } else if deletes_unattributed {
                None
            } else {
                Some(owned_segments.iter().any(|segment| {
                    segment
                        .deltalogs
                        .iter()
                        .any(|delta| !delta.binlogs.is_empty())
                }))
            };

            // The per-entry TSO is authoritative; older writers omit it there
            // and only stamp the backup level, which covers the whole tree.
            let backup_timestamp = coll.backup_timestamp.or(backup_meta.backup_timestamp);

            let fields = coll.schema.map(|schema| {
                schema
                    .fields
                    .unwrap_or_default()
                    .into_iter()
                    .map(|field| {
                        let dim = field
                            .type_params
                            .unwrap_or_default()
                            .into_iter()
                            .filter(|kv| kv.key.as_deref() == Some("dim"))
                            .find_map(|kv| kv.value.and_then(|value| value.parse::<i64>().ok()));
                        FieldEntry {
                            name: field.name,
                            field_id: field.field_id,
                            data_type: field.data_type.map(data_type_name),
                            is_primary_key: field.is_primary_key,
                            element_type: field.element_type.map(data_type_name),
                            dim,
                            auto_id: field.auto_id,
                            is_dynamic: field.is_dynamic,
                        }
                    })
                    .collect()
            });

            let indexes = coll.index_infos.map(|infos| {
                infos
                    .into_iter()
                    .map(|info| IndexEntry {
                        index_name: info.index_name,
                        index_type: info.index_type,
                        field_name: info.field_name,
                    })
                    .collect()
            });

            CollectionRow {
                db_name: coll.db_name,
                collection_name: coll.collection_name,
                collection_id,
                backup_timestamp,
                backup_physical_timestamp: coll.backup_physical_timestamp,
                shards_num: coll.shards_num,
                consistency_level: coll.consistency_level.map(consistency_level_name),
                size_bytes: coll.size,
                partition_count,
                segment_count,
                num_rows_inserted,
                has_delete_logs,
                has_index: coll.has_index,
                aliases: coll.aliases,
                partition_names,
                indexes,
                fields,
            }
        })
        .collect()
}

/// Names for the `DataType` proto enum. Unknown codes (a newer writer on an
/// older extension) stay visible as their raw decimal value instead of being
/// nulled away.
fn data_type_name(code: i64) -> String {
    match code {
        0 => "None",
        1 => "Bool",
        2 => "Int8",
        3 => "Int16",
        4 => "Int32",
        5 => "Int64",
        10 => "Float",
        11 => "Double",
        20 => "String",
        21 => "VarChar",
        22 => "Array",
        23 => "JSON",
        24 => "Geometry",
        25 => "Text",
        26 => "Timestamptz",
        100 => "BinaryVector",
        101 => "FloatVector",
        102 => "Float16Vector",
        103 => "BFloat16Vector",
        104 => "SparseFloatVector",
        105 => "Int8Vector",
        106 => "ArrayOfVector",
        200 => "ArrayOfStruct",
        201 => "Struct",
        other => return other.to_string(),
    }
    .to_string()
}

/// Names for the `ConsistencyLevel` proto enum, same unknown-code policy.
fn consistency_level_name(code: i64) -> String {
    match code {
        0 => "Strong",
        1 => "Session",
        2 => "Bounded",
        3 => "Eventually",
        4 => "Customized",
        other => return other.to_string(),
    }
    .to_string()
}

#[derive(Deserialize)]
struct BackupMeta {
    /// The name recorded by the writer; the single-argument overload uses it
    /// for the `backup_name` join key.
    name: Option<String>,
    format: Option<String>,
    /// Fallback for collections whose entry omits the per-collection TSO.
    backup_timestamp: Option<u64>,
}

#[derive(Deserialize)]
struct CollectionInfo {
    collection_id: Option<i64>,
    db_name: Option<String>,
    collection_name: Option<String>,
    schema: Option<SchemaInfo>,
    shards_num: Option<i64>,
    consistency_level: Option<i64>,
    /// Bytes this collection's binlogs occupy, summed by the writer.
    size: Option<i64>,
    has_index: Option<bool>,
    index_infos: Option<Vec<IndexInfo>>,
    backup_timestamp: Option<u64>,
    /// Seconds since epoch.
    backup_physical_timestamp: Option<u64>,
    aliases: Option<Vec<String>>,
    /// L0 delta segments live at collection level, not under partitions.
    l0_segments: Option<Vec<serde_json::Value>>,
}

#[derive(Deserialize)]
struct SchemaInfo {
    fields: Option<Vec<FieldInfo>>,
}

#[derive(Deserialize)]
struct FieldInfo {
    name: Option<String>,
    // The JSON mixes camelCase (fieldID, autoID) with snake_case; accept both.
    #[serde(alias = "fieldID")]
    field_id: Option<i64>,
    is_primary_key: Option<bool>,
    data_type: Option<i64>,
    element_type: Option<i64>,
    type_params: Option<Vec<KeyValuePair>>,
    #[serde(alias = "autoID", alias = "autoId")]
    auto_id: Option<bool>,
    is_dynamic: Option<bool>,
}

#[derive(Deserialize)]
struct KeyValuePair {
    key: Option<String>,
    value: Option<String>,
}

#[derive(Deserialize)]
struct IndexInfo {
    index_name: Option<String>,
    index_type: Option<String>,
    field_name: Option<String>,
}

#[derive(Deserialize)]
struct PartitionInfo {
    partition_id: Option<i64>,
    partition_name: Option<String>,
    collection_id: Option<i64>,
}

#[derive(Deserialize)]
struct SegmentInfo {
    collection_id: Option<i64>,
    partition_id: Option<i64>,
    num_of_rows: Option<i64>,
    #[serde(default)]
    deltalogs: Vec<DeltaLog>,
}

#[derive(Deserialize)]
struct DeltaLog {
    #[serde(default)]
    binlogs: Vec<serde_json::Value>,
}

pub(crate) fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "milvus_backup_collections",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("root_path"))
            .param(FunctionParameter::new::<String>("backup_name"))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .add_category("milvus")
                    .description(
                        "List the collections stored inside one Milvus backup, addressed by \
                         backup root path and backup name. One row per collection; reads only \
                         the meta JSON files, never binlog data.",
                    )
                    .add_example(
                        "list collections of a backup",
                        "SELECT * FROM milvus_backup_collections('/mnt/milvus-backup', \
                         'backup_20260101')",
                    ),
            ),
        CollectionsFunction,
    )
    .register(extension)?;

    // Overload: address the backup directory directly, without naming a root.
    TableFunctionBuilder::new(
        "milvus_backup_collections",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("path"))
            .documentation(
                FunctionDocumentationBuilder::new()
                    .add_category("milvus")
                    .description(
                        "List the collections stored inside the Milvus backup directory given \
                         by path (the directory containing meta/backup_meta.json). One row per \
                         collection; reads only the meta JSON files, never binlog data.",
                    )
                    .add_example(
                        "list collections of a backup directory",
                        "SELECT * FROM \
                         milvus_backup_collections('/mnt/milvus-backup/backup_20260101')",
                    ),
            ),
        CollectionsAtPathFunction,
    )
    .register(extension)?;

    Ok(())
}
