//! `milvus_backup_scan(root_path, backup_name, collection)` — read the rows
//! of one collection straight out of a Milvus backup.
//!
//! Bind reads only the meta JSON (same helpers as the other functions),
//! resolves the collection's segments, and declares the output schema from
//! the collection schema. Execution claims one segment per batch (claim_batch
//! + a shared task counter), opens only the binlog files of projected columns
//! (plus pk/timestamp when delete filtering is on), decodes them with the
//! in-house parquet reader, and applies delta-log tombstones by default so
//! the output matches what a restored Milvus would return. Reads go through
//! DuckDB's FileSystem, so `s3://` paths work via httpfs credentials.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use serde::Deserialize;

use duckdb_neo::{
    DuckDBType, Error, Parameters, Result, ToValue,
    bind_arguments::BindArgument,
    connection::{Context, FFILink},
    data_chunk::DataChunkRef,
    extension::Extension,
    file::{File, FileBuilder, FileSystem},
    logical_type::LogicalType,
    signature::{
        FunctionDocumentationBuilder, FunctionParameter, FunctionParameterKind, SignatureBuilder,
    },
    table_function::{
        BindFunctionHandle, ExecColumnInfo, InitColumnData, OrderPreservation,
        TableClaimBatchCallbacks, TableFunctionBuilder, TableFunctionCallbacks,
    },
    types::{Array, BlobValue, List, Struct, StructSchema, StructValue, StructWrite},
    value::Value,
};

use crate::binlog::{self, BinlogKind};
use crate::meta::{
    backup_dir, constant_string, normalize_root, parse_json, read_full_meta_l0, read_infos,
    read_required,
};
use crate::parquet::{self, ChunkReader, ColumnData, FileView, Values};
use crate::paths::backup_log_paths;
use crate::schema::{self, MilvusField, TargetKind};
use crate::tombstone::{Pk, Tombstones};

pub(crate) struct ScanFunction;

/// Maximum rows produced per exec call; matches DuckDB's standard vector
/// size, keeping output chunks aligned with the engine's expectations.
const CHUNK_ROWS: usize = 2048;

// --- bind-time data ---------------------------------------------------------

pub(crate) struct BindData {
    /// Declared output columns in order: system columns first (when
    /// requested), then user fields in schema order.
    columns: Vec<OutputColumn>,
    /// Data segments sorted by segment id for deterministic claiming; L0
    /// segments are tombstone sources, not tasks.
    tasks: Vec<SegmentTask>,
    /// Sum of segment num_of_rows; an upper bound (deletes shrink it).
    estimated_rows: u64,
    apply_deletes: bool,
    /// Backup-relative deltalog candidate lists (deduped), read once by
    /// init_global_state when delete filtering is active.
    delta_sources: Vec<Vec<String>>,
    pk_field_id: Option<i64>,
    dir: String,
}

struct OutputColumn {
    name: String,
    source: ColumnSource,
    target: TargetKind,
}

#[derive(Clone)]
enum ColumnSource {
    /// Field 0 (RowID), exposed by `include_system`.
    RowId,
    /// Field 1 (Timestamp), exposed by `include_system`.
    Timestamp,
    Field(MilvusField),
}

impl ColumnSource {
    fn field_id(&self) -> i64 {
        match self {
            ColumnSource::RowId => 0,
            ColumnSource::Timestamp => 1,
            ColumnSource::Field(field) => field.field_id,
        }
    }
}

struct SegmentTask {
    segment_id: i64,
    /// Every insert-log file of the segment, in meta order. Milvus 2.6+
    /// writes column-group ids (or nothing) into the meta's `fieldID`, so the
    /// true field mapping is discovered by opening each file at task-cursor
    /// build time (envelope descriptor / parquet field ids); `declared` is
    /// only a fallback for footers without field-id annotations.
    files: Vec<SegmentFile>,
}

struct SegmentFile {
    candidates: Vec<String>,
    /// fieldID the meta attributes this file to, when present.
    declared: Option<i64>,
}

// --- duckdb-neo plumbing ----------------------------------------------------

pub(crate) struct GlobalState {
    next_task: AtomicUsize,
    rows_done: AtomicU64,
    estimated_rows: u64,
    /// Built once per scan when delete filtering is active and the backup has
    /// delta logs; empty otherwise.
    tombstones: OnceLock<Arc<Tombstones>>,
}

pub(crate) struct LocalState {
    task_index: Option<usize>,
    cursor: Option<TaskCursor>,
}

impl TableFunctionCallbacks for ScanFunction {
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
        let collection = constant_string(&arguments, 2, "collection")?;
        let apply_deletes = optional_bool(&arguments, 3, "apply_deletes", true)?;
        let include_system = optional_bool(&arguments, 4, "include_system", false)?;
        let dir = backup_dir(&root, &backup_name)?;
        bind_impl(
            context,
            &dir,
            &collection,
            apply_deletes,
            include_system,
            bind_handle,
        )
    }

    fn init_global_state(
        &self,
        bind_data: &Self::BindData,
        context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<(Option<Self::GlobalState>, Option<usize>)> {
        let tombstones = OnceLock::new();
        if bind_data.apply_deletes && !bind_data.delta_sources.is_empty() {
            let built = build_tombstones(context, &bind_data.dir, &bind_data.delta_sources)?;
            tombstones
                .set(built)
                .map_err(|_| Error::api_error("tombstones already built"))?;
        }
        let global = GlobalState {
            next_task: AtomicUsize::new(0),
            rows_done: AtomicU64::new(0),
            estimated_rows: bind_data.estimated_rows,
            tombstones,
        };
        let threads = (!bind_data.tasks.is_empty()).then(|| bind_data.tasks.len());
        Ok((Some(global), threads))
    }

    fn init_local_state(
        &self,
        _bind_data: &Self::BindData,
        _global_state: Option<&Self::GlobalState>,
        _context: &Context,
        _column_data: InitColumnData<'_>,
    ) -> Result<Option<Self::LocalState>> {
        Ok(Some(LocalState {
            task_index: None,
            cursor: None,
        }))
    }

    fn progress(
        &self,
        _bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        _context: &Context,
    ) -> Result<Option<f64>> {
        let Some(global) = global_state else {
            return Ok(None);
        };
        if global.estimated_rows == 0 {
            return Ok(None);
        }
        let done = global.rows_done.load(Ordering::SeqCst);
        Ok(Some((done as f64 / global.estimated_rows as f64).min(1.0)))
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
        exec_chunk(bind_data, global_state, local_state, context, output, column_info)
    }
}

impl TableClaimBatchCallbacks for ScanFunction {
    fn claim_batch(
        &self,
        bind_data: &Self::BindData,
        global_state: Option<&Self::GlobalState>,
        local_state: Option<&mut Self::LocalState>,
        _context: &Context,
    ) -> Result<bool> {
        let Some(local) = local_state else {
            return Err(Error::api_error("scan local state missing"));
        };
        let Some(global) = global_state else {
            return Err(Error::api_error("scan global state missing"));
        };
        // claim_batch runs before a thread's first exec and after every exec
        // that produced zero rows (task finished) — each call advances to the
        // next task. File opening happens lazily in exec so this callback
        // never blocks other threads on I/O.
        let claimed = global.next_task.fetch_add(1, Ordering::SeqCst);
        if claimed >= bind_data.tasks.len() {
            local.task_index = None;
            return Ok(false);
        }
        local.task_index = Some(claimed);
        local.cursor = None;
        Ok(true)
    }
}

// --- bind -------------------------------------------------------------------

fn bind_impl(
    context: &Context,
    dir: &str,
    collection_arg: &str,
    apply_deletes: bool,
    include_system: bool,
    bind_handle: BindFunctionHandle<'_>,
) -> Result<BindData> {
    let fs = FileSystem::new(context)?;

    let backup_meta_path = format!("{dir}/meta/backup_meta.json");
    if !fs.exists(&backup_meta_path)? {
        return Err(Error::invalid_input(format!(
            "backup not found: {dir}; expected a backup directory containing \
             meta/backup_meta.json"
        )));
    }
    let backup_meta: BackupMeta =
        parse_json(&read_required(&fs, &backup_meta_path)?, &backup_meta_path)?;
    if backup_meta.format.as_deref() == Some("snapshot") {
        return Err(Error::invalid_input(
            "snapshot-format backups carry no segment meta; milvus_backup_scan does not \
             support them yet",
        ));
    }

    let colls: Vec<CollectionEntry> = read_infos(&fs, dir, "collection_meta")?.ok_or_else(|| {
        Error::invalid_input(format!("file not found: {dir}/meta/collection_meta.json"))
    })?;
    let target = resolve_collection(&colls, collection_arg)?;
    let collection_id = target.collection_id.ok_or_else(|| {
        Error::invalid_input(format!(
            "collection {collection_arg} has no collection id in the meta; segment data \
             cannot be attributed (backup too old for milvus_backup_scan)"
        ))
    })?;

    let fields = schema::parse_fields(target.schema.clone().and_then(|schema| schema.fields))?;
    if fields.is_empty() {
        return Err(Error::invalid_input(format!(
            "collection {collection_arg} has an empty schema"
        )));
    }
    let pk_field_id = fields
        .iter()
        .find(|field| field.is_primary_key)
        .map(|field| field.field_id);

    // L0 (delete-only) segments: level-file entries, falling back to
    // full_meta.json where newer writers record them exclusively.
    let full_meta_l0: Vec<SegmentEntry> = if target.l0_segments.is_none() {
        resolve_full_meta_l0(&fs, dir, collection_id)?
    } else {
        Vec::new()
    };
    let collection_l0: &[SegmentEntry] = match &target.l0_segments {
        Some(l0) => l0,
        None => &full_meta_l0,
    };

    let mut columns: Vec<OutputColumn> = Vec::new();
    if include_system {
        columns.push(OutputColumn {
            name: "__row_id".to_string(),
            source: ColumnSource::RowId,
            target: TargetKind::BigInt,
        });
        columns.push(OutputColumn {
            name: "__timestamp".to_string(),
            source: ColumnSource::Timestamp,
            target: TargetKind::BigInt,
        });
    }
    for field in &fields {
        columns.push(OutputColumn {
            name: field.name.clone(),
            source: ColumnSource::Field(field.clone()),
            target: schema::target_kind(field)?,
        });
    }

    if apply_deletes
        && pk_field_id.is_none()
        && collection_l0.iter().any(|l0| !delta_paths(l0).is_empty())
    {
        return Err(Error::invalid_input(format!(
            "collection {collection_arg} has delete logs but its schema has no primary key \
             field; pass apply_deletes = false to read raw rows"
        )));
    }

    // Segment attribution mirrors milvus_backup_collections: by collection
    // id, falling back to the owning partition's id.
    let partitions: Vec<PartitionEntry> = read_infos(&fs, dir, "partition_meta")?.unwrap_or_default();
    let segments: Vec<SegmentEntry> = read_infos(&fs, dir, "segment_meta")?.unwrap_or_default();
    let partition_owner: HashMap<i64, i64> = partitions
        .iter()
        .filter_map(|partition| Some((partition.partition_id?, partition.collection_id?)))
        .collect();

    let mut tasks = Vec::new();
    let mut delta_sources: Vec<Vec<String>> = Vec::new();
    let mut estimated_rows = 0u64;
    for segment in &segments {
        let owner = segment
            .collection_id
            .or_else(|| partition_owner.get(&segment.partition_id?).copied());
        if owner != Some(collection_id) {
            continue;
        }
        // Delete-only segments (L0 flagged or legacy delta-only entries
        // without an is_l0 key) carry no insert logs; they are tombstone
        // sources, not data tasks.
        let has_insert_logs = segment
            .binlogs
            .iter()
            .any(|field| field.binlogs.iter().any(|entry| entry.log_path.is_some()));
        if segment.is_l0.unwrap_or(false) || !has_insert_logs {
            collect_delta_candidates(segment, &mut delta_sources);
            continue;
        }
        let mut files: Vec<SegmentFile> = Vec::new();
        for field_log in &segment.binlogs {
            for binlog_entry in &field_log.binlogs {
                let Some(log_path) = &binlog_entry.log_path else { continue };
                let candidates =
                    backup_log_paths(log_path, segment.group_id, segment.partition_id)?;
                files.push(SegmentFile {
                    candidates,
                    declared: field_log.field_id,
                });
            }
        }
        collect_delta_candidates(segment, &mut delta_sources);
        estimated_rows += segment.num_of_rows.unwrap_or(0).max(0) as u64;
        tasks.push(SegmentTask {
            segment_id: segment.segment_id.unwrap_or(-1),
            files,
        });
    }
    for l0 in collection_l0 {
        collect_delta_candidates(l0, &mut delta_sources);
    }
    tasks.sort_by_key(|task| task.segment_id);

    bind_handle.set_cardinality(estimated_rows as usize, true)?;
    bind_handle.set_order(OrderPreservation::None)?;
    for column in &columns {
        let logical = logical_type_for(context, &column.target)?;
        bind_handle.add_result_column(&column.name, &logical)?;
    }

    Ok(BindData {
        columns,
        tasks,
        estimated_rows,
        apply_deletes,
        delta_sources,
        pk_field_id,
        dir: dir.to_string(),
    })
}

fn resolve_collection<'a>(
    colls: &'a [CollectionEntry],
    collection_arg: &str,
) -> Result<&'a CollectionEntry> {
    let matches: Vec<&CollectionEntry> = colls
        .iter()
        .filter(|entry| collection_matches(entry, collection_arg))
        .collect();
    match matches.len() {
        0 => {
            let available = colls
                .iter()
                .filter_map(qualified_name)
                .collect::<Vec<_>>()
                .join(", ");
            Err(Error::invalid_input(format!(
                "collection {collection_arg:?} not found in backup (available: {available})"
            )))
        }
        1 => Ok(matches[0]),
        _ => {
            let names = matches
                .iter()
                .filter_map(|entry| qualified_name(entry))
                .collect::<Vec<_>>()
                .join(", ");
            Err(Error::invalid_input(format!(
                "collection name {collection_arg:?} is ambiguous across databases; qualify \
                 it as db.collection (matches: {names})"
            )))
        }
    }
}

fn collection_matches(entry: &CollectionEntry, arg: &str) -> bool {
    if entry.collection_name.as_deref() == Some(arg) {
        return true;
    }
    match (&entry.db_name, &entry.collection_name) {
        (Some(db), Some(name)) if !db.is_empty() => format!("{db}.{name}") == arg,
        _ => false,
    }
}

fn qualified_name(entry: &CollectionEntry) -> Option<String> {
    let name = entry.collection_name.as_ref()?;
    match entry.db_name.as_deref() {
        Some(db) if !db.is_empty() => Some(format!("{db}.{name}")),
        _ => Some(name.clone()),
    }
}

/// L0 segments of one collection from `full_meta.json`, parsed into
/// [`SegmentEntry`]. Empty when the file is absent or records none.
fn resolve_full_meta_l0(
    fs: &FileSystem<'_>,
    dir: &str,
    collection_id: i64,
) -> Result<Vec<SegmentEntry>> {
    let Some(values) = read_full_meta_l0(fs, dir)?.remove(&collection_id) else {
        return Ok(Vec::new());
    };
    values
        .iter()
        .map(|value| {
            serde_json::from_value(value.clone()).map_err(|error| {
                Error::invalid_input(format!(
                    "failed to parse an L0 segment of collection {collection_id} in \
                     full_meta.json: {error}"
                ))
            })
        })
        .collect()
}

fn delta_paths(segment: &SegmentEntry) -> Vec<&String> {
    segment
        .deltalogs
        .iter()
        .flat_map(|delta| delta.binlogs.iter())
        .filter_map(|binlog| binlog.log_path.as_ref())
        .collect()
}

fn collect_delta_candidates(segment: &SegmentEntry, out: &mut Vec<Vec<String>>) {
    for field_log in &segment.deltalogs {
        for binlog_entry in &field_log.binlogs {
            let Some(log_path) = &binlog_entry.log_path else { continue };
            if let Ok(candidates) =
                backup_log_paths(log_path, segment.group_id, segment.partition_id)
            {
                if !out.iter().any(|existing| existing.first() == candidates.first()) {
                    out.push(candidates);
                }
            }
        }
    }
}

/// Build the tombstone map from every deltalog file, once per scan.
fn build_tombstones(
    context: &Context,
    dir: &str,
    delta_sources: &[Vec<String>],
) -> Result<Arc<Tombstones>> {
    let fs = FileSystem::new(context)?;
    let mut tombstones = Tombstones::default();
    for candidates in delta_sources {
        let (path, file) = open_first_candidate(&fs, dir, candidates)?;
        let view = binlog::open_view(file)?;
        let entries = Tombstones::read_file(&view).map_err(|error| {
            Error::invalid_input(format!("failed to read delta log {path}: {error}"))
        })?;
        for (pk, ts) in entries {
            tombstones.record(pk, ts);
        }
    }
    Ok(Arc::new(tombstones))
}

fn open_first_candidate(
    fs: &FileSystem<'_>,
    dir: &str,
    candidates: &[String],
) -> Result<(String, File)> {
    let mut last_error = None;
    for candidate in candidates {
        let path = format!("{dir}/{candidate}");
        match FileBuilder::new(fs, &path)?.read().open() {
            Ok(file) => return Ok((path, file)),
            Err(error) => last_error = Some(error),
        }
    }
    Err(Error::invalid_input(match last_error {
        Some(error) => format!(
            "binlog file not found in backup; tried {} candidate paths, last error: {error}",
            candidates.len()
        ),
        None => "binlog file has no candidate paths".to_string(),
    }))
}

// --- execution --------------------------------------------------------------

/// Per-segment execution state: one cursor per declared column (AllNull for
/// fields without files), plus pk/timestamp cursors when delete filtering is
/// active. Filter cursors are appended after the declared columns.
struct TaskCursor {
    segment_id: i64,
    columns: Vec<ColumnCursor>,
    /// Indices into `columns` of the (pk, ts) filter cursors.
    filter: Option<(usize, usize)>,
}

enum ColumnCursor {
    /// The field has no binlog in this segment (schema evolution): every row
    /// is NULL. Never drives row termination.
    AllNull,
    Field(FieldCursor),
}

impl ColumnCursor {
    fn has_row(&mut self) -> Result<bool> {
        match self {
            ColumnCursor::AllNull => Ok(false),
            ColumnCursor::Field(cursor) => {
                cursor.ensure_loaded()?;
                Ok(cursor.has_row())
            }
        }
    }

    fn peek(&self) -> Option<&Cell> {
        match self {
            ColumnCursor::AllNull => None,
            ColumnCursor::Field(cursor) => cursor.cells.front(),
        }
    }

    fn advance(&mut self) {
        if let ColumnCursor::Field(cursor) = self {
            cursor.cells.pop_front();
        }
    }
}

/// One insert-log file opened and resolved at task-cursor build time, so
/// every field cursor can pick up its share without reopening the file.
struct ResolvedFile {
    view: FileView,
    /// v1 envelope: the single field the file carries. Bare parquet: every
    /// field id found in the footer; when the footer carries no ids, the
    /// meta-declared field.
    fields: UnattributedFields,
    path: String,
}

#[derive(Debug)]
enum UnattributedFields {
    Single(i64),
    Bare(Vec<i64>),
}

/// One already-opened insert-log file (resolved once per task, shared by the
/// field cursors that it serves).
struct FileSource {
    view: FileView,
    path: String,
}

struct FieldCursor {
    field_id: i64,
    field_name: String,
    target: TargetKind,
    /// Remaining files of this field, in meta order.
    files: VecDeque<FileSource>,
    /// Remaining (view, leaf, chunk) triples of the current unit.
    chunks: VecDeque<(FileView, parquet::SchemaLeaf, parquet::ColumnChunkMeta)>,
    cells: VecDeque<Cell>,
    exhausted: bool,
}

impl FieldCursor {
    fn has_row(&self) -> bool {
        !self.cells.is_empty()
    }

    /// Decode the next available chunk of rows into `cells`.
    fn ensure_loaded(&mut self) -> Result<()> {
        while self.cells.is_empty() && !self.exhausted {
            if self.chunks.is_empty() {
                self.open_next_file()?;
            }
            while self.cells.is_empty() {
                let Some((view, leaf, chunk)) = self.chunks.pop_front() else { break };
                let mut reader = ChunkReader::open(view, leaf.clone(), chunk)
                    .map_err(|error| field_error(&self.field_name, error))?;
                let data = parquet::read_chunk(&mut reader)
                    .map_err(|error| field_error(&self.field_name, error))?;
                self.cells.extend(cells_from_data(
                    &data,
                    &leaf,
                    &self.target,
                    &self.field_name,
                )?);
            }
            if self.cells.is_empty() && self.chunks.is_empty() && self.files.is_empty() {
                self.exhausted = true;
            }
        }
        Ok(())
    }

    fn open_next_file(&mut self) -> Result<()> {
        while let Some(source) = self.files.pop_front() {
            if self.process_view(source.view, &source.path)? {
                return Ok(());
            }
        }
        Ok(())
    }

    /// Queue the chunks of `view` that serve this cursor's field. The file's
    /// served fields were resolved at task-cursor build time; this only
    /// double-checks the envelope case, where the descriptor is authoritative.
    fn process_view(&mut self, view: FileView, _path: &str) -> Result<bool> {
        match binlog::sniff(&view)? {
            BinlogKind::Envelope => {
                let envelope = binlog::parse_envelope(&view)?;
                if envelope.field_id != self.field_id {
                    return Ok(false);
                }
                for index in 0..envelope.events.len() {
                    let payload = envelope.event_payload(&view, index)?;
                    let meta = parquet::read_footer(&payload)?;
                    for group in &meta.row_groups {
                        self.chunks.push_back((
                            payload.clone(),
                            meta.leaves[0].clone(),
                            group.columns[0].clone(),
                        ));
                    }
                }
                Ok(!self.chunks.is_empty())
            }
            BinlogKind::BareParquet => {
                let meta = parquet::read_footer(&view)?;
                let Some(leaf_index) = meta
                    .leaves
                    .iter()
                    .position(|leaf| leaf.field_id == Some(self.field_id))
                else {
                    // Storage-v2 column groups: a field lives in exactly one
                    // group file; absence here is normal.
                    return Ok(false);
                };
                for group in &meta.row_groups {
                    self.chunks.push_back((
                        view.clone(),
                        meta.leaves[leaf_index].clone(),
                        group.columns[leaf_index].clone(),
                    ));
                }
                Ok(true)
            }
        }
    }
}

fn field_error(field_name: &str, error: duckdb_neo::Error) -> Error {
    Error::invalid_input(format!("segment field {field_name}: {error}"))
}

fn exec_chunk(
    bind_data: &BindData,
    global_state: Option<&GlobalState>,
    local_state: Option<&mut LocalState>,
    context: &Context,
    mut output: DataChunkRef<'_>,
    column_info: ExecColumnInfo<'_>,
) -> Result<()> {
    let Some(local) = local_state else {
        return Err(Error::api_error("scan local state missing"));
    };
    let Some(global) = global_state else {
        return Err(Error::api_error("scan global state missing"));
    };
    let Some(task_index) = local.task_index.filter(|index| *index < bind_data.tasks.len()) else {
        // Claimed past the end: nothing to produce.
        for out_index in 0..column_info.count()? {
            let bind_index = column_info.get_column_index(out_index)?;
            set_column_size(
                &mut output,
                out_index,
                &bind_data.columns[bind_index].target,
                0,
            )?;
        }
        return Ok(());
    };

    if local.cursor.is_none() {
        let fs = FileSystem::new(context)?;
        let projected: Vec<usize> = (0..column_info.count()?)
            .map(|position| column_info.get_column_index(position))
            .collect::<Result<Vec<_>>>()?;
        // Delete filtering reads pk + timestamp even when not projected.
        let tombstones = global.tombstones.get();
        let filtering = bind_data.apply_deletes
            && tombstones.is_some_and(|tombstones| !tombstones.is_empty());
        local.cursor = Some(build_task_cursor(
            &fs,
            &bind_data.dir,
            bind_data,
            task_index,
            &projected,
            filtering,
        )?);
    }
    let cursor = local.cursor.as_mut().expect("cursor built above");
    let tombstones = global.tombstones.get();

    // Output position -> declared column index.
    let projection: Vec<usize> = (0..column_info.count()?)
        .map(|position| column_info.get_column_index(position))
        .collect::<Result<Vec<_>>>()?;

    let mut rows = 0usize;
    loop {
        if rows == CHUNK_ROWS {
            break;
        }
        let mut any_row = false;
        for column in &mut cursor.columns {
            if column.has_row()? {
                any_row = true;
            }
        }
        if !any_row {
            break;
        }
        // Cross-field alignment: every field-backed cursor must move in
        // lockstep; a cursor that ran dry while others still have rows means
        // a corrupt segment.
        for column in &mut cursor.columns {
            if let ColumnCursor::Field(field) = column {
                if !field.has_row() {
                    return Err(Error::invalid_input(format!(
                        "segment {}: row count mismatch between field logs ({} ran out \
                         while other fields still have rows)",
                        cursor.segment_id, field.field_name
                    )));
                }
            }
        }

        // Delete filtering.
        if let Some((pk_index, ts_index)) = cursor.filter {
            let Some(tombstones) = tombstones else {
                return Err(Error::api_error("filter active without tombstones"));
            };
            let (pk_cell, ts_cell) = match (cursor.columns[pk_index].peek(), cursor.columns[ts_index].peek()) {
                (Some(pk), Some(ts)) => (pk, ts),
                _ => {
                    return Err(Error::invalid_input(format!(
                        "segment {}: pk/timestamp logs ended before data rows",
                        cursor.segment_id
                    )))
                }
            };
            if tombstone_kills(tombstones, pk_cell, ts_cell) {
                for column in &mut cursor.columns {
                    column.advance();
                }
                continue;
            }
        }

        // Emit the surviving row into the projected output vectors.
        for (out_index, bind_index) in projection.iter().enumerate() {
            let cell = cursor.columns[*bind_index].peek();
            write_cell(&mut output, out_index, rows, &bind_data.columns[*bind_index].target, cell)?;
        }
        for column in &mut cursor.columns {
            column.advance();
        }
        rows += 1;
    }

    for out_index in 0..projection.len() {
        let bind_index = projection[out_index];
        set_column_size(&mut output, out_index, &bind_data.columns[bind_index].target, rows)?;
    }
    global
        .rows_done
        .fetch_add(rows as u64, Ordering::SeqCst);
    Ok(())
}

/// Resize one projected output vector; needed because `get_vector_at_mut`
/// validates the declared column type.
fn set_column_size(
    output: &mut DataChunkRef<'_>,
    out_index: usize,
    target: &TargetKind,
    size: usize,
) -> Result<()> {
    macro_rules! resize {
        ($ty:ty) => {{
            let mut vector = output.get_vector_at_mut::<$ty>(out_index)?;
            vector.set_size(size)?;
        }};
    }
    match target {
        TargetKind::Boolean => resize!(bool),
        TargetKind::TinyInt => resize!(i8),
        TargetKind::SmallInt => resize!(i16),
        TargetKind::Int => resize!(i32),
        TargetKind::BigInt => resize!(i64),
        TargetKind::Float => resize!(f32),
        TargetKind::Double => resize!(f64),
        TargetKind::VarChar => resize!(String),
        TargetKind::Blob => resize!(BlobValue),
        TargetKind::FloatArray(_) => resize!(Array<f32>),
        TargetKind::SparseStruct => resize!(Struct),
        TargetKind::List(element) => match element.as_ref() {
            TargetKind::Boolean => resize!(List<bool>),
            TargetKind::TinyInt => resize!(List<i8>),
            TargetKind::SmallInt => resize!(List<i16>),
            TargetKind::Int => resize!(List<i32>),
            TargetKind::BigInt => resize!(List<i64>),
            TargetKind::Float => resize!(List<f32>),
            TargetKind::Double => resize!(List<f64>),
            TargetKind::VarChar => resize!(List<String>),
            TargetKind::Blob => resize!(List<BlobValue>),
            other => {
                return Err(Error::invalid_input(format!(
                    "nested {other:?} list elements are not supported yet"
                )))
            }
        },
    }
    Ok(())
}

fn tombstone_kills(tombstones: &Tombstones, pk_cell: &Cell, ts_cell: &Cell) -> bool {
    let pk = match pk_cell {
        Cell::Int(value) => Pk::Int(*value),
        Cell::Bytes(bytes) => match std::str::from_utf8(bytes) {
            Ok(text) => Pk::Str(text.to_string()),
            Err(_) => return false, // non-UTF8 pk cannot match a string tombstone
        },
        _ => return false,
    };
    let insert_ts = match ts_cell {
        Cell::Int(value) => *value as u64,
        _ => return false,
    };
    tombstones.is_deleted(&pk, insert_ts)
}

fn build_task_cursor(
    fs: &FileSystem<'_>,
    dir: &str,
    bind_data: &BindData,
    task_index: usize,
    projected: &[usize],
    filtering: bool,
) -> Result<TaskCursor> {
    let task = &bind_data.tasks[task_index];
    // Open every insert-log file once and discover which fields it actually
    // serves. The meta's fieldID cannot be trusted for this: Milvus 2.6
    // column-group logs store the group id (or nothing) there, while the
    // authoritative mapping lives in the parquet field-id annotations.
    let mut resolved: Vec<ResolvedFile> = Vec::new();
    for file in &task.files {
        let (path, file_handle) = open_first_candidate(fs, dir, &file.candidates)?;
        let view = binlog::open_view(file_handle)?;
        let fields = match binlog::sniff(&view)? {
            BinlogKind::Envelope => {
                let envelope = binlog::parse_envelope(&view)?.field_id;
                if let Some(declared) = file.declared
                    && declared != envelope
                {
                    return Err(Error::invalid_input(format!(
                        "binlog {path} carries field {envelope} but the meta assigns it to \
                         field {declared}"
                    )));
                }
                UnattributedFields::Single(envelope)
            }
            BinlogKind::BareParquet => {
                let leaves = parquet::read_footer(&view)?.leaves;
                let ids: Vec<i64> =
                    leaves.iter().filter_map(|leaf| leaf.field_id).collect();
                if ids.is_empty() {
                    // Old writer: bare parquet without field-id annotations;
                    // the meta attribution is all we have.
                    match file.declared {
                        Some(declared) => UnattributedFields::Single(declared),
                        None => UnattributedFields::Bare(Vec::new()),
                    }
                } else {
                    UnattributedFields::Bare(ids)
                }
            }
        };
        resolved.push(ResolvedFile { view, fields, path });
    }

    let mut columns: Vec<ColumnCursor> = Vec::new();
    let mut filter: Option<(usize, usize)> = None;

    for (bind_index, column) in bind_data.columns.iter().enumerate() {
        let needed = projected.contains(&bind_index);
        if !needed {
            columns.push(ColumnCursor::AllNull);
            continue;
        }
        columns.push(build_column_cursor(&resolved, column)?);
    }
    if filtering {
        let pk_field_id = bind_data.pk_field_id.ok_or_else(|| {
            Error::api_error("delete filtering without a primary key field")
        })?;
        let pk_column = OutputColumn {
            name: format!("pk:{pk_field_id}"),
            source: ColumnSource::Field(
                bind_data
                    .columns
                    .iter()
                    .find_map(|column| match &column.source {
                        ColumnSource::Field(field) if field.field_id == pk_field_id => {
                            Some(field.clone())
                        }
                        _ => None,
                    })
                    .ok_or_else(|| {
                        Error::invalid_input(format!(
                            "segment {}: primary key field {pk_field_id} is not in the schema",
                            task.segment_id
                        ))
                    })?,
            ),
            target: TargetKind::BigInt,
        };
        // The pk target must match the field's actual type (Int64 or VarChar).
        let mut pk_column = pk_column;
        if let ColumnSource::Field(field) = &pk_column.source {
            pk_column.target = schema::target_kind(field)?;
        }
        let ts_column = OutputColumn {
            name: "__timestamp".to_string(),
            source: ColumnSource::Timestamp,
            target: TargetKind::BigInt,
        };
        let pk_cursor = build_column_cursor(&resolved, &pk_column)?;
        if matches!(pk_cursor, ColumnCursor::AllNull) {
            return Err(Error::invalid_input(format!(
                "segment {}: primary key field {pk_field_id} has no binlog file",
                task.segment_id
            )));
        }
        let ts_cursor = build_column_cursor(&resolved, &ts_column)?;
        if matches!(ts_cursor, ColumnCursor::AllNull) {
            return Err(Error::invalid_input(format!(
                "segment {}: has delete logs to apply but no timestamp log (field 1)",
                task.segment_id
            )));
        }
        filter = Some((columns.len(), columns.len() + 1));
        columns.push(pk_cursor);
        columns.push(ts_cursor);
    }

    Ok(TaskCursor {
        segment_id: task.segment_id,
        columns,
        filter,
    })
}

fn build_column_cursor(
    resolved: &[ResolvedFile],
    column: &OutputColumn,
) -> Result<ColumnCursor> {
    let field_id = column.source.field_id();
    // Files whose discovered contents serve this field, in meta order. Empty
    // => the field has no data in this segment (schema evolution): every row
    // is NULL.
    let mut files: VecDeque<FileSource> = VecDeque::new();
    for file in resolved {
        let serves = match &file.fields {
            UnattributedFields::Single(single) => *single == field_id,
            UnattributedFields::Bare(fields) => fields.contains(&field_id),
        };
        if serves {
            files.push_back(FileSource {
                view: file.view.clone(),
                path: file.path.clone(),
            });
        }
    }
    if files.is_empty() {
        return Ok(ColumnCursor::AllNull);
    }
    let field_name = match &column.source {
        ColumnSource::Field(field) => field.name.clone(),
        ColumnSource::RowId => "__row_id".to_string(),
        ColumnSource::Timestamp => "__timestamp".to_string(),
    };
    Ok(ColumnCursor::Field(FieldCursor {
        field_id,
        field_name,
        target: column.target.clone(),
        files,
        chunks: VecDeque::new(),
        cells: VecDeque::new(),
        exhausted: false,
    }))
}

// --- row assembly -----------------------------------------------------------

/// One decoded row value, converted from the parquet page representation.
enum Cell {
    Null,
    Bool(bool),
    Int(i64),
    Float(f32),
    Double(f64),
    Bytes(Vec<u8>),
    Sparse(Vec<u32>, Vec<f32>),
    List(Vec<Cell>),
}

/// Convert one decoded chunk into row cells following the declared target.
fn cells_from_data(
    data: &ColumnData,
    leaf: &parquet::SchemaLeaf,
    target: &TargetKind,
    field_name: &str,
) -> Result<VecDeque<Cell>> {
    if matches!(target, TargetKind::List(_)) {
        return list_cells(data, leaf, target, field_name);
    }
    let mut cells = VecDeque::with_capacity(data.rows());
    let mut value_index = 0usize;
    for row in 0..data.rows() {
        if !data.is_valid(row) {
            cells.push_back(Cell::Null);
            continue;
        }
        cells.push_back(value_cell(&data.values, value_index, target, field_name)?);
        value_index += 1;
    }
    Ok(cells)
}

fn value_cell(
    values: &parquet::Values,
    index: usize,
    target: &TargetKind,
    field_name: &str,
) -> Result<Cell> {
    let mismatch = |expected: &str| {
        Error::invalid_input(format!(
            "field {field_name}: parquet page holds {values:?}, incompatible with {expected}"
        ))
    };
    Ok(match target {
        TargetKind::Boolean => match values {
            parquet::Values::Bool(values) => {
                Cell::Bool(*values.get(index).ok_or_else(|| mismatch("BOOLEAN"))?)
            }
            other => return Err(mismatch(&format!("BOOLEAN (got {other:?})"))),
        },
        TargetKind::TinyInt | TargetKind::SmallInt | TargetKind::Int => match values {
            parquet::Values::Int32(values) => {
                let value = *values.get(index).ok_or_else(|| mismatch("INT"))?;
                let (min, max, name): (i64, i64, &str) = match target {
                    TargetKind::TinyInt => (i8::MIN as i64, i8::MAX as i64, "INT8"),
                    TargetKind::SmallInt => (i16::MIN as i64, i16::MAX as i64, "INT16"),
                    _ => (i32::MIN as i64, i32::MAX as i64, "INT32"),
                };
                if (value as i64) < min || (value as i64) > max {
                    return Err(Error::invalid_input(format!(
                        "field {field_name}: value {value} does not fit {name}"
                    )));
                }
                Cell::Int(value as i64)
            }
            other => return Err(mismatch(&format!("INT (got {other:?})"))),
        },
        TargetKind::BigInt => match values {
            parquet::Values::Int64(values) => {
                Cell::Int(*values.get(index).ok_or_else(|| mismatch("BIGINT"))?)
            }
            other => return Err(mismatch(&format!("BIGINT (got {other:?})"))),
        },
        TargetKind::Float => match values {
            parquet::Values::Float(values) => {
                Cell::Float(*values.get(index).ok_or_else(|| mismatch("FLOAT"))?)
            }
            other => return Err(mismatch(&format!("FLOAT (got {other:?})"))),
        },
        TargetKind::Double => match values {
            parquet::Values::Double(values) => {
                Cell::Double(*values.get(index).ok_or_else(|| mismatch("DOUBLE"))?)
            }
            other => return Err(mismatch(&format!("DOUBLE (got {other:?})"))),
        },
        TargetKind::VarChar => Cell::Bytes(take_bytes(values, index, field_name)?),
        TargetKind::Blob => Cell::Bytes(take_bytes(values, index, field_name)?),
        TargetKind::FloatArray(dim) => {
            let bytes = take_bytes(values, index, field_name)?;
            if bytes.len() != dim * 4 {
                return Err(Error::invalid_input(format!(
                    "field {field_name}: vector row is {} bytes, expected {} (dim {dim} x float32)",
                    bytes.len(),
                    dim * 4
                )));
            }
            Cell::Bytes(bytes)
        }
        TargetKind::SparseStruct => {
            let bytes = take_bytes(values, index, field_name)?;
            let (indices, values) = decode_sparse(&bytes, field_name)?;
            Cell::Sparse(indices, values)
        }
        TargetKind::List(_) => {
            return Err(Error::invalid_input(format!(
                "field {field_name}: LIST column reached non-list decoding"
            )))
        }
    })
}

fn take_bytes(values: &parquet::Values, index: usize, field_name: &str) -> Result<Vec<u8>> {
    let Values::Bytes { data, offsets } = values else {
        return Err(Error::invalid_input(format!(
            "field {field_name}: expected a byte-array page, found {values:?}"
        )));
    };
    let start = *offsets
        .get(index)
        .ok_or_else(|| Error::invalid_input(format!("field {field_name}: page row out of range")))?
        as usize;
    let end = *offsets
        .get(index + 1)
        .ok_or_else(|| Error::invalid_input(format!("field {field_name}: page row out of range")))?
        as usize;
    Ok(data[start..end].to_vec())
}

/// SparseFloatVector blob: consecutive little-endian `<u32 index><f32 value>`
/// pairs (verified against real BM25 backups; no header, no prefix).
fn decode_sparse(bytes: &[u8], field_name: &str) -> Result<(Vec<u32>, Vec<f32>)> {
    if bytes.len() % 8 != 0 {
        return Err(Error::invalid_input(format!(
            "field {field_name}: sparse vector blob is {} bytes, not a multiple of 8",
            bytes.len()
        )));
    }
    let mut indices = Vec::with_capacity(bytes.len() / 8);
    let mut values = Vec::with_capacity(bytes.len() / 8);
    for pair in bytes.chunks_exact(8) {
        indices.push(u32::from_le_bytes(pair[..4].try_into().unwrap()));
        values.push(f32::from_le_bytes(pair[4..].try_into().unwrap()));
    }
    Ok((indices, values))
}

/// Assemble LIST rows from definition + repetition levels (native Array
/// fields), following the standard parquet 3-level LIST mapping. Known
/// limitation: with an OPTIONAL list group AND optional element, a list whose
/// first element is NULL is indistinguishable from an empty list at the same
/// definition level; this reader reports the empty list.
fn list_cells(
    data: &ColumnData,
    leaf: &parquet::SchemaLeaf,
    target: &TargetKind,
    field_name: &str,
) -> Result<VecDeque<Cell>> {
    let Some(rep_levels) = &data.rep_levels else {
        return Err(Error::invalid_input(format!(
            "field {field_name}: LIST column has no repetition levels"
        )));
    };
    let Some(def_levels) = &data.def_levels else {
        return Err(Error::invalid_input(format!(
            "field {field_name}: LIST column has no definition levels"
        )));
    };
    let TargetKind::List(element) = target else {
        return Err(Error::invalid_input(format!(
            "field {field_name}: expected a LIST target"
        )));
    };

    let mut out = VecDeque::new();
    let mut current: Vec<Cell> = Vec::new();
    let mut current_null = false;
    let mut leading_null = false;
    let mut value_index = 0usize;

    for row in 0..data.rows() {
        if rep_levels[row] == 0 && row > 0 {
            out.push_back(if current_null {
                Cell::Null
            } else {
                Cell::List(std::mem::take(&mut current))
            });
            current = Vec::new();
            current_null = false;
            leading_null = false;
        }
        if leading_null && rep_levels[row] > 0 {
            // The rep==0 entry was a null FIRST element, not an empty list.
            current.push(Cell::Null);
            leading_null = false;
        }
        let level = def_levels[row];
        // Definition levels accumulate one bit per OPTIONAL node: the LIST
        // group (when optional), the repeated `list` wrapper, and the element
        // (when optional). Empty lists therefore sit at max_def - 2 when the
        // element is optional and at max_def - 1 when it is required.
        let element_optional =
            data.max_def - 1 - u16::from(leaf.list_group_optional) == 1;
        if level == data.max_def {
            current.push(value_cell(&data.values, value_index, element, field_name)?);
            value_index += 1;
        } else if level == 0 {
            if !leaf.list_group_optional {
                return Err(Error::invalid_input(format!(
                    "field {field_name}: null LIST row in a required list group"
                )));
            }
            current_null = true;
        } else if element_optional && level + 1 == data.max_def {
            if rep_levels[row] == 0 && !leaf.list_group_optional {
                // Required group: an empty list and a null first element
                // encode identically; disambiguate from later entries.
                leading_null = true;
            } else {
                current.push(Cell::Null);
            }
        } else if (element_optional && leaf.list_group_optional && level + 2 == data.max_def)
            || (!element_optional && level + 1 == data.max_def)
        {
            // Empty list: no element to append; the row closes empty.
        } else {
            return Err(Error::invalid_input(format!(
                "field {field_name}: unexpected definition level {level} (max {})",
                data.max_def
            )));
        }
    }
    out.push_back(if current_null {
        Cell::Null
    } else {
        Cell::List(current)
    });
    Ok(out)
}

// --- output writing ---------------------------------------------------------

/// Borrowed accessors with tied input/output lifetimes; closure shorthand
/// cannot express the elision the vector writers need.
fn bytes_of(cell: &Cell) -> Option<&[u8]> {
    match cell {
        Cell::Bytes(bytes) => Some(bytes),
        _ => None,
    }
}

fn str_of(cell: &Cell) -> Option<&str> {
    match cell {
        Cell::Bytes(bytes) => std::str::from_utf8(bytes).ok(),
        _ => None,
    }
}

/// Write one cell into the output vector at (`out_index`, `row`).
fn write_cell(
    output: &mut DataChunkRef<'_>,
    out_index: usize,
    row: usize,
    target: &TargetKind,
    cell: Option<&Cell>,
) -> Result<()> {
    macro_rules! scalar {
        ($ty:ty, $extract:expr) => {{
            let mut vector = output.get_vector_at_mut::<$ty>(out_index)?;
            vector.set_size(row + 1)?;
            let value = match cell {
                None | Some(Cell::Null) => None,
                Some(value) => match $extract(value) {
                    Some(converted) => Some(converted),
                    None => {
                        return Err(Error::api_error(format!(
                            "scan output column {out_index}: unexpected cell for {}",
                            std::any::type_name::<$ty>()
                        )))
                    }
                },
            };
            vector.writer()?.set(row, value)?;
        }};
    }

    match target {
        TargetKind::Boolean => scalar!(bool, |cell: &Cell| match cell {
            Cell::Bool(value) => Some(*value),
            _ => None,
        }),
        TargetKind::TinyInt => scalar!(i8, |cell: &Cell| match cell {
            Cell::Int(value) => i8::try_from(*value).ok(),
            _ => None,
        }),
        TargetKind::SmallInt => scalar!(i16, |cell: &Cell| match cell {
            Cell::Int(value) => i16::try_from(*value).ok(),
            _ => None,
        }),
        TargetKind::Int => scalar!(i32, |cell: &Cell| match cell {
            Cell::Int(value) => i32::try_from(*value).ok(),
            _ => None,
        }),
        TargetKind::BigInt => scalar!(i64, |cell: &Cell| match cell {
            Cell::Int(value) => Some(*value),
            _ => None,
        }),
        TargetKind::Float => scalar!(f32, |cell: &Cell| match cell {
            Cell::Float(value) => Some(*value),
            _ => None,
        }),
        TargetKind::Double => scalar!(f64, |cell: &Cell| match cell {
            Cell::Double(value) => Some(*value),
            _ => None,
        }),
        TargetKind::VarChar => {
            let mut vector = output.get_vector_at_mut::<String>(out_index)?;
            vector.set_size(row + 1)?;
            let value = match cell {
                None | Some(Cell::Null) => None,
                Some(Cell::Bytes(bytes)) => {
                    let text = std::str::from_utf8(bytes).map_err(|error| {
                        Error::invalid_input(format!("VARCHAR column holds invalid UTF-8: {error}"))
                    })?;
                    Some(text)
                }
                Some(_) => {
                    return Err(Error::api_error("scan output: non-string cell in VARCHAR column"))
                }
            };
            vector.writer()?.set(row, value)?;
        }
        TargetKind::Blob => scalar!(BlobValue, bytes_of),
        TargetKind::FloatArray(dim) => {
            let mut vector = output.get_vector_at_mut::<Array<f32>>(out_index)?;
            vector.set_size(row + 1)?;
            let value = match cell {
                None | Some(Cell::Null) => None,
                Some(Cell::Bytes(bytes)) => {
                    let floats = bytes
                        .chunks_exact(4)
                        .map(|chunk| Some(f32::from_le_bytes(chunk.try_into().unwrap())))
                        .collect::<Vec<_>>();
                    if floats.len() != *dim {
                        return Err(Error::invalid_input(format!(
                            "vector row has {} elements, expected {dim}",
                            floats.len()
                        )));
                    }
                    Some(floats)
                }
                Some(_) => return Err(Error::api_error("scan output: non-blob cell in vector column")),
            };
            vector.writer()?.set(row, value)?;
        }
        TargetKind::SparseStruct => {
            let mut vector = output.get_vector_at_mut::<Struct>(out_index)?;
            vector.set_size(row + 1)?;
            let value = match cell {
                None | Some(Cell::Null) => None,
                Some(Cell::Sparse(indices, values)) => Some(
                    StructWrite::default()
                        .field::<List<u32>>(Some(indices.iter().map(|&index| Some(index)).collect()))
                        .field::<List<f32>>(Some(values.iter().map(|&value| Some(value)).collect())),
                ),
                Some(_) => {
                    return Err(Error::api_error("scan output: non-sparse cell in sparse column"))
                }
            };
            vector.writer()?.set(row, value)?;
        }
        TargetKind::List(element) => write_list_cell(output, out_index, row, element, cell)?,
    }
    Ok(())
}

fn write_list_cell(
    output: &mut DataChunkRef<'_>,
    out_index: usize,
    row: usize,
    element: &TargetKind,
    cell: Option<&Cell>,
) -> Result<()> {
    let elements: Option<&Vec<Cell>> = match cell {
        None | Some(Cell::Null) => None,
        Some(Cell::List(elements)) => Some(elements),
        Some(_) => return Err(Error::api_error("scan output: non-list cell in LIST column")),
    };
    // `$convert` maps a non-null Cell to the element's write value (None when
    // the cell variant does not fit the declared element type — an internal
    // inconsistency reported as an error).
    macro_rules! list {
        ($ty:ty, $elem:ty, $convert:expr) => {{
            let mut vector = output.get_vector_at_mut::<List<$ty>>(out_index)?;
            vector.set_size(row + 1)?;
            let value = match elements {
                None => None,
                Some(rows) => {
                    let converted = rows
                        .iter()
                        .map(|cell| match cell {
                            Cell::Null => Some(None),
                            cell => ($convert)(cell).map(Some),
                        })
                        .collect::<Option<Vec<Option<$elem>>>>();
                    Some(converted.ok_or_else(|| {
                        Error::api_error(
                            "scan output: list element does not match the declared column type",
                        )
                    })?)
                }
            };
            vector.writer()?.set(row, value)?;
        }};
    }
    match element {
        TargetKind::Boolean => {
            list!(bool, bool, |cell: &Cell| match cell {
                Cell::Bool(value) => Some(*value),
                _ => None,
            })
        }
        TargetKind::TinyInt => {
            list!(i8, i8, |cell: &Cell| match cell {
                Cell::Int(value) => i8::try_from(*value).ok(),
                _ => None,
            })
        }
        TargetKind::SmallInt => {
            list!(i16, i16, |cell: &Cell| match cell {
                Cell::Int(value) => i16::try_from(*value).ok(),
                _ => None,
            })
        }
        TargetKind::Int => {
            list!(i32, i32, |cell: &Cell| match cell {
                Cell::Int(value) => i32::try_from(*value).ok(),
                _ => None,
            })
        }
        TargetKind::BigInt => {
            list!(i64, i64, |cell: &Cell| match cell {
                Cell::Int(value) => Some(*value),
                _ => None,
            })
        }
        TargetKind::Float => {
            list!(f32, f32, |cell: &Cell| match cell {
                Cell::Float(value) => Some(*value),
                _ => None,
            })
        }
        TargetKind::Double => {
            list!(f64, f64, |cell: &Cell| match cell {
                Cell::Double(value) => Some(*value),
                _ => None,
            })
        }
        TargetKind::VarChar => {
            list!(String, &str, str_of)
        }
        TargetKind::Blob => {
            list!(BlobValue, &[u8], bytes_of)
        }
        other => {
            return Err(Error::invalid_input(format!(
                "nested {other:?} list elements are not supported yet"
            )))
        }
    }
    Ok(())
}

// --- logical types ----------------------------------------------------------

/// DuckDB STRUCT schema of the sparse-vector output column.
struct SparseStructSchema;

impl StructSchema for SparseStructSchema {
    fn fields<C: FFILink + ?Sized>(link: &C) -> Result<Vec<(&'static str, LogicalType)>> {
        Ok(vec![
            ("indices", Vec::<u32>::logical_type(link)?),
            ("values", Vec::<f32>::logical_type(link)?),
        ])
    }
}

fn logical_type_for(link: &Context, target: &TargetKind) -> Result<LogicalType> {
    Ok(match target {
        TargetKind::Boolean => bool::logical_type(link)?,
        TargetKind::TinyInt => i8::logical_type(link)?,
        TargetKind::SmallInt => i16::logical_type(link)?,
        TargetKind::Int => i32::logical_type(link)?,
        TargetKind::BigInt => i64::logical_type(link)?,
        TargetKind::Float => f32::logical_type(link)?,
        TargetKind::Double => f64::logical_type(link)?,
        TargetKind::VarChar => String::logical_type(link)?,
        TargetKind::Blob => BlobValue::logical_type(link)?,
        TargetKind::FloatArray(dim) => {
            let child = f32::logical_type(link)?;
            let child_value = Value::from_logical_type(link, &child)?;
            let length = (*dim as u64).value(link)?;
            link.logical_type_create("ARRAY", Parameters::positional(&[&child_value, &length]))?
        }
        TargetKind::SparseStruct => StructValue::<SparseStructSchema>::logical_type(link)?,
        TargetKind::List(element) => {
            let child = logical_type_for(link, element)?;
            let child_value = Value::from_logical_type(link, &child)?;
            link.logical_type_create("LIST", Parameters::positional(&[&child_value]))?
        }
    })
}

// --- meta JSON shapes -------------------------------------------------------

#[derive(Deserialize)]
struct BackupMeta {
    #[allow(dead_code)]
    name: Option<String>,
    format: Option<String>,
}

#[derive(Deserialize)]
struct CollectionEntry {
    collection_id: Option<i64>,
    db_name: Option<String>,
    collection_name: Option<String>,
    schema: Option<SchemaEntry>,
    /// Collection-level L0 segments (pure delta data).
    l0_segments: Option<Vec<SegmentEntry>>,
}

#[derive(Deserialize, Clone)]
struct SchemaEntry {
    fields: Option<Vec<schema::RawField>>,
}

#[derive(Deserialize)]
struct PartitionEntry {
    partition_id: Option<i64>,
    collection_id: Option<i64>,
}

#[derive(Deserialize)]
struct SegmentEntry {
    segment_id: Option<i64>,
    collection_id: Option<i64>,
    partition_id: Option<i64>,
    num_of_rows: Option<i64>,
    /// Insert logs, grouped per field.
    #[serde(default)]
    binlogs: Vec<FieldBinlog>,
    /// Delete logs; non-empty stubs `[{ }]` are tolerated.
    #[serde(default)]
    deltalogs: Vec<FieldBinlog>,
    is_l0: Option<bool>,
    group_id: Option<i64>,
}

#[derive(Deserialize)]
struct FieldBinlog {
    #[serde(alias = "fieldID", alias = "field_id")]
    field_id: Option<i64>,
    #[serde(default)]
    binlogs: Vec<BinlogRef>,
}

#[derive(Deserialize)]
struct BinlogRef {
    log_path: Option<String>,
}

// --- helpers ----------------------------------------------------------------

/// Named boolean option; absent or NULL falls back to the default.
fn optional_bool(
    arguments: &[BindArgument],
    index: usize,
    name: &str,
    default: bool,
) -> Result<bool> {
    let Some(argument) = arguments.get(index) else {
        return Ok(default);
    };
    let Some(value) = argument.value.as_ref() else {
        return Ok(default);
    };
    value.get::<bool>()?.ok_or_else(|| {
        Error::invalid_input(format!("{name} must not be NULL"))
    })
}

// --- registration -----------------------------------------------------------

pub(crate) fn register(extension: &Extension) -> Result<()> {
    TableFunctionBuilder::new(
        "milvus_backup_scan",
        SignatureBuilder::new()
            .param(FunctionParameter::new::<String>("root_path"))
            .param(FunctionParameter::new::<String>("backup_name"))
            .param(FunctionParameter::new::<String>("collection"))
            .param(
                FunctionParameter::with_default::<bool>("apply_deletes", true)
                    .kind(FunctionParameterKind::Named),
            )
            .param(
                FunctionParameter::with_default::<bool>("include_system", false)
                    .kind(FunctionParameterKind::Named),
            )
            .documentation(
                FunctionDocumentationBuilder::new()
                    .add_category("milvus")
                    .description(
                        "Read the rows of one collection inside a Milvus backup as a table. \
                         Delta-log tombstones are applied by default (pass apply_deletes = \
                         false to see raw inserted rows); vector columns decode to FLOAT[dim] \
                         arrays and sparse vectors to STRUCT(indices, values). Reads binlog \
                         data, so it touches every matching file.",
                    )
                    .add_example(
                        "scan a collection",
                        "SELECT * FROM milvus_backup_scan('/mnt/milvus-backup', \
                         'backup_20260101', 'messages')",
                    )
                    .add_example(
                        "count rows without decoding vectors",
                        "SELECT count(*) FROM milvus_backup_scan('/mnt/milvus-backup', \
                         'backup_20260101', 'messages')",
                    ),
            ),
        ScanFunction,
    )
    .with_projection_pushdown(true)
    .with_claim_batch()
    .with_progress()
    .register(extension)?;

    Ok(())
}
