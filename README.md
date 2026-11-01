# duckdb-milvus-backup

A DuckDB extension for reading [Milvus](https://github.com/milvus-io/milvus) backups with SQL. Point it at a backup directory — on local disk or `s3://` — and inspect collections, schemas, and row data before (or instead of) restoring it into a running Milvus instance.

Both on-disk formats are supported: the classic binlog layout written by [milvus-backup](https://github.com/zilliztech/milvus-backup), and the snapshot bundles exported by Milvus 3.0 servers.

## Why

Looking inside a backup the normal way means restoring it first: a running Milvus of the matching version, load time, and mutations to a live instance — all to answer questions that are read-only. This extension answers them directly from the backup files:

- What backups do I have, how big are they, how old, and what do they cover?
- Is this backup trustworthy — right schema, indexes recorded, row counts sane — before I restore it somewhere that matters?
- What does the data actually look like? Export it to parquet/CSV, spot-check rows, join it against other tables, or diff two backups.
- Which rows were deleted or upserted away? The raw inserted rows and the hidden system columns (`__row_id`, `__timestamp`) are both reachable.

## Quick Start

You need a Rust toolchain, Python 3 with venv support, Make, and Git to build, plus the DuckDB **v2.0.0** CLI to load the result. The CLI version must match the extension's target version exactly — download it from the [DuckDB v2.0.0 release](https://github.com/duckdb/duckdb/releases/tag/v2.0.0) and put it on your `PATH` (or at `./build/duckdb`; the CLI is not produced by the extension build).

```shell
git clone --recurse-submodules <repo-url>
cd duckdb-milvus-backup

# Creates a Python venv, installs the DuckDB v2.0.0 test client, detects the platform
DUCKDB_TEST_VERSION=v2.0.0 make configure

# Builds with cargo and appends extension metadata; artifacts land in build/debug
make debug        # or: make release
```

Local extensions are unsigned, so start DuckDB with `-unsigned` and `LOAD` the file by absolute path:

```shell
duckdb -unsigned
```

```sql
LOAD '/abs/path/to/build/debug/extension/milvus_backup/milvus_backup.duckdb_extension';

SELECT milvus_backup_version();
```

```
┌─────────────────────────┐
│ milvus_backup_version() │
│         varchar         │
├─────────────────────────┤
│ milvus_backup 0.1.0     │
└─────────────────────────┘
```

## A Walkthrough

Suppose `/mnt/backups` holds two backups produced by `milvus-backup`: `backup_20261027` (binlog format) and `backup_20261101` (snapshot format). The same three steps work identically against an `s3://` root.

### 1. Inventory the backups

One row per backup, from meta JSON only — binlog files are never touched at this stage:

```sql
SELECT name, milvus_version, format, databases, collection_names,
       num_rows_inserted, has_delete_logs, has_rbac
FROM milvus_backups('/mnt/backups');
```

```
┌─────────────────┬────────────────┬──────────┬────────────┬───────────────────────────┬───────────────────┬─────────────────┬──────────┐
│      name       │ milvus_version │  format  │ databases  │     collection_names      │ num_rows_inserted │ has_delete_logs │ has_rbac │
│     varchar     │    varchar     │ varchar  │ varchar[]  │         varchar[]         │       int64       │     boolean     │ boolean  │
├─────────────────┼────────────────┼──────────┼────────────┼───────────────────────────┼───────────────────┼─────────────────┼──────────┤
│ backup_20261027 │ v3.0.0         │ binlog   │ [db1, db2] │ [db1.messages, db1.users] │                 7 │ true            │ true     │
│ backup_20261101 │ v3.0.0         │ snapshot │ [db9]      │ [db9.docs]                │              NULL │ false           │ NULL     │
└─────────────────┴────────────────┴──────────┴────────────┴───────────────────────────┴───────────────────┴─────────────────┴──────────┘
```

The NULLs are deliberate: snapshot bundles do not record `has_rbac`, and meta-only inspection cannot know their row count. Information the meta does not carry comes back as NULL, never as a fabricated zero.

### 2. Inspect one backup

One row per collection, with the schema as `LIST(STRUCT)`:

```sql
SELECT collection_name, db_name, num_rows_inserted,
       has_delete_logs, has_index,
       fields[1].name AS pk_field, fields[2].name AS vec_field, fields[2].dim AS vector_dim
FROM milvus_backup_collections('/mnt/backups', 'backup_20261027');
```

```
┌─────────────────┬─────────┬───────────────────┬─────────────────┬──────────┬──────────┬───────────┬────────────┐
│ collection_name │ db_name │ num_rows_inserted │ has_delete_logs │ has_index │ pk_field │ vec_field │ vector_dim │
│     varchar     │ varchar │       int64       │     boolean     │ boolean  │ varchar  │  varchar  │   int64    │
├─────────────────┼─────────┼───────────────────┼─────────────────┼──────────┼──────────┼───────────┼────────────┤
│ messages        │ db1     │                 7 │ true            │ true     │ id       │ vec       │          4 │
│ users           │ db1     │              NULL │ NULL            │ NULL     │ id       │ NULL      │       NULL │
└─────────────────┴─────────┴───────────────────┴─────────────────┴──────────┴──────────┴───────────┴────────────┘
```

The `users` row comes from an older backup whose partition/segment entries predate collection ids, so nothing can be attributed to it — the counts stay NULL rather than pretending to be zero. Pass the backup directory itself to address one backup directly: `milvus_backup_collections('/mnt/backups/backup_20261027')`.

### 3. Read the rows

`milvus_backup_scan` returns one column per schema field. Vectors arrive as `FLOAT[dim]` arrays, so they slice and unnest like any DuckDB array:

```sql
-- Final rows: delta-log tombstones are applied by default, matching what a
-- restored Milvus would return.
SELECT count(*) FROM milvus_backup_scan('/mnt/backups', 'backup_20261027', 'messages');

-- Spot-check rows; vector slices work, JSON fields stay VARCHAR.
-- (id 1 is tombstoned, see step 4, so it does not appear.)
SELECT id, vec[1:4], meta
FROM milvus_backup_scan('/mnt/backups', 'backup_20261027', 'messages')
WHERE id < 3;
```

```
┌──────────────┐
│ count_star() │
│    int64     │
├──────────────┤
│      5       │
└──────────────┘
┌───────┬───────────────────────┬─────────────────┐
│  id   │       vec[1:4]        │      meta       │
│ int64 │      float[4]         │     varchar     │
├───────┼───────────────────────┼─────────────────┤
│     0 │ [0.1, 0.2, 0.3, 0.4]  │ {"lang": "en"}  │
│     2 │ [0.5, 0.6, 0.7, 0.8]  │ {"lang": "zh"}  │
└───────┴───────────────────────┴─────────────────┘
```

Only the columns you reference are read — `SELECT count(*)` never decodes vector data, and `LIMIT` stops early.

### 4. Look at deleted rows

This backup has delete logs, and two of the seven inserted rows were tombstoned (one by a segment-local delta log, one by a collection-level L0 segment). Turn tombstones off, or add the system columns, to see the raw picture:

```sql
-- Raw inserted rows, including those later deleted or upserted away.
SELECT count(*)
FROM milvus_backup_scan('/mnt/backups', 'backup_20261027', 'messages', apply_deletes = false);

-- Row identity and the TSO Milvus assigned each row.
SELECT id, __row_id, __timestamp
FROM milvus_backup_scan('/mnt/backups', 'backup_20261027', 'messages',
                        include_system = true, apply_deletes = false)
WHERE id <= 1;
```

```
┌──────────────┐
│ count_star() │
│    int64     │
├──────────────┤
│      7       │
└──────────────┘
┌───────┬──────────┬─────────────┐
│  id   │ __row_id │ __timestamp │
│ int64 │  int64   │    int64    │
├───────┼──────────┼─────────────┤
│     0 │     1000 │          50 │
│     1 │     1001 │          55 │
└───────┴──────────┴─────────────┘
```

## Reading from S3

Every function accepts `s3://` roots because they read through DuckDB's file system. Load `httpfs`, provide credentials, and pass URIs instead of paths:

```sql
LOAD 'httpfs';

CREATE SECRET (
    TYPE S3,
    KEY_ID 'AKIA...',
    SECRET '...',
    REGION 'us-east-1'
);
-- Or reuse the ambient provider chain:
-- CREATE SECRET (TYPE S3, PROVIDER CREDENTIAL_CHAIN);

SELECT name, format, num_rows_inserted
FROM milvus_backups('s3://my-bucket/milvus-backups');

SELECT count(*)
FROM milvus_backup_scan('s3://my-bucket/milvus-backups', 'backup_20261027', 'messages');
```

## Function Reference

All four functions are read-only. Metadata functions (`milvus_backups`, `milvus_backup_collections`) read only meta JSON and never touch binlogs; only `milvus_backup_scan` reads data files. Anything the meta cannot determine is NULL.

### `milvus_backups(root_path)`

One row per backup found under `root_path` (a directory containing one or more backups). Unknown roots scan to zero rows, since the path is treated as a glob.

| Column | Type | Description |
|---|---|---|
| `path` | `VARCHAR` | Backup directory as resolved under the root |
| `name` | `VARCHAR` | Backup name (directory name) |
| `id` | `VARCHAR` | Backup id from `backup_meta.json` |
| `milvus_version` | `VARCHAR` | Milvus version the backup was taken with |
| `format` | `VARCHAR` | `binlog` or `snapshot` |
| `created_at` | `TIMESTAMPTZ` | Backup start time, with fallbacks for older writers |
| `backup_timestamp` | `UBIGINT` | Backup TSO (hybrid timestamp) |
| `size_bytes` | `BIGINT` | Total backup size recorded in meta |
| `databases` | `VARCHAR[]` | Databases covered, from `database_backups` when present |
| `collection_names` | `VARCHAR[]` | Qualified `db.collection` names |
| `collection_count` | `BIGINT` | |
| `partition_count` | `BIGINT` | Snapshot backups: 0 (no partition level) |
| `segment_count` | `BIGINT` | Snapshot backups: 0 (no segment level in meta) |
| `num_rows_inserted` | `BIGINT` | NULL when the meta does not record it (snapshots) |
| `has_delete_logs` | `BOOLEAN` | NULL when undeterminable; stub delta-log entries do not count |
| `has_rbac` | `BOOLEAN` | NULL when the backup carries no rbac_meta |

### `milvus_backup_collections(root_path, backup_name)`

One row per collection inside a single backup. Unlike `milvus_backups`, an unknown backup name is an error, because the backup directory is located exactly. A single-argument overload `milvus_backup_collections(path)` takes the backup directory itself and reads the backup name from its meta.

| Column | Type | Description |
|---|---|---|
| `backup_name` | `VARCHAR` | |
| `db_name` | `VARCHAR` | |
| `collection_name` | `VARCHAR` | |
| `collection_id` | `BIGINT` | NULL in old backups that did not record it |
| `backup_timestamp` | `UBIGINT` | Backup TSO, falling back to the backup-level value |
| `backup_physical_timestamp` | `UBIGINT` | Unix seconds, where the writer recorded it |
| `shards_num` | `BIGINT` | |
| `consistency_level` | `VARCHAR` | Proto enum rendered as a name, e.g. `Bounded` |
| `size_bytes` | `BIGINT` | |
| `partition_count` | `BIGINT` | NULL when entries cannot be attributed to this collection |
| `segment_count` | `BIGINT` | Same attribution rule |
| `num_rows_inserted` | `BIGINT` | Same attribution rule; NULL for snapshot backups |
| `has_delete_logs` | `BOOLEAN` | From delta logs and L0 segments attributed to the collection |
| `has_index` | `BOOLEAN` | |
| `aliases` | `VARCHAR[]` | |
| `partition_names` | `VARCHAR[]` | |
| `fields` | `STRUCT(name, field_id, data_type, is_primary_key, element_type, dim, auto_id, is_dynamic)[]` | Schema fields; `data_type` is the Milvus type name, `dim` set for vectors |
| `indexes` | `STRUCT(index_name, index_type, field_name)[]` | Recorded index metadata |

### `milvus_backup_scan(root_path, backup_name, collection)`

Reads the data files of one collection and returns one column per schema field, named by field name. A two-argument overload `milvus_backup_scan(path, collection)` takes the backup directory itself. Named options:

| Parameter | Default | Description |
|---|---|---|
| `apply_deletes` | `true` | Apply delta-log tombstones with Milvus's semantics: a row dies when a delete record for its primary key carries a strictly larger timestamp |
| `include_system` | `false` | Prepend `__row_id` (field 0) and `__timestamp` (raw TSO, field 1), both `BIGINT` |

`collection` may be ambiguous across databases; qualify it as `db.collection` when it is. Projection pushdown means only referenced files are read, and `LIMIT` stops early.

### Milvus-to-DuckDB types

How field types map in scan output:

| Milvus type | DuckDB type |
|---|---|
| `Bool` | `BOOLEAN` |
| `Int8` / `Int16` / `Int32` / `Int64` | `TINYINT` / `SMALLINT` / `INTEGER` / `BIGINT` |
| `Float` / `Double` | `FLOAT` / `DOUBLE` |
| `String` / `VarChar` / `Text` / `JSON` | `VARCHAR` (cast with `::JSON` or query with `->>` when needed) |
| `FloatVector(dim)` | `FLOAT[dim]` fixed-size array |
| `SparseFloatVector` | `STRUCT(indices UINTEGER[], values FLOAT[])` |
| `Array` | `LIST(element_type)`; vector elements degrade to `BLOB` |
| `BinaryVector`, `Float16Vector`, `BFloat16Vector`, `Int8Vector`, `Geometry` | `BLOB` (raw bytes, lossless: float16 is `2·dim` little-endian bytes, binary vector is `(dim+7)/8` bytes) |

## Snapshot-Format Backups

Milvus 3.0 servers export backups as self-contained snapshot bundles instead of copied binlogs. All functions understand them; scanning walks the bundle's metadata JSON, per-segment avro manifests, and milvus-storage packed parquet data files, and applies L0 tombstones exactly like the binlog format.

```sql
SELECT * FROM milvus_backup_collections('/mnt/backups', 'backup_20261101');
SELECT count(*) FROM milvus_backup_scan('/mnt/backups', 'backup_20261101', 'docs');
```

Two writer-side caveats:

- Data files are the exact bytes the Milvus server wrote, so scanning requires parquet — the upstream default (`dataNode.storage.format = parquet`). Clusters configured with `vortex` cannot be scanned; the error names the column-group format. TEXT fields stored through milvus-storage's LOB layer decode when inline and error on oversized LOB references.
- `milvus_backup_collections` reports bundle counts (partitions, segments) without reading the per-segment manifests, so `num_rows_inserted` and `has_delete_logs` stay NULL — scan the collection for real rows.

## Development

The repository was initialized from the experimental [duckdb/extension-template-rs](https://github.com/duckdb/extension-template-rs) template. It is a pure Rust extension based on DuckDB C Extension API v2 (`duckdb-neo` + `duckdb-loadable-macros`, targeting DuckDB **v2.0.0**; versions must match exactly), requiring neither a DuckDB source build nor C/C++ code; the `extension-ci-tools` CI/CD pipeline is preconfigured.

### Testing

Tests use the SQLLogicTest format and live in `test/sql/`. They run through the DuckDB Python client installed by `make configure`, pinned to v2.0.0:

```shell
make test_debug    # Or make test_release
```

### Debugging

VS Code breakpoint debugging is preconfigured in `.vscode/`. Install the CodeLLDB extension and press **F5**: "run version query" LOADs the extension automatically and hits breakpoints, "interactive" runs SQL manually in the integrated terminal. The debug configurations use `./build/duckdb`.

macOS notes:

- The official `build/duckdb` binary uses Hardened Runtime, which prevents debugger attachment. Re-sign it ad hoc with `codesign --force --sign - ./build/duckdb` (repeat after downloading the CLI again).
- If LOAD is killed by SIGKILL (exit 137) after `make debug`, remove the old extension artifacts before rebuilding. This works around a signed-page poisoning issue caused by inode reuse.
- DuckDB internals are not shipped with symbols, so only extension code can be debugged. If breakpoints jump around during multithreaded execution, use `SET threads TO 1`.

### WASM

In addition to the native prerequisites, install and configure the Emscripten SDK (which provides `emcc`) and the Rust target:

```shell
rustup target add wasm32-unknown-emscripten
make wasm_eh
```

The output is `./build/wasm_eh/release/milvus_backup.duckdb_extension.wasm`, which DuckDB Wasm can load in Node.js or a browser.

### Project Structure

The Cargo workspace contains three crates:

- `crates/core` (`milvus_backup_core`): function implementations and registration; all extension logic lives here.
  Convention: **one function per file** (`version.rs`, `backups.rs`, `collections.rs`, etc.). Each file provides its own `pub(crate) fn register`; `lib.rs` contains only module declarations and the top-level `register`.
- `crates/native` (`milvus_backup`): the native `cdylib` and DuckDB extension entry point.
- `crates/wasm` (`milvus_backup_wasm`): a Wasm `staticlib` using the same entry point.

Dependency versions are managed centrally in the root `Cargo.toml`; `upgrade.sh` upgrades duckdb-neo, duckdb-loadable-macros, and the DuckDB version together.

### Changing the DuckDB Target Version

`DUCKDB_TEST_VERSION` controls only the test client version; **it does not change the extension build target**. To change the target version, use `upgrade.sh` to update duckdb-neo, duckdb-loadable-macros, the `Makefile`, the workflow, and the extension entry point together. Both crates must correspond to the same DuckDB version.

The example below uses the crate versions for v2.0.0. To target another release, update the first three variables:

```shell
NEO_VERSION=0.20000.0
MACROS_VERSION=1.20000.0
DUCKDB_VERSION=v2.0.0

./upgrade.sh "$NEO_VERSION" "$MACROS_VERSION"
cargo update
make clean_all
DUCKDB_TEST_VERSION="$DUCKDB_VERSION" make configure
make debug
make test_debug
```

`make clean_all` deletes the entire `build/` directory. If the CLI is located at `build/duckdb`, download it again and repeat the ad-hoc signing step before debugging.

### Roadmap

On the horizon: multi-collection scans with schema consistency checks, and a `milvus_backup_verify()` for integrity checks.
