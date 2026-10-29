# duckdb-milvus-backup

A DuckDB extension for reading [Milvus](https://github.com/milvus-io/milvus) backups with SQL—inspect collections, partitions, and binlog data before restoring a backup into a running Milvus instance.

This repository was initialized from the experimental [duckdb/extension-template-rs](https://github.com/duckdb/extension-template-rs) template. It is a pure Rust extension based on DuckDB C Extension API v2 (`duckdb-neo` + `duckdb-loadable-macros`, targeting DuckDB **v2.0.0**; versions must match exactly). It requires neither a DuckDB source build nor C/C++ code, and the `extension-ci-tools` CI/CD pipeline is preconfigured.

## Roadmap

| Phase | Function | Status | Description |
|---|---|---|---|
| 0 | `milvus_backup_version()` | ✅ Implemented | Returns the extension version embedded at compile time; also validates the end-to-end build/load/test pipeline |
| 1 | `milvus_backups(root_path)` | 🚧 Planned | Backup inventory: one row per backup with flat scalar columns (name, id, milvus_version, created_at, size_bytes, collection_count, etc.). Reads only meta JSON and never touches binlogs |
| 2 | `milvus_backup_collections(root_path, backup_name)` | 🚧 Planned | One row per collection for WHERE/ORDER BY/JOIN; secondary details such as schemas are returned only through explicitly selected columns |
| 3 | `milvus_backup_scan(root_path, backup_name, collection)` | 🚧 Planned | Scans binlog data with per-file parallelism via `claim_batch`, projection/filter pushdown, and vector columns represented as fixed-size `FLOAT[dim]` arrays |
| Future | Multi-collection scans (schema consistency checks), S3 support (endpoint/credentials via named parameters), `milvus_backup_verify()` | 💡 | |

Design principles: `SELECT *` should answer "Can this backup be trusted and used?". Information that cannot be determined from meta, such as the final row count or integrity, is represented as NULL or a boolean rather than a fabricated 0 or success status. Metadata functions perform only meta I/O and never scan binlogs.

## Quick Start

### Prerequisites

Native builds require a Rust toolchain, Python 3 with venv support, Make, and Git.

Loading the extension also requires the DuckDB **v2.0.0** CLI, exactly matching the target version. Download it from the [DuckDB v2.0.0 release](https://github.com/duckdb/duckdb/releases/tag/v2.0.0) and add it to your `PATH`. The commands below use `duckdb`; alternatively, place the CLI at `./build/duckdb` and substitute that path in the commands. The CLI is not produced by the extension build.

### Build the Extension

```shell
git clone --recurse-submodules <repo-url>
cd duckdb-milvus-backup

# Create a Python venv, install the DuckDB v2.0.0 test client, and detect the platform
DUCKDB_TEST_VERSION=v2.0.0 make configure

# Build with cargo and append extension metadata; artifacts are written to build/debug
make debug
```

For an optimized release build, run:

```shell
make release
```

### Load and Verify

Start DuckDB with `-unsigned` to load an unsigned local extension:

```shell
duckdb -unsigned
```

Use the absolute path to the extension file when loading it. The CLI version must be v2.0.0:

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

The following interfaces are planned and have not been implemented yet:

```sql
SELECT * FROM milvus_backups('/mnt/backups');
SELECT * FROM milvus_backup_collections('/mnt/backups', 'backup_20261027');
SELECT * FROM milvus_backup_scan('/mnt/backups', 'backup_20261027', collection = 'messages');
```

## Project Structure

The Cargo workspace contains three crates:

- `crates/core` (`milvus_backup_core`): function implementations and registration; all extension logic lives here.
  Convention: **one function per file** (`version.rs`, and later `backups.rs`, `collections.rs`, etc.). Each file provides its own `pub(crate) fn register`; `lib.rs` contains only module declarations and the top-level `register`.
- `crates/native` (`milvus_backup`): the native `cdylib` and DuckDB extension entry point.
- `crates/wasm` (`milvus_backup_wasm`): a Wasm `staticlib` using the same entry point.

Dependency versions are managed centrally in the root `Cargo.toml`; `upgrade.sh` upgrades duckdb-neo, duckdb-loadable-macros, and the DuckDB version together.

## WASM

In addition to the native build prerequisites, install and configure the Emscripten SDK (which provides `emcc`) and the Rust `wasm32-unknown-emscripten` target:

```shell
rustup target add wasm32-unknown-emscripten
make wasm_eh
```

The output is `./build/wasm_eh/release/milvus_backup.duckdb_extension.wasm`, which DuckDB Wasm can load in Node.js or a browser.

## Testing

Tests use the SQLLogicTest format and are located in `test/sql/`. They run through the DuckDB Python client installed by `make configure`. The test client must match the extension target version; the Quick Start command pins it to v2.0.0.

```shell
make test_debug    # Or make test_release
```

## Debugging

VS Code breakpoint debugging is preconfigured in `.vscode/`. Install the CodeLLDB extension and press **F5**. Choose "run version query" to LOAD the extension automatically and hit breakpoints, or "interactive" to run SQL manually in the integrated terminal. The debug configurations use `./build/duckdb`; prepare the CLI as described in Quick Start.

macOS notes:

- The official `build/duckdb` binary uses Hardened Runtime, which prevents debugger attachment. Re-sign it ad hoc with `codesign --force --sign - ./build/duckdb` (repeat after downloading the CLI again).
- If LOAD is killed by SIGKILL (exit 137) after `make debug`, remove the old extension artifacts before rebuilding. This works around a signed-page poisoning issue caused by inode reuse.
- DuckDB internals are not shipped with symbols, so only extension code can be debugged. If breakpoints jump around during multithreaded execution, use `SET threads TO 1`.

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
