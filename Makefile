.PHONY: clean clean_all mingw_libgcc_eh_stub

PROJ_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

EXTENSION_NAME=milvus_backup

# Mark the extension as C_STRUCT_UNSTABLE: DuckDB 2.x requires the _init_c_api_v2 entrypoint.
# Binaries load only in the exact TARGET_DUCKDB_VERSION, not in newer DuckDB releases.
USE_UNSTABLE_C_API=1
CAPI_VERSION=2

# Target DuckDB version
TARGET_DUCKDB_VERSION=v2.0.1

all: configure debug

# Include makefiles from DuckDB
include extension-ci-tools/makefiles/c_api_extensions/base.Makefile
include extension-ci-tools/makefiles/c_api_extensions/rust.Makefile

# Build workspace libraries rather than the CI tools' Wasm example target.
IS_EXAMPLE=
ifneq ($(DUCKDB_WASM_PLATFORM),)
	TARGET_INFO=--target $(TARGET) --package milvus_backup_wasm
	# Side modules (-sSIDE_MODULE=2) require PIC object files; cc-rs omits
	# -fPIC on wasm32 by default, which breaks linking C deps like zstd-sys.
	export CFLAGS_wasm32_unknown_emscripten ?= -fPIC
else ifneq ($(filter windows_amd64_mingw windows_amd64_rtools,$(DUCKDB_PLATFORM)),)
	# CI installs the GNU Rust target and sets CC=gcc for MinGW, but
	# rust.Makefile leaves TARGET unset so Cargo defaults to MSVC and fails
	# linking MinGW-compiled C deps (zstd-sys) under link.exe.
	# Rtools' static MinGW toolchain folds unwind symbols into libgcc.a and
	# omits a separate libgcc_eh.a, so provide an empty stub archive when absent.
	TARGET=x86_64-pc-windows-gnu
	TARGET_INFO=--target $(TARGET) --package milvus_backup
	TARGET_PATH=./target/$(TARGET)
	export CARGO_TARGET_X86_64_PC_WINDOWS_GNU_RUSTFLAGS ?= -Lnative=build/mingw-stub
build_extension_library_debug build_extension_library_release: mingw_libgcc_eh_stub
else
	TARGET_INFO+=--package milvus_backup
endif

configure: venv platform extension_version

mingw_libgcc_eh_stub: check_configure
	$(PYTHON_VENV_BIN) -c "import subprocess; from pathlib import Path; p = Path('build/mingw-stub'); p.mkdir(parents=True, exist_ok=True); stub = p / 'libgcc_eh.a'; out = subprocess.check_output(['x86_64-w64-mingw32-gcc', '-print-file-name=libgcc_eh.a'], text=True).strip(); stub.write_bytes(b'!<arch>' + bytes([10])) if out == 'libgcc_eh.a' else stub.unlink(missing_ok=True)"

debug: build_extension_library_debug build_extension_with_metadata_debug
release: build_extension_library_release build_extension_with_metadata_release

test: test_debug
test_debug: test_extension_debug
test_release: test_extension_release

clean: clean_build clean_rust
clean_all: clean_configure clean
