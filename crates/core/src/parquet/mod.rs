//! Minimal parquet reader covering exactly the subset Milvus binlogs use
//! (verified against real backups and the milvus writer sources):
//! physical types BOOLEAN/INT32/INT64/FLOAT/DOUBLE/BYTE_ARRAY/FLBA, logical
//! STRING and INT(width), DATA_PAGE V1/V2, PLAIN + dictionary encodings,
//! RLE/bit-packed hybrid levels, ZSTD/SNAPPY/UNCOMPRESSED, and standard
//! 3-level LIST columns.
//!
//! Files are read through a [`FileView`] — an (offset, length) window over a
//! byte source — so the same reader handles whole parquet files and parquet
//! payloads embedded in storage-v1 binlog envelopes, and reads go through
//! DuckDB's FileSystem (S3 range requests via httpfs included).

mod codec;
mod decode;
mod footer;
mod rle;
mod thrift;

#[cfg(test)]
mod tests;

use std::sync::Arc;

use duckdb_neo::{Error, Result, file::File};

pub(crate) use decode::{ChunkReader, ColumnData, Values, read_chunk};
pub(crate) use footer::{ColumnChunkMeta, PhysicalType, SchemaLeaf, read_footer};

/// Underlying byte source of a scan region. `Mem` backs the unit tests.
#[allow(dead_code)]
pub(crate) enum BlockSource {
    File(File),
    Mem(Arc<Vec<u8>>),
}

// SAFETY: `File` wraps an opaque duckdb file-handle pointer; duckdb file
// handles are safe for concurrent reads (the engine's own parallel readers
// share one handle across worker threads), and `Mem` is immutable.
unsafe impl Send for BlockSource {}
unsafe impl Sync for BlockSource {}

impl BlockSource {
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        match self {
            BlockSource::File(file) => file.read_exact_at(offset as usize, buf),
            BlockSource::Mem(bytes) => {
                let start = offset as usize;
                let end = start + buf.len();
                let src = bytes.get(start..end).ok_or_else(|| {
                    Error::invalid_input(format!(
                        "read {start}..{end} beyond in-memory source of {} bytes",
                        bytes.len()
                    ))
                })?;
                buf.copy_from_slice(src);
                Ok(())
            }
        }
    }

    #[allow(dead_code)] // used by whole-file callers and tests
    fn size(&self) -> Result<u64> {
        Ok(match self {
            BlockSource::File(file) => file.size()?,
            BlockSource::Mem(bytes) => bytes.len() as u64,
        })
    }
}

/// An (offset, length) window over a [`BlockSource`]. Cloning shares the
/// source; `slice` derives sub-windows (one per row group) cheaply.
#[derive(Clone)]
pub(crate) struct FileView {
    source: Arc<BlockSource>,
    base: u64,
    len: u64,
}

impl FileView {
    pub(crate) fn new(source: Arc<BlockSource>, base: u64, len: u64) -> Self {
        FileView { source, base, len }
    }

    pub(crate) fn size(&self) -> u64 {
        self.len
    }

    /// A sub-window; used to open one row group of a larger file.
    pub(crate) fn slice(&self, base: u64, len: u64) -> Result<FileView> {
        if base + len > self.len {
            return Err(Error::invalid_input(format!(
                "slice {base}..{} beyond file view of {} bytes",
                base + len,
                self.len
            )));
        }
        Ok(FileView {
            source: Arc::clone(&self.source),
            base: self.base + base,
            len,
        })
    }

    pub(crate) fn read(&self, offset: u64, buf: &mut [u8]) -> Result<()> {
        if offset + buf.len() as u64 > self.len {
            return Err(Error::invalid_input(format!(
                "read {}..{} beyond file view of {} bytes",
                offset,
                offset + buf.len() as u64,
                self.len
            )));
        }
        self.source.read_at(self.base + offset, buf)
    }

    pub(crate) fn read_vec(&self, offset: u64, len: usize) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len];
        self.read(offset, &mut buf)?;
        Ok(buf)
    }
}
