//! Page compression codecs. Milvus binlogs are ZSTD (parquet-go writer) or
//! UNCOMPRESSED; SNAPPY covers arrow-written storage-v2 files. Other codecs
//! fail with the codec id visible rather than being silently misread.

use duckdb_neo::{Error, Result};

/// parquet `CompressionCodec` enum values we handle.
pub(crate) const CODEC_UNCOMPRESSED: i32 = 0;
pub(crate) const CODEC_SNAPPY: i32 = 1;
pub(crate) const CODEC_ZSTD: i32 = 6;

pub(crate) fn decompress(codec: i32, src: &[u8], uncompressed_size: usize) -> Result<Vec<u8>> {
    match codec {
        CODEC_UNCOMPRESSED => {
            if src.len() < uncompressed_size {
                return Err(Error::invalid_input(format!(
                    "uncompressed page shorter than header size: {} < {uncompressed_size}",
                    src.len()
                )));
            }
            Ok(src[..uncompressed_size].to_vec())
        }
        CODEC_SNAPPY => snap::raw::Decoder::new()
            .decompress_vec(src)
            .map_err(|error| Error::invalid_input(format!("snappy page decode: {error}"))),
        CODEC_ZSTD => zstd::bulk::decompress(src, uncompressed_size).map_err(|error| {
            Error::invalid_input(format!("zstd page decode: {error}"))
        }),
        other => Err(Error::invalid_input(format!(
            "unsupported parquet compression codec {other} (expected UNCOMPRESSED, \
             SNAPPY or ZSTD; Milvus only writes those)"
        ))),
    }
}
