//! RLE/bit-packed-hybrid decoding (parquet spec section "Encoding").
//!
//! Verified against real Milvus binlogs written by parquet-go 12.0.1: literal
//! runs are counted in 8-value groups (`header = groups << 1 | 1`, payload =
//! ceil(groups * 8 * bit_width / 8) bytes, LSB-first), RLE runs are counted
//! in values (`header = count << 1`, one fixed-width little-endian value).

use duckdb_neo::{Error, Result};

/// Bit width needed to represent `max_level`.
pub(crate) fn level_bit_width(max_level: u16) -> u8 {
    if max_level == 0 {
        0
    } else {
        32 - (max_level as u32).leading_zeros() as u8
    }
}

/// Decode a hybrid stream into exactly `count` values. Some writers pad the
/// final literal run, so decoding stops as soon as `count` is reached.
pub(crate) fn decode_hybrid(mut data: &[u8], bit_width: u8, count: usize) -> Result<Vec<u64>> {
    let mut out = Vec::with_capacity(count);
    while out.len() < count {
        if data.is_empty() {
            return Err(Error::invalid_input(format!(
                "RLE hybrid stream truncated: have {} of {} values",
                out.len(),
                count
            )));
        }
        // Varint header, LSB = run kind.
        let mut header: u64 = 0;
        let mut shift = 0u32;
        loop {
            let byte = *data
                .first()
                .ok_or_else(|| Error::invalid_input("RLE hybrid header truncated"))?;
            data = &data[1..];
            header |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                break;
            }
            shift += 7;
            if shift > 63 {
                return Err(Error::invalid_input("RLE hybrid header too long"));
            }
        }

        if header & 1 == 0 {
            // RLE run: one value repeated `count` times.
            let run_len = (header >> 1) as usize;
            let value_len = (bit_width as usize).div_ceil(8);
            if data.len() < value_len {
                return Err(Error::invalid_input("RLE run value truncated"));
            }
            let mut value: u64 = 0;
            for (index, byte) in data[..value_len].iter().enumerate() {
                value |= (*byte as u64) << (8 * index);
            }
            data = &data[value_len..];
            for _ in 0..run_len.min(count - out.len()) {
                out.push(value);
            }
        } else {
            // Literal run: groups of 8 values, bit-packed LSB-first.
            let groups = (header >> 1) as usize;
            let values = groups * 8;
            let payload_len = values * bit_width as usize / 8;
            if data.len() < payload_len {
                return Err(Error::invalid_input("RLE literal payload truncated"));
            }
            let payload = &data[..payload_len];
            data = &data[payload_len..];
            let base = out.len();
            for index in base..(base + values).min(count) {
                out.push(read_bits(payload, index - base, bit_width));
            }
        }
    }
    Ok(out)
}

/// Read the `index`-th bit_width value of an LSB-first bit-packed stream.
fn read_bits(data: &[u8], index: usize, bit_width: u8) -> u64 {
    if bit_width == 0 {
        return 0;
    }
    let bit_offset = index * bit_width as usize;
    let byte_offset = bit_offset / 8;
    let bit_shift = (bit_offset % 8) as u32;

    let mut value: u64 = 0;
    // A value spans at most 9 bytes (bit_width <= 64 after +7 bit shift).
    for extra in 0..((bit_width as usize + 7) / 8 + 1) {
        let byte = data
            .get(byte_offset + extra)
            .copied()
            .unwrap_or(0); // last group may be padded
        value |= (byte as u64) << (8 * extra);
    }
    (value >> bit_shift) & ((1u64 << bit_width) - 1)
}
