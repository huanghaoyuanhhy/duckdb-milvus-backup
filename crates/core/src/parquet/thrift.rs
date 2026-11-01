//! Minimal Thrift compact-protocol reader, just enough for parquet footers
//! and page headers: varints, zigzag integers, binary strings, and
//! fields/lists/structs with skip support for unknown members. Parquet never
//! uses sets/maps in the structures we parse, so the skip path covers them
//! only coarsely.

use duckdb_neo::{Error, Result};

pub(crate) const T_STOP: u8 = 0;
pub(crate) const T_BOOL_TRUE: u8 = 1;
pub(crate) const T_BOOL_FALSE: u8 = 2;
pub(crate) const T_BYTE: u8 = 3;
pub(crate) const T_I16: u8 = 4;
pub(crate) const T_I32: u8 = 5;
pub(crate) const T_I64: u8 = 6;
pub(crate) const T_DOUBLE: u8 = 7;
pub(crate) const T_BINARY: u8 = 8;
pub(crate) const T_LIST: u8 = 9;
pub(crate) const T_SET: u8 = 10;
pub(crate) const T_MAP: u8 = 11;
pub(crate) const T_STRUCT: u8 = 12;

const MAX_VARINT_LEN: usize = 10;

pub(crate) struct ThriftReader<'a> {
    data: &'a [u8],
    pos: usize,
    /// Field id of the previous field header inside the current struct;
    /// compact protocol encodes field ids as deltas from it. Nested structs
    /// each get their own baseline, restored when their STOP byte arrives.
    last_field_id: i16,
    /// Saved `last_field_id` baselines of enclosing structs.
    struct_stack: Vec<i16>,
}

impl<'a> ThriftReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        ThriftReader {
            data,
            pos: 0,
            last_field_id: 0,
            struct_stack: Vec::new(),
        }
    }

    fn byte(&mut self) -> Result<u8> {
        let byte = *self
            .data
            .get(self.pos)
            .ok_or_else(|| Error::invalid_input("truncated thrift data"))?;
        self.pos += 1;
        Ok(byte)
    }

    /// ULEB128 varint.
    pub(crate) fn varint(&mut self) -> Result<u64> {
        let mut value: u64 = 0;
        let mut shift = 0u32;
        for _ in 0..MAX_VARINT_LEN {
            let byte = self.byte()?;
            value |= ((byte & 0x7f) as u64) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
            shift += 7;
        }
        Err(Error::invalid_input("thrift varint too long"))
    }

    /// Zigzag-decoded i32/i64 (thrift compact writes both the same way).
    pub(crate) fn zigzag(&mut self) -> Result<i64> {
        let raw = self.varint()?;
        Ok(((raw >> 1) as i64) ^ -((raw & 1) as i64))
    }

    pub(crate) fn binary(&mut self) -> Result<&'a [u8]> {
        let len = self.varint()? as usize;
        let bytes = self
            .data
            .get(self.pos..self.pos + len)
            .ok_or_else(|| Error::invalid_input("truncated thrift binary"))?;
        self.pos += len;
        Ok(bytes)
    }

    /// Read one field header. Returns `(field_id, type)`; the end of a struct
    /// is `(0, T_STOP)`, which also restores the enclosing struct's field-id
    /// baseline. Boolean values are encoded in the header type itself.
    pub(crate) fn field_header(&mut self) -> Result<(i16, u8)> {
        let byte = self.byte()?;
        if byte == T_STOP {
            self.last_field_id = self.struct_stack.pop().unwrap_or(0);
            return Ok((0, T_STOP));
        }
        let compact_type = byte & 0x0f;
        let delta = byte >> 4;
        let field_id = if delta == 0 {
            self.zigzag()? as i16
        } else {
            self.last_field_id + delta as i16
        };
        self.last_field_id = field_id;
        Ok((field_id, compact_type))
    }

    /// Reset the field-id delta baseline when entering a nested struct.
    pub(crate) fn enter_struct(&mut self) {
        self.struct_stack.push(self.last_field_id);
        self.last_field_id = 0;
    }

    /// List/set header: `(element_type, element_count)`.
    pub(crate) fn list_header(&mut self) -> Result<(u8, usize)> {
        let byte = self.byte()?;
        let element_type = byte & 0x0f;
        let mut size = (byte >> 4) as usize;
        if size == 0xf {
            size = self.varint()? as usize;
        }
        Ok((element_type, size))
    }

    pub(crate) fn skip(&mut self, compact_type: u8) -> Result<()> {
        match compact_type {
            T_STOP => Ok(()),
            T_BOOL_TRUE | T_BOOL_FALSE => Ok(()), // value lives in the header
            T_BYTE => {
                self.byte()?;
                Ok(())
            }
            T_I16 | T_I32 | T_I64 => {
                self.zigzag()?;
                Ok(())
            }
            T_DOUBLE => {
                self.pos += 8;
                if self.pos > self.data.len() {
                    return Err(Error::invalid_input("truncated thrift double"));
                }
                Ok(())
            }
            T_BINARY => {
                self.binary()?;
                Ok(())
            }
            T_LIST | T_SET => {
                let (element_type, size) = self.list_header()?;
                for _ in 0..size {
                    self.skip(element_type)?;
                }
                Ok(())
            }
            T_MAP => {
                // Compact map: kv-type byte (unless empty), varint size, pairs.
                let byte = self.byte()?;
                if byte == 0 {
                    return Ok(());
                }
                let size = self.varint()? as usize;
                for _ in 0..size {
                    self.skip(byte >> 4)?;
                    self.skip(byte & 0x0f)?;
                }
                Ok(())
            }
            T_STRUCT => {
                self.enter_struct();
                loop {
                    let (_, nested_type) = self.field_header()?;
                    if nested_type == T_STOP {
                        return Ok(());
                    }
                    self.skip(nested_type)?;
                }
            }
            other => Err(Error::invalid_input(format!(
                "unsupported thrift type {other}"
            ))),
        }
    }

    /// Bytes consumed so far (page-header callers use this to find the page
    /// body start).
    pub(crate) fn consumed(&self) -> usize {
        self.pos
    }
}
