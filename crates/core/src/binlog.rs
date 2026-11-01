//! Milvus binlog file layer. A backup's binlog files come in two shapes,
//! distinguished by the leading magic:
//!
//! * storage v1 (storage_version 0/1, Milvus ≤2.4 and l0compact output): a
//!   "v1 event envelope" — magic `0xfffabc`, a descriptor event, then data
//!   events, each event header followed by a bare parquet payload. Byte
//!   layout verified against `milvus-backup/internal/l0compact/envelope.go`
//!   and real backups.
//! * storage v2 (storage_version 2/3): a bare parquet file whose columns
//!   carry thrift field ids matching the Milvus field ids.
//!
//! All integers are little-endian; Go's binary.Read writes sequential
//! (unaligned) struct fields.

use std::sync::Arc;

use duckdb_neo::{Error, Result, file::File};
use serde::Deserialize;

use crate::parquet::FileView;

pub(crate) const V1_MAGIC: i32 = 0xfffabc;

/// Event type codes inside the v1 envelope.
pub(crate) const EVENT_INSERT: i8 = 1;
#[allow(dead_code)] // tests assert it; the reader itself sniffs via the magic
pub(crate) const EVENT_DELETE: i8 = 2;

const EVENT_HEADER_SIZE: usize = 17;
const DATA_EVENT_FIX_PART: usize = 16;
const DESCRIPTOR_FIX_PART: usize = 52;
const DESCRIPTOR_POST_HEADER: usize = 8;

/// What the leading magic says about the file.
pub(crate) enum BinlogKind {
    /// Storage-v1 event envelope with embedded parquet payloads.
    Envelope,
    /// Storage-v2 bare parquet file.
    BareParquet,
}

/// One data event of a v1 envelope; the payload is a complete parquet file.
pub(crate) struct EnvelopeEvent {
    #[allow(dead_code)] // part of the parsed envelope; tests assert it
    pub(crate) type_code: i8,
    pub(crate) payload_offset: u64,
    pub(crate) payload_len: u64,
}

impl EnvelopeEvent {
    #[allow(dead_code)] // insert/delete distinction kept for diagnostics
    pub(crate) fn is_insert(&self) -> bool {
        self.type_code == EVENT_INSERT
    }
}

#[derive(Deserialize)]
struct EnvelopeExtras {
    #[serde(default)]
    nullable: Option<bool>,
    #[serde(default)]
    version: Option<String>,
}

/// Parsed v1 envelope descriptor + event index.
pub(crate) struct Envelope {
    pub(crate) field_id: i64,
    /// Milvus `DataType` code of the payload column (deltalogs carry String
    /// for the JSON variant or Int64 for MULTI_FIELD); parsed for
    /// completeness, the scan layer trusts the collection schema instead.
    #[allow(dead_code)]
    pub(crate) payload_data_type: i32,
    /// Parsed from the descriptor extras; the parquet def levels are
    /// authoritative for decoding.
    #[allow(dead_code)]
    pub(crate) nullable: Option<bool>,
    /// Descriptor extras `version == "MULTI_FIELD"`: payload is a 2-column
    /// {pk, ts} parquet instead of DeleteLog JSON strings.
    pub(crate) multi_field: bool,
    pub(crate) events: Vec<EnvelopeEvent>,
}

impl Envelope {
    /// The payload views must be derived from the same file view.
    pub(crate) fn event_payload(&self, view: &FileView, index: usize) -> Result<FileView> {
        let event = self
            .events
            .get(index)
            .ok_or_else(|| Error::invalid_input(format!("envelope event {index} out of range")))?;
        view.slice(event.payload_offset, event.payload_len)
    }
}

/// Sniff the binlog kind from the leading magic.
pub(crate) fn sniff(view: &FileView) -> Result<BinlogKind> {
    let magic = view.read_vec(0, 4)?;
    if magic == *b"PAR1" {
        Ok(BinlogKind::BareParquet)
    } else if magic == V1_MAGIC.to_le_bytes() {
        Ok(BinlogKind::Envelope)
    } else {
        Err(Error::invalid_input(format!(
            "not a Milvus binlog file: leading bytes {:?} are neither PAR1 nor the \
             storage-v1 envelope magic",
            magic
        )))
    }
}

/// Open a backup binlog file as a view over its bytes.
pub(crate) fn open_view(file: File) -> Result<FileView> {
    let len = file.size()?;
    Ok(FileView::new(Arc::new(crate::parquet::BlockSource::File(file)), 0, len))
}

/// Parse the v1 envelope: descriptor + data events. Payload bytes are left in
/// place; callers derive `FileView`s via [`Envelope::event_payload`].
pub(crate) fn parse_envelope(view: &FileView) -> Result<Envelope> {
    let mut cursor = Cursor {
        view,
        pos: 0,
        size: view.size(),
    };
    let magic = cursor.i32_le()?;
    if magic != V1_MAGIC {
        return Err(Error::invalid_input(format!(
            "bad envelope magic {magic:#x} (want {V1_MAGIC:#x})"
        )));
    }

    // Descriptor event header (17 bytes) + fix part (52) + post header (8)
    // + extras. Only FieldID and PayloadDataType matter for decoding.
    cursor.u64_le()?; // timestamp
    let type_code = cursor.i8()?;
    if type_code != 0 {
        return Err(Error::invalid_input(format!(
            "first envelope event is not the descriptor (type {type_code})"
        )));
    }
    let event_length = cursor.i32_le()? as usize;
    let _next_position = cursor.i32_le()?;
    cursor.i64_le()?; // collection id
    cursor.i64_le()?; // partition id
    cursor.i64_le()?; // segment id
    let field_id = cursor.i64_le()?;
    cursor.u64_le()?; // start ts
    cursor.u64_le()?; // end ts
    let payload_data_type = cursor.i32_le()?;
    cursor.skip(DESCRIPTOR_POST_HEADER)?;
    let extras_len = cursor.i32_le()? as usize;
    let extras_bytes = cursor.read(extras_len)?;

    let extras: EnvelopeExtras = if extras_len > 0 {
        serde_json::from_slice(&extras_bytes)
            .map_err(|error| Error::invalid_input(format!("bad envelope extras JSON: {error}")))?
    } else {
        EnvelopeExtras {
            nullable: None,
            version: None,
        }
    };
    // Sanity: the descriptor's own length must cover what we consumed.
    if event_length < 4 + EVENT_HEADER_SIZE + DESCRIPTOR_FIX_PART + DESCRIPTOR_POST_HEADER + 4 {
        return Err(Error::invalid_input(format!(
            "descriptor event length {event_length} too small"
        )));
    }
    // (The extras may extend beyond the nominal descriptor length in old
    // writers; the event loop below walks from the declared position.)
    let mut pos = 4 + event_length as u64;

    let multi_field = extras.version.as_deref() == Some("MULTI_FIELD");
    let mut events = Vec::new();
    while pos < view.size() {
        let mut header = Cursor {
            view,
            pos,
            size: view.size(),
        };
        let _timestamp = header.u64_le()?;
        let type_code = header.i8()?;
        let event_length = header.i32_le()? as u64;
        let next_position = header.i32_le()?;
        let _ = next_position;
        if event_length < (EVENT_HEADER_SIZE + DATA_EVENT_FIX_PART) as u64 {
            return Err(Error::invalid_input(format!(
                "data event length {event_length} too small at offset {pos}"
            )));
        }
        let payload_len = event_length - EVENT_HEADER_SIZE as u64 - DATA_EVENT_FIX_PART as u64;
        if pos + event_length > view.size() {
            return Err(Error::invalid_input(format!(
                "data event at {pos} overruns the file"
            )));
        }
        events.push(EnvelopeEvent {
            type_code,
            payload_offset: pos + EVENT_HEADER_SIZE as u64 + DATA_EVENT_FIX_PART as u64,
            payload_len,
        });
        pos += event_length;
    }
    if events.is_empty() {
        return Err(Error::invalid_input("envelope carries no data events"));
    }

    Ok(Envelope {
        field_id,
        payload_data_type,
        nullable: extras.nullable,
        multi_field,
        events,
    })
}

/// Small sequential reader over a [`FileView`] for fixed-width fields.
struct Cursor<'v> {
    view: &'v FileView,
    pos: u64,
    size: u64,
}

impl Cursor<'_> {
    fn read(&mut self, len: usize) -> Result<Vec<u8>> {
        if self.pos + len as u64 > self.size {
            return Err(Error::invalid_input("binlog file truncated"));
        }
        let bytes = self.view.read_vec(self.pos, len)?;
        self.pos += len as u64;
        Ok(bytes)
    }

    fn skip(&mut self, len: usize) -> Result<()> {
        self.read(len).map(|_| ())
    }

    fn i8(&mut self) -> Result<i8> {
        Ok(self.read(1)?[0] as i8)
    }

    fn i32_le(&mut self) -> Result<i32> {
        let bytes = self.read(4)?;
        Ok(i32::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn i64_le(&mut self) -> Result<i64> {
        let bytes = self.read(8)?;
        Ok(i64::from_le_bytes(bytes.try_into().unwrap()))
    }

    fn u64_le(&mut self) -> Result<u64> {
        let bytes = self.read(8)?;
        Ok(u64::from_le_bytes(bytes.try_into().unwrap()))
    }
}
