//! End-to-end tests against real Milvus binlog files committed under
//! `crates/core/testdata/` (copied verbatim from examplebackup). They pin the
//! envelope layout, footer parsing, and page decoding in one pass.

use std::sync::Arc;

use super::{BlockSource, FileView, read_footer};
use super::decode::{ChunkReader, Values};
use super::footer::{LogicalKind, PhysicalType};
use crate::binlog::{self, BinlogKind};

fn view(name: &str) -> FileView {
    let path = format!("{}/testdata/{name}", env!("CARGO_MANIFEST_DIR"));
    let bytes = Arc::new(std::fs::read(path).expect("testdata file"));
    let len = bytes.len() as u64;
    FileView::new(Arc::new(BlockSource::Mem(bytes)), 0, len)
}

/// Drain a chunk reader into (values, def_levels-per-page).
fn drain(reader: &mut ChunkReader) -> (Values, Vec<Option<Vec<u16>>>) {
    let mut values: Option<Values> = None;
    let mut defs = Vec::new();
    while let Some(page) = reader.next_page().expect("next page") {
        let len = page.values.len();
        assert_eq!(page.num_values as usize >= len, true, "page has fewer values than levels");
        defs.push(page.def_levels.clone());
        match values.as_mut() {
            None => values = Some(page.values),
            Some(existing) => match (existing, page.values) {
                (Values::Int64(a), Values::Int64(b)) => a.extend(b),
                (Values::Bytes { data: ad, offsets: ao }, Values::Bytes { data: bd, offsets: bo }) => {
                    let base = ad.len() as u32;
                    ad.extend(bd);
                    ao.extend(bo[1..].iter().map(|offset| offset + base));
                }
                (left, right) => panic!("mismatched page value variants: {left:?} vs {right:?}"),
            },
        }
    }
    (values.expect("at least one page"), defs)
}

#[test]
fn rle_hybrid_hand_built_runs() {
    // RLE run: header (5 << 1) = 10, bit_width 3, value 0b101.
    let decoded = super::rle::decode_hybrid(&[0x0a, 0x05], 3, 5).unwrap();
    assert_eq!(decoded, vec![5; 5]);

    // Literal run: header (1 group << 1) | 1 = 3, bit_width 1, values
    // 1,0,1,1,0,0,1,0 packed LSB-first into one byte 0b01001101.
    let decoded = super::rle::decode_hybrid(&[0x03, 0x4d], 1, 8).unwrap();
    assert_eq!(decoded, vec![1, 0, 1, 1, 0, 0, 1, 0]);

    // Literal run trimmed to fewer than 8 values (last group partial).
    let decoded = super::rle::decode_hybrid(&[0x03, 0x4d], 1, 3).unwrap();
    assert_eq!(decoded, vec![1, 0, 1]);

    // Zero bit width: all zeros, count from the run header.
    let decoded = super::rle::decode_hybrid(&[0x0c], 0, 6).unwrap();
    assert_eq!(decoded, vec![0; 6]);

    assert_eq!(super::rle::level_bit_width(0), 0);
    assert_eq!(super::rle::level_bit_width(1), 1);
    assert_eq!(super::rle::level_bit_width(2), 2);
    assert_eq!(super::rle::level_bit_width(4), 3);
}

#[test]
fn storage_v1_timestamp_dict_encoded() {
    let file = view("insert_field1_timestamp.envelope");
    assert!(matches!(binlog::sniff(&file).unwrap(), BinlogKind::Envelope));
    let envelope = binlog::parse_envelope(&file).unwrap();
    // System timestamp field, payload type Int64 (5), single insert event.
    assert_eq!(envelope.field_id, 1);
    assert_eq!(envelope.payload_data_type, 5);
    assert_eq!(envelope.nullable, Some(false));
    assert!(!envelope.multi_field);
    assert_eq!(envelope.events.len(), 1);
    assert!(envelope.events[0].is_insert());

    let payload = envelope.event_payload(&file, 0).unwrap();
    assert!(payload.read_vec(payload.size() - 4, 4).unwrap().ends_with(b"PAR1"));
    let meta = read_footer(&payload).unwrap();
    assert!(meta.created_by.as_deref().unwrap().contains("parquet-go"));
    assert_eq!(meta.leaves.len(), 1);
    assert_eq!(meta.leaves[0].path, vec!["1"]);
    assert_eq!(meta.leaves[0].physical, PhysicalType::Int64);
    assert_eq!(meta.leaves[0].max_def, 1, "parquet-go marks leaves OPTIONAL");
    assert_eq!(meta.leaves[0].max_rep, 0);
    assert_eq!(meta.row_groups.len(), 1);
    assert_eq!(meta.row_groups[0].num_rows, 1500);

    let mut reader = ChunkReader::open(
        payload.slice(0, payload.size()).unwrap(),
        meta.leaves[0].clone(),
        meta.row_groups[0].columns[0].clone(),
    )
    .unwrap();
    let (values, defs) = drain(&mut reader);
    match &values {
        Values::Int64(values) => {
            assert_eq!(values.len(), 1500);
            assert!(values.iter().all(|&value| value == values[0]));
        }
        other => panic!("expected Int64 values, got {other:?}"),
    }
    // All 1500 definition levels are 1 (no nulls in this file).
    let defs = defs.into_iter().flatten().flatten().collect::<Vec<_>>();
    assert_eq!(defs.len(), 1500);
    assert!(defs.iter().all(|&level| level == 1));
}

#[test]
fn storage_v1_pk_plain_encoded() {
    let file = view("insert_field100_pk.envelope");
    let envelope = binlog::parse_envelope(&file).unwrap();
    assert_eq!(envelope.field_id, 100);
    assert_eq!(envelope.payload_data_type, 5);

    let payload = envelope.event_payload(&file, 0).unwrap();
    let meta = read_footer(&payload).unwrap();
    assert_eq!(meta.leaves[0].path, vec!["100"]);
    // The pk chunk is PLAIN-encoded (no dictionary page offset).
    assert!(meta.row_groups[0].columns[0].dictionary_page_offset.is_none());

    let mut reader = ChunkReader::open(
        payload.slice(0, payload.size()).unwrap(),
        meta.leaves[0].clone(),
        meta.row_groups[0].columns[0].clone(),
    )
    .unwrap();
    let (values, _) = drain(&mut reader);
    match &values {
        Values::Int64(values) => {
            assert_eq!(values.len(), 1500);
            assert_eq!(values[..5], [0, 1, 2, 3, 4]);
            assert_eq!(values[1499], 1499);
        }
        other => panic!("expected Int64 values, got {other:?}"),
    }
}

#[test]
fn storage_v1_l0_delta_json_strings() {
    let file = view("l0_delta_json.envelope");
    let envelope = binlog::parse_envelope(&file).unwrap();
    // Delta logs carry field_id -1 and a String (20) JSON payload column.
    assert_eq!(envelope.field_id, -1);
    assert_eq!(envelope.payload_data_type, 20);
    assert!(envelope.events[0].type_code == binlog::EVENT_DELETE);
    assert!(!envelope.multi_field);

    let payload = envelope.event_payload(&file, 0).unwrap();
    let meta = read_footer(&payload).unwrap();
    assert_eq!(meta.leaves.len(), 1);
    assert_eq!(meta.leaves[0].physical, PhysicalType::ByteArray);
    assert_eq!(meta.leaves[0].logical, LogicalKind::String);

    let mut reader = ChunkReader::open(
        payload.slice(0, payload.size()).unwrap(),
        meta.leaves[0].clone(),
        meta.row_groups[0].columns[0].clone(),
    )
    .unwrap();
    let (values, _) = drain(&mut reader);
    let Values::Bytes { data, offsets } = &values else {
        panic!("expected string values, got {values:?}")
    };
    assert_eq!(offsets.len() - 1, 9, "l0 delta deletes 9 rows");
    let first = &data[offsets[0] as usize..offsets[1] as usize];
    let last = &data[offsets[8] as usize..offsets[9] as usize];
    let first = std::str::from_utf8(first).unwrap();
    let last = std::str::from_utf8(last).unwrap();
    assert_eq!(
        first,
        r#"{"pk":0,"ts":456769326808039425,"pkType":5}"#,
        "unexpected first delete row: {first}"
    );
    assert!(last.contains(r#""pk":8"#), "unexpected last delete row: {last}");
}

#[test]
fn storage_v2_bare_multi_column_parquet() {
    let file = view("storage_v2_group0.parquet");
    assert!(matches!(binlog::sniff(&file).unwrap(), BinlogKind::BareParquet));
    let meta = read_footer(&file).unwrap();
    assert_eq!(meta.leaves.len(), 5);
    let field_ids = meta
        .leaves
        .iter()
        .map(|leaf| leaf.field_id)
        .collect::<Vec<_>>();
    assert_eq!(field_ids, vec![Some(100), Some(102), Some(103), Some(0), Some(1)]);
    let names = meta
        .leaves
        .iter()
        .map(|leaf| leaf.path.join("."))
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["id", "int_2", "int_3", "RowID", "Timestamp"]);
    // int_2/int_3 are Int16 fields stored as INT32 + logical INT(16).
    for leaf in &meta.leaves[1..3] {
        assert_eq!(leaf.physical, PhysicalType::Int32);
        assert_eq!(leaf.logical, LogicalKind::Int { bits: 16, signed: true });
    }
    assert_eq!(meta.row_groups.len(), 1);
    assert_eq!(meta.row_groups[0].num_rows, 100);
    // The arrow writer emits UNCOMPRESSED pages.
    assert_eq!(meta.row_groups[0].columns[0].codec, super::footer::Codec::Uncompressed);

    let expected_first: [(usize, i64); 5] = [
        (0, 459559709168059710),
        (1, -5501),
        (2, 21268),
        (3, 459559709168059710),
        (4, 459559768298881043),
    ];
    for (column, expected_value) in expected_first {
        let mut reader = ChunkReader::open(
            file.slice(0, file.size()).unwrap(),
            meta.leaves[column].clone(),
            meta.row_groups[0].columns[column].clone(),
        )
        .unwrap();
        let (values, defs) = drain(&mut reader);
        match &values {
            Values::Int64(values) => {
                assert_eq!(values[0], expected_value, "column {column} first row");
            }
            Values::Int32(values) => {
                assert_eq!(values[0] as i64, expected_value, "column {column} first row");
            }
            other => panic!("column {column}: unexpected values {other:?}"),
        }
        let defs = defs.into_iter().flatten().flatten().collect::<Vec<_>>();
        assert!(defs.iter().all(|&level| level == 1), "column {column} all required");
    }
}

/// A duckdb-COPY-written file with a nested LIST schema exercises the
/// schema-tree walk on a second writer (the real backups cover parquet-go and
/// arrow C++, which are flat or single-list).
#[test]
fn duckdb_written_nested_list_schema() {
    let path = format!(
        "{}/../../test/fixtures/scan/scan_v2/binlogs/insert_log/2/20/201/201/0/2001",
        env!("CARGO_MANIFEST_DIR")
    );
    let bytes = Arc::new(std::fs::read(&path).expect("fixture file"));
    let len = bytes.len() as u64;
    let file = FileView::new(Arc::new(BlockSource::Mem(bytes)), 0, len);
    let meta = read_footer(&file).unwrap();
    let paths = meta
        .leaves
        .iter()
        .map(|leaf| leaf.path.join("."))
        .collect::<Vec<_>>();
    assert_eq!(
        paths,
        vec!["id", "small", "arr.list.element", "sparse", "meta", "geom", "flag"]
    );
    let list_leaf = &meta.leaves[2];
    assert_eq!(list_leaf.max_rep, 1);
    assert_eq!(list_leaf.max_def, 3, "optional list + repeated list + optional element");
    assert!(list_leaf.list_group_optional);
    assert_eq!(meta.row_groups[0].num_rows, 4);

    // Decode the LIST column: [1,2], NULL, [], [3].
    use super::decode::{ChunkReader, read_chunk};
    let mut reader = ChunkReader::open(
        file.slice(0, file.size()).unwrap(),
        meta.leaves[2].clone(),
        meta.row_groups[0].columns[2].clone(),
    )
    .unwrap();
    let data = read_chunk(&mut reader).unwrap();
    assert_eq!(data.rows(), 5);
    assert_eq!(data.rep_levels.as_ref().unwrap(), &[0, 1, 0, 0, 0]);
    // [1,2], NULL list, empty list, [3].
    assert_eq!(data.def_levels.as_ref().unwrap(), &[3, 3, 0, 1, 3]);
    assert_eq!(data.values, super::decode::Values::Int32(vec![1, 2, 3]));
}
