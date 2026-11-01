#!/usr/bin/env python3
"""Generate the synthetic Milvus backups used by test/sql/milvus_backup_scan.test.

Creates two backups under test/fixtures/scan/:
  scan_v1  — storage-v1 envelope files (hand-packed), covering multi-file
             fields, NULL via parquet definition levels, segment-local delta
             logs (JSON variant), a collection-level L0 delta, an old-layout
             segment without the group path level, and include_system columns.
  scan_v2  — storage-v2 bare parquet (written by the repo's duckdb CLI), one
             multi-column group file, covering SMALLINT NULLs, LIST columns,
             sparse-vector blobs, JSON-as-VARCHAR and geometry blobs.

Data files are deterministic; run from the repo root:
    python3 -I test/fixtures/generate_scan_fixtures.py
The generated binaries are committed; rerun only to regenerate them.
"""

import json
import os
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BACKUPS = os.path.join(ROOT, "test", "fixtures", "scan")
DUCKDB = os.path.join(ROOT, "build", "duckdb")

MAGIC = 0xFFFABC


def envelope(field_id: int, payload_type: int, payload: bytes, coll: int, part: int, seg: int, event_type: int) -> bytes:
    """Pack a parquet payload into a storage-v1 binlog envelope."""
    extras = json.dumps({"nullable": False, "original_size": str(len(payload))}).encode()
    desc_data_len = 52 + 8 + 4 + len(extras)
    desc_event_len = 17 + desc_data_len
    out = struct.pack("<i", MAGIC)
    # descriptor event: header + fixpart + postheader + extras
    out += struct.pack("<Qbii", 0, 0, desc_event_len, 4 + desc_event_len)
    out += struct.pack("<qqqqQQi", coll, part, seg, field_id, 0, 0, payload_type)
    out += bytes([52, 16, 16, 16, 16, 16, 16, 16])
    out += struct.pack("<i", len(extras)) + extras
    # single data event carrying the whole parquet payload
    data_event_len = 17 + 16 + len(payload)
    out += struct.pack("<Qbii", 0, event_type, data_event_len, -1)
    out += struct.pack("<QQ", 0, 0)
    out += payload
    return out


def write_parquet(sql: str, path: str, options: str = "") -> None:
    """Materialize one parquet file via the repo's duckdb CLI."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    result = subprocess.run(
        [DUCKDB, "-unsigned", "-noheader", "-list", "-c",
         f"COPY ({sql}) TO '{path}' (FORMAT parquet{options})"],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not os.path.exists(path):
        sys.exit(f"duckdb COPY failed for {path}:\n{result.stdout}\n{result.stderr}")


def write_json(path: str, payload) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, separators=(",", ":"))


def write_binlog(path: str, blob: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(blob)


def meta_level(path: str, infos) -> None:
    write_json(path, {"infos": infos})


def build_scan_v1() -> None:
    base = os.path.join(BACKUPS, "scan_v1")
    coll, part = 1, 10

    def insert_log_path(seg: int, field: int, log: int) -> str:
        return f"files/insert_log/{coll}/{part}/{seg}/{field}/{log}"

    def delta_log_path(seg: int, log: int) -> str:
        return f"files/delta_log/{coll}/{part}/{seg}/{log}"

    # --- segment 101 (modern layout): id split across two files, one NULL tag
    id_a = f"SELECT * FROM (VALUES (0::BIGINT), (1)) t(id)"
    id_b = f"SELECT * FROM (VALUES (2::BIGINT)) t(id)"
    score_a = "SELECT * FROM (VALUES (10::INTEGER), (11), (12)) t(score)"
    tag_a = "SELECT * FROM (VALUES ('a'::VARCHAR), (NULL), ('c')) t(tag)"
    seg101_logs = [
        # RowID entry: real writers omit fieldID when it is 0 (proto default),
        # so this file reaches the reader unattributed.
        (insert_log_path(101, 0, 1000), 5, "SELECT * FROM (VALUES (1000::BIGINT), (1001), (1002)) t(rowid)", None),
        (insert_log_path(101, 1, 1001), 5, "SELECT * FROM (VALUES (50::BIGINT), (55), (60)) t(ts)", 1),
        (insert_log_path(101, 100, 1002), 5, id_a, 100),
        (insert_log_path(101, 100, 1003), 5, id_b, 100),
        (insert_log_path(101, 101, 1004), 4, score_a, 101),
        (insert_log_path(101, 102, 1005), 20, tag_a, 102),
    ]
    # --- segment 102 (modern layout)
    seg102_logs = [
        (insert_log_path(102, 0, 1100), 5, "SELECT * FROM (VALUES (1003::BIGINT), (1004)) t(rowid)", None),
        (insert_log_path(102, 1, 1101), 5, "SELECT * FROM (VALUES (65::BIGINT), (66)) t(ts)", 1),
        (insert_log_path(102, 100, 1102), 5, "SELECT * FROM (VALUES (3::BIGINT), (4)) t(id)", 100),
        (insert_log_path(102, 101, 1103), 4, "SELECT * FROM (VALUES (13::INTEGER), (14)) t(score)", 101),
        (insert_log_path(102, 102, 1104), 20, "SELECT * FROM (VALUES ('d'::VARCHAR), ('e')) t(tag)", 102),
    ]
    # --- segment 103 (pre-group_id layout: files without the doubled segment dir)
    seg103_logs = [
        (insert_log_path(103, 0, 1200), 5, "SELECT * FROM (VALUES (1005::BIGINT), (1006)) t(rowid)", None),
        (insert_log_path(103, 1, 1201), 5, "SELECT * FROM (VALUES (70::BIGINT), (71)) t(ts)", 1),
        (insert_log_path(103, 100, 1202), 5, "SELECT * FROM (VALUES (5::BIGINT), (6)) t(id)", 100),
        (insert_log_path(103, 101, 1203), 4, "SELECT * FROM (VALUES (15::INTEGER), (16)) t(score)", 101),
        (insert_log_path(103, 102, 1204), 20, "SELECT * FROM (VALUES (NULL::VARCHAR), ('f')) t(tag)", 102),
    ]

    delta_seg101 = "SELECT * FROM (VALUES ('{\"pk\":1,\"ts\":100,\"pkType\":5}'::VARCHAR)) t(delta)"
    delta_l0 = "SELECT * FROM (VALUES ('{\"pk\":3,\"ts\":99,\"pkType\":5}'::VARCHAR)) t(delta)"

    def emit(log_path: str, payload_type: int, sql: str, field_id: int, seg_id: int, event_type: int, old_layout: bool) -> None:
        rel = log_path[log_path.index("insert_log/") + len("insert_log/"):] if "insert_log/" in log_path \
            else log_path[log_path.index("delta_log/") + len("delta_log/"):]
        coll_id, part_id, seg, tail = rel.split("/", 3)
        tmp = os.path.join(base, "_tmp.parquet")
        write_parquet(sql, tmp)
        with open(tmp, "rb") as handle:
            payload = handle.read()
        os.remove(tmp)
        if "insert_log/" in log_path:
            field_dir, log_id = tail.split("/")
            # Modern layout doubles the segment dir (segment/group); the old
            # writer wrote files without the group level.
            parts = (coll_id, part_id, seg, seg, field_dir, log_id) if not old_layout \
                else (coll_id, part_id, seg, field_dir, log_id)
            dest = os.path.join(base, "binlogs", "insert_log", *parts)
        else:
            dest_dir = (coll_id, part_id, seg, tail) if old_layout else (coll_id, part_id, seg, seg, tail)
            dest = os.path.join(base, "binlogs", "delta_log", *dest_dir)
        # fieldID omitted in meta means field 0 (RowID); the envelope
        # descriptor always records the true field id.
        write_binlog(dest, envelope(field_id if field_id is not None else 0,
                                    payload_type, payload, coll, part, seg_id, event_type))

    for log_path, payload_type, sql, field_id in seg101_logs + seg102_logs:
        seg_id = int(log_path.split("/")[2])
        emit(log_path, payload_type, sql, field_id, seg_id, 1, old_layout=False)
    for log_path, payload_type, sql, field_id in seg103_logs:
        emit(log_path, payload_type, sql, field_id, 103, 1, old_layout=True)

    # delta logs (JSON variant): segment-local for 101, collection-level L0.
    emit(delta_log_path(101, 1005), 20, delta_seg101, 0, 101, 2, old_layout=False)
    # L0 delta lands at binlogs/delta_log/{coll}/-1/{seg}/{log} (no group level).
    l0_dest = os.path.join(base, "binlogs", "delta_log", str(coll), "-1", "900", "9001")
    tmp = os.path.join(base, "_tmp.parquet")
    write_parquet(delta_l0, tmp)
    with open(tmp, "rb") as handle:
        payload = handle.read()
    os.remove(tmp)
    write_binlog(l0_dest, envelope(0, 20, payload, coll, -1, 900, 2))

    write_json(
        os.path.join(base, "meta", "backup_meta.json"),
        {"name": "scan_v1", "milvus_version": "v2.4.1", "backup_timestamp": 455275622782205954},
    )
    meta_level(
        os.path.join(base, "meta", "collection_meta.json"),
        [{
            "collection_id": coll,
            "db_name": "default",
            "collection_name": "items",
            "shards_num": 1,
            "schema": {"fields": [
                {"fieldID": 100, "name": "id", "data_type": 5, "is_primary_key": True},
                {"fieldID": 101, "name": "score", "data_type": 4},
                {"fieldID": 102, "name": "tag", "data_type": 21, "nullable": True},
            ]},
            "l0_segments": [{
                "segment_id": 900, "collection_id": coll, "partition_id": -1,
                "is_l0": True,
                "deltalogs": [{"fieldID": 901, "binlogs": [{"log_path": f"files/delta_log/{coll}/-1/900/9001"}]}],
            }],
        }],
    )
    meta_level(
        os.path.join(base, "meta", "partition_meta.json"),
        [{"partition_id": part, "collection_id": coll, "partition_name": "_default"}],
    )
    meta_level(
        os.path.join(base, "meta", "segment_meta.json"),
        [
            {
                "segment_id": 101, "collection_id": coll, "partition_id": part,
                "num_of_rows": 3, "group_id": 101,
                "binlogs": [
                    {"binlogs": [{"log_path": insert_log_path(101, 0, 1000)}]},
                    {"fieldID": 1, "binlogs": [{"log_path": insert_log_path(101, 1, 1001)}]},
                    {"fieldID": 100, "binlogs": [
                        {"log_path": insert_log_path(101, 100, 1002)},
                        {"log_path": insert_log_path(101, 100, 1003)},
                    ]},
                    {"fieldID": 101, "binlogs": [{"log_path": insert_log_path(101, 101, 1004)}]},
                    {"fieldID": 102, "binlogs": [{"log_path": insert_log_path(101, 102, 1005)}]},
                ],
                "deltalogs": [{"fieldID": 105, "binlogs": [{"log_path": delta_log_path(101, 1005)}]}],
            },
            {
                "segment_id": 102, "collection_id": coll, "partition_id": part,
                "num_of_rows": 2, "group_id": 102,
                "binlogs": [
                    {"binlogs": [{"log_path": insert_log_path(102, 0, 1100)}]},
                    {"fieldID": 1, "binlogs": [{"log_path": insert_log_path(102, 1, 1101)}]},
                    {"fieldID": 100, "binlogs": [{"log_path": insert_log_path(102, 100, 1102)}]},
                    {"fieldID": 101, "binlogs": [{"log_path": insert_log_path(102, 101, 1103)}]},
                    {"fieldID": 102, "binlogs": [{"log_path": insert_log_path(102, 102, 1104)}]},
                ],
                "deltalogs": [{}],
            },
            {
                # Old writer: no group_id, files without the group level.
                "segment_id": 103, "collection_id": coll, "partition_id": part,
                "num_of_rows": 2,
                "binlogs": [
                    {"binlogs": [{"log_path": insert_log_path(103, 0, 1200)}]},
                    {"fieldID": 1, "binlogs": [{"log_path": insert_log_path(103, 1, 1201)}]},
                    {"fieldID": 100, "binlogs": [{"log_path": insert_log_path(103, 100, 1202)}]},
                    {"fieldID": 101, "binlogs": [{"log_path": insert_log_path(103, 101, 1203)}]},
                    {"fieldID": 102, "binlogs": [{"log_path": insert_log_path(103, 102, 1204)}]},
                ],
                "deltalogs": [{}],
            },
        ],
    )


def build_scan_v2() -> None:
    base = os.path.join(BACKUPS, "scan_v2")
    coll, part, seg = 2, 20, 201
    log_path = f"files/insert_log/{coll}/{part}/{seg}/0/2001"
    dest = os.path.join(base, "binlogs", "insert_log", str(coll), str(part), str(seg), str(seg), "0", "2001")

    sparse_rows = [
        struct.pack("<If", 0, 1.5) + struct.pack("<If", 5, 2.5),
        struct.pack("<If", 3, 0.5),
        b"",
        struct.pack("<If", 1, 1.0),
    ]
    blob_literals = ",".join(
        "'" + "".join(f"\\x{byte:02x}" for byte in row) + "'::BLOB" for row in sparse_rows
    )
    sql = (
        "SELECT * FROM (VALUES "
        "(0::BIGINT, NULL::SMALLINT, [1, 2]::INTEGER[], "
        f"{blob_literals.split(',')[0]}, "
        "'{\"a\": 1}'::VARCHAR, '\\x01\\x01\\x00\\x00'::BLOB, true::BOOLEAN), "
        "(1, 7, NULL, "
        f"{blob_literals.split(',')[1]}, "
        "'{\"b\": [2]}'::VARCHAR, '\\x01\\x02\\x00\\x00'::BLOB, false), "
        "(2, NULL, []::INTEGER[], "
        f"{blob_literals.split(',')[2]}, "
        "'{}'::VARCHAR, '\\x01\\x03\\x00\\x00'::BLOB, true), "
        "(3, 9, [3], "
        f"{blob_literals.split(',')[3]}, "
        "'{\"c\": {\"d\": 4}}'::VARCHAR, '\\x01\\x04\\x00\\x00'::BLOB, false)"
        ") t(id, small, arr, sparse, meta, geom, flag)"
    )
    # Real storage-v2 files tag every column with its Milvus field id; the
    # reader resolves columns through those ids.
    write_parquet(
        sql, dest,
        options=", FIELD_IDS {id: 100, small: 102, arr: 103, sparse: 104, meta: 105, geom: 106, flag: 107}",
    )

    write_json(os.path.join(base, "meta", "backup_meta.json"),
               {"name": "scan_v2", "milvus_version": "v2.5.0"})
    meta_level(
        os.path.join(base, "meta", "collection_meta.json"),
        [{
            "collection_id": coll,
            "db_name": "default",
            "collection_name": "metrics",
            "shards_num": 1,
            "schema": {"fields": [
                {"fieldID": 100, "name": "id", "data_type": 5, "is_primary_key": True},
                {"fieldID": 102, "name": "small", "data_type": 3, "nullable": True},
                {"fieldID": 103, "name": "arr", "data_type": 22, "element_type": 4},
                {"fieldID": 104, "name": "sparse", "data_type": 104},
                {"fieldID": 105, "name": "meta", "data_type": 23},
                {"fieldID": 106, "name": "geom", "data_type": 24},
                {"fieldID": 107, "name": "flag", "data_type": 1},
            ]},
        }],
    )
    meta_level(
        os.path.join(base, "meta", "partition_meta.json"),
        [{"partition_id": part, "collection_id": coll, "partition_name": "_default"}],
    )
    meta_level(
        os.path.join(base, "meta", "segment_meta.json"),
        [{
            "segment_id": seg, "collection_id": coll, "partition_id": part,
            "num_of_rows": 4, "storage_version": 2,
            # storage-v2 writers emit the multi-field group file without a
            # fieldID (proto default omission), like the real v2_04 backup.
            "binlogs": [{"binlogs": [{"log_path": log_path}]}],
            "deltalogs": [{}],
        }],
    )


def main() -> None:
    if not os.path.exists(DUCKDB):
        sys.exit(f"duckdb CLI not found at {DUCKDB}; run make debug first")
    build_scan_v1()
    build_scan_v2()
    print("generated scan_v1 and scan_v2 under", BACKUPS)


if __name__ == "__main__":
    main()
