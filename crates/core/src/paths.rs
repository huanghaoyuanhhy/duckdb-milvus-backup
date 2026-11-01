//! Reconstructing backup-relative binlog paths from the source-cluster
//! object keys recorded in segment_meta.json.
//!
//! `log_path` values are the SOURCE Milvus bucket keys
//! (`{rootPath}/insert_log/{coll}/{part}/{seg}/{field}/{log_id}`), not paths
//! inside the backup. milvus-backup copies files verbatim to
//! `binlogs/insert_log/{coll}/{part}/{group}/{seg}/{field}/{log_id}` (delta
//! analogously); the `{group}` level only exists in newer backups, equals the
//! segment's group_id, and is absent for L0 delta logs — which may sit under
//! partition `-1` in the backup even when the source key used a real
//! partition id. See milvus-backup `internal/storage/mpath/path.go`. Because
//! the group level and partition components vary across writer generations,
//! callers get an ordered candidate list and try each on open (a miss is one
//! cheap 404).

use duckdb_neo::{Error, Result};

const INSERT_MARKER: &str = "insert_log/";
const DELTA_MARKER: &str = "delta_log/";
const BACKUP_INSERT_PREFIX: &str = "binlogs/insert_log/";
const BACKUP_DELTA_PREFIX: &str = "binlogs/delta_log/";

/// Ordered backup-relative candidate paths for one source log key. The first
/// entry matches current milvus-backup output; later entries cover older or
/// L0-specific layouts.
pub(crate) fn backup_log_paths(
    log_path: &str,
    group_id: Option<i64>,
    partition_id: Option<i64>,
) -> Result<Vec<String>> {
    let (marker, parts) = split_source_path(log_path)?;
    let is_insert = marker == INSERT_MARKER;
    // insert: {coll} {part} {seg} {field} {log}; delta: {coll} {part} {seg} {log}.
    let expected = if is_insert { 5 } else { 4 };
    if parts.len() != expected {
        return Err(Error::invalid_input(format!(
            "unexpected {marker}key shape ({}/{} components): {log_path}",
            parts.len(),
            expected
        )));
    }
    let collection = parts[0];
    let segment = parts[2];
    let group = group_id.filter(|group| *group != 0);

    // Partition components worth trying: the source key's own first (all
    // generations agree for normal segments), then the raw meta partition
    // (covers L0 delta logs recorded under partition -1).
    let mut partitions: Vec<String> = vec![parts[1].to_string()];
    if let Some(partition) = partition_id {
        let partition = partition.to_string();
        if !partitions.contains(&partition) {
            partitions.push(partition);
        }
    }

    let mut candidates = Vec::with_capacity(partitions.len() * 2);
    let mut push = |with_group: bool, partition: &str| {
        let mut path = String::from(if is_insert {
            BACKUP_INSERT_PREFIX
        } else {
            BACKUP_DELTA_PREFIX
        });
        path.push_str(collection);
        path.push('/');
        path.push_str(partition);
        path.push('/');
        let group_level = if with_group {
            match group {
                Some(group) => Some(group.to_string()),
                // Insert logs of every modern backup carry the group level
                // (it equals the segment id); only very old backups lack it.
                None if is_insert => Some(segment.to_string()),
                None => None,
            }
        } else {
            None
        };
        if let Some(group_level) = group_level {
            path.push_str(&group_level);
            path.push('/');
        }
        for part in &parts[2..] {
            path.push_str(part);
            path.push('/');
        }
        path.pop(); // trailing slash
        if !candidates.contains(&path) {
            candidates.push(path);
        }
    };

    for partition in &partitions {
        push(true, partition);
        push(false, partition);
    }
    Ok(candidates)
}

/// Split a source key at the last `insert_log/` or `delta_log/` marker,
/// returning the marker found and the components after it.
fn split_source_path(log_path: &str) -> Result<(&'static str, Vec<&str>)> {
    let insert = log_path
        .rfind(INSERT_MARKER)
        .map(|index| (index, INSERT_MARKER));
    let delta = log_path
        .rfind(DELTA_MARKER)
        .map(|index| (index, DELTA_MARKER));
    let (marker_index, marker) = insert
        .into_iter()
        .chain(delta)
        .max_by_key(|(index, _)| *index)
        .ok_or_else(|| {
            Error::invalid_input(format!(
                "log_path {log_path:?} contains neither insert_log/ nor delta_log/"
            ))
        })?;
    let tail = &log_path[marker_index + marker.len()..];
    let parts: Vec<&str> = tail.split('/').filter(|part| !part.is_empty()).collect();
    Ok((marker, parts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modern_insert_log_gets_the_group_level() {
        // group_id present (= segment id in practice): doubled segment dir.
        let candidates =
            backup_log_paths("files/insert_log/45/46/47/100/48", Some(47), Some(46)).unwrap();
        assert_eq!(
            candidates,
            vec![
                "binlogs/insert_log/45/46/47/47/100/48".to_string(),
                "binlogs/insert_log/45/46/47/100/48".to_string(),
            ]
        );
    }

    #[test]
    fn insert_log_without_group_id_defaults_to_the_segment_dir() {
        let candidates = backup_log_paths("files/insert_log/45/46/47/100/48", None, Some(46)).unwrap();
        assert_eq!(candidates[0], "binlogs/insert_log/45/46/47/47/100/48");
    }

    #[test]
    fn delta_log_uses_the_meta_partition_and_group() {
        let candidates =
            backup_log_paths("files/delta_log/45/46/47/48", Some(47), Some(46)).unwrap();
        assert_eq!(
            candidates,
            vec![
                "binlogs/delta_log/45/46/47/47/48".to_string(),
                "binlogs/delta_log/45/46/47/48".to_string(),
            ]
        );
    }

    #[test]
    fn l0_delta_falls_back_to_partition_minus_one() {
        // Source key uses a real partition; the backup places L0 delta logs
        // under the meta partition (-1) without a group level.
        let candidates = backup_log_paths("files/delta_log/45/46/47/48", Some(0), Some(-1)).unwrap();
        assert_eq!(
            candidates,
            vec![
                "binlogs/delta_log/45/46/47/48".to_string(),
                "binlogs/delta_log/45/-1/47/48".to_string(),
            ]
        );
    }

    #[test]
    fn root_path_prefix_is_stripped_at_the_last_marker() {
        let candidates =
            backup_log_paths("files/insert_log/1/2/3/100/4", Some(3), Some(2)).unwrap();
        assert!(candidates[0].starts_with("binlogs/insert_log/1/"));
    }

    #[test]
    fn foreign_paths_are_rejected() {
        assert!(backup_log_paths("unrelated/parquet/file", None, None).is_err());
        assert!(backup_log_paths("files/insert_log/45/46/47", Some(47), Some(46)).is_err());
    }
}
