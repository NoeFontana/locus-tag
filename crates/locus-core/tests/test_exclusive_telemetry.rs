//! Tests for mutually exclusive telemetry and profiling.
mod common;

use std::path::Path;

#[test]
fn test_telemetry_json_mode_creates_file() {
    let test_id = "test_json_mode";
    let log_path = format!("../../target/profiling/{test_id}_events.json");
    let _ = std::fs::remove_file(&log_path);

    let guard = common::telemetry::init_with_mode(test_id, "json");
    tracing::info!("Test JSON mode");
    drop(guard);

    assert!(
        Path::new(&log_path).exists(),
        "JSON file should be created in json mode"
    );

    let content = std::fs::read_to_string(&log_path).expect("failed to read log file");
    assert!(content.contains("Test JSON mode"));
}

#[test]
fn test_telemetry_silent_mode_no_file() {
    let test_id = "test_silent_mode";
    let log_path = format!("../../target/profiling/{test_id}_events.json");
    let _ = std::fs::remove_file(&log_path);

    let guard = common::telemetry::init_with_mode(test_id, "silent");
    tracing::info!("Test silent mode");
    drop(guard);

    assert!(
        !Path::new(&log_path).exists(),
        "JSON file should not be created in silent mode"
    );
}
