//! Tests for mutually exclusive telemetry and profiling.
#![allow(unsafe_code)]
mod common;

use std::env;
use std::path::Path;

#[test]
fn test_telemetry_json_mode_creates_file() {
    let test_id = "test_json_mode";
    let log_path = format!("../../target/profiling/{test_id}_events.json");
    let _ = std::fs::remove_file(&log_path);

    // SAFETY: We are in a test environment setting a telemetry mode.
    unsafe { env::set_var("TELEMETRY_MODE", "json") };

    let guard = common::telemetry::init(test_id);
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

    // SAFETY: We are in a test environment removing a telemetry mode.
    unsafe { env::remove_var("TELEMETRY_MODE") };

    let guard = common::telemetry::init(test_id);
    tracing::info!("Test silent mode");
    drop(guard);

    assert!(
        !Path::new(&log_path).exists(),
        "JSON file should not be created in silent mode"
    );
}
