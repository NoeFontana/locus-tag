//! Tests for telemetry initialization and log file creation.
mod common;
use std::env;

#[test]
fn test_telemetry_initialization_creates_log_file() {
    let log_path = "../../target/profiling/test_telemetry_events.json";
    // Ensure the file is deleted before the test
    let _ = std::fs::remove_file(log_path);

    // Explicitly set JSON mode for this test, otherwise if CI runs with TRACY_NO_INVARIANT_CHECK
    // but without TELEMETRY_MODE=json, it defaults to silent mode and fails.
    #[allow(unsafe_code)]
    unsafe {
        env::set_var("TELEMETRY_MODE", "json")
    };

    // This should initialize the telemetry and return a guard
    let guard = common::telemetry::init("test_telemetry");

    // Log an event
    tracing::info!("Telemetry test event");

    // Drop the guard to flush
    drop(guard);

    // Verify the file exists and contains the log
    assert!(
        std::path::Path::new(log_path).exists(),
        "Telemetry log file was not created"
    );

    let content = std::fs::read_to_string(log_path).expect("failed to read telemetry log");
    assert!(
        content.contains("Telemetry test event"),
        "Log event not found in file"
    );
}
