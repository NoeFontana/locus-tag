use std::fs::File;
use std::path::PathBuf;
use tracing_subscriber::{Registry, layer::SubscriberExt};

/// Represents the active guard. If None, telemetry is not writing to file.
pub struct TelemetryGuard {
    _worker: Option<tracing_appender::non_blocking::WorkerGuard>,
    _default_guard: Option<tracing::subscriber::DefaultGuard>,
}

pub fn init(test_id: &str) -> TelemetryGuard {
    let mode = std::env::var("TELEMETRY_MODE").unwrap_or_default();
    init_with_mode(test_id, &mode)
}

pub fn init_with_mode(test_id: &str, mode: &str) -> TelemetryGuard {
    if mode == "json" {
        let log_path = PathBuf::from(format!("../../target/profiling/{test_id}_events.json"));
        if let Some(parent) = log_path.parent() {
            std::fs::create_dir_all(parent).expect("Failed to create log directory");
        }

        let file = File::create(&log_path).expect("Failed to create log file");
        let (non_blocking, guard) = tracing_appender::non_blocking(file);

        let json_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE)
            .with_writer(non_blocking);

        let subscriber = Registry::default().with(json_layer);
        let default_guard = tracing::subscriber::set_default(subscriber);

        TelemetryGuard {
            _worker: Some(guard),
            _default_guard: Some(default_guard),
        }
    } else if mode == "tracy" {
        #[cfg(feature = "tracy")]
        {
            let subscriber = Registry::default().with(tracing_tracy::TracyLayer::default());
            let default_guard = tracing::subscriber::set_default(subscriber);
            TelemetryGuard {
                _worker: None,
                _default_guard: Some(default_guard),
            }
        }
        #[cfg(not(feature = "tracy"))]
        TelemetryGuard {
            _worker: None,
            _default_guard: None,
        }
    } else {
        TelemetryGuard {
            _worker: None,
            _default_guard: None,
        }
    }
}
