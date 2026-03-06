use std::fs::File;
use std::path::PathBuf;
use tracing_subscriber::{layer::SubscriberExt, Registry};

pub fn init() -> tracing_appender::non_blocking::WorkerGuard {
    let log_path = PathBuf::from("../../target/profiling/regression_events.json");
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent).expect("Failed to create log directory");
    }
    
    let file = File::create(&log_path)
        .expect("Failed to create log file");
    
    let (non_blocking, guard) = tracing_appender::non_blocking(file);
    
    let json_layer = tracing_subscriber::fmt::layer()
        .json()
        .with_writer(non_blocking);
        
    let subscriber = Registry::default().with(json_layer);

    #[cfg(feature = "tracy")]
    let subscriber = subscriber.with(tracing_tracy::TracyLayer::default());

    let _ = tracing::subscriber::set_global_default(subscriber);

    guard
}
