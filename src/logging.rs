use anyhow::Result;
use std::path::Path;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Initialize the logging system
/// Logs will be written to the logs/ directory only (no console output)
pub fn init_logging() -> Result<()> {
    // Create logs directory if it doesn't exist
    std::fs::create_dir_all("logs")?;

    // File appender - daily rotation in logs/ folder
    let file_appender = RollingFileAppender::new(
        Rotation::DAILY,
        "logs",
        "agent.log",
    );

    // Create file layer
    let file_layer = fmt::layer()
        .with_writer(file_appender)
        .with_ansi(false)
        .with_target(true)
        .with_thread_ids(true)
        .with_line_number(true);

    // Set up environment filter
    // Default to INFO level, can be overridden with RUST_LOG env var
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    // Combine layers - only file layer, no stdout
    tracing_subscriber::registry()
        .with(env_filter)
        .with(file_layer)
        .init();

    tracing::info!("Logging system initialized");
    tracing::info!("Log files location: logs/agent.log");

    Ok(())
}

/// Check if logs directory exists
pub fn logs_dir_exists() -> bool {
    Path::new("logs").exists()
}
