//! Continuum Tools
//!
//! CLI tools for working with Continuum worlds.

pub mod analyze;
pub mod ipc_protocol;

use tracing_subscriber::{EnvFilter, fmt};

/// Initialize logging with a default filter.
///
/// Use `RUST_LOG` environment variable to override the default filter.
/// Default is `info` for continuum crates and `warn` for others.
pub fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("warn,world_ipc=info,continuum_tools=warn,continuum_runtime=warn,continuum_compiler=warn,continuum_ir=warn,continuum_dsl=warn")
    });

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time() // Remove timestamps as requested
        .with_level(true)
        .init();
}
