//! Continuum Tools
//!
//! CLI tools for working with Continuum worlds.

pub mod analyze;

use tracing_subscriber::{EnvFilter, fmt};

/// Initialize logging with a default filter.
///
/// Use `RUST_LOG` environment variable to override the default filter.
/// Default is `info` for continuum crates and `warn` for others.
pub fn init_logging() {
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new("info,continuum_tools=debug,continuum_runtime=debug,continuum_compiler=debug,continuum_ir=info,continuum_dsl=info")
    });

    fmt()
        .with_env_filter(filter)
        .with_target(false)
        .without_time() // Remove timestamps as requested
        .with_level(true)
        .init();
}
