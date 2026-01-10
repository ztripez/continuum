//! Error types for GPU compute operations.

use thiserror::Error;

/// Errors that can occur during GPU compute operations.
#[derive(Error, Debug)]
pub enum GpuError {
    /// No compatible GPU adapter was found.
    #[error("no compatible GPU adapter found")]
    NoAdapter,

    /// Failed to request GPU device.
    #[error("failed to request GPU device: {0}")]
    DeviceRequest(String),

    /// Shader compilation failed.
    #[error("shader compilation failed: {0}")]
    ShaderCompilation(String),

    /// Buffer creation failed.
    #[error("buffer creation failed: {0}")]
    BufferCreation(String),

    /// Compute dispatch failed.
    #[error("compute dispatch failed: {0}")]
    DispatchFailed(String),

    /// Buffer mapping failed.
    #[error("buffer mapping failed: {0}")]
    BufferMapping(String),

    /// GPU feature not enabled.
    #[error("GPU feature not enabled - compile with --features gpu")]
    FeatureNotEnabled,
}
