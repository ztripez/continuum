//! Lens error types.

use continuum_foundation::FieldId;
use thiserror::Error;

/// Lens error types.
#[derive(Debug, Error)]
pub enum LensError {
    /// Configuration validation failure.
    #[error("Invalid lens config: {0}")]
    InvalidConfig(String),
    /// Requested field is not present.
    #[error("Field not found: {0}")]
    FieldNotFound(FieldId),
    /// No samples available for the requested field and tick.
    #[error("No samples for field {field} at tick {tick}")]
    NoSamplesAtTick { field: FieldId, tick: u64 },
    /// Refinement queue capacity exceeded.
    #[error("Refinement queue full")]
    RefinementQueueFull,
    /// GPU backend was not configured.
    #[error("GPU backend not configured")]
    GpuUnavailable,
    /// GPU query failed.
    #[error("GPU query failed: {0}")]
    GpuQuery(String),
    /// GPU batch queries only support scalar samples.
    #[error("Non-scalar sample encountered for GPU query: {0}")]
    NonScalarSample(FieldId),
}

impl LensError {
    /// Create a NoSamplesAtTick error (DRY helper).
    pub fn no_samples(field: FieldId, tick: u64) -> Self {
        Self::NoSamplesAtTick { field, tick }
    }
}
