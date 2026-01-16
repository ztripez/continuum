//! Lens configuration types.

use crate::error::LensError;

/// Lens configuration (observer-only).
///
/// These settings affect observation only and must not influence simulation results.
#[derive(Debug, Clone, Copy)]
pub struct FieldLensConfig {
    /// Maximum number of frames to retain per field.
    pub max_frames_per_field: usize,
    /// Maximum cached reconstructions per field.
    pub max_cached_per_field: usize,
    /// Maximum refinement requests buffered.
    pub max_refinement_queue: usize,
}

impl FieldLensConfig {
    /// Validate configuration values.
    pub fn validate(&self) -> Result<(), LensError> {
        if self.max_frames_per_field == 0 {
            return Err(LensError::InvalidConfig(
                "max_frames_per_field must be > 0".to_string(),
            ));
        }
        if self.max_cached_per_field == 0 {
            return Err(LensError::InvalidConfig(
                "max_cached_per_field must be > 0".to_string(),
            ));
        }
        if self.max_refinement_queue == 0 {
            return Err(LensError::InvalidConfig(
                "max_refinement_queue must be > 0".to_string(),
            ));
        }
        Ok(())
    }
}

impl Default for FieldLensConfig {
    fn default() -> Self {
        Self {
            max_frames_per_field: 1000,
            max_cached_per_field: 32,
            max_refinement_queue: 1024,
        }
    }
}

/// Per-field overrides for Lens behavior.
#[derive(Debug, Clone, Default)]
pub struct FieldConfig {
    /// Optional override for max cached reconstructions for this field.
    pub max_cached_per_field: Option<usize>,
}
