//! Spatial field definitions and sampling structures.
//!
//! Fields represent spatially distributed data (e.g., elevation, temperature,
//! pressure) that can be sampled at arbitrary coordinates.

use serde::{Deserialize, Serialize};

use crate::Value;

/// A single sample from a spatial field.
///
/// Pairs a 3D position with the value measured at that point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSample {
    /// Position in the field's coordinate space [x, y, z].
    pub position: [f64; 3],
    /// The value measured at the specified position.
    pub value: Value,
}
