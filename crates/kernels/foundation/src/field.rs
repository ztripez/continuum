use serde::{Deserialize, Serialize};

use crate::Value;

/// A single field sample (position + value).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldSample {
    /// Position in field's coordinate space.
    pub position: [f64; 3],
    /// Sample value.
    pub value: Value,
}
