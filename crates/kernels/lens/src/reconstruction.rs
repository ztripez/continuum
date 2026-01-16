//! Field reconstruction from discrete samples.
//!
//! Reconstruction is mandatory for observer use - fields are functions,
//! not raw sample data.

use continuum_foundation::FieldSample;

/// Reconstructed field interface (observer-only).
///
/// Implementations must be deterministic given the same samples.
pub trait FieldReconstruction: Send + Sync {
    /// Query scalar value at position.
    fn query(&self, position: [f64; 3]) -> f64;

    /// Query vector value at position (default: scalar -> zero vector).
    fn query_vector(&self, position: [f64; 3]) -> [f64; 3] {
        let v = self.query(position);
        [v, 0.0, 0.0]
    }
    // Raw sample access intentionally omitted to enforce observer boundary.
}

/// Nearest-neighbor reconstruction (MVP).
///
/// Returns the value of the closest sample by Euclidean distance.
pub struct NearestNeighborReconstruction {
    samples: Vec<FieldSample>,
}

impl NearestNeighborReconstruction {
    /// Create a nearest-neighbor reconstruction from samples.
    pub fn new(samples: Vec<FieldSample>) -> Self {
        Self { samples }
    }
}

impl FieldReconstruction for NearestNeighborReconstruction {
    fn query(&self, position: [f64; 3]) -> f64 {
        find_nearest(&self.samples, position, |s| {
            s.value.as_scalar().unwrap_or(0.0)
        })
    }

    fn query_vector(&self, position: [f64; 3]) -> [f64; 3] {
        find_nearest(&self.samples, position, |s| {
            s.value
                .as_vec3()
                .map(|v| [v[0], v[1], v[2]])
                .unwrap_or([0.0, 0.0, 0.0])
        })
    }
}

/// Find the nearest sample and extract a value from it (DRY helper).
fn find_nearest<T>(
    samples: &[FieldSample],
    position: [f64; 3],
    extract: impl Fn(&FieldSample) -> T,
) -> T
where
    T: Default,
{
    let mut best_dist = f64::MAX;
    let mut best_sample: Option<&FieldSample> = None;

    for sample in samples {
        let dx = sample.position[0] - position[0];
        let dy = sample.position[1] - position[1];
        let dz = sample.position[2] - position[2];
        let dist = dx * dx + dy * dy + dz * dz;
        if dist < best_dist {
            best_dist = dist;
            best_sample = Some(sample);
        }
    }

    best_sample.map(extract).unwrap_or_default()
}
