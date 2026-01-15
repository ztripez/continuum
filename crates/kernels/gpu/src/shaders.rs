//! WGSL shader generation for field compute.
//!
//! This module generates WGSL compute shaders from DSL field expressions.
//! The shaders are compiled via Naga (built into wgpu) to the target backend.
//!
//! # Future Work
//!
//! - Integrate with compiled expression system
//! - Generate optimized shaders for common patterns
//! - Support topology-specific dispatch strategies

// Shader templates and topology types are infrastructure for future GPU field compute
#![allow(dead_code)]

/// Shader templates for common field patterns.
pub mod templates {
    /// Simple accumulator: sum all signal inputs.
    pub const ACCUMULATE: &str = r#"
@group(0) @binding(0) var<storage, read> signals: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }

    var sum: f32 = 0.0;
    let count = arrayLength(&signals);
    for (var i: u32 = 0u; i < count; i = i + 1u) {
        sum = sum + signals[i];
    }

    output[idx] = sum;
}
"#;

    /// Decay pattern: prev * decay_factor + collected.
    pub const DECAY_ACCUMULATE: &str = r#"
struct DecayParams {
    decay_factor: f32,
    collected_offset: u32,
    prev_offset: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> signals: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: DecayParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }

    let prev = signals[params.prev_offset + idx];
    let collected = signals[params.collected_offset + idx];

    output[idx] = prev * params.decay_factor + collected;
}
"#;

    /// Linear interpolation/smooth pattern.
    pub const SMOOTH: &str = r#"
struct SmoothParams {
    factor: f32,
    target_offset: u32,
    prev_offset: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> signals: array<f32>;
@group(0) @binding(1) var<storage, read> positions: array<vec3<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<uniform> params: SmoothParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= arrayLength(&output)) {
        return;
    }

    let prev = signals[params.prev_offset + idx];
    let target = signals[params.target_offset + idx];

    output[idx] = prev + (target - prev) * params.factor;
}
"#;

    /// Sphere surface field with lat/lon positioning.
    pub const SPHERE_SURFACE: &str = r#"
struct SphereParams {
    signal_value: f32,
    radius: f32,
    lat_samples: u32,
    lon_samples: u32,
}

@group(0) @binding(0) var<storage, read> signals: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: SphereParams;

const PI: f32 = 3.14159265359;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let lat_idx = global_id.x;
    let lon_idx = global_id.y;

    if (lat_idx >= params.lat_samples || lon_idx >= params.lon_samples) {
        return;
    }

    let output_idx = lat_idx * params.lon_samples + lon_idx;

    // Convert grid position to spherical coordinates
    let lat = (f32(lat_idx) / f32(params.lat_samples - 1u) - 0.5) * PI;
    let lon = f32(lon_idx) / f32(params.lon_samples) * 2.0 * PI;

    // Sample field at this location (placeholder)
    output[output_idx] = params.signal_value;
}
"#;
}

/// Topology types for field dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FieldTopology {
    /// Scalar field (single value).
    Scalar,
    /// Point cloud (arbitrary positions).
    PointCloud,
    /// Sphere surface (lat/lon grid).
    SphereSurface {
        /// Number of latitude samples.
        lat_samples: u32,
        /// Number of longitude samples.
        lon_samples: u32,
    },
    /// 3D volume grid.
    Volume {
        /// X dimension samples.
        x_samples: u32,
        /// Y dimension samples.
        y_samples: u32,
        /// Z dimension samples.
        z_samples: u32,
    },
}

impl FieldTopology {
    /// Get the total sample count for this topology.
    pub fn sample_count(&self) -> usize {
        match self {
            FieldTopology::Scalar => 1,
            FieldTopology::PointCloud => 0, // Dynamic
            FieldTopology::SphereSurface {
                lat_samples,
                lon_samples,
            } => (*lat_samples as usize) * (*lon_samples as usize),
            FieldTopology::Volume {
                x_samples,
                y_samples,
                z_samples,
            } => (*x_samples as usize) * (*y_samples as usize) * (*z_samples as usize),
        }
    }

    /// Get the workgroup dispatch dimensions.
    pub fn dispatch_dimensions(&self, workgroup_size: u32) -> [u32; 3] {
        match self {
            FieldTopology::Scalar => [1, 1, 1],
            FieldTopology::PointCloud => [0, 1, 1], // Set dynamically
            FieldTopology::SphereSurface {
                lat_samples,
                lon_samples,
            } => {
                let x = (*lat_samples + 15) / 16;
                let y = (*lon_samples + 15) / 16;
                [x, y, 1]
            }
            FieldTopology::Volume {
                x_samples,
                y_samples,
                z_samples,
            } => {
                let x = (*x_samples + workgroup_size - 1) / workgroup_size;
                let y = (*y_samples + workgroup_size - 1) / workgroup_size;
                let z = (*z_samples + workgroup_size - 1) / workgroup_size;
                [x, y, z]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topology_sample_count() {
        assert_eq!(FieldTopology::Scalar.sample_count(), 1);

        let sphere = FieldTopology::SphereSurface {
            lat_samples: 64,
            lon_samples: 128,
        };
        assert_eq!(sphere.sample_count(), 64 * 128);

        let volume = FieldTopology::Volume {
            x_samples: 32,
            y_samples: 32,
            z_samples: 32,
        };
        assert_eq!(volume.sample_count(), 32 * 32 * 32);
    }

    #[test]
    fn test_topology_dispatch() {
        let sphere = FieldTopology::SphereSurface {
            lat_samples: 64,
            lon_samples: 128,
        };
        let dispatch = sphere.dispatch_dimensions(16);
        assert_eq!(dispatch, [4, 8, 1]);
    }
}
