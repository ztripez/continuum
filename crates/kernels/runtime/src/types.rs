//! Core runtime types for simulation execution.
//!
//! This module defines the fundamental types used during simulation runtime.
//! These types represent the execution model and are populated from compiled IR.
//!
//! # Key Types
//!
//! - [`Phase`] - The five execution phases (Configure, Collect, Resolve, Fracture, Measure)
//! - [`Value`] - Runtime signal values (Scalar, Vec2, Vec3, Vec4)
//! - [`StratumState`] - Stratum activation within an era (Active, Gated, Strided)
//! - [`Dt`] - Time step wrapper for the current tick
//! - [`TickContext`] - Execution context for a single tick
//!
//! # ID Types
//!
//! This module re-exports foundational ID types from [`continuum_foundation`]:
//! [`SignalId`], [`FieldId`], [`EraId`], [`StratumId`], etc.

// Re-export foundational ID types
pub use continuum_foundation::{
    EntityId, EraId, FieldId, FractureId, ImpulseId, InstanceId, OperatorId, SignalId, StratumId,
};

use serde::{Deserialize, Serialize};

/// The five execution phases that occur each simulation tick.
///
/// Phases execute in strict order and define what operations are permitted
/// at each stage. This ensures deterministic execution regardless of how
/// signals and operators are declared.
///
/// # Phase Order
///
/// 1. **Configure** - Freeze execution context (dt, era, tick number)
/// 2. **Collect** - Accumulate impulse inputs and inter-signal contributions
/// 3. **Resolve** - Compute new signal values from expressions
/// 4. **Fracture** - Detect tension conditions and emit responses
/// 5. **Measure** - Emit field values for observer consumption
///
/// # Phase Boundaries
///
/// Operations are restricted by phase:
/// - Signal writes only occur during Resolve
/// - Field emission only occurs during Measure
/// - Fracture detection only occurs during Fracture
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Phase {
    /// Freeze execution context for the tick.
    Configure,
    /// Accumulate inputs from impulses and other sources.
    Collect,
    /// Compute new signal values from resolver expressions.
    Resolve,
    /// Detect tension conditions and emit fracture responses.
    Fracture,
    /// Emit field values for external observation.
    Measure,
}

impl Phase {
    /// All phases in execution order
    pub const ALL: [Phase; 5] = [
        Phase::Configure,
        Phase::Collect,
        Phase::Resolve,
        Phase::Fracture,
        Phase::Measure,
    ];
}

/// Stratum activation state within an era.
///
/// Strata can be configured to run at different rates or be paused entirely.
/// This allows multi-rate simulation where fast-changing phenomena (weather)
/// run every tick while slow phenomena (geology) run less frequently.
///
/// # Example
///
/// ```
/// use continuum_runtime::StratumState;
///
/// let fast = StratumState::Active;
/// let slow = StratumState::ActiveWithStride(100);
/// let paused = StratumState::Gated;
///
/// // Check if stratum runs on a given tick
/// assert!(fast.is_eligible(42));
/// assert!(slow.is_eligible(100));  // Multiple of 100
/// assert!(!slow.is_eligible(42));  // Not a multiple
/// assert!(!paused.is_eligible(42)); // Never runs
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StratumState {
    /// Executes every tick. Use for fast-changing phenomena.
    Active,
    /// Executes every N ticks. Use for slower phenomena.
    ActiveWithStride(u32),
    /// Paused entirely; state is preserved but not updated.
    Gated,
}

impl StratumState {
    /// Check if stratum should execute on given tick
    pub fn is_eligible(&self, tick: u64) -> bool {
        match self {
            StratumState::Active => true,
            StratumState::ActiveWithStride(stride) => tick.is_multiple_of(*stride as u64),
            StratumState::Gated => false,
        }
    }
}

/// Runtime value types for simulation signals.
///
/// Values are the fundamental data type stored in signals and passed between
/// resolvers. The DSL type system handles units at compile time; at runtime
/// we work with dimensionless numbers.
///
/// # Variants
///
/// - `Scalar` - A single floating-point value (temperature, pressure, etc.)
/// - `Vec2` - 2D vector (texture coordinates, 2D positions)
/// - `Vec3` - 3D vector (position, velocity, RGB color)
/// - `Vec4` - 4D vector (RGBA color, quaternion components)
///
/// # Example
///
/// ```
/// use continuum_runtime::Value;
///
/// // Creating values
/// let temp = Value::Scalar(300.0);
/// let pos = Value::Vec3([1.0, 2.0, 3.0]);
///
/// // Extracting scalars
/// assert_eq!(temp.as_scalar(), Some(300.0));
///
/// // Component access
/// assert_eq!(pos.component("x"), Some(1.0));
/// assert_eq!(pos.component("z"), Some(3.0));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Single scalar value (e.g., temperature, pressure, density).
    Scalar(f64),
    /// 2D vector (e.g., UV coordinates, 2D position).
    Vec2([f64; 2]),
    /// 3D vector (e.g., position, velocity, force).
    Vec3([f64; 3]),
    /// 4D vector (e.g., quaternion, RGBA color).
    Vec4([f64; 4]),
    // TODO: Mat4, Tensor, Grid, Seq
}

impl Value {
    /// Attempt to get the value as a scalar.
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Value::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    /// Attempt to get the value as a 3D vector.
    pub fn as_vec3(&self) -> Option<[f64; 3]> {
        match self {
            Value::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    /// Get a component by name (x, y, z, w)
    pub fn component(&self, name: &str) -> Option<f64> {
        match (self, name) {
            (Value::Scalar(v), _) => Some(*v),
            (Value::Vec2(v), "x") => Some(v[0]),
            (Value::Vec2(v), "y") => Some(v[1]),
            (Value::Vec3(v), "x") => Some(v[0]),
            (Value::Vec3(v), "y") => Some(v[1]),
            (Value::Vec3(v), "z") => Some(v[2]),
            (Value::Vec4(v), "x") => Some(v[0]),
            (Value::Vec4(v), "y") => Some(v[1]),
            (Value::Vec4(v), "z") => Some(v[2]),
            (Value::Vec4(v), "w") => Some(v[3]),
            _ => None,
        }
    }
}

impl Default for Value {
    fn default() -> Self {
        Value::Scalar(0.0)
    }
}

/// Time step for the current tick
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Dt(pub f64);

impl Dt {
    /// Get the time step in seconds.
    pub fn seconds(&self) -> f64 {
        self.0
    }
}

/// Context available during tick execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickContext {
    /// Current tick number
    pub tick: u64,
    /// Time step for this tick
    pub dt: Dt,
    /// Current era
    pub era: EraId,
}

/// Configuration for warmup execution
#[derive(Debug, Clone)]
pub struct WarmupConfig {
    /// Maximum iterations before failure
    pub max_iterations: u32,
    /// Convergence threshold (None = run all iterations)
    pub convergence_epsilon: Option<f64>,
}

impl Default for WarmupConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            convergence_epsilon: None,
        }
    }
}

/// Result of warmup execution
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Number of iterations executed
    pub iterations: u32,
    /// Whether convergence was achieved
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_component_vec3() {
        let v = Value::Vec3([1.0, 2.0, 3.0]);
        assert_eq!(v.component("x"), Some(1.0));
        assert_eq!(v.component("y"), Some(2.0));
        assert_eq!(v.component("z"), Some(3.0));
        assert_eq!(v.component("w"), None);
    }

    #[test]
    fn test_value_component_scalar() {
        let v = Value::Scalar(42.0);
        assert_eq!(v.component("x"), Some(42.0));
        assert_eq!(v.component("y"), Some(42.0));
    }
}
