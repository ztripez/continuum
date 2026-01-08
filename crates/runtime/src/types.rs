//! Core runtime types
//!
//! These types represent the execution model at runtime.
//! They are populated from the compiled IR.

// Re-export foundational ID types
pub use continuum_foundation::{
    EraId, FieldId, FractureId, ImpulseId, OperatorId, SignalId, StratumId,
};

/// Execution phases in order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Phase {
    Configure,
    Collect,
    Resolve,
    Fracture,
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

/// Stratum activation state within an era
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StratumState {
    /// Executes every tick
    Active,
    /// Executes every N ticks
    ActiveWithStride(u32),
    /// Paused, state preserved
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

/// Runtime value types
///
/// For now, using f64 for everything. The DSL type system
/// handles units at compile time; at runtime we just have numbers.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scalar(f64),
    Vec2([f64; 2]),
    Vec3([f64; 3]),
    Vec4([f64; 4]),
    // TODO: Mat4, Tensor, Grid, Seq
}

impl Value {
    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            Value::Scalar(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_vec3(&self) -> Option<[f64; 3]> {
        match self {
            Value::Vec3(v) => Some(*v),
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dt(pub f64);

impl Dt {
    pub fn seconds(&self) -> f64 {
        self.0
    }
}

/// Context available during tick execution
#[derive(Debug, Clone)]
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
