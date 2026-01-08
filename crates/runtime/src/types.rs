//! Core runtime types
//!
//! These types represent the execution model at runtime.
//! They are populated from the compiled IR.

use std::fmt;

/// Unique identifier for a signal
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SignalId(pub String);

impl fmt::Display for SignalId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for SignalId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Unique identifier for a stratum
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StratumId(pub String);

impl fmt::Display for StratumId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for StratumId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Unique identifier for an era
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EraId(pub String);

impl fmt::Display for EraId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<&str> for EraId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Unique identifier for a field
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FieldId(pub String);

impl fmt::Display for FieldId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for an operator
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct OperatorId(pub String);

impl fmt::Display for OperatorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for an impulse
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ImpulseId(pub String);

impl fmt::Display for ImpulseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for a fracture
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FractureId(pub String);

impl fmt::Display for FractureId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

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
            StratumState::ActiveWithStride(stride) => tick % (*stride as u64) == 0,
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
