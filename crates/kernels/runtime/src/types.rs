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

// Re-export foundational ID types and StratumState
pub use continuum_foundation::{
    AssertionSeverity, DeterminismPolicy, EntityId, EraId, FaultPolicy, FieldId, FractureId,
    ImpulseId, InstanceId, OperatorId, Phase, SignalId, StratumId, StratumState, Value,
    WorldPolicy,
};

use serde::{Deserialize, Serialize};

/// The internal state machine phases for a simulation tick.
///
/// While [`Phase`] defines the logical simulation phases (the 5 kernels),
/// [`TickPhase`] includes the runtime plumbing required to manage the tick
/// lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TickPhase {
    /// Logical simulation phase (Configure, Collect, Resolve, Fracture, Measure)
    Simulation(Phase),
    /// Transition to another era if conditions met.
    EraTransition,
    /// Post-tick state advancement (tick++, advance buffers).
    PostTick,
}

impl TickPhase {
    /// The first phase of a tick.
    pub const START: TickPhase = TickPhase::Simulation(Phase::Configure);

    /// Get the next phase in the tick lifecycle.
    pub fn next(&self) -> Self {
        match self {
            TickPhase::Simulation(Phase::CollectConfig)
            | TickPhase::Simulation(Phase::Initialize)
            | TickPhase::Simulation(Phase::WarmUp) => TickPhase::Simulation(Phase::Configure),
            TickPhase::Simulation(Phase::Configure) => TickPhase::Simulation(Phase::Collect),
            TickPhase::Simulation(Phase::Collect) => TickPhase::Simulation(Phase::Resolve),
            TickPhase::Simulation(Phase::Resolve) => TickPhase::Simulation(Phase::Fracture),
            TickPhase::Simulation(Phase::Fracture) => TickPhase::Simulation(Phase::Measure),
            TickPhase::Simulation(Phase::Measure) => TickPhase::Simulation(Phase::Assert),
            TickPhase::Simulation(Phase::Assert) => TickPhase::EraTransition,
            TickPhase::EraTransition => TickPhase::PostTick,
            TickPhase::PostTick => TickPhase::Simulation(Phase::Configure),
        }
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
    /// Accumulated simulation time in seconds
    pub sim_time: f64,
    /// Time step for this tick
    pub dt: Dt,
    /// Current era
    pub era: EraId,
}

/// Result of an execution step (phase or partial phase)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepResult {
    /// Execution continues normally
    Continue,
    /// A breakpoint was hit
    Breakpoint {
        /// The signal that triggered the breakpoint
        signal: SignalId,
    },
    /// A tick was completed
    TickCompleted(TickContext),
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

/// Result of a warmup phase
#[derive(Debug, Clone)]
pub struct WarmupResult {
    /// Total iterations performed
    pub iterations: u32,
    /// Whether convergence was achieved
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_component_scalar() {
        let val = Value::Scalar(42.0);
        assert_eq!(val.component("x"), Some(42.0));
        assert_eq!(val.component("y"), Some(42.0)); // Scalar returns its value for any component
    }

    #[test]
    fn test_value_component_vec3() {
        let val = Value::Vec3([1.0, 2.0, 3.0]);
        assert_eq!(val.component("x"), Some(1.0));
        assert_eq!(val.component("y"), Some(2.0));
        assert_eq!(val.component("z"), Some(3.0));
        assert_eq!(val.component("w"), None);
    }
}
