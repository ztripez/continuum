//! Execution policy configuration.
//!
//! Defines how the simulation engine enforces determinism and handles faults.
//! Policies are declared in the world manifest and applied at runtime.

use serde::{Deserialize, Serialize};

/// Execution policy configuration for a world.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorldPolicy {
    /// Determinism enforcement level
    pub determinism: DeterminismPolicy,

    /// Behavior when assertions or engine faults occur
    pub faults: FaultPolicy,
}

/// Determinism enforcement level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum DeterminismPolicy {
    /// Enable extra validation and hashing to guarantee determinism.
    Strict,
    /// Standard deterministic execution without extra validation overhead.
    #[default]
    Relaxed,
}

/// Behavior when assertions or engine faults occur.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum FaultPolicy {
    /// Halt simulation immediately on any fault.
    Fatal,
    /// Log the fault and continue simulation.
    #[default]
    Warn,
    /// Silently continue simulation.
    Ignore,
}

impl Default for WorldPolicy {
    fn default() -> Self {
        Self {
            determinism: DeterminismPolicy::Relaxed,
            faults: FaultPolicy::Warn,
        }
    }
}
