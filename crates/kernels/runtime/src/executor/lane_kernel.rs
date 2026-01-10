//! Lane kernel abstraction for two-level execution model.
//!
//! This module implements the second level of the two-level execution model:
//! - **Level 1: World-Level DAG** - Operators and signals scheduled by dependencies
//! - **Level 2: Lane Kernels** - Internal execution of entity/member signal nodes
//!
//! Member signal nodes in the DAG can be "lowered" to different execution strategies:
//! - L1: Instance-parallel tasks (chunked CPU parallelism)
//! - L2: Vector kernel (SSA lowering, SIMD, GPU compute)
//! - L3: Sub-DAG (internal dependency graph for complex logic)
//!
//! The lowering choice is transparent to the world-level DAG - a member signal
//! node completes before its dependents execute regardless of which strategy
//! was used internally.
//!
//! # Architecture
//!
//! ```text
//! World-Level DAG (Level 1)
//! ┌─────────────────────────────────────────────────────────┐
//! │  [SignalResolve] ──► [MemberSignalResolve] ──► [Measure]│
//! │                            │                            │
//! │                     ┌──────┴──────┐                     │
//! │                     │ Lane Kernel │ (Level 2)           │
//! │                     │ ┌─────────┐ │                     │
//! │                     │ │L1/L2/L3 │ │                     │
//! │                     │ └─────────┘ │                     │
//! │                     └─────────────┘                     │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! use continuum_runtime::executor::{LaneKernel, LoweringStrategy, ScalarL1Kernel};
//!
//! // Create an L1 kernel for a scalar member signal
//! let kernel = ScalarL1Kernel::new(
//!     member_signal_id,
//!     Arc::new(|ctx| ctx.prev + 1.0),
//!     population_hint,
//! );
//!
//! // Execute the kernel (handles all instances in parallel)
//! kernel.execute(&signals, &mut population, dt)?;
//! ```

use crate::soa_storage::PopulationStorage;
use crate::storage::SignalStorage;
use crate::types::Dt;
use crate::vectorized::MemberSignalId;

// Re-export for backward compatibility
pub use super::lowering_strategy::LoweringStrategy;

// ============================================================================
// Lane Kernel Trait
// ============================================================================

/// Result of lane kernel execution.
#[derive(Debug)]
pub struct LaneKernelResult {
    /// Number of instances processed.
    pub instances_processed: usize,
    /// Execution time in nanoseconds (if profiled).
    pub execution_ns: Option<u64>,
}

/// A compiled lane kernel that executes member signal resolution.
///
/// Lane kernels encapsulate a specific lowering strategy for a member signal.
/// They appear as single nodes in the world-level DAG but internally may
/// execute using parallel tasks, SIMD, GPU, or sub-DAG scheduling.
///
/// # Key Properties
///
/// - **Opaque to DAG** - The world-level scheduler only sees barriers
/// - **Deterministic** - Same inputs produce bitwise-identical outputs
/// - **Read-only inputs** - Kernels read signals/members, write to buffer
///
/// # Thread Safety
///
/// Lane kernels must be `Send + Sync` because the world-level DAG may
/// execute multiple strata/phases in parallel (where safe).
pub trait LaneKernel: Send + Sync {
    /// The lowering strategy this kernel uses.
    fn strategy(&self) -> LoweringStrategy;

    /// The member signal this kernel resolves.
    fn member_signal_id(&self) -> &MemberSignalId;

    /// Expected population size (for profiling/heuristics).
    fn population_hint(&self) -> usize;

    /// Execute the kernel, writing new values to the population storage.
    ///
    /// # Arguments
    ///
    /// * `signals` - Read-only access to global signals
    /// * `population` - Read-write access to population storage
    /// * `dt` - Time step for this tick
    ///
    /// # Returns
    ///
    /// Result containing execution statistics, or an error.
    fn execute(
        &self,
        signals: &SignalStorage,
        population: &mut PopulationStorage,
        dt: Dt,
    ) -> Result<LaneKernelResult, LaneKernelError>;
}

/// Error during lane kernel execution.
#[derive(Debug)]
pub enum LaneKernelError {
    /// Member signal buffer not found in population storage.
    SignalNotFound(String),
    /// Numeric error during computation (NaN, Inf).
    NumericError { index: usize, message: String },
    /// Kernel execution failed.
    ExecutionFailed(String),
}

impl std::fmt::Display for LaneKernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LaneKernelError::SignalNotFound(name) => {
                write!(f, "member signal not found: {}", name)
            }
            LaneKernelError::NumericError { index, message } => {
                write!(f, "numeric error at index {}: {}", index, message)
            }
            LaneKernelError::ExecutionFailed(msg) => {
                write!(f, "kernel execution failed: {}", msg)
            }
        }
    }
}

impl std::error::Error for LaneKernelError {}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lane_kernel_error_display() {
        let err = LaneKernelError::SignalNotFound("test.sig".to_string());
        assert!(err.to_string().contains("not found"));

        let err = LaneKernelError::NumericError {
            index: 42,
            message: "NaN".to_string(),
        };
        assert!(err.to_string().contains("42"));

        let err = LaneKernelError::ExecutionFailed("test".to_string());
        assert!(err.to_string().contains("test"));
    }
}
