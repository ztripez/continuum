//! Runtime errors for simulation execution.
//!
//! This module defines errors that can occur during simulation runtime,
//! including signal resolution failures, DAG cycle detection, assertion
//! failures, and numeric instabilities.
//!
//! # Error Categories
//!
//! - **Lookup errors**: [`Error::SignalNotFound`], [`Error::StratumNotFound`], [`Error::EraNotFound`]
//! - **DAG errors**: [`Error::CycleDetected`]
//! - **Execution errors**: [`Error::AssertionFailed`], [`Error::NumericError`], [`Error::PhaseViolation`]
//! - **Value errors**: [`Error::InvalidValue`], [`Error::WarmupDivergence`]
//!
//! # Error Handling Policy
//!
//! Runtime errors are designed to be explicit and actionable. The simulation
//! will not silently correct or mask errors—any invalid state is surfaced
//! immediately. This aligns with the "fail loudly" principle of Continuum.

use thiserror::Error;

use crate::types::{EraId, Phase, SignalId, StratumId};

/// Runtime result type alias.
///
/// Convenience type for functions that may fail with a runtime error.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during simulation runtime execution.
///
/// These errors represent failures that occur after successful compilation,
/// during actual tick execution. They indicate problems with the simulation
/// state or configuration that prevent correct execution.
#[derive(Debug, Error)]
pub enum Error {
    /// A signal was referenced that does not exist in storage.
    ///
    /// This typically indicates a bug in the compiler or an uninitialized
    /// signal. All signals should be initialized before execution begins.
    #[error("signal not found: {0}")]
    SignalNotFound(SignalId),

    /// A stratum was referenced that does not exist in the era configuration.
    ///
    /// Strata define execution scheduling. This error means the runtime
    /// tried to execute or query a stratum that wasn't registered.
    #[error("stratum not found: {0}")]
    StratumNotFound(StratumId),

    /// An era was referenced that does not exist in the world configuration.
    ///
    /// Eras define time step sizes and stratum activation. This error
    /// means the runtime tried to transition to an undefined era.
    #[error("era not found: {0}")]
    EraNotFound(EraId),

    /// A cycle was detected in the signal dependency graph.
    ///
    /// Signals must form a directed acyclic graph (DAG) for deterministic
    /// resolution. If signal A depends on B, and B depends on A (directly
    /// or transitively), execution order cannot be determined.
    ///
    /// The `signals` field contains the IDs of signals involved in the cycle.
    #[error("cycle detected in DAG: {signals:?}")]
    CycleDetected {
        /// Signal IDs involved in the dependency cycle.
        signals: Vec<SignalId>,
    },

    /// A user-defined assertion failed during signal resolution.
    ///
    /// Assertions are declared in DSL signal definitions to validate
    /// invariants. This error is raised when an assertion condition
    /// evaluates to false.
    #[error("assertion failed in {signal}: {message}")]
    AssertionFailed {
        /// The signal whose assertion failed.
        signal: SignalId,
        /// Description of the failed assertion.
        message: String,
    },

    /// A numeric error occurred during expression evaluation.
    ///
    /// This includes division by zero, NaN results, infinity, or other
    /// floating-point edge cases that indicate unstable computation.
    #[error("numeric error in {signal}: {message}")]
    NumericError {
        /// The signal where the numeric error occurred.
        signal: SignalId,
        /// Description of the numeric problem.
        message: String,
    },

    /// An operation was attempted in an invalid execution phase.
    ///
    /// The five-phase execution model restricts what operations can occur
    /// in each phase. For example, signal reads are not allowed during
    /// the Configure phase.
    #[error("phase violation: {operation} not allowed in {phase:?}")]
    PhaseViolation {
        /// Description of the attempted operation.
        operation: String,
        /// The phase in which the violation occurred.
        phase: Phase,
    },

    /// A signal value violated its declared constraints.
    ///
    /// This occurs when a resolved value falls outside the signal's
    /// declared range or doesn't match its type specification.
    #[error("invalid value for {signal}: {message}")]
    InvalidValue {
        /// The signal with the invalid value.
        signal: SignalId,
        /// Description of the constraint violation.
        message: String,
    },

    /// Warmup iteration failed to converge or produced invalid values.
    ///
    /// During warmup, signals iterate toward stable initial values.
    /// This error indicates the warmup process diverged (values grew
    /// unboundedly) or hit the iteration limit without converging.
    #[error("warmup divergence in {signal} at iteration {iteration}: {message}")]
    WarmupDivergence {
        /// The signal that failed to converge.
        signal: SignalId,
        /// The iteration number when divergence was detected.
        iteration: u32,
        /// Description of the divergence.
        message: String,
    },

    /// Generic error message.
    #[error("{0}")]
    Generic(String),
}
