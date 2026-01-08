//! Runtime errors

use thiserror::Error;

use crate::types::{SignalId, StratumId, EraId, Phase};

/// Runtime result type
pub type Result<T> = std::result::Result<T, Error>;

/// Runtime errors
#[derive(Debug, Error)]
pub enum Error {
    #[error("signal not found: {0}")]
    SignalNotFound(SignalId),

    #[error("stratum not found: {0}")]
    StratumNotFound(StratumId),

    #[error("era not found: {0}")]
    EraNotFound(EraId),

    #[error("cycle detected in DAG: {signals:?}")]
    CycleDetected { signals: Vec<SignalId> },

    #[error("assertion failed in {signal}: {message}")]
    AssertionFailed { signal: SignalId, message: String },

    #[error("numeric error in {signal}: {message}")]
    NumericError { signal: SignalId, message: String },

    #[error("phase violation: {operation} not allowed in {phase:?}")]
    PhaseViolation { operation: String, phase: Phase },

    #[error("invalid value for {signal}: {message}")]
    InvalidValue { signal: SignalId, message: String },
}
