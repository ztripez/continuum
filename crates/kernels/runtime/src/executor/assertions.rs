//! Assertion evaluation and checking
//! Assertions provide runtime validation of simulation invariants.
//!
//! Unlike engine-level checks (which detect bugs in the runtime), assertions
//! validate user-defined rules within the simulation (e.g., "temperature
//! must not exceed melting point").
//!
//! # Assertion Lifecycle
//!
//! 1. **Declaration** - Assertions are declared in the DSL within execution blocks.
//! 2. **Registration** - During runtime initialization, assertions are attached to their target signals.
//! 3. **Evaluation** - After a signal is resolved, all associated assertions are evaluated.
//! 4. **Reporting** - Failures are recorded in the `AssertionChecker` and can trigger a simulation halt.
//!
//! # Severities
//!
//! | Severity | Behavior |
//! |----------|----------|
//! | `Note`   | Purely informational, logged only. |
//! | `Warn`   | Significant deviation, logged but simulation continues. |
//! | `Error`  | Invalid state, triggers a `RunError` and halts the simulation. |
//! | `Fatal`  | Critical invariant violation, triggers an immediate panic. |
//!
//! # Observer Boundary
//!
//! Assertions are strictly **non-causal**. They can read simulation state to
//! validate conditions, but they can NEVER mutate values or influence the
//! execution path. This ensures that a simulation with all assertions disabled
//! produces the exact same results as one with all assertions enabled.

use serde::{Deserialize, Serialize};
use tracing::{debug, error, warn};

use crate::error::{Error, Result};
use crate::storage::{EntityStorage, SignalStorage};
use crate::types::{AssertionSeverity, Dt, FaultPolicy, SignalId, Value};

use super::context::AssertContext;

/// Function that evaluates an assertion condition

pub type AssertionFn = Box<dyn Fn(&AssertContext) -> bool + Send + Sync>;

/// Represents a single violation of a simulation invariant detected at runtime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionFailure {
    /// The unique identifier of the signal that failed the assertion.
    pub signal: SignalId,
    /// Severity level determining the runtime response (Note/Warn/Error/Fatal).
    pub severity: AssertionSeverity,
    /// Human-readable description of the failure condition.
    pub message: String,
    /// The simulation tick when the failure occurred.
    pub tick: u64,
    /// The name of the era active during the failure.
    pub era: String,
    /// Total elapsed simulation time in seconds.
    pub sim_time: f64,
}

/// Registered assertion for a signal.
pub struct SignalAssertion {
    pub signal: SignalId,
    pub condition: AssertionFn,
    pub severity: AssertionSeverity,
    pub message: Option<String>,
}

/// Assertion checker for the runtime
#[derive(Default)]
pub struct AssertionChecker {
    /// Registered assertions
    assertions: Vec<SignalAssertion>,
    /// Recent assertion failures (circular buffer)
    failures: Vec<AssertionFailure>,
    /// Maximum number of failures to retain
    max_failures: usize,
    /// Fault policy for Error-level assertions
    policy: FaultPolicy,
}

impl AssertionChecker {
    /// Create a new assertion checker
    pub fn new() -> Self {
        Self {
            assertions: Vec::new(),
            failures: Vec::new(),
            max_failures: 1000,
            policy: FaultPolicy::Warn,
        }
    }

    /// Create a new assertion checker with a specific failure buffer size
    pub fn with_capacity(max_failures: usize) -> Self {
        Self {
            assertions: Vec::new(),
            failures: Vec::new(),
            max_failures,
            policy: FaultPolicy::Warn,
        }
    }

    /// Set the fault policy
    pub fn set_policy(&mut self, policy: FaultPolicy) {
        self.policy = policy;
    }

    /// Register an assertion for a signal
    pub fn register(
        &mut self,
        signal: SignalId,
        condition: AssertionFn,
        severity: AssertionSeverity,
        message: Option<String>,
    ) {
        debug!(signal = %signal, ?severity, "assertion registered");
        self.assertions.push(SignalAssertion {
            signal,
            condition,
            severity,
            message,
        });
    }

    /// Check all assertions for a specific signal
    ///
    /// Returns Ok(()) if all assertions pass or only warnings were emitted.
    /// Returns Err if any Error or Fatal assertion failed.
    pub fn check_signal(
        &mut self,
        signal: &SignalId,
        current: &Value,
        prev: &Value,
        signals: &SignalStorage,
        entities: &EntityStorage,
        dt: Dt,
        sim_time: f64,
        tick: u64,
        era: &str,
    ) -> Result<()> {
        let ctx = AssertContext {
            current,
            prev,
            signals,
            entities,
            dt,
            sim_time,
        };

        for assertion in self.assertions.iter().filter(|a| &a.signal == signal) {
            let passed = (assertion.condition)(&ctx);

            if !passed {
                let message = assertion
                    .message
                    .clone()
                    .unwrap_or_else(|| format!("assertion failed for signal {}", signal));

                // Record the failure
                let failure = AssertionFailure {
                    signal: signal.clone(),
                    severity: assertion.severity,
                    message: message.clone(),
                    tick,
                    era: era.to_string(),
                    sim_time,
                };

                self.failures.push(failure);
                if self.failures.len() > self.max_failures {
                    self.failures.remove(0);
                }

                match assertion.severity {
                    AssertionSeverity::Warn => {
                        warn!(signal = %signal, message = %message, "assertion warning");
                    }
                    AssertionSeverity::Error => match self.policy {
                        FaultPolicy::Fatal => {
                            error!(signal = %signal, message = %message, "assertion error (Policy: Fatal)");
                            return Err(Error::AssertionFailed {
                                signal: signal.clone(),
                                message,
                            });
                        }
                        FaultPolicy::Warn => {
                            error!(
                                signal = %signal,
                                message = %message,
                                "assertion error (continuing per policy)"
                            );
                        }
                        FaultPolicy::Ignore => {
                            debug!(
                                signal = %signal,
                                message = %message,
                                "assertion error (ignored per policy)"
                            );
                        }
                    },
                    AssertionSeverity::Fatal => {
                        panic!(
                            "FATAL ASSERTION FAILURE: signal={}, message={:?}, tick={}, era={}, time={}",
                            signal, message, tick, era, sim_time
                        );
                    }
                }
            }
        }

        Ok(())
    }

    /// Check all assertions for all signals that were resolved
    pub fn check_all(
        &mut self,
        resolved_signals: &[(SignalId, Value)],
        signals: &SignalStorage,
        entities: &EntityStorage,
        dt: Dt,
        sim_time: f64,
        tick: u64,
        era: &str,
    ) -> Result<()> {
        for (signal, current) in resolved_signals {
            let prev = signals.get_prev(signal).unwrap_or(current);
            self.check_signal(
                signal, current, prev, signals, entities, dt, sim_time, tick, era,
            )?;
        }
        Ok(())
    }

    /// Check if any assertions are registered
    pub fn is_empty(&self) -> bool {
        self.assertions.is_empty()
    }

    /// Get number of registered assertions
    pub fn len(&self) -> usize {
        self.assertions.len()
    }

    /// Get all registered assertions
    pub fn assertions(&self) -> &[SignalAssertion] {
        &self.assertions
    }

    /// Get all recent failures
    pub fn failures(&self) -> &[AssertionFailure] {
        &self.failures
    }

    /// Get failures for a specific signal
    pub fn failures_for_signal(&self, signal: &SignalId) -> Vec<&AssertionFailure> {
        self.failures
            .iter()
            .filter(|f| &f.signal == signal)
            .collect()
    }

    /// Clear all recorded failures
    pub fn clear_failures(&mut self) {
        self.failures.clear();
    }

    /// Drain all recorded failures, returning them and clearing the buffer
    pub fn drain_failures(&mut self) -> Vec<AssertionFailure> {
        std::mem::take(&mut self.failures)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assertion_pass() {
        let mut checker = AssertionChecker::new();
        let signal: SignalId = "test.signal".into();

        // Assertion: value must be positive
        checker.register(
            signal.clone(),
            Box::new(|ctx| ctx.current.as_scalar().unwrap_or(0.0) > 0.0),
            AssertionSeverity::Error,
            Some("value must be positive".to_string()),
        );

        let signals = SignalStorage::default();
        let entities = EntityStorage::default();
        let current = Value::Scalar(10.0);
        let prev = Value::Scalar(5.0);

        let result = checker.check_signal(
            &signal,
            &current,
            &prev,
            &signals,
            &entities,
            Dt(1.0),
            0.0,
            0,
            "test",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_assertion_fail() {
        let mut checker = AssertionChecker::new();
        checker.set_policy(FaultPolicy::Fatal);
        let signal: SignalId = "test.signal".into();

        // Assertion: value must be positive
        checker.register(
            signal.clone(),
            Box::new(|ctx| ctx.current.as_scalar().unwrap_or(0.0) > 0.0),
            AssertionSeverity::Error,
            Some("value must be positive".to_string()),
        );

        let signals = SignalStorage::default();
        let entities = EntityStorage::default();
        let current = Value::Scalar(-5.0);
        let prev = Value::Scalar(5.0);

        let result = checker.check_signal(
            &signal,
            &current,
            &prev,
            &signals,
            &entities,
            Dt(1.0),
            0.0,
            0,
            "test",
        );
        assert!(matches!(result, Err(Error::AssertionFailed { .. })));
    }

    #[test]
    fn test_assertion_warn_continues() {
        let mut checker = AssertionChecker::new();
        let signal: SignalId = "test.signal".into();

        // Warning assertion: value should be below 100
        checker.register(
            signal.clone(),
            Box::new(|ctx| ctx.current.as_scalar().unwrap_or(0.0) < 100.0),
            AssertionSeverity::Warn,
            Some("value exceeds recommended maximum".to_string()),
        );

        let signals = SignalStorage::default();
        let entities = EntityStorage::default();
        let current = Value::Scalar(150.0); // Exceeds 100
        let prev = Value::Scalar(50.0);

        // Should succeed despite warning
        let result = checker.check_signal(
            &signal,
            &current,
            &prev,
            &signals,
            &entities,
            Dt(1.0),
            0.0,
            0,
            "test",
        );
        assert!(result.is_ok());
    }
}
