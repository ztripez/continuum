//! Assertion evaluation and checking
//!
//! Assertions are evaluated after signal resolution to validate invariants.

use tracing::{debug, error, warn};

use crate::error::{Error, Result};
use crate::storage::{EntityStorage, SignalStorage};
use crate::types::{AssertionSeverity, Dt, SignalId, Value};

use super::context::AssertContext;

/// Function that evaluates an assertion condition

pub type AssertionFn = Box<dyn Fn(&AssertContext) -> bool + Send + Sync>;

/// Record of an assertion failure
#[derive(Debug, Clone)]
pub struct AssertionFailure {
    pub signal: SignalId,
    pub severity: AssertionSeverity,
    pub message: String,
    pub tick: u64,
    pub era: String,
    pub sim_time: f64,
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
}

impl AssertionChecker {
    /// Create a new assertion checker
    pub fn new() -> Self {
        Self {
            assertions: Vec::new(),
            failures: Vec::new(),
            max_failures: 1000,
        }
    }

    /// Create a new assertion checker with a specific failure buffer size
    pub fn with_capacity(max_failures: usize) -> Self {
        Self {
            assertions: Vec::new(),
            failures: Vec::new(),
            max_failures,
        }
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
                    AssertionSeverity::Error => {
                        error!(signal = %signal, message = %message, "assertion error");
                        return Err(Error::AssertionFailed {
                            signal: signal.clone(),
                            message,
                        });
                    }
                    AssertionSeverity::Fatal => {
                        error!(signal = %signal, message = %message, "assertion fatal");
                        return Err(Error::AssertionFailed {
                            signal: signal.clone(),
                            message,
                        });
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
