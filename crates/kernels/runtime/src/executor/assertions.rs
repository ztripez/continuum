//! Assertion evaluation and checking
//!
//! Assertions are evaluated after signal resolution to validate invariants.

use tracing::{debug, error, warn};

use crate::error::{Error, Result};
use crate::storage::SignalStorage;
use crate::types::{Dt, SignalId, Value};

use super::context::AssertContext;

/// Severity of an assertion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssertionSeverity {
    /// Warning only, execution continues
    Warn,
    /// Error, may halt based on policy
    Error,
    /// Fatal, always halts
    Fatal,
}

impl Default for AssertionSeverity {
    fn default() -> Self {
        Self::Error
    }
}

/// A registered assertion for a signal
pub struct SignalAssertion {
    /// The signal this assertion belongs to
    pub signal: SignalId,
    /// Function that evaluates the assertion condition
    pub condition: AssertionFn,
    /// Severity of the assertion
    pub severity: AssertionSeverity,
    /// Optional message for failure
    pub message: Option<String>,
}

/// Function that evaluates an assertion condition
pub type AssertionFn = Box<dyn Fn(&AssertContext) -> bool + Send + Sync>;

/// Assertion checker for the runtime
#[derive(Default)]
pub struct AssertionChecker {
    /// Registered assertions
    assertions: Vec<SignalAssertion>,
}

impl AssertionChecker {
    /// Create a new assertion checker
    pub fn new() -> Self {
        Self::default()
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
        &self,
        signal: &SignalId,
        current: &Value,
        prev: &Value,
        signals: &SignalStorage,
        dt: Dt,
    ) -> Result<()> {
        let ctx = AssertContext {
            current,
            prev,
            signals,
            dt,
        };

        for assertion in self.assertions.iter().filter(|a| &a.signal == signal) {
            let passed = (assertion.condition)(&ctx);

            if !passed {
                let message = assertion
                    .message
                    .clone()
                    .unwrap_or_else(|| format!("assertion failed for signal {}", signal));

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
        &self,
        resolved_signals: &[(SignalId, Value)],
        signals: &SignalStorage,
        dt: Dt,
    ) -> Result<()> {
        for (signal, current) in resolved_signals {
            let prev = signals.get_prev(signal).unwrap_or(current);
            self.check_signal(signal, current, prev, signals, dt)?;
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
        let current = Value::Scalar(10.0);
        let prev = Value::Scalar(5.0);

        let result = checker.check_signal(&signal, &current, &prev, &signals, Dt(1.0));
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
        let current = Value::Scalar(-5.0);
        let prev = Value::Scalar(5.0);

        let result = checker.check_signal(&signal, &current, &prev, &signals, Dt(1.0));
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
        let current = Value::Scalar(150.0); // Exceeds 100
        let prev = Value::Scalar(50.0);

        // Should succeed despite warning
        let result = checker.check_signal(&signal, &current, &prev, &signals, Dt(1.0));
        assert!(result.is_ok());
    }
}
