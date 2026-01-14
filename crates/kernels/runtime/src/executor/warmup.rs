//! Warmup phase execution
//!
//! Executes pre-causal equilibration before simulation begins.

use tracing::{debug, error, info, instrument, trace};

use crate::error::{Error, Result};
use crate::storage::{EntityStorage, SignalStorage};
use crate::types::{SignalId, Value, WarmupConfig, WarmupResult};

use super::context::WarmupContext;

/// Function that computes a warmup iteration for a signal
pub type WarmupFn = Box<dyn Fn(&WarmupContext) -> Value + Send + Sync>;

/// Registered warmup function with its configuration
pub struct RegisteredWarmup {
    /// The signal to warm up.
    pub signal: SignalId,
    /// The function that computes the next value.
    pub function: WarmupFn,
    /// Configuration for this warmup.
    pub config: WarmupConfig,
}

/// Warmup executor
pub struct WarmupExecutor {
    /// Registered warmup functions
    warmups: Vec<RegisteredWarmup>,
    /// Whether warmup has been executed
    complete: bool,
}

impl Default for WarmupExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl WarmupExecutor {
    /// Create a new warmup executor
    pub fn new() -> Self {
        Self {
            warmups: Vec::new(),
            complete: false,
        }
    }

    /// Register a warmup function for a signal
    pub fn register(&mut self, signal: SignalId, function: WarmupFn, config: WarmupConfig) {
        debug!(signal = %signal, max_iter = config.max_iterations, "warmup registered");
        self.warmups.push(RegisteredWarmup {
            signal,
            function,
            config,
        });
    }

    /// Check if warmup has been executed
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Check if any warmup functions are registered
    pub fn is_empty(&self) -> bool {
        self.warmups.is_empty()
    }

    /// Execute warmup phase
    ///
    /// Runs all registered warmup functions until convergence or max iterations.
    #[instrument(skip(self, signals, entities), name = "warmup")]
    pub fn execute(
        &mut self,
        signals: &mut SignalStorage,
        entities: &EntityStorage,
        sim_time: f64,
    ) -> Result<WarmupResult> {
        if self.complete {
            return Ok(WarmupResult {
                iterations: 0,
                converged: true,
            });
        }

        if self.warmups.is_empty() {
            info!("no warmup functions registered");
            self.complete = true;
            return Ok(WarmupResult {
                iterations: 0,
                converged: true,
            });
        }

        // Find the maximum iterations needed
        let max_iterations = self
            .warmups
            .iter()
            .map(|w| w.config.max_iterations)
            .max()
            .unwrap_or(0);

        info!(
            signals = self.warmups.len(),
            max_iterations, "warmup starting"
        );

        let mut iteration = 0;
        let mut converged = false;

        while iteration < max_iterations {
            trace!(iteration, "warmup iteration");

            let mut all_converged = true;
            let mut max_delta: f64 = 0.0;

            // Execute each warmup function
            for warmup in &self.warmups {
                // Skip if this signal's iterations are exhausted
                if iteration >= warmup.config.max_iterations {
                    continue;
                }

                let prev = signals
                    .get(&warmup.signal)
                    .ok_or_else(|| Error::SignalNotFound(warmup.signal.clone()))?;

                let ctx = WarmupContext {
                    prev,
                    signals,
                    entities,
                    iteration,
                    sim_time,
                };

                let new_value = (warmup.function)(&ctx);

                // Check for numeric errors
                if let Value::Scalar(v) = &new_value {
                    if v.is_nan() {
                        error!(signal = %warmup.signal, iteration, "warmup NaN");
                        return Err(Error::WarmupDivergence {
                            signal: warmup.signal.clone(),
                            iteration,
                            message: "NaN result".to_string(),
                        });
                    }
                    if v.is_infinite() {
                        error!(signal = %warmup.signal, iteration, "warmup infinite");
                        return Err(Error::WarmupDivergence {
                            signal: warmup.signal.clone(),
                            iteration,
                            message: "Infinite result".to_string(),
                        });
                    }

                    // Check convergence
                    if let Some(epsilon) = warmup.config.convergence_epsilon {
                        if let Value::Scalar(prev_v) = prev {
                            let delta = (v - prev_v).abs();
                            max_delta = max_delta.max(delta);
                            if delta >= epsilon {
                                all_converged = false;
                            }
                        }
                    } else {
                        all_converged = false;
                    }
                }

                signals.set_current(warmup.signal.clone(), new_value);
            }

            // Advance iteration state
            signals.advance_tick();
            iteration += 1;

            // Check if all signals with convergence criteria have converged
            if all_converged
                && self
                    .warmups
                    .iter()
                    .any(|w| w.config.convergence_epsilon.is_some())
            {
                debug!(iteration, max_delta, "warmup converged");
                converged = true;
                break;
            }
        }

        // Check for divergence (didn't converge within iterations)
        let any_requires_convergence = self
            .warmups
            .iter()
            .any(|w| w.config.convergence_epsilon.is_some());

        if any_requires_convergence && !converged {
            let first_signal = &self.warmups[0].signal;
            error!(iterations = iteration, "warmup failed to converge");
            return Err(Error::WarmupDivergence {
                signal: first_signal.clone(),
                iteration,
                message: "Failed to converge within max iterations".to_string(),
            });
        }

        info!(iterations = iteration, converged, "warmup complete");
        self.complete = true;

        Ok(WarmupResult {
            iterations: iteration,
            converged,
        })
    }
}
