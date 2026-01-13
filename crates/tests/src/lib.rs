//! Integration test harness for Continuum.
//!
//! This crate provides utilities for end-to-end testing of the full
//! simulation pipeline: Parse → Lower → Compile → Execute → Verify.

use std::collections::HashMap;
use std::path::PathBuf;

use continuum_compiler::ir::{
    CompiledWorld, build_assertion, build_era_configs, build_field_measure, build_fracture,
    build_signal_resolver, build_warmup_fn, compile, convert_assertion_severity,
    get_initial_signal_value,
};
use continuum_foundation::{EraId, FieldId, SignalId};
use continuum_runtime::executor::Runtime;
use continuum_runtime::types::{Value, WarmupConfig};

// Ensure functions are registered
use continuum_functions as _;

/// Test harness for running Continuum simulations from DSL source.
pub struct TestHarness {
    runtime: Runtime,
    world: CompiledWorld,
}

impl TestHarness {
    /// Create a new test harness from DSL source code.
    ///
    /// # Panics
    ///
    /// Panics if parsing, lowering, or compilation fails.
    pub fn from_source(source: &str) -> Self {
        // Use unified compiler
        let mut source_map = HashMap::new();
        source_map.insert(PathBuf::from("test.cdsl"), source);

        let compile_result = continuum_compiler::compile(&source_map);
        let world = match compile_result.success() {
            Ok(w) => w,
            Err(diagnostics) => {
                panic!("Compilation failed: {:?}", diagnostics);
            }
        };

        // Compile to DAGs
        let compilation = compile(&world).expect("DAG compilation failed");

        let signals = world.signals();
        let eras = world.eras();
        let fields = world.fields();
        let fractures = world.fractures();

        // Find initial era
        let initial_era = eras
            .iter()
            .find(|(_, era)| era.is_initial)
            .map(|(id, _)| id.clone())
            .unwrap_or_else(|| {
                eras.keys()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| EraId::from("default"))
            });

        // Build era configs
        let era_configs = build_era_configs(&world);

        // Create runtime
        let mut runtime = Runtime::new(initial_era, era_configs, compilation.dags);

        // Register resolvers
        for (signal_id, signal) in &signals {
            if let Some(resolver) = build_signal_resolver(signal, &world) {
                runtime.register_resolver(resolver);
            }

            // Register warmup if present
            if let Some(ref warmup) = signal.warmup {
                let warmup_fn = build_warmup_fn(&warmup.iterate, &world.constants, &world.config);
                let config = WarmupConfig {
                    max_iterations: warmup.iterations,
                    convergence_epsilon: warmup.convergence,
                };
                runtime.register_warmup(signal_id.clone(), warmup_fn, config);
            }
        }

        // Register assertions
        for (signal_id, signal) in &signals {
            for assertion in &signal.assertions {
                let assertion_fn = build_assertion(&assertion.condition, &world);
                let severity = convert_assertion_severity(assertion.severity);
                runtime.register_assertion(
                    signal_id.clone(),
                    assertion_fn,
                    severity,
                    assertion.message.clone(),
                );
            }
        }

        // Register field measure functions
        // Skip fields with entity expressions (aggregates, etc.)
        for (field_id, field) in &fields {
            if let Some(ref expr) = field.measure {
                let runtime_id = field_id.clone();
                if let Some(measure_fn) = build_field_measure(&runtime_id, expr, &world) {
                    runtime.register_measure_op(measure_fn);
                }
            }
        }

        // Register fracture detectors
        for (_fracture_id, fracture) in &fractures {
            let fracture_fn = build_fracture(fracture, &world);
            runtime.register_fracture(fracture_fn);
        }

        // Initialize signals
        for (signal_id, _signal) in &signals {
            let value = get_initial_signal_value(&world, signal_id);
            runtime.init_signal(signal_id.clone(), value);
        }

        Self { runtime, world }
    }

    /// Execute a single tick.
    ///
    /// # Panics
    ///
    /// Panics if tick execution fails.
    pub fn tick(&mut self) {
        if !self.runtime.is_warmup_complete() {
            self.runtime.execute_warmup().expect("Warmup failed");
        }
        self.runtime.execute_tick().expect("Tick failed");
    }

    /// Execute multiple ticks.
    pub fn run_ticks(&mut self, count: u64) {
        for _ in 0..count {
            self.tick();
        }
    }

    /// Get a signal's current value.
    pub fn get_signal(&self, name: &str) -> Option<&Value> {
        self.runtime.get_signal(&SignalId::from(name))
    }

    /// Get a signal's scalar value.
    pub fn get_scalar(&self, name: &str) -> Option<f64> {
        self.get_signal(name).and_then(|v| v.as_scalar())
    }

    /// Get the current era.
    pub fn current_era(&self) -> &EraId {
        self.runtime.era()
    }

    /// Get the current tick number.
    pub fn current_tick(&self) -> u64 {
        self.runtime.tick()
    }

    /// Get access to the compiled world for verification.
    pub fn world(&self) -> &CompiledWorld {
        &self.world
    }

    /// Drain field samples (clears the buffer).
    pub fn drain_fields(
        &mut self,
    ) -> indexmap::IndexMap<FieldId, Vec<continuum_runtime::storage::FieldSample>> {
        self.runtime.drain_fields()
    }
}
