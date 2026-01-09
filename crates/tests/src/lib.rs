//! Integration test harness for Continuum.
//!
//! This crate provides utilities for end-to-end testing of the full
//! simulation pipeline: Parse → Lower → Compile → Execute → Verify.

use continuum_dsl::parse;
use continuum_foundation::{EraId, FieldId, SignalId};
use continuum_ir::{
    build_assertion, build_era_configs, build_field_measure, build_fracture, build_resolver,
    compile, convert_assertion_severity, get_initial_signal_value, lower, CompiledWorld,
};
use continuum_runtime::executor::Runtime;
use continuum_runtime::types::Value;

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
        // Parse
        let (unit, errors) = parse(source);
        if !errors.is_empty() {
            panic!("Parse errors: {:?}", errors);
        }
        let unit = unit.expect("Parsing failed");

        // Lower
        let world = lower(&unit).expect("Lowering failed");

        // Compile
        let compilation = compile(&world).expect("Compilation failed");

        // Find initial era
        let initial_era = world
            .eras
            .iter()
            .find(|(_, era)| era.is_initial)
            .map(|(id, _)| id.clone())
            .unwrap_or_else(|| {
                world
                    .eras
                    .keys()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| EraId::from("default"))
            });

        // Build era configs
        let era_configs = build_era_configs(&world);

        // Create runtime
        let mut runtime = Runtime::new(initial_era, era_configs, compilation.dags);

        // Register resolvers
        for (_signal_id, signal) in &world.signals {
            if let Some(ref expr) = signal.resolve {
                let resolver = build_resolver(expr, &world, signal.uses_dt_raw);
                runtime.register_resolver(resolver);
            }
        }

        // Register assertions
        for (signal_id, signal) in &world.signals {
            for assertion in &signal.assertions {
                let assertion_fn = build_assertion(&assertion.condition, &world);
                let severity = convert_assertion_severity(assertion.severity);
                runtime.register_assertion(
                    SignalId(signal_id.0.clone()),
                    assertion_fn,
                    severity,
                    assertion.message.clone(),
                );
            }
        }

        // Register field measure functions
        for (field_id, field) in &world.fields {
            if let Some(ref expr) = field.measure {
                let runtime_id = FieldId(field_id.0.clone());
                let measure_fn = build_field_measure(&runtime_id, expr, &world);
                runtime.register_measure_op(measure_fn);
            }
        }

        // Register fracture detectors
        for (_fracture_id, fracture) in &world.fractures {
            let fracture_fn = build_fracture(fracture, &world);
            runtime.register_fracture(fracture_fn);
        }

        // Initialize signals
        for (signal_id, _signal) in &world.signals {
            let value = get_initial_signal_value(&world, signal_id);
            runtime.init_signal(SignalId(signal_id.0.clone()), value);
        }

        Self { runtime, world }
    }

    /// Execute a single tick.
    ///
    /// # Panics
    ///
    /// Panics if tick execution fails.
    pub fn tick(&mut self) {
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
        self.runtime.get_signal(&SignalId(name.to_string()))
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
    pub fn drain_fields(&mut self) -> indexmap::IndexMap<FieldId, Vec<continuum_runtime::storage::FieldSample>> {
        self.runtime.drain_fields()
    }
}
