//! Tick executor
//!
//! Orchestrates DAG execution through phases.

mod assertions;
mod context;
mod phases;
mod warmup;

use std::collections::HashMap;

use tracing::{error, info, instrument, trace};

use crate::dag::DagSet;
use crate::error::{Error, Result};
use crate::storage::{FieldBuffer, FieldSample, FractureQueue, InputChannels, SignalStorage};
use crate::types::{
    Dt, EraId, FieldId, SignalId, StratumId, StratumState, TickContext, Value, WarmupConfig,
    WarmupResult,
};

// Re-export public types
pub use assertions::{AssertionChecker, AssertionFn, AssertionSeverity, SignalAssertion};
pub use context::{
    AssertContext, CollectContext, FractureContext, ImpulseContext, MeasureContext,
    ResolveContext, WarmupContext,
};
pub use phases::{CollectFn, FractureFn, ImpulseFn, MeasureFn, PhaseExecutor, ResolverFn};
pub use warmup::{RegisteredWarmup, WarmupExecutor, WarmupFn};

/// Function that evaluates era transition conditions
pub type TransitionFn = Box<dyn Fn(&SignalStorage) -> Option<EraId> + Send + Sync>;

/// Era configuration
pub struct EraConfig {
    /// Time step for this era
    pub dt: Dt,
    /// Stratum states in this era
    pub strata: HashMap<StratumId, StratumState>,
    /// Transition condition (returns Some(next_era) if should transition)
    pub transition: Option<TransitionFn>,
}

/// Runtime state for a simulation
pub struct Runtime {
    /// Signal storage
    signals: SignalStorage,
    /// Input channels for Collect phase
    input_channels: InputChannels,
    /// Field buffer for Measure phase
    field_buffer: FieldBuffer,
    /// Fracture outputs queued for next tick
    fracture_queue: FractureQueue,
    /// Current tick number
    tick: u64,
    /// Current era
    current_era: EraId,
    /// Era configurations
    eras: HashMap<EraId, EraConfig>,
    /// Execution DAGs
    dags: DagSet,
    /// Phase executor
    phase_executor: PhaseExecutor,
    /// Warmup executor
    warmup_executor: WarmupExecutor,
    /// Assertion checker
    assertion_checker: AssertionChecker,
    /// Pending impulses to apply in next Collect phase (handler_idx, payload)
    pending_impulses: Vec<(usize, Value)>,
}

impl Runtime {
    /// Create a new runtime
    pub fn new(initial_era: EraId, eras: HashMap<EraId, EraConfig>, dags: DagSet) -> Self {
        info!(era = %initial_era, "runtime created");
        Self {
            signals: SignalStorage::default(),
            input_channels: InputChannels::default(),
            field_buffer: FieldBuffer::default(),
            fracture_queue: FractureQueue::default(),
            tick: 0,
            current_era: initial_era,
            eras,
            dags,
            phase_executor: PhaseExecutor::new(),
            warmup_executor: WarmupExecutor::new(),
            assertion_checker: AssertionChecker::new(),
            pending_impulses: Vec::new(),
        }
    }

    /// Register a resolver function, returns its index
    pub fn register_resolver(&mut self, resolver: ResolverFn) -> usize {
        self.phase_executor.register_resolver(resolver)
    }

    /// Register a collect operator, returns its index
    pub fn register_collect_op(&mut self, op: CollectFn) -> usize {
        self.phase_executor.register_collect_op(op)
    }

    /// Register a fracture function, returns its index
    pub fn register_fracture(&mut self, fracture: FractureFn) -> usize {
        self.phase_executor.register_fracture(fracture)
    }

    /// Register a measure operator, returns its index
    pub fn register_measure_op(&mut self, op: MeasureFn) -> usize {
        self.phase_executor.register_measure_op(op)
    }

    /// Register an impulse handler, returns its index
    pub fn register_impulse(&mut self, handler: ImpulseFn) -> usize {
        self.phase_executor.register_impulse(handler)
    }

    /// Inject an impulse to be applied in the next tick's Collect phase
    pub fn inject_impulse(&mut self, handler_idx: usize, payload: Value) {
        tracing::debug!(handler_idx, ?payload, "impulse injected");
        self.pending_impulses.push((handler_idx, payload));
    }

    /// Register a warmup function for a signal
    pub fn register_warmup(&mut self, signal: SignalId, warmup_fn: WarmupFn, config: WarmupConfig) {
        self.warmup_executor.register(signal, warmup_fn, config);
    }

    /// Register an assertion for a signal
    pub fn register_assertion(
        &mut self,
        signal: SignalId,
        condition: AssertionFn,
        severity: AssertionSeverity,
        message: Option<String>,
    ) {
        self.assertion_checker
            .register(signal, condition, severity, message);
    }

    /// Initialize a signal with a value
    pub fn init_signal(&mut self, id: SignalId, value: Value) {
        tracing::debug!(signal = %id, ?value, "signal initialized");
        self.signals.init(id, value);
    }

    /// Get current tick number
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Get current era
    pub fn era(&self) -> &EraId {
        &self.current_era
    }

    /// Get a signal's last resolved value
    pub fn get_signal(&self, id: &SignalId) -> Option<&Value> {
        self.signals.get_resolved(id)
    }

    /// Check if warmup has been executed
    pub fn is_warmup_complete(&self) -> bool {
        self.warmup_executor.is_complete()
    }

    /// Get access to the field buffer (for observer consumption)
    pub fn field_buffer(&self) -> &FieldBuffer {
        &self.field_buffer
    }

    /// Drain the field buffer (for observer consumption)
    pub fn drain_fields(&mut self) -> indexmap::IndexMap<FieldId, Vec<FieldSample>> {
        self.field_buffer.drain()
    }

    /// Execute warmup phase (pre-causal equilibration)
    ///
    /// Must be called before execute_tick. Runs all registered warmup
    /// functions until convergence or max iterations.
    pub fn execute_warmup(&mut self) -> Result<WarmupResult> {
        self.warmup_executor.execute(&mut self.signals)
    }

    /// Execute a single tick
    #[instrument(skip(self), fields(tick = self.tick, era = %self.current_era))]
    pub fn execute_tick(&mut self) -> Result<TickContext> {
        trace!("tick start");

        // Extract needed config values to avoid borrow issues
        let (dt, strata_states) = {
            let era_config = self
                .eras
                .get(&self.current_era)
                .ok_or_else(|| Error::EraNotFound(self.current_era.clone()))?;
            (era_config.dt, era_config.strata.clone())
        };

        let ctx = TickContext {
            tick: self.tick,
            dt,
            era: self.current_era.clone(),
        };

        // Verify era DAGs exist
        if !self.dags.eras.contains_key(&self.current_era) {
            error!(era = %self.current_era, "era DAGs not found");
            return Err(Error::EraNotFound(self.current_era.clone()));
        }

        // Phase 2: Collect
        self.phase_executor.execute_collect(
            &self.current_era,
            self.tick,
            dt,
            &strata_states,
            &self.dags,
            &self.signals,
            &mut self.input_channels,
            &mut self.pending_impulses,
        )?;

        // Phase 3: Resolve
        self.phase_executor.execute_resolve(
            &self.current_era,
            self.tick,
            dt,
            &strata_states,
            &self.dags,
            &mut self.signals,
            &mut self.input_channels,
            &self.assertion_checker,
        )?;

        // Phase 4: Fracture
        self.phase_executor.execute_fracture(
            &self.current_era,
            dt,
            &self.dags,
            &self.signals,
            &mut self.fracture_queue,
        )?;

        // Phase 5: Measure
        self.phase_executor.execute_measure(
            &self.current_era,
            self.tick,
            dt,
            &strata_states,
            &self.dags,
            &self.signals,
            &mut self.field_buffer,
        )?;

        // Post-tick: check era transitions
        self.check_era_transition()?;

        // Advance state
        self.signals.advance_tick();
        self.fracture_queue.drain_into(&mut self.input_channels);
        self.tick += 1;

        trace!("tick complete");
        Ok(ctx)
    }

    fn check_era_transition(&mut self) -> Result<()> {
        let era_config = self.eras.get(&self.current_era).unwrap();
        if let Some(ref transition) = era_config.transition
            && let Some(next_era) = transition(&self.signals)
        {
            if !self.eras.contains_key(&next_era) {
                error!(era = %next_era, "transition to unknown era");
                return Err(Error::EraNotFound(next_era));
            }
            info!(from = %self.current_era, to = %next_era, "era transition");
            self.current_era = next_era;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{DagBuilder, DagNode, EraDags, NodeId, NodeKind};
    use crate::types::Phase;
    use std::collections::HashSet;

    #[test]
    fn test_simple_tick_execution() {
        // Create a simple world with one signal that increments
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "default".into();
        let signal_id: SignalId = "counter".into();

        // Build DAG with one resolve node
        let mut builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
        builder.add_node(DagNode {
            id: NodeId("counter_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_id.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_id.clone(),
                resolver_idx: 0,
            },
        });
        let dag = builder.build().unwrap();

        let mut era_dags = EraDags::default();
        era_dags.insert(dag);

        let mut dags = DagSet::default();
        dags.insert_era(era_id.clone(), era_dags);

        // Create era config
        let mut strata = HashMap::new();
        strata.insert(stratum_id, StratumState::Active);
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = HashMap::new();
        eras.insert(era_id.clone(), era_config);

        // Create runtime
        let mut runtime = Runtime::new(era_id, eras, dags);

        // Register resolver: prev + 1
        runtime.register_resolver(Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap_or(0.0);
            Value::Scalar(prev + 1.0)
        }));

        // Initialize signal
        runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

        // Execute ticks
        runtime.execute_tick().unwrap();
        assert_eq!(
            runtime.get_signal(&signal_id),
            Some(&Value::Scalar(1.0))
        );

        runtime.execute_tick().unwrap();
        assert_eq!(
            runtime.get_signal(&signal_id),
            Some(&Value::Scalar(2.0))
        );

        runtime.execute_tick().unwrap();
        assert_eq!(
            runtime.get_signal(&signal_id),
            Some(&Value::Scalar(3.0))
        );
    }

    #[test]
    fn test_stratum_stride() {
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "slow".into();
        let signal_id: SignalId = "counter".into();

        let mut builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
        builder.add_node(DagNode {
            id: NodeId("counter_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_id.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_id.clone(),
                resolver_idx: 0,
            },
        });
        let dag = builder.build().unwrap();

        let mut era_dags = EraDags::default();
        era_dags.insert(dag);

        let mut dags = DagSet::default();
        dags.insert_era(era_id.clone(), era_dags);

        // Stratum executes every 2 ticks
        let mut strata = HashMap::new();
        strata.insert(stratum_id, StratumState::ActiveWithStride(2));
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = HashMap::new();
        eras.insert(era_id.clone(), era_config);

        let mut runtime = Runtime::new(era_id, eras, dags);
        runtime.register_resolver(Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap_or(0.0);
            Value::Scalar(prev + 1.0)
        }));
        runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

        // Tick 0: stride 2, 0 % 2 == 0, executes
        runtime.execute_tick().unwrap();
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(1.0)));

        // Tick 1: 1 % 2 != 0, skipped
        runtime.execute_tick().unwrap();
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(1.0)));

        // Tick 2: 2 % 2 == 0, executes
        runtime.execute_tick().unwrap();
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(2.0)));
    }

    fn create_minimal_runtime(era_id: EraId) -> Runtime {
        let dags = DagSet::default();
        let mut eras = HashMap::new();
        eras.insert(
            era_id.clone(),
            EraConfig {
                dt: Dt(1.0),
                strata: HashMap::new(),
                transition: None,
            },
        );
        Runtime::new(era_id, eras, dags)
    }

    #[test]
    fn test_warmup_fixed_iterations() {
        let era_id: EraId = "test".into();
        let signal_id: SignalId = "temp".into();

        let mut runtime = create_minimal_runtime(era_id);
        runtime.init_signal(signal_id.clone(), Value::Scalar(1000.0));

        // Warmup: halve the value each iteration
        runtime.register_warmup(
            signal_id.clone(),
            Box::new(|ctx| {
                let prev = ctx.prev.as_scalar().unwrap();
                Value::Scalar(prev * 0.5)
            }),
            WarmupConfig {
                max_iterations: 5,
                convergence_epsilon: None,
            },
        );

        let result = runtime.execute_warmup().unwrap();

        assert_eq!(result.iterations, 5);
        assert!(!result.converged);

        // 1000 * 0.5^5 = 31.25
        let final_value = runtime.get_signal(&signal_id).unwrap().as_scalar().unwrap();
        assert!((final_value - 31.25).abs() < 0.001);
    }

    #[test]
    fn test_warmup_convergence() {
        let era_id: EraId = "test".into();
        let signal_id: SignalId = "equilibrium".into();

        let mut runtime = create_minimal_runtime(era_id);
        runtime.init_signal(signal_id.clone(), Value::Scalar(100.0));

        // Warmup: converge toward 50 by halving the distance each iteration
        runtime.register_warmup(
            signal_id.clone(),
            Box::new(|ctx| {
                let prev = ctx.prev.as_scalar().unwrap();
                let target = 50.0;
                Value::Scalar(prev + (target - prev) * 0.5)
            }),
            WarmupConfig {
                max_iterations: 100,
                convergence_epsilon: Some(0.01),
            },
        );

        let result = runtime.execute_warmup().unwrap();

        assert!(result.converged);
        assert!(result.iterations < 100); // Should converge before max

        let final_value = runtime.get_signal(&signal_id).unwrap().as_scalar().unwrap();
        assert!((final_value - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_warmup_divergence_nan() {
        let era_id: EraId = "test".into();
        let signal_id: SignalId = "bad".into();

        let mut runtime = create_minimal_runtime(era_id);
        runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

        // Warmup: produce NaN
        runtime.register_warmup(
            signal_id.clone(),
            Box::new(|_ctx| Value::Scalar(f64::NAN)),
            WarmupConfig {
                max_iterations: 10,
                convergence_epsilon: None,
            },
        );

        let result = runtime.execute_warmup();
        assert!(matches!(result, Err(Error::WarmupDivergence { .. })));
    }

    #[test]
    fn test_warmup_no_functions() {
        let era_id: EraId = "test".into();
        let mut runtime = create_minimal_runtime(era_id);

        let result = runtime.execute_warmup().unwrap();

        assert_eq!(result.iterations, 0);
        assert!(result.converged);
        assert!(runtime.is_warmup_complete());
    }

    #[test]
    fn test_warmup_multiple_signals() {
        let era_id: EraId = "test".into();
        let signal_a: SignalId = "a".into();
        let signal_b: SignalId = "b".into();

        let mut runtime = create_minimal_runtime(era_id);
        runtime.init_signal(signal_a.clone(), Value::Scalar(100.0));
        runtime.init_signal(signal_b.clone(), Value::Scalar(0.0));

        // Signal A: decays toward 50
        runtime.register_warmup(
            signal_a.clone(),
            Box::new(|ctx| {
                let prev = ctx.prev.as_scalar().unwrap();
                Value::Scalar(prev + (50.0 - prev) * 0.5)
            }),
            WarmupConfig {
                max_iterations: 50,
                convergence_epsilon: Some(0.1),
            },
        );

        // Signal B: grows toward 50
        runtime.register_warmup(
            signal_b.clone(),
            Box::new(|ctx| {
                let prev = ctx.prev.as_scalar().unwrap();
                Value::Scalar(prev + (50.0 - prev) * 0.5)
            }),
            WarmupConfig {
                max_iterations: 50,
                convergence_epsilon: Some(0.1),
            },
        );

        let result = runtime.execute_warmup().unwrap();

        assert!(result.converged);

        let a_val = runtime.get_signal(&signal_a).unwrap().as_scalar().unwrap();
        let b_val = runtime.get_signal(&signal_b).unwrap().as_scalar().unwrap();

        assert!((a_val - 50.0).abs() < 1.0);
        assert!((b_val - 50.0).abs() < 1.0);
    }

    #[test]
    fn test_measure_phase_field_emission() {
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "default".into();
        let signal_id: SignalId = "temperature".into();
        let field_id: FieldId = "temp_field".into();

        // Build DAG for Resolve phase
        let mut resolve_builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
        resolve_builder.add_node(DagNode {
            id: NodeId("temp_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_id.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_id.clone(),
                resolver_idx: 0,
            },
        });
        let resolve_dag = resolve_builder.build().unwrap();

        // Build DAG for Measure phase
        let mut measure_builder = DagBuilder::new(Phase::Measure, stratum_id.clone());
        measure_builder.add_node(DagNode {
            id: NodeId("temp_measure".to_string()),
            reads: [signal_id.clone()].into_iter().collect(),
            writes: None,
            kind: NodeKind::OperatorMeasure { operator_idx: 0 },
        });
        let measure_dag = measure_builder.build().unwrap();

        let mut era_dags = EraDags::default();
        era_dags.insert(resolve_dag);
        era_dags.insert(measure_dag);

        let mut dags = DagSet::default();
        dags.insert_era(era_id.clone(), era_dags);

        let mut strata = HashMap::new();
        strata.insert(stratum_id, StratumState::Active);
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = HashMap::new();
        eras.insert(era_id.clone(), era_config);

        let mut runtime = Runtime::new(era_id, eras, dags);

        // Register resolver: temperature increments by 10 each tick
        runtime.register_resolver(Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap_or(0.0);
            Value::Scalar(prev + 10.0)
        }));

        // Register measure operator: emit temperature to field
        let signal_id_clone = signal_id.clone();
        let field_id_clone = field_id.clone();
        runtime.register_measure_op(Box::new(move |ctx| {
            let temp = ctx.signals.get(&signal_id_clone).unwrap().as_scalar().unwrap();
            ctx.fields.emit_scalar(field_id_clone.clone(), temp);
        }));

        runtime.init_signal(signal_id, Value::Scalar(100.0));

        // Execute tick
        runtime.execute_tick().unwrap();

        // Check field buffer has the emitted value
        let samples = runtime.field_buffer().get_samples(&field_id).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].value.as_scalar(), Some(110.0));

        // Drain and verify empty
        let drained = runtime.drain_fields();
        assert_eq!(drained.len(), 1);
        assert!(runtime.field_buffer().is_empty());

        // Execute another tick
        runtime.execute_tick().unwrap();

        let samples = runtime.field_buffer().get_samples(&field_id).unwrap();
        assert_eq!(samples[0].value.as_scalar(), Some(120.0));
    }

    #[test]
    fn test_impulse_injection() {
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "default".into();
        let signal_id: SignalId = "energy".into();

        // Build DAG for Resolve phase
        let mut resolve_builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
        resolve_builder.add_node(DagNode {
            id: NodeId("energy_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_id.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_id.clone(),
                resolver_idx: 0,
            },
        });
        let resolve_dag = resolve_builder.build().unwrap();

        let mut era_dags = EraDags::default();
        era_dags.insert(resolve_dag);

        let mut dags = DagSet::default();
        dags.insert_era(era_id.clone(), era_dags);

        let mut strata = HashMap::new();
        strata.insert(stratum_id, StratumState::Active);
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = HashMap::new();
        eras.insert(era_id.clone(), era_config);

        let mut runtime = Runtime::new(era_id, eras, dags);

        // Register resolver: prev + sum(inputs)
        runtime.register_resolver(Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap_or(0.0);
            Value::Scalar(prev + ctx.inputs)
        }));

        // Register impulse handler: adds payload value to signal
        let signal_id_clone = signal_id.clone();
        let handler_idx = runtime.register_impulse(Box::new(move |ctx, payload| {
            let value = payload.as_scalar().unwrap();
            ctx.channels.accumulate(&signal_id_clone, value);
        }));

        runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

        // Tick 1: no impulse, signal stays at 0
        runtime.execute_tick().unwrap();
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(0.0)));

        // Inject impulse for next tick
        runtime.inject_impulse(handler_idx, Value::Scalar(100.0));

        // Tick 2: impulse adds 100
        runtime.execute_tick().unwrap();
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(100.0)));

        // Tick 3: no impulse, signal stays at 100
        runtime.execute_tick().unwrap();
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(100.0)));

        // Inject multiple impulses
        runtime.inject_impulse(handler_idx, Value::Scalar(25.0));
        runtime.inject_impulse(handler_idx, Value::Scalar(25.0));

        // Tick 4: both impulses add 50 total
        runtime.execute_tick().unwrap();
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(150.0)));
    }

    #[test]
    fn test_assertion_during_resolve() {
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "default".into();
        let signal_id: SignalId = "bounded".into();

        let mut builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
        builder.add_node(DagNode {
            id: NodeId("bounded_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_id.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_id.clone(),
                resolver_idx: 0,
            },
        });
        let dag = builder.build().unwrap();

        let mut era_dags = EraDags::default();
        era_dags.insert(dag);

        let mut dags = DagSet::default();
        dags.insert_era(era_id.clone(), era_dags);

        let mut strata = HashMap::new();
        strata.insert(stratum_id, StratumState::Active);
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = HashMap::new();
        eras.insert(era_id.clone(), era_config);

        let mut runtime = Runtime::new(era_id, eras, dags);

        // Register resolver that produces negative values after tick 2
        let tick_counter = std::sync::atomic::AtomicU64::new(0);
        runtime.register_resolver(Box::new(move |_ctx| {
            let tick = tick_counter.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if tick >= 2 {
                Value::Scalar(-10.0)
            } else {
                Value::Scalar(10.0)
            }
        }));

        // Register assertion: value must be positive
        runtime.register_assertion(
            signal_id.clone(),
            Box::new(|ctx| ctx.current.as_scalar().unwrap_or(0.0) > 0.0),
            AssertionSeverity::Error,
            Some("value must be positive".to_string()),
        );

        runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

        // First two ticks should succeed
        runtime.execute_tick().unwrap();
        runtime.execute_tick().unwrap();

        // Third tick should fail assertion
        let result = runtime.execute_tick();
        assert!(matches!(result, Err(Error::AssertionFailed { .. })));
    }
}
