//! Tick executor
//!
//! Orchestrates DAG execution through phases.

use std::collections::HashMap;

use rayon::prelude::*;
use tracing::{debug, error, info, instrument, trace};

use crate::dag::{DagSet, NodeKind};
use crate::error::{Error, Result};
use crate::storage::{FieldBuffer, FractureQueue, InputChannels, SignalStorage};
use crate::types::{
    Dt, EraId, Phase, SignalId, StratumId, StratumState, TickContext, Value, WarmupConfig,
    WarmupResult,
};

/// Function that resolves a signal value
pub type ResolverFn = Box<dyn Fn(&ResolveContext) -> Value + Send + Sync>;

/// Function that executes a collect operator
pub type CollectFn = Box<dyn Fn(&CollectContext) + Send + Sync>;

/// Function that evaluates a fracture condition and emits
pub type FractureFn = Box<dyn Fn(&FractureContext) -> Option<Vec<(SignalId, f64)>> + Send + Sync>;

/// Function that evaluates era transition conditions
pub type TransitionFn = Box<dyn Fn(&SignalStorage) -> Option<EraId> + Send + Sync>;

/// Function that computes a warmup iteration for a signal
pub type WarmupFn = Box<dyn Fn(&WarmupContext) -> Value + Send + Sync>;

/// Function that executes a measure-phase operator
pub type MeasureFn = Box<dyn Fn(&mut MeasureContext) + Send + Sync>;

/// Function that applies an impulse with a typed payload
pub type ImpulseFn = Box<dyn Fn(&ImpulseContext, &Value) + Send + Sync>;

/// Context available to warmup functions
pub struct WarmupContext<'a> {
    /// Current warmup value for this signal
    pub prev: &'a Value,
    /// Access to other signals (current iteration if resolved, else previous)
    pub signals: &'a SignalStorage,
    /// Current warmup iteration (0-indexed)
    pub iteration: u32,
}

/// Context available to resolver functions
pub struct ResolveContext<'a> {
    /// Previous tick's value for this signal
    pub prev: &'a Value,
    /// Access to other signals (current tick if resolved, else previous)
    pub signals: &'a SignalStorage,
    /// Accumulated inputs for this signal
    pub inputs: f64,
    /// Time step
    pub dt: Dt,
}

/// Context available to collect operators
pub struct CollectContext<'a> {
    /// Access to signals (previous tick values)
    pub signals: &'a SignalStorage,
    /// Channel to write inputs
    pub channels: &'a mut InputChannels,
    /// Time step
    pub dt: Dt,
}

/// Context available to fracture evaluation
pub struct FractureContext<'a> {
    /// Access to signals (current tick values)
    pub signals: &'a SignalStorage,
    /// Time step
    pub dt: Dt,
}

/// Context available to measure operators
pub struct MeasureContext<'a> {
    /// Access to signals (current tick values, post-resolve)
    pub signals: &'a SignalStorage,
    /// Field buffer for emission
    pub fields: &'a mut FieldBuffer,
    /// Time step
    pub dt: Dt,
}

/// Context available to impulse application
pub struct ImpulseContext<'a> {
    /// Access to signals (previous tick values)
    pub signals: &'a SignalStorage,
    /// Channel to write inputs
    pub channels: &'a mut InputChannels,
}

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
    /// Resolver functions indexed by resolver_idx
    resolvers: Vec<ResolverFn>,
    /// Collect operator functions
    collect_ops: Vec<CollectFn>,
    /// Measure operator functions
    measure_ops: Vec<MeasureFn>,
    /// Fracture functions
    fractures: Vec<FractureFn>,
    /// Impulse handler functions
    impulse_handlers: Vec<ImpulseFn>,
    /// Pending impulses to apply in next Collect phase (handler_idx, payload)
    pending_impulses: Vec<(usize, Value)>,
    /// Warmup functions indexed by signal
    warmup_fns: Vec<(SignalId, WarmupFn, WarmupConfig)>,
    /// Whether warmup has been executed
    warmup_complete: bool,
}

impl Runtime {
    /// Create a new runtime
    pub fn new(
        initial_era: EraId,
        eras: HashMap<EraId, EraConfig>,
        dags: DagSet,
    ) -> Self {
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
            resolvers: Vec::new(),
            collect_ops: Vec::new(),
            measure_ops: Vec::new(),
            fractures: Vec::new(),
            impulse_handlers: Vec::new(),
            pending_impulses: Vec::new(),
            warmup_fns: Vec::new(),
            warmup_complete: false,
        }
    }

    /// Register a resolver function, returns its index
    pub fn register_resolver(&mut self, resolver: ResolverFn) -> usize {
        let idx = self.resolvers.len();
        self.resolvers.push(resolver);
        idx
    }

    /// Register a collect operator, returns its index
    pub fn register_collect_op(&mut self, op: CollectFn) -> usize {
        let idx = self.collect_ops.len();
        self.collect_ops.push(op);
        idx
    }

    /// Register a fracture function, returns its index
    pub fn register_fracture(&mut self, fracture: FractureFn) -> usize {
        let idx = self.fractures.len();
        self.fractures.push(fracture);
        idx
    }

    /// Register a measure operator, returns its index
    pub fn register_measure_op(&mut self, op: MeasureFn) -> usize {
        let idx = self.measure_ops.len();
        self.measure_ops.push(op);
        idx
    }

    /// Register an impulse handler, returns its index
    pub fn register_impulse(&mut self, handler: ImpulseFn) -> usize {
        let idx = self.impulse_handlers.len();
        self.impulse_handlers.push(handler);
        idx
    }

    /// Inject an impulse to be applied in the next tick's Collect phase
    pub fn inject_impulse(&mut self, handler_idx: usize, payload: Value) {
        debug!(handler_idx, ?payload, "impulse injected");
        self.pending_impulses.push((handler_idx, payload));
    }

    /// Register a warmup function for a signal
    pub fn register_warmup(&mut self, signal: SignalId, warmup_fn: WarmupFn, config: WarmupConfig) {
        debug!(signal = %signal, max_iter = config.max_iterations, "warmup registered");
        self.warmup_fns.push((signal, warmup_fn, config));
    }

    /// Initialize a signal with a value
    pub fn init_signal(&mut self, id: SignalId, value: Value) {
        debug!(signal = %id, ?value, "signal initialized");
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
        self.warmup_complete
    }

    /// Get access to the field buffer (for observer consumption)
    pub fn field_buffer(&self) -> &FieldBuffer {
        &self.field_buffer
    }

    /// Drain the field buffer (for observer consumption)
    pub fn drain_fields(&mut self) -> indexmap::IndexMap<crate::types::FieldId, Vec<crate::storage::FieldSample>> {
        self.field_buffer.drain()
    }

    /// Execute warmup phase (pre-causal equilibration)
    ///
    /// Must be called before execute_tick. Runs all registered warmup
    /// functions until convergence or max iterations.
    #[instrument(skip(self), name = "warmup")]
    pub fn execute_warmup(&mut self) -> Result<WarmupResult> {
        if self.warmup_complete {
            return Ok(WarmupResult {
                iterations: 0,
                converged: true,
            });
        }

        if self.warmup_fns.is_empty() {
            info!("no warmup functions registered");
            self.warmup_complete = true;
            return Ok(WarmupResult {
                iterations: 0,
                converged: true,
            });
        }

        // Find the maximum iterations needed
        let max_iterations = self
            .warmup_fns
            .iter()
            .map(|(_, _, cfg)| cfg.max_iterations)
            .max()
            .unwrap_or(0);

        info!(signals = self.warmup_fns.len(), max_iterations, "warmup starting");

        let mut iteration = 0;
        let mut converged = false;

        while iteration < max_iterations {
            trace!(iteration, "warmup iteration");

            let mut all_converged = true;
            let mut max_delta: f64 = 0.0;

            // Execute each warmup function
            for (signal_id, warmup_fn, config) in &self.warmup_fns {
                // Skip if this signal's iterations are exhausted
                if iteration >= config.max_iterations {
                    continue;
                }

                let prev = self
                    .signals
                    .get(signal_id)
                    .ok_or_else(|| Error::SignalNotFound(signal_id.clone()))?;

                let ctx = WarmupContext {
                    prev,
                    signals: &self.signals,
                    iteration,
                };

                let new_value = warmup_fn(&ctx);

                // Check for numeric errors
                if let Value::Scalar(v) = &new_value {
                    if v.is_nan() {
                        error!(signal = %signal_id, iteration, "warmup NaN");
                        return Err(Error::WarmupDivergence {
                            signal: signal_id.clone(),
                            iteration,
                            message: "NaN result".to_string(),
                        });
                    }
                    if v.is_infinite() {
                        error!(signal = %signal_id, iteration, "warmup infinite");
                        return Err(Error::WarmupDivergence {
                            signal: signal_id.clone(),
                            iteration,
                            message: "Infinite result".to_string(),
                        });
                    }

                    // Check convergence
                    if let Some(epsilon) = config.convergence_epsilon {
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

                self.signals.set_current(signal_id.clone(), new_value);
            }

            // Advance iteration state
            self.signals.advance_tick();
            iteration += 1;

            // Check if all signals with convergence criteria have converged
            if all_converged
                && self
                    .warmup_fns
                    .iter()
                    .any(|(_, _, cfg)| cfg.convergence_epsilon.is_some())
            {
                debug!(iteration, max_delta, "warmup converged");
                converged = true;
                break;
            }
        }

        // Check for divergence (didn't converge within iterations)
        let any_requires_convergence = self
            .warmup_fns
            .iter()
            .any(|(_, _, cfg)| cfg.convergence_epsilon.is_some());

        if any_requires_convergence && !converged {
            let first_signal = &self.warmup_fns[0].0;
            error!(iterations = iteration, "warmup failed to converge");
            return Err(Error::WarmupDivergence {
                signal: first_signal.clone(),
                iteration,
                message: "Failed to converge within max iterations".to_string(),
            });
        }

        info!(iterations = iteration, converged, "warmup complete");
        self.warmup_complete = true;

        Ok(WarmupResult {
            iterations: iteration,
            converged,
        })
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
        self.execute_collect_phase(&strata_states, dt)?;

        // Phase 3: Resolve
        self.execute_resolve_phase(&strata_states, dt)?;

        // Phase 4: Fracture
        self.execute_fracture_phase(dt)?;

        // Phase 5: Measure
        self.execute_measure_phase(&strata_states, dt)?;

        // Post-tick: check era transitions
        self.check_era_transition()?;

        // Advance state
        self.signals.advance_tick();
        self.fracture_queue.drain_into(&mut self.input_channels);
        self.tick += 1;

        trace!("tick complete");
        Ok(ctx)
    }

    #[instrument(skip_all, name = "collect")]
    fn execute_collect_phase(
        &mut self,
        strata_states: &HashMap<StratumId, StratumState>,
        dt: Dt,
    ) -> Result<()> {
        // Apply pending impulses first
        let impulses = std::mem::take(&mut self.pending_impulses);
        for (handler_idx, payload) in impulses {
            let handler = &self.impulse_handlers[handler_idx];
            let mut ctx = ImpulseContext {
                signals: &self.signals,
                channels: &mut self.input_channels,
            };
            trace!(handler_idx, "applying impulse");
            handler(&mut ctx, &payload);
        }

        let era_dags = self.dags.get_era(&self.current_era).unwrap();

        for dag in era_dags.for_phase(Phase::Collect) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or(StratumState::Active);

            if !stratum_state.is_eligible(self.tick) {
                trace!(stratum = %dag.stratum, "stratum gated");
                continue;
            }

            trace!(stratum = %dag.stratum, nodes = dag.node_count(), "executing stratum");

            for level in &dag.levels {
                for node in &level.nodes {
                    if let NodeKind::OperatorCollect { operator_idx } = &node.kind {
                        let op = &self.collect_ops[*operator_idx];
                        let ctx = CollectContext {
                            signals: &self.signals,
                            channels: &mut self.input_channels,
                            dt,
                        };
                        op(&ctx);
                    }
                }
            }
        }
        Ok(())
    }

    #[instrument(skip_all, name = "resolve")]
    fn execute_resolve_phase(
        &mut self,
        strata_states: &HashMap<StratumId, StratumState>,
        dt: Dt,
    ) -> Result<()> {
        let era_dags = self.dags.get_era(&self.current_era).unwrap();

        for dag in era_dags.for_phase(Phase::Resolve) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or(StratumState::Active);

            if !stratum_state.is_eligible(self.tick) {
                trace!(stratum = %dag.stratum, "stratum gated");
                continue;
            }

            trace!(stratum = %dag.stratum, levels = dag.levels.len(), "resolving stratum");

            for level in &dag.levels {
                // Collect signal IDs and drain inputs BEFORE parallel iteration
                let signal_inputs: Vec<(SignalId, usize, f64)> = level
                    .nodes
                    .iter()
                    .filter_map(|node| {
                        if let NodeKind::SignalResolve {
                            signal,
                            resolver_idx,
                        } = &node.kind
                        {
                            let inputs = self.input_channels.drain_sum(signal);
                            Some((signal.clone(), *resolver_idx, inputs))
                        } else {
                            None
                        }
                    })
                    .collect();

                // Parallelize resolution
                let results: Vec<Result<(SignalId, Value)>> = signal_inputs
                    .par_iter()
                    .map(|(signal, resolver_idx, inputs)| {
                        let prev = self
                            .signals
                            .get_prev(signal)
                            .ok_or_else(|| Error::SignalNotFound(signal.clone()))?;
                        let resolver = &self.resolvers[*resolver_idx];
                        let ctx = ResolveContext {
                            prev,
                            signals: &self.signals,
                            inputs: *inputs,
                            dt,
                        };
                        let value = resolver(&ctx);
                        Ok((signal.clone(), value))
                    })
                    .collect();

                // Apply results sequentially for determinism
                for result in results {
                    let (signal, value) = result?;
                    if let Value::Scalar(v) = &value {
                        if v.is_nan() {
                            error!(signal = %signal, "NaN result");
                            return Err(Error::NumericError {
                                signal: signal.clone(),
                                message: "NaN result".to_string(),
                            });
                        }
                        if v.is_infinite() {
                            error!(signal = %signal, "infinite result");
                            return Err(Error::NumericError {
                                signal: signal.clone(),
                                message: "Infinite result".to_string(),
                            });
                        }
                    }
                    trace!(signal = %signal, ?value, "signal resolved");
                    self.signals.set_current(signal, value);
                }
            }
        }
        Ok(())
    }

    #[instrument(skip_all, name = "fracture")]
    fn execute_fracture_phase(&mut self, dt: Dt) -> Result<()> {
        let era_dags = self.dags.get_era(&self.current_era).unwrap();

        for dag in era_dags.for_phase(Phase::Fracture) {
            for level in &dag.levels {
                for node in &level.nodes {
                    if let NodeKind::Fracture { fracture_idx } = &node.kind {
                        let fracture = &self.fractures[*fracture_idx];
                        let ctx = FractureContext {
                            signals: &self.signals,
                            dt,
                        };
                        if let Some(outputs) = fracture(&ctx) {
                            debug!(outputs = outputs.len(), "fracture emitted");
                            for (signal, value) in outputs {
                                self.fracture_queue.queue(signal, value);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[instrument(skip_all, name = "measure")]
    fn execute_measure_phase(
        &mut self,
        strata_states: &HashMap<StratumId, StratumState>,
        dt: Dt,
    ) -> Result<()> {
        let era_dags = self.dags.get_era(&self.current_era).unwrap();

        for dag in era_dags.for_phase(Phase::Measure) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or(StratumState::Active);

            if !stratum_state.is_eligible(self.tick) {
                trace!(stratum = %dag.stratum, "stratum gated");
                continue;
            }

            trace!(stratum = %dag.stratum, nodes = dag.node_count(), "measuring stratum");

            for level in &dag.levels {
                for node in &level.nodes {
                    match &node.kind {
                        NodeKind::OperatorMeasure { operator_idx } => {
                            let op = &self.measure_ops[*operator_idx];
                            let mut ctx = MeasureContext {
                                signals: &self.signals,
                                fields: &mut self.field_buffer,
                                dt,
                            };
                            op(&mut ctx);
                        }
                        NodeKind::FieldEmit { field_idx: _ } => {
                            // FieldEmit nodes are placeholders for dependency tracking
                            // The actual emission happens via MeasureContext.fields.emit()
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
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
    use crate::dag::{DagBuilder, DagNode, EraDags, NodeId};
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
        use crate::types::FieldId;

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
}
