//! Phase execution
//!
//! Executes individual simulation phases: Collect, Resolve, Fracture, Measure.

use indexmap::IndexMap;

use rayon::prelude::*;
use tracing::{debug, error, instrument, trace};

use crate::dag::{DagSet, NodeKind};
use crate::error::{Error, Result};
use crate::storage::{EventBuffer, FieldBuffer, FractureQueue, InputChannels, SignalStorage};
use crate::types::{Dt, EraId, Phase, SignalId, StratumId, StratumState, Value};

use super::assertions::AssertionChecker;
use super::context::{
    ChronicleContext, CollectContext, FractureContext, ImpulseContext, MeasureContext,
    ResolveContext,
};

/// Function that resolves a signal value
pub type ResolverFn = Box<dyn Fn(&ResolveContext) -> Value + Send + Sync>;

/// Function that executes a collect operator
pub type CollectFn = Box<dyn Fn(&CollectContext) + Send + Sync>;

/// Function that evaluates a fracture condition and emits
pub type FractureFn = Box<dyn Fn(&FractureContext) -> Option<Vec<(SignalId, f64)>> + Send + Sync>;

/// Function that executes a measure-phase operator
pub type MeasureFn = Box<dyn Fn(&mut MeasureContext) + Send + Sync>;

/// Function that applies an impulse with a typed payload
pub type ImpulseFn = Box<dyn Fn(&mut ImpulseContext, &Value) + Send + Sync>;

/// An event emitted by a chronicle handler.
///
/// Chronicles observe simulation state and emit events for logging and analytics.
/// These events are non-causal and do not affect simulation outcomes.
#[derive(Debug, Clone)]
pub struct EmittedEvent {
    /// The event name (e.g., "climate.alert")
    pub name: String,
    /// Event fields as key-value pairs
    pub fields: Vec<(String, Value)>,
}

/// Function that evaluates chronicle handlers and emits events.
///
/// Returns a vector of events to emit. Empty vector means no events triggered.
pub type ChronicleFn = Box<dyn Fn(&ChronicleContext) -> Vec<EmittedEvent> + Send + Sync>;

/// Configuration for Measure phase parallelism
#[derive(Debug, Clone)]
pub struct MeasureParallelConfig {
    /// Minimum number of operators to trigger parallel execution.
    /// Below this threshold, operators execute sequentially to avoid overhead.
    pub parallel_threshold: usize,
}

impl Default for MeasureParallelConfig {
    fn default() -> Self {
        Self {
            // Default: parallelize when there are 2+ operators
            parallel_threshold: 2,
        }
    }
}

/// Configuration for Fracture phase parallelism
#[derive(Debug, Clone)]
pub struct FractureParallelConfig {
    /// Minimum number of fractures to trigger parallel evaluation.
    /// Below this threshold, fractures evaluate sequentially to avoid overhead.
    pub parallel_threshold: usize,
}

impl Default for FractureParallelConfig {
    fn default() -> Self {
        Self {
            // Default: parallelize when there are 2+ fractures
            parallel_threshold: 2,
        }
    }
}

/// Phase executor handles individual phase execution
pub struct PhaseExecutor {
    /// Resolver functions indexed by resolver_idx
    pub resolvers: Vec<ResolverFn>,
    /// Collect operator functions
    pub collect_ops: Vec<CollectFn>,
    /// Measure operator functions
    pub measure_ops: Vec<MeasureFn>,
    /// Fracture functions
    pub fractures: Vec<FractureFn>,
    /// Impulse handler functions
    pub impulse_handlers: Vec<ImpulseFn>,
    /// Chronicle handler functions
    pub chronicle_handlers: Vec<ChronicleFn>,
    /// Configuration for Measure phase parallelism
    pub measure_config: MeasureParallelConfig,
    /// Configuration for Fracture phase parallelism
    pub fracture_config: FractureParallelConfig,
}

impl Default for PhaseExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseExecutor {
    /// Create a new phase executor with default configuration
    pub fn new() -> Self {
        Self::with_configs(
            MeasureParallelConfig::default(),
            FractureParallelConfig::default(),
        )
    }

    /// Create a new phase executor with custom Measure phase configuration
    pub fn with_measure_config(measure_config: MeasureParallelConfig) -> Self {
        Self::with_configs(measure_config, FractureParallelConfig::default())
    }

    /// Create a new phase executor with custom Fracture phase configuration
    pub fn with_fracture_config(fracture_config: FractureParallelConfig) -> Self {
        Self::with_configs(MeasureParallelConfig::default(), fracture_config)
    }

    /// Create a new phase executor with custom configurations for both phases
    pub fn with_configs(
        measure_config: MeasureParallelConfig,
        fracture_config: FractureParallelConfig,
    ) -> Self {
        Self {
            resolvers: Vec::new(),
            collect_ops: Vec::new(),
            measure_ops: Vec::new(),
            fractures: Vec::new(),
            impulse_handlers: Vec::new(),
            chronicle_handlers: Vec::new(),
            measure_config,
            fracture_config,
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

    /// Register a chronicle handler, returns its index
    pub fn register_chronicle(&mut self, handler: ChronicleFn) -> usize {
        let idx = self.chronicle_handlers.len();
        self.chronicle_handlers.push(handler);
        idx
    }

    /// Execute the Collect phase
    #[instrument(skip_all, name = "collect")]
    pub fn execute_collect(
        &self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        strata_states: &IndexMap<StratumId, StratumState>,
        dags: &DagSet,
        signals: &SignalStorage,
        input_channels: &mut InputChannels,
        pending_impulses: &mut Vec<(usize, Value)>,
    ) -> Result<()> {
        // Apply pending impulses first
        let impulses = std::mem::take(pending_impulses);
        for (handler_idx, payload) in impulses {
            let handler = &self.impulse_handlers[handler_idx];
            let mut ctx = ImpulseContext {
                signals,
                channels: input_channels,
            };
            trace!(handler_idx, "applying impulse");
            handler(&mut ctx, &payload);
        }

        let era_dags = dags.get_era(era).unwrap();

        for dag in era_dags.for_phase(Phase::Collect) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or_else(|| panic!("stratum {:?} not found in strata_states", dag.stratum));

            if !stratum_state.is_eligible(tick) {
                trace!(stratum = %dag.stratum, "stratum gated");
                continue;
            }

            trace!(stratum = %dag.stratum, nodes = dag.node_count(), "executing stratum");

            for level in &dag.levels {
                for node in &level.nodes {
                    if let NodeKind::OperatorCollect { operator_idx } = &node.kind {
                        let op = &self.collect_ops[*operator_idx];
                        let ctx = CollectContext {
                            signals,
                            channels: input_channels,
                            dt,
                        };
                        op(&ctx);
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute the Resolve phase
    #[instrument(skip_all, name = "resolve")]
    pub fn execute_resolve(
        &self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        strata_states: &IndexMap<StratumId, StratumState>,
        dags: &DagSet,
        signals: &mut SignalStorage,
        input_channels: &mut InputChannels,
        assertion_checker: &AssertionChecker,
    ) -> Result<()> {
        let era_dags = dags.get_era(era).unwrap();

        for dag in era_dags.for_phase(Phase::Resolve) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or_else(|| panic!("stratum {:?} not found in strata_states", dag.stratum));

            if !stratum_state.is_eligible(tick) {
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
                            let inputs = input_channels.drain_sum(signal);
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
                        let prev = signals
                            .get_prev(signal)
                            .ok_or_else(|| Error::SignalNotFound(signal.clone()))?;
                        let resolver = &self.resolvers[*resolver_idx];
                        let ctx = ResolveContext {
                            prev,
                            signals,
                            inputs: *inputs,
                            dt,
                        };
                        let value = resolver(&ctx);
                        Ok((signal.clone(), value))
                    })
                    .collect();

                // Apply results sequentially for determinism and check assertions
                for result in results {
                    let (signal, value) = result?;

                    // Check for numeric errors
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

                    // Check assertions before committing the value
                    let prev = signals.get_prev(&signal).unwrap_or(&value);
                    assertion_checker.check_signal(&signal, &value, prev, signals, dt)?;

                    trace!(signal = %signal, ?value, "signal resolved");
                    signals.set_current(signal, value);
                }
            }
        }
        Ok(())
    }

    /// Execute the Fracture phase
    ///
    /// Fracture condition evaluation is read-only, so we can parallelize it.
    /// Results are sorted by fracture index before emitting to maintain
    /// deterministic ordering.
    #[instrument(skip_all, name = "fracture")]
    pub fn execute_fracture(
        &self,
        era: &EraId,
        dt: Dt,
        dags: &DagSet,
        signals: &SignalStorage,
        fracture_queue: &mut FractureQueue,
    ) -> Result<()> {
        let era_dags = dags.get_era(era).unwrap();

        // Collect all fracture indices across all DAGs and levels
        let all_fracture_indices: Vec<usize> = era_dags
            .for_phase(Phase::Fracture)
            .flat_map(|dag| {
                dag.levels.iter().flat_map(|level| {
                    level.nodes.iter().filter_map(|node| match &node.kind {
                        NodeKind::Fracture { fracture_idx } => Some(*fracture_idx),
                        _ => None,
                    })
                })
            })
            .collect();

        let total_fractures = all_fracture_indices.len();

        if total_fractures == 0 {
            return Ok(());
        }

        trace!(
            fractures = total_fractures,
            threshold = self.fracture_config.parallel_threshold,
            "fracture evaluation"
        );

        if total_fractures < self.fracture_config.parallel_threshold {
            // Below threshold - evaluate sequentially to avoid parallelism overhead
            for &fracture_idx in &all_fracture_indices {
                let fracture = &self.fractures[fracture_idx];
                let ctx = FractureContext { signals, dt };
                if let Some(outputs) = fracture(&ctx) {
                    debug!(fracture_idx, outputs = outputs.len(), "fracture emitted");
                    for (signal, value) in outputs {
                        fracture_queue.queue(signal, value);
                    }
                }
            }
        } else {
            // At or above threshold - evaluate all in parallel
            // Results include fracture index for deterministic ordering
            let mut triggered: Vec<(usize, Vec<(SignalId, f64)>)> = all_fracture_indices
                .par_iter()
                .filter_map(|&fracture_idx| {
                    let fracture = &self.fractures[fracture_idx];
                    let ctx = FractureContext { signals, dt };
                    fracture(&ctx).map(|outputs| (fracture_idx, outputs))
                })
                .collect();

            // Sort by fracture index for deterministic emit ordering
            triggered.sort_by_key(|(idx, _)| *idx);

            // Sequential emit to maintain determinism
            for (fracture_idx, outputs) in triggered {
                debug!(fracture_idx, outputs = outputs.len(), "fracture emitted");
                for (signal, value) in outputs {
                    fracture_queue.queue(signal, value);
                }
            }
        }

        Ok(())
    }

    /// Execute the Measure phase
    ///
    /// Since Measure is non-causal (observers only read signals), we can use
    /// aggressive parallelism:
    /// - All strata execute in parallel
    /// - All levels within a stratum execute in parallel (no barriers)
    /// - All operators within a level execute in parallel
    ///
    /// Each parallel task writes to its own local buffer, which are merged
    /// after all execution completes.
    #[instrument(skip_all, name = "measure")]
    pub fn execute_measure(
        &self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        strata_states: &IndexMap<StratumId, StratumState>,
        dags: &DagSet,
        signals: &SignalStorage,
        field_buffer: &mut FieldBuffer,
    ) -> Result<()> {
        let era_dags = dags.get_era(era).unwrap();

        // Collect all eligible DAGs for parallel execution
        let eligible_dags: Vec<_> = era_dags
            .for_phase(Phase::Measure)
            .filter(|dag| {
                let stratum_state = strata_states.get(&dag.stratum).copied().unwrap_or_else(|| {
                    panic!("stratum {:?} not found in strata_states", dag.stratum)
                });
                stratum_state.is_eligible(tick)
            })
            .collect();

        if eligible_dags.is_empty() {
            return Ok(());
        }

        // Collect all measure operators across all strata and levels
        let all_operator_indices: Vec<usize> = eligible_dags
            .iter()
            .flat_map(|dag| {
                dag.levels.iter().flat_map(|level| {
                    level.nodes.iter().filter_map(|node| match &node.kind {
                        NodeKind::OperatorMeasure { operator_idx } => Some(*operator_idx),
                        _ => None,
                    })
                })
            })
            .collect();

        let total_operators = all_operator_indices.len();

        if total_operators == 0 {
            return Ok(());
        }

        trace!(
            strata = eligible_dags.len(),
            operators = total_operators,
            threshold = self.measure_config.parallel_threshold,
            "measure execution"
        );

        if total_operators < self.measure_config.parallel_threshold {
            // Below threshold - execute sequentially to avoid parallelism overhead
            for &operator_idx in &all_operator_indices {
                let op = &self.measure_ops[operator_idx];
                let mut ctx = MeasureContext {
                    signals,
                    fields: field_buffer,
                    dt,
                };
                op(&mut ctx);
            }
        } else {
            // At or above threshold - execute all in parallel
            let local_buffers: Vec<FieldBuffer> = all_operator_indices
                .par_iter()
                .map(|&operator_idx| {
                    let mut local_buffer = FieldBuffer::default();
                    let op = &self.measure_ops[operator_idx];
                    let mut ctx = MeasureContext {
                        signals,
                        fields: &mut local_buffer,
                        dt,
                    };
                    op(&mut ctx);
                    local_buffer
                })
                .collect();

            // Merge all local buffers into the main buffer
            for buffer in local_buffers {
                field_buffer.merge(buffer);
            }
        }

        Ok(())
    }

    /// Execute chronicles in the Measure phase.
    ///
    /// Chronicles observe resolved signals and conditionally emit events.
    /// They are non-causal: removing all chronicles must not change simulation results.
    ///
    /// Chronicle handlers can run in parallel since they only read signals.
    /// Results are sorted by chronicle index before emitting to maintain
    /// deterministic event ordering.
    #[instrument(skip_all, name = "chronicles")]
    pub fn execute_chronicles(
        &self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        strata_states: &IndexMap<StratumId, StratumState>,
        dags: &DagSet,
        signals: &SignalStorage,
        event_buffer: &mut EventBuffer,
    ) -> Result<()> {
        let era_dags = dags.get_era(era).unwrap();

        // Collect all chronicle indices across all Measure DAGs
        let all_chronicle_indices: Vec<usize> = era_dags
            .for_phase(Phase::Measure)
            .filter(|dag| {
                let stratum_state = strata_states.get(&dag.stratum).copied().unwrap_or_else(|| {
                    panic!("stratum {:?} not found in strata_states", dag.stratum)
                });
                stratum_state.is_eligible(tick)
            })
            .flat_map(|dag| {
                dag.levels.iter().flat_map(|level| {
                    level.nodes.iter().filter_map(|node| match &node.kind {
                        NodeKind::ChronicleObserve { chronicle_idx } => Some(*chronicle_idx),
                        _ => None,
                    })
                })
            })
            .collect();

        let total_chronicles = all_chronicle_indices.len();

        if total_chronicles == 0 {
            return Ok(());
        }

        trace!(chronicles = total_chronicles, "chronicle execution");

        // Execute chronicles and collect events
        // Using parallel execution since chronicles are read-only
        let mut all_events: Vec<(usize, Vec<EmittedEvent>)> = all_chronicle_indices
            .par_iter()
            .filter_map(|&chronicle_idx| {
                let handler = &self.chronicle_handlers[chronicle_idx];
                let ctx = ChronicleContext { signals, dt };
                let events = handler(&ctx);
                if events.is_empty() {
                    None
                } else {
                    Some((chronicle_idx, events))
                }
            })
            .collect();

        // Sort by chronicle index for deterministic event ordering
        all_events.sort_by_key(|(idx, _)| *idx);

        // Sequential emit to maintain determinism
        for (chronicle_idx, events) in all_events {
            for event in events {
                debug!(chronicle_idx, event_name = %event.name, "chronicle event emitted");
                event_buffer.emit(event.name, event.fields);
            }
        }

        Ok(())
    }
}
