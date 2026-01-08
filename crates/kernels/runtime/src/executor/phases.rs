//! Phase execution
//!
//! Executes individual simulation phases: Collect, Resolve, Fracture, Measure.

use std::collections::HashMap;

use rayon::prelude::*;
use tracing::{debug, error, instrument, trace};

use crate::dag::{DagSet, NodeKind};
use crate::error::{Error, Result};
use crate::storage::{FieldBuffer, FractureQueue, InputChannels, SignalStorage};
use crate::types::{Dt, EraId, Phase, SignalId, StratumId, StratumState, Value};

use super::assertions::AssertionChecker;
use super::context::{CollectContext, FractureContext, ImpulseContext, MeasureContext, ResolveContext};

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
}

impl Default for PhaseExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseExecutor {
    /// Create a new phase executor
    pub fn new() -> Self {
        Self {
            resolvers: Vec::new(),
            collect_ops: Vec::new(),
            measure_ops: Vec::new(),
            fractures: Vec::new(),
            impulse_handlers: Vec::new(),
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

    /// Execute the Collect phase
    #[instrument(skip_all, name = "collect")]
    pub fn execute_collect(
        &self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        strata_states: &HashMap<StratumId, StratumState>,
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
                .unwrap_or(StratumState::Active);

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
        strata_states: &HashMap<StratumId, StratumState>,
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
                .unwrap_or(StratumState::Active);

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

        for dag in era_dags.for_phase(Phase::Fracture) {
            for level in &dag.levels {
                for node in &level.nodes {
                    if let NodeKind::Fracture { fracture_idx } = &node.kind {
                        let fracture = &self.fractures[*fracture_idx];
                        let ctx = FractureContext { signals, dt };
                        if let Some(outputs) = fracture(&ctx) {
                            debug!(outputs = outputs.len(), "fracture emitted");
                            for (signal, value) in outputs {
                                fracture_queue.queue(signal, value);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute the Measure phase
    #[instrument(skip_all, name = "measure")]
    pub fn execute_measure(
        &self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        strata_states: &HashMap<StratumId, StratumState>,
        dags: &DagSet,
        signals: &SignalStorage,
        field_buffer: &mut FieldBuffer,
    ) -> Result<()> {
        let era_dags = dags.get_era(era).unwrap();

        for dag in era_dags.for_phase(Phase::Measure) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or(StratumState::Active);

            if !stratum_state.is_eligible(tick) {
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
                                signals,
                                fields: field_buffer,
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
}
