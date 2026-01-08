//! Tick executor
//!
//! Orchestrates DAG execution through phases.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::dag::{DagSet, NodeKind};
use crate::error::{Error, Result};
use crate::storage::{FractureQueue, InputChannels, SignalStorage};
use crate::types::{Dt, EraId, Phase, SignalId, StratumId, StratumState, TickContext, Value};

/// Function that resolves a signal value
pub type ResolverFn = Box<dyn Fn(&ResolveContext) -> Value + Send + Sync>;

/// Function that executes a collect operator
pub type CollectFn = Box<dyn Fn(&CollectContext) + Send + Sync>;

/// Function that evaluates a fracture condition and emits
pub type FractureFn = Box<dyn Fn(&FractureContext) -> Option<Vec<(SignalId, f64)>> + Send + Sync>;

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

/// Era configuration
pub struct EraConfig {
    /// Time step for this era
    pub dt: Dt,
    /// Stratum states in this era
    pub strata: HashMap<StratumId, StratumState>,
    /// Transition condition (returns Some(next_era) if should transition)
    pub transition: Option<Box<dyn Fn(&SignalStorage) -> Option<EraId> + Send + Sync>>,
}

/// Runtime state for a simulation
pub struct Runtime {
    /// Signal storage
    signals: SignalStorage,
    /// Input channels for Collect phase
    input_channels: InputChannels,
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
    /// Fracture functions
    fractures: Vec<FractureFn>,
}

impl Runtime {
    /// Create a new runtime
    pub fn new(
        initial_era: EraId,
        eras: HashMap<EraId, EraConfig>,
        dags: DagSet,
    ) -> Self {
        Self {
            signals: SignalStorage::new(),
            input_channels: InputChannels::new(),
            fracture_queue: FractureQueue::new(),
            tick: 0,
            current_era: initial_era,
            eras,
            dags,
            resolvers: Vec::new(),
            collect_ops: Vec::new(),
            fractures: Vec::new(),
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

    /// Initialize a signal with a value
    pub fn init_signal(&mut self, id: SignalId, value: Value) {
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

    /// Execute a single tick
    pub fn execute_tick(&mut self) -> Result<TickContext> {
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
            return Err(Error::EraNotFound(self.current_era.clone()));
        }

        // Phase 1: Configure (internal - prepare context)
        // Already done above

        // Phase 2: Collect
        self.execute_collect_phase(&strata_states, dt)?;

        // Phase 3: Resolve
        self.execute_resolve_phase(&strata_states, dt)?;

        // Phase 4: Fracture
        self.execute_fracture_phase(dt)?;

        // Phase 5: Measure (TODO: fields and chronicles)

        // Post-tick: check era transitions
        self.check_era_transition()?;

        // Advance state
        self.signals.advance_tick();
        self.fracture_queue.drain_into(&mut self.input_channels);
        self.tick += 1;

        Ok(ctx)
    }

    fn execute_collect_phase(
        &mut self,
        strata_states: &HashMap<StratumId, StratumState>,
        dt: Dt,
    ) -> Result<()> {
        let era_dags = self.dags.get_era(&self.current_era).unwrap();

        for dag in era_dags.for_phase(Phase::Collect) {
            // Check if stratum is eligible this tick
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or(StratumState::Active);

            if !stratum_state.is_eligible(self.tick) {
                continue;
            }

            // Execute levels sequentially, nodes within level could parallelize
            // but collect ops mutate input_channels, so we serialize for now
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
                continue;
            }

            // Execute levels sequentially
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

                // Now parallelize the actual resolution
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

                // Apply results (must be sequential to maintain determinism)
                for result in results {
                    let (signal, value) = result?;
                    // Validate value
                    if let Value::Scalar(v) = &value {
                        if v.is_nan() {
                            return Err(Error::NumericError {
                                signal: signal.clone(),
                                message: "NaN result".to_string(),
                            });
                        }
                        if v.is_infinite() {
                            return Err(Error::NumericError {
                                signal: signal.clone(),
                                message: "Infinite result".to_string(),
                            });
                        }
                    }
                    self.signals.set_current(signal, value);
                }
            }
        }
        Ok(())
    }

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

    fn check_era_transition(&mut self) -> Result<()> {
        let era_config = self.eras.get(&self.current_era).unwrap();
        if let Some(ref transition) = era_config.transition {
            if let Some(next_era) = transition(&self.signals) {
                if !self.eras.contains_key(&next_era) {
                    return Err(Error::EraNotFound(next_era));
                }
                self.current_era = next_era;
            }
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

        let mut era_dags = EraDags::new();
        era_dags.insert(dag);

        let mut dags = DagSet::new();
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

        let mut era_dags = EraDags::new();
        era_dags.insert(dag);

        let mut dags = DagSet::new();
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
}
