//! Tick executor
//!
//! Orchestrates DAG execution through phases.

mod assertions;
mod context;
pub mod cost_model;
pub mod kernel_registry;
pub mod l1_kernels;
pub mod l3_kernel;
pub mod lane_kernel;
pub mod lowering_strategy;
pub mod member_executor;
mod phases;
mod warmup;

use indexmap::IndexMap;

use tracing::{error, info, instrument, trace};

use crate::dag::DagSet;
use crate::error::{Error, Result};
use crate::soa_storage::{MemberSignalBuffer, ValueType as MemberValueType};
use crate::storage::{
    EntityInstances, EntityStorage, FieldBuffer, FieldSample, FractureQueue, InputChannels,
    SignalStorage,
};
use crate::types::{
    Dt, EntityId, EraId, FieldId, SignalId, StratumId, StratumState, TickContext, Value,
    WarmupConfig, WarmupResult,
};

// Re-export public types
pub use assertions::{AssertionChecker, AssertionFn, AssertionSeverity, SignalAssertion};
pub use context::{
    AssertContext, CollectContext, FractureContext, ImpulseContext, MeasureContext, ResolveContext,
    WarmupContext,
};
pub use kernel_registry::LaneKernelRegistry;
pub use l1_kernels::{ScalarKernelFn, ScalarL1Kernel, Vec3KernelFn, Vec3L1Kernel};
pub use l3_kernel::{
    L3Kernel, L3KernelBuilder, L3MemberResolver, MemberDag, MemberDagError, MemberEdge,
    ScalarL3MemberResolver, ScalarL3ResolveContext, ScalarL3ResolverFn, Vec3L3MemberResolver,
    Vec3L3ResolveContext, Vec3L3ResolverFn,
};
pub use lane_kernel::{LaneKernel, LaneKernelError, LaneKernelResult};
pub use lowering_strategy::{LoweringHeuristics, LoweringStrategy};
pub use member_executor::{
    ChunkConfig, MemberResolveContext, MemberSignalResolver, ScalarL1Resolver,
    ScalarResolveContext, ScalarResolverFn, Vec3L1Resolver, Vec3ResolveContext, Vec3ResolverFn,
};
pub use phases::{
    CollectFn, FractureFn, FractureParallelConfig, ImpulseFn, MeasureFn, MeasureParallelConfig,
    PhaseExecutor, ResolverFn,
};
pub use warmup::{RegisteredWarmup, WarmupExecutor, WarmupFn};

/// Function that evaluates era transition conditions
pub type TransitionFn = Box<dyn Fn(&SignalStorage, f64) -> Option<EraId> + Send + Sync>;

/// Function that evaluates aggregate expressions over member signals.
///
/// These resolvers run after member signal resolution (Phase 3c) and can access
/// both global signals and member signal data for aggregate computations like
/// `sum(entity.particle, self.mass)`.
pub type AggregateResolverFn =
    Box<dyn Fn(&SignalStorage, &MemberSignalBuffer, Dt, f64) -> Value + Send + Sync>;

/// Era configuration
pub struct EraConfig {
    /// Time step for this era
    pub dt: Dt,
    /// Stratum states in this era (IndexMap for deterministic iteration order)
    pub strata: IndexMap<StratumId, StratumState>,
    /// Transition condition (returns Some(next_era) if should transition)
    pub transition: Option<TransitionFn>,
}

/// Runtime state for a simulation
pub struct Runtime {
    /// Signal storage
    signals: SignalStorage,
    /// Entity storage for per-instance state
    entities: EntityStorage,
    /// Member signal storage (SoA for vectorized execution)
    member_signals: MemberSignalBuffer,
    /// Scalar member resolver functions indexed by signal name
    /// Uses ScalarResolverFn which captures constants/config at build time
    member_resolvers: IndexMap<String, ScalarResolverFn>,
    /// Vec3 member resolver functions indexed by signal name
    /// Uses Vec3ResolverFn which captures constants/config at build time
    vec3_member_resolvers: IndexMap<String, Vec3ResolverFn>,
    /// Aggregate resolvers for signals that depend on member signal data
    /// Maps signal ID to (resolver_fn) - runs after member signal resolution
    aggregate_resolvers: IndexMap<SignalId, AggregateResolverFn>,
    /// Input channels for Collect phase
    input_channels: InputChannels,
    /// Field buffer for Measure phase
    field_buffer: FieldBuffer,
    /// Fracture outputs queued for next tick
    fracture_queue: FractureQueue,
    /// Current tick number
    tick: u64,
    /// Accumulated simulation time in seconds
    sim_time: f64,
    /// Current era
    current_era: EraId,
    /// Era configurations (IndexMap for deterministic iteration order)
    eras: IndexMap<EraId, EraConfig>,
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
    pub fn new(initial_era: EraId, eras: IndexMap<EraId, EraConfig>, dags: DagSet) -> Self {
        info!(era = %initial_era, "runtime created");
        Self {
            signals: SignalStorage::default(),
            entities: EntityStorage::default(),
            member_signals: MemberSignalBuffer::new(),
            member_resolvers: IndexMap::new(),
            vec3_member_resolvers: IndexMap::new(),
            aggregate_resolvers: IndexMap::new(),
            input_channels: InputChannels::default(),
            field_buffer: FieldBuffer::default(),
            fracture_queue: FractureQueue::default(),
            tick: 0,
            sim_time: 0.0,
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

    /// Initialize an entity type with its instances
    pub fn init_entity(&mut self, id: EntityId, instances: EntityInstances) {
        let count = instances.count();
        tracing::debug!(entity = %id, count, "entity initialized");
        self.entities.init_entity(id, instances);
    }

    /// Register a member signal type
    ///
    /// Must be called before `init_member_instances` for all member signals.
    pub fn register_member_signal(&mut self, signal_name: &str, value_type: MemberValueType) {
        tracing::debug!(
            signal = signal_name,
            ?value_type,
            "member signal registered"
        );
        self.member_signals
            .register_signal(signal_name.to_string(), value_type);
    }

    /// Initialize storage for all registered member signals
    ///
    /// Must be called after all member signals are registered with `register_member_signal`.
    pub fn init_member_instances(&mut self, instance_count: usize) {
        tracing::debug!(count = instance_count, "member instances initialized");
        self.member_signals.init_instances(instance_count);
    }

    /// Register the instance count for a specific entity.
    ///
    /// Call this after `init_member_instances` to track per-entity instance counts
    /// for aggregate operations. Aggregates will use this count instead of the
    /// global instance count when iterating over entity members.
    pub fn register_entity_count(&mut self, entity_id: &str, count: usize) {
        tracing::debug!(
            entity = entity_id,
            count,
            "entity instance count registered"
        );
        self.member_signals.register_entity_count(entity_id, count);
    }

    /// Register a scalar member resolver function
    ///
    /// The resolver should be built using `build_member_resolver` from the IR,
    /// which captures constants and config at build time.
    pub fn register_member_resolver(&mut self, signal_name: String, resolver: ScalarResolverFn) {
        tracing::debug!(signal = %signal_name, "scalar member resolver registered");
        self.member_resolvers.insert(signal_name, resolver);
    }

    /// Register a Vec3 member resolver function
    ///
    /// The resolver should be built using `build_vec3_member_resolver` from the IR,
    /// which captures constants and config at build time.
    pub fn register_vec3_member_resolver(&mut self, signal_name: String, resolver: Vec3ResolverFn) {
        tracing::debug!(signal = %signal_name, "vec3 member resolver registered");
        self.vec3_member_resolvers.insert(signal_name, resolver);
    }

    /// Register an aggregate resolver for a signal that depends on member signal data.
    ///
    /// These resolvers run after member signal resolution (Phase 3c) and can compute
    /// aggregates like `sum(entity.particle, self.mass)`.
    pub fn register_aggregate_resolver(
        &mut self,
        signal_id: SignalId,
        resolver: AggregateResolverFn,
    ) {
        tracing::debug!(signal = %signal_id, "aggregate resolver registered");
        self.aggregate_resolvers.insert(signal_id, resolver);
    }

    /// Get a member signal value for a specific instance
    pub fn get_member_signal(&self, signal_name: &str, instance_idx: usize) -> Option<Value> {
        self.member_signals.get_current(signal_name, instance_idx)
    }

    /// Set a member signal value for a specific instance.
    ///
    /// Used for initializing member signals with non-zero values before execution starts.
    pub fn set_member_signal(&mut self, signal_name: &str, instance_idx: usize, value: Value) {
        self.member_signals
            .set_current(signal_name, instance_idx, value);
    }

    /// Commit member initial values by advancing the buffer.
    ///
    /// After setting initial values with `set_member_signal`, call this to make
    /// those values available as "previous" values for resolvers that read `prev`.
    pub fn commit_member_initials(&mut self) {
        self.member_signals.advance_tick();
    }

    /// Get access to member signal buffer
    pub fn member_signals(&self) -> &MemberSignalBuffer {
        &self.member_signals
    }

    /// Get the number of instances for an entity type
    pub fn entity_count(&self, id: &EntityId) -> usize {
        self.entities.count(id)
    }

    /// Get access to entity storage (for aggregate computations)
    pub fn entities(&self) -> &EntityStorage {
        &self.entities
    }

    /// Get mutable access to entity storage
    pub fn entities_mut(&mut self) -> &mut EntityStorage {
        &mut self.entities
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

    /// Get current tick context (tick, dt, era)
    ///
    /// # Panics
    ///
    /// Panics if the current era is not found in the era configurations.
    /// This indicates a configuration bug since the runtime should not be
    /// in an era that was never registered.
    pub fn tick_context(&self) -> TickContext {
        let dt = self
            .eras
            .get(&self.current_era)
            .map(|c| c.dt)
            .unwrap_or_else(|| {
                panic!(
                    "Era '{}' not found in runtime configuration - cannot get tick context for unregistered era",
                    self.current_era
                )
            });
        TickContext {
            tick: self.tick,
            dt,
            era: self.current_era.clone(),
        }
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
        self.warmup_executor
            .execute(&mut self.signals, self.sim_time)
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
            self.sim_time,
            &strata_states,
            &self.dags,
            &self.signals,
            &mut self.input_channels,
            &mut self.pending_impulses,
        )?;

        // Phase 3: Resolve global signals
        self.phase_executor.execute_resolve(
            &self.current_era,
            self.tick,
            dt,
            self.sim_time,
            &strata_states,
            &self.dags,
            &mut self.signals,
            &mut self.input_channels,
            &self.assertion_checker,
        )?;

        // Phase 3b: Resolve member signals (after global signals are available)
        self.execute_member_resolve(dt)?;

        // Phase 3c: Resolve aggregate signals (after member signals are available)
        self.execute_aggregate_resolve(dt)?;

        // Phase 4: Fracture
        self.phase_executor.execute_fracture(
            &self.current_era,
            dt,
            self.sim_time,
            &self.dags,
            &self.signals,
            &mut self.fracture_queue,
        )?;

        // Phase 5: Measure
        self.phase_executor.execute_measure(
            &self.current_era,
            self.tick,
            dt,
            self.sim_time,
            &strata_states,
            &self.dags,
            &self.signals,
            &mut self.field_buffer,
        )?;

        // Post-tick: check era transitions
        self.check_era_transition()?;

        // Advance state
        self.signals.advance_tick();
        self.entities.advance_tick();
        self.member_signals.advance_tick();
        self.fracture_queue.drain_into(&mut self.input_channels);
        self.sim_time += dt.seconds();
        self.tick += 1;

        trace!("tick complete");
        Ok(ctx)
    }

    /// Execute member signal resolution using L1 parallel execution.
    ///
    /// Iterates over all registered member resolvers (scalar and Vec3) and executes them
    /// for each instance in parallel using the chunked L1 execution strategy. This runs
    /// after global signal resolution so that member expressions can access resolved
    /// global signals.
    #[instrument(skip(self), name = "member_resolve")]
    fn execute_member_resolve(&mut self, dt: Dt) -> Result<()> {
        use member_executor::{ChunkConfig, resolve_scalar_l1, resolve_vec3_l1};

        if self.member_resolvers.is_empty() && self.vec3_member_resolvers.is_empty() {
            return Ok(());
        }

        let instance_count = self.member_signals.instance_count();
        if instance_count == 0 {
            return Ok(());
        }

        trace!(
            scalar_resolvers = self.member_resolvers.len(),
            vec3_resolvers = self.vec3_member_resolvers.len(),
            instances = instance_count,
            "resolving member signals"
        );

        // Execute scalar member resolvers
        let scalar_signal_names: Vec<String> = self.member_resolvers.keys().cloned().collect();

        for signal_name in scalar_signal_names {
            // Get the correct instance count for this member's entity
            let signal_instance_count = self.member_signals.instance_count_for_signal(&signal_name);

            // Collect previous values as Vec<f64>
            let prev_values: Vec<f64> = (0..signal_instance_count)
                .map(|i| {
                    self.member_signals
                        .get_previous(&signal_name, i)
                        .and_then(|v| v.as_scalar())
                        .unwrap_or(0.0)
                })
                .collect();

            let resolver = self.member_resolvers.get(&signal_name).unwrap();
            let config = ChunkConfig::auto(signal_instance_count);

            // Execute in parallel using L1 strategy
            let results = resolve_scalar_l1(
                &prev_values,
                |ctx| resolver(ctx),
                &self.signals,
                &self.member_signals,
                dt,
                self.sim_time,
                config,
            );

            // Validate and write results back
            for (instance_idx, value) in results.into_iter().enumerate() {
                // Check for numeric errors
                if value.is_nan() {
                    error!(signal = %signal_name, instance = instance_idx, "NaN result in member signal");
                    return Err(Error::NumericError {
                        signal: SignalId(signal_name.clone()),
                        message: format!("NaN result for instance {}", instance_idx),
                    });
                }
                if value.is_infinite() {
                    error!(signal = %signal_name, instance = instance_idx, "infinite result in member signal");
                    return Err(Error::NumericError {
                        signal: SignalId(signal_name.clone()),
                        message: format!("Infinite result for instance {}", instance_idx),
                    });
                }

                trace!(signal = %signal_name, instance = instance_idx, value, "member signal resolved");
                self.member_signals
                    .set_current(&signal_name, instance_idx, Value::Scalar(value));
            }
        }

        // Execute Vec3 member resolvers
        let vec3_signal_names: Vec<String> = self.vec3_member_resolvers.keys().cloned().collect();

        for signal_name in vec3_signal_names {
            // Get the correct instance count for this member's entity
            let signal_instance_count = self.member_signals.instance_count_for_signal(&signal_name);

            // Collect previous values as Vec<[f64; 3]>
            let prev_values: Vec<[f64; 3]> = (0..signal_instance_count)
                .map(|i| {
                    self.member_signals
                        .get_previous(&signal_name, i)
                        .and_then(|v| v.as_vec3())
                        .unwrap_or([0.0, 0.0, 0.0])
                })
                .collect();

            let resolver = self.vec3_member_resolvers.get(&signal_name).unwrap();
            let config = ChunkConfig::auto(signal_instance_count);

            // Execute in parallel using L1 strategy
            let results = resolve_vec3_l1(
                &prev_values,
                |ctx| resolver(ctx),
                &self.signals,
                &self.member_signals,
                dt,
                self.sim_time,
                config,
            );

            // Validate and write results back
            for (instance_idx, value) in results.into_iter().enumerate() {
                // Check for numeric errors (any component)
                for (comp_idx, &comp) in value.iter().enumerate() {
                    if comp.is_nan() {
                        error!(signal = %signal_name, instance = instance_idx, component = comp_idx, "NaN result in Vec3 member signal");
                        return Err(Error::NumericError {
                            signal: SignalId(signal_name.clone()),
                            message: format!(
                                "NaN result in component {} for instance {}",
                                comp_idx, instance_idx
                            ),
                        });
                    }
                    if comp.is_infinite() {
                        error!(signal = %signal_name, instance = instance_idx, component = comp_idx, "infinite result in Vec3 member signal");
                        return Err(Error::NumericError {
                            signal: SignalId(signal_name.clone()),
                            message: format!(
                                "Infinite result in component {} for instance {}",
                                comp_idx, instance_idx
                            ),
                        });
                    }
                }

                trace!(signal = %signal_name, instance = instance_idx, ?value, "Vec3 member signal resolved");
                self.member_signals
                    .set_current(&signal_name, instance_idx, Value::Vec3(value));
            }
        }

        Ok(())
    }

    /// Execute aggregate signal resolution (Phase 3c).
    ///
    /// Runs after member signal resolution so that aggregate expressions like
    /// `sum(entity.particle, self.mass)` have access to resolved member values.
    #[instrument(skip(self), name = "aggregate_resolve")]
    fn execute_aggregate_resolve(&mut self, dt: Dt) -> Result<()> {
        if self.aggregate_resolvers.is_empty() {
            return Ok(());
        }

        trace!(
            resolvers = self.aggregate_resolvers.len(),
            "resolving aggregate signals"
        );

        // Copy signal IDs to avoid borrow issues
        let signal_ids: Vec<SignalId> = self.aggregate_resolvers.keys().cloned().collect();

        for signal_id in signal_ids {
            let resolver = self.aggregate_resolvers.get(&signal_id).unwrap();
            let value = resolver(&self.signals, &self.member_signals, dt, self.sim_time);

            // Validate numeric results
            if let Some(scalar) = value.as_scalar() {
                if scalar.is_nan() {
                    error!(signal = %signal_id, "NaN result in aggregate signal");
                    return Err(Error::NumericError {
                        signal: signal_id.clone(),
                        message: "NaN result in aggregate".to_string(),
                    });
                }
                if scalar.is_infinite() {
                    error!(signal = %signal_id, "infinite result in aggregate signal");
                    return Err(Error::NumericError {
                        signal: signal_id.clone(),
                        message: "Infinite result in aggregate".to_string(),
                    });
                }
            }

            trace!(signal = %signal_id, ?value, "aggregate signal resolved");
            self.signals.set_current(signal_id, value);
        }

        Ok(())
    }

    fn check_era_transition(&mut self) -> Result<()> {
        let era_config = self.eras.get(&self.current_era).unwrap();
        if let Some(ref transition) = era_config.transition
            && let Some(next_era) = transition(&self.signals, self.sim_time)
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
        let mut strata = IndexMap::new();
        strata.insert(stratum_id, StratumState::Active);
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = IndexMap::new();
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
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(1.0)));

        runtime.execute_tick().unwrap();
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(2.0)));

        runtime.execute_tick().unwrap();
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(3.0)));
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
        let mut strata = IndexMap::new();
        strata.insert(stratum_id, StratumState::ActiveWithStride(2));
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = IndexMap::new();
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
        let mut eras = IndexMap::new();
        eras.insert(
            era_id.clone(),
            EraConfig {
                dt: Dt(1.0),
                strata: IndexMap::new(),
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

        let mut strata = IndexMap::new();
        strata.insert(stratum_id, StratumState::Active);
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = IndexMap::new();
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
            let temp = ctx
                .signals
                .get(&signal_id_clone)
                .unwrap()
                .as_scalar()
                .unwrap();
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

        let mut strata = IndexMap::new();
        strata.insert(stratum_id, StratumState::Active);
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = IndexMap::new();
        eras.insert(era_id.clone(), era_config);

        let mut runtime = Runtime::new(era_id, eras, dags);

        // Register resolver: prev + collected
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

        let mut strata = IndexMap::new();
        strata.insert(stratum_id, StratumState::Active);
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = IndexMap::new();
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

    #[test]
    fn test_era_transition() {
        let era_a: EraId = "era_a".into();
        let era_b: EraId = "era_b".into();
        let stratum_id: StratumId = "default".into();
        let signal_id: SignalId = "counter".into();

        // Build DAG for Resolve phase
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

        let mut era_dags_a = EraDags::default();
        era_dags_a.insert(dag);

        // Build same DAG for era B
        let mut builder_b = DagBuilder::new(Phase::Resolve, stratum_id.clone());
        builder_b.add_node(DagNode {
            id: NodeId("counter_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_id.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_id.clone(),
                resolver_idx: 0,
            },
        });
        let dag_b = builder_b.build().unwrap();
        let mut era_dags_b = EraDags::default();
        era_dags_b.insert(dag_b);

        let mut dags = DagSet::default();
        dags.insert_era(era_a.clone(), era_dags_a);
        dags.insert_era(era_b.clone(), era_dags_b);

        // Era A transitions to Era B when counter >= 5
        let mut strata_a = IndexMap::new();
        strata_a.insert(stratum_id.clone(), StratumState::Active);

        let era_b_clone = era_b.clone();
        let signal_id_clone = signal_id.clone();
        let era_a_config = EraConfig {
            dt: Dt(1.0),
            strata: strata_a.clone(),
            transition: Some(Box::new(move |signals, _sim_time| {
                if let Some(value) = signals.get(&signal_id_clone) {
                    if value.as_scalar().unwrap_or(0.0) >= 5.0 {
                        return Some(era_b_clone.clone());
                    }
                }
                None
            })),
        };

        let era_b_config = EraConfig {
            dt: Dt(10.0), // Different dt to verify transition
            strata: strata_a,
            transition: None,
        };

        let mut eras = IndexMap::new();
        eras.insert(era_a.clone(), era_a_config);
        eras.insert(era_b.clone(), era_b_config);

        let mut runtime = Runtime::new(era_a.clone(), eras, dags);

        // Register resolver: increment by 1 each tick
        runtime.register_resolver(Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap_or(0.0);
            Value::Scalar(prev + 1.0)
        }));

        runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

        // Start in era A
        assert_eq!(runtime.era(), &era_a);

        // Ticks 0-4: counter goes from 0 to 5, still in era A until tick ends
        for i in 0..5 {
            let ctx = runtime.execute_tick().unwrap();
            assert_eq!(ctx.dt.0, 1.0, "tick {} should have dt=1.0", i);
        }

        // After tick 4, counter is 5, transition should happen
        assert_eq!(runtime.era(), &era_b);

        // Next tick should be in era B with dt=10
        let ctx = runtime.execute_tick().unwrap();
        assert_eq!(ctx.dt.0, 10.0, "should now use era B's dt");
        assert_eq!(runtime.era(), &era_b);
    }

    #[test]
    fn test_stratum_gating() {
        let era_id: EraId = "test".into();
        let active_stratum: StratumId = "active".into();
        let gated_stratum: StratumId = "gated".into();
        let active_signal: SignalId = "active_counter".into();
        let gated_signal: SignalId = "gated_counter".into();

        // Build DAGs for both strata
        let mut active_builder = DagBuilder::new(Phase::Resolve, active_stratum.clone());
        active_builder.add_node(DagNode {
            id: NodeId("active_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(active_signal.clone()),
            kind: NodeKind::SignalResolve {
                signal: active_signal.clone(),
                resolver_idx: 0,
            },
        });

        let mut gated_builder = DagBuilder::new(Phase::Resolve, gated_stratum.clone());
        gated_builder.add_node(DagNode {
            id: NodeId("gated_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(gated_signal.clone()),
            kind: NodeKind::SignalResolve {
                signal: gated_signal.clone(),
                resolver_idx: 1,
            },
        });

        let mut era_dags = EraDags::default();
        era_dags.insert(active_builder.build().unwrap());
        era_dags.insert(gated_builder.build().unwrap());

        let mut dags = DagSet::default();
        dags.insert_era(era_id.clone(), era_dags);

        // Configure: active stratum is Active, gated stratum is Gated
        let mut strata = IndexMap::new();
        strata.insert(active_stratum, StratumState::Active);
        strata.insert(gated_stratum, StratumState::Gated);

        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = IndexMap::new();
        eras.insert(era_id.clone(), era_config);

        let mut runtime = Runtime::new(era_id, eras, dags);

        // Register resolvers
        runtime.register_resolver(Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap_or(0.0);
            Value::Scalar(prev + 1.0)
        }));
        runtime.register_resolver(Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap_or(0.0);
            Value::Scalar(prev + 10.0) // Gated signal would increment by 10
        }));

        runtime.init_signal(active_signal.clone(), Value::Scalar(0.0));
        runtime.init_signal(gated_signal.clone(), Value::Scalar(0.0));

        // Execute tick - only active stratum should run
        runtime.execute_tick().unwrap();

        // Active signal should have incremented
        assert_eq!(
            runtime.get_signal(&active_signal),
            Some(&Value::Scalar(1.0))
        );

        // Gated signal should NOT have changed (gated stratum skipped)
        assert_eq!(runtime.get_signal(&gated_signal), Some(&Value::Scalar(0.0)));
    }

    #[test]
    fn test_parallel_level_signals() {
        // Two independent signals in the same level should both execute
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "default".into();
        let signal_a: SignalId = "a".into();
        let signal_b: SignalId = "b".into();

        let mut builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
        // Two nodes with no dependencies - same level
        builder.add_node(DagNode {
            id: NodeId("a_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_a.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_a.clone(),
                resolver_idx: 0,
            },
        });
        builder.add_node(DagNode {
            id: NodeId("b_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_b.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_b.clone(),
                resolver_idx: 1,
            },
        });

        let dag = builder.build().unwrap();
        // Verify they're in the same level
        assert_eq!(dag.levels.len(), 1, "both should be in same level");
        assert_eq!(dag.levels[0].nodes.len(), 2, "two nodes in level");

        let mut era_dags = EraDags::default();
        era_dags.insert(dag);

        let mut dags = DagSet::default();
        dags.insert_era(era_id.clone(), era_dags);

        let mut strata = IndexMap::new();
        strata.insert(stratum_id, StratumState::Active);
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = IndexMap::new();
        eras.insert(era_id.clone(), era_config);

        let mut runtime = Runtime::new(era_id, eras, dags);

        runtime.register_resolver(Box::new(|ctx| {
            Value::Scalar(ctx.prev.as_scalar().unwrap_or(0.0) + 1.0)
        }));
        runtime.register_resolver(Box::new(|ctx| {
            Value::Scalar(ctx.prev.as_scalar().unwrap_or(0.0) + 100.0)
        }));

        runtime.init_signal(signal_a.clone(), Value::Scalar(0.0));
        runtime.init_signal(signal_b.clone(), Value::Scalar(0.0));

        runtime.execute_tick().unwrap();

        // Both signals should have been resolved
        assert_eq!(runtime.get_signal(&signal_a), Some(&Value::Scalar(1.0)));
        assert_eq!(runtime.get_signal(&signal_b), Some(&Value::Scalar(100.0)));
    }

    #[test]
    fn test_dependency_chain_levels() {
        // Signal chain: A -> B -> C (three levels)
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "default".into();
        let signal_a: SignalId = "a".into();
        let signal_b: SignalId = "b".into();
        let signal_c: SignalId = "c".into();

        let mut builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
        builder.add_node(DagNode {
            id: NodeId("a_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_a.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_a.clone(),
                resolver_idx: 0,
            },
        });
        builder.add_node(DagNode {
            id: NodeId("b_resolve".to_string()),
            reads: [signal_a.clone()].into_iter().collect(),
            writes: Some(signal_b.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_b.clone(),
                resolver_idx: 1,
            },
        });
        builder.add_node(DagNode {
            id: NodeId("c_resolve".to_string()),
            reads: [signal_b.clone()].into_iter().collect(),
            writes: Some(signal_c.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_c.clone(),
                resolver_idx: 2,
            },
        });

        let dag = builder.build().unwrap();
        // Should have 3 levels
        assert_eq!(dag.levels.len(), 3, "chain should produce 3 levels");

        let mut era_dags = EraDags::default();
        era_dags.insert(dag);

        let mut dags = DagSet::default();
        dags.insert_era(era_id.clone(), era_dags);

        let mut strata = IndexMap::new();
        strata.insert(stratum_id, StratumState::Active);
        let era_config = EraConfig {
            dt: Dt(1.0),
            strata,
            transition: None,
        };

        let mut eras = IndexMap::new();
        eras.insert(era_id.clone(), era_config);

        let mut runtime = Runtime::new(era_id, eras, dags);

        // A: returns 10
        runtime.register_resolver(Box::new(|_| Value::Scalar(10.0)));
        // B: reads A and doubles it
        runtime.register_resolver(Box::new(|ctx| {
            let a = ctx.signals.get(&"a".into()).unwrap().as_scalar().unwrap();
            Value::Scalar(a * 2.0)
        }));
        // C: reads B and doubles it
        runtime.register_resolver(Box::new(|ctx| {
            let b = ctx.signals.get(&"b".into()).unwrap().as_scalar().unwrap();
            Value::Scalar(b * 2.0)
        }));

        runtime.init_signal(signal_a.clone(), Value::Scalar(0.0));
        runtime.init_signal(signal_b.clone(), Value::Scalar(0.0));
        runtime.init_signal(signal_c.clone(), Value::Scalar(0.0));

        runtime.execute_tick().unwrap();

        // Chain: A=10, B=20, C=40
        assert_eq!(runtime.get_signal(&signal_a), Some(&Value::Scalar(10.0)));
        assert_eq!(runtime.get_signal(&signal_b), Some(&Value::Scalar(20.0)));
        assert_eq!(runtime.get_signal(&signal_c), Some(&Value::Scalar(40.0)));
    }
}
