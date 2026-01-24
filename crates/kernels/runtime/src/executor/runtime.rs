//! Runtime core - orchestrates simulation execution.
//!
//! The Runtime struct manages all simulation state and coordinates execution
//! through phases. This module contains:
//! - Runtime struct definition and all its implementation
//! - EraConfig and related types
//! - All Runtime methods (construction, registration, execution, checkpointing)
//! - Runtime tests

use indexmap::IndexMap;
use tracing::{debug, error, info, instrument, trace, warn};

use crate::bytecode::CompiledBlock;
use crate::dag::DagSet;
use crate::error::{Error, Result};
use crate::soa_storage::{MemberSignalBuffer, ValueType as MemberValueType};
use crate::storage::{
    EmittedEventRecord, EntityInstances, EntityStorage, EventBuffer, FieldBuffer, FieldSample,
    FractureQueue, InputChannels, SignalStorage,
};
use crate::types::{
    DeterminismPolicy, Dt, EntityId, EraId, FieldId, ImpulseId, Phase, SignalId, StratumId,
    StratumState, TickContext, Value, WarmupConfig, WarmupResult, WorldPolicy,
};

use super::assertions::{AssertionChecker, AssertionFn};
use super::context::ImpulseContext;
use super::member_executor::{ScalarResolverFn, Vec3ResolverFn};
use crate::types::AssertionSeverity;

// ============================================================================
// Era and Stratum Configuration
// ============================================================================

/// Era configuration
pub struct EraConfig {
    /// Time step for this era
    pub dt: Dt,
    /// Stratum states in this era (IndexMap for deterministic iteration order)
    pub strata: IndexMap<StratumId, StratumState>,
    /// Transition condition (returns Some(next_era) if should transition)
    pub transition: Option<TransitionFn>,
}

/// Function that evaluates era transition
pub type TransitionFn =
    Box<dyn Fn(&SignalStorage, &EntityStorage, f64) -> Option<EraId> + Send + Sync>;

/// Function that resolves an aggregate value for a signal
pub type AggregateResolverFn = Box<
    dyn Fn(&SignalStorage, &EntityStorage, &MemberSignalBuffer, Dt, f64) -> Value + Send + Sync,
>;

/// The Continuum runtime orchestrates simulation execution.
pub struct Runtime {
    /// Storage for global signals
    signals: SignalStorage,
    /// Storage for entity instances
    entities: EntityStorage,
    /// Buffers for member signals (SoA storage)
    member_signals: MemberSignalBuffer,
    /// Channels for accumulating inputs before resolution
    input_channels: InputChannels,
    /// Buffer for emitted fields
    field_buffer: FieldBuffer,
    /// Buffer for emitted events
    event_buffer: EventBuffer,
    /// Queue for signals emitted during the fracture phase
    fracture_queue: FractureQueue,
    /// Current simulation tick
    tick: u64,
    /// Accumulated simulation time in seconds
    sim_time: f64,
    /// Current era
    current_era: EraId,
    /// Current internal phase within the tick
    current_phase: crate::types::TickPhase,
    /// Era configurations (IndexMap for deterministic iteration order)
    eras: IndexMap<EraId, EraConfig>,
    /// Execution DAGs
    dags: DagSet,
    /// Compiled bytecode blocks indexed by DAG node
    bytecode_blocks: Vec<CompiledBlock>,
    /// Bytecode phase executor
    bytecode_executor: crate::executor::bytecode::BytecodePhaseExecutor,
    /// Phase executor (procedural handlers)
    phase_executor: crate::executor::phases::PhaseExecutor,
    /// Warmup executor
    warmup_executor: crate::executor::warmup::WarmupExecutor,
    /// Assertion checker
    assertion_checker: AssertionChecker,
    /// Registered breakpoints (SignalId)
    breakpoints: std::collections::HashSet<SignalId>,
    /// Active tick context (frozen at start of tick)
    active_tick_ctx: Option<TickContext>,
    /// Pending impulses to apply in next Collect phase (handler_idx, payload)
    pending_impulses: Vec<(usize, Value)>,
    /// Background checkpoint writer (optional)
    checkpoint_writer: Option<crate::checkpoint::CheckpointWriter>,
    /// World IR hash for checkpoint validation (set once at initialization)
    world_ir_hash: Option<[u8; 32]>,
    /// Initial seed for determinism
    initial_seed: u64,
    /// Mapping from impulse ID to bytecode block index (for interactive injection)
    impulse_map: std::collections::HashMap<ImpulseId, usize>,
    /// Execution policy for the world
    policy: WorldPolicy,
}

impl Runtime {
    /// Create a new runtime
    pub fn new(
        initial_era: EraId,
        eras: IndexMap<EraId, EraConfig>,
        dags: DagSet,
        bytecode_blocks: Vec<CompiledBlock>,
        policy: WorldPolicy,
    ) -> Self {
        info!(era = %initial_era, "runtime created");
        let mut assertion_checker = AssertionChecker::new();
        assertion_checker.set_policy(policy.faults);

        Self {
            signals: SignalStorage::default(),
            entities: EntityStorage::default(),
            member_signals: MemberSignalBuffer::new(),
            input_channels: InputChannels::default(),
            field_buffer: FieldBuffer::default(),
            event_buffer: EventBuffer::default(),
            fracture_queue: FractureQueue::default(),
            tick: 0,
            sim_time: 0.0,
            current_era: initial_era,
            current_phase: crate::types::TickPhase::START,
            eras,
            dags,
            bytecode_blocks,
            bytecode_executor: crate::executor::bytecode::BytecodePhaseExecutor::new(),
            phase_executor: crate::executor::phases::PhaseExecutor::new(),
            warmup_executor: crate::executor::warmup::WarmupExecutor::new(),
            assertion_checker,
            breakpoints: std::collections::HashSet::new(),
            active_tick_ctx: None,
            pending_impulses: Vec::new(),
            checkpoint_writer: None,
            world_ir_hash: None,
            initial_seed: 0,
            impulse_map: std::collections::HashMap::new(),
            policy,
        }
    }

    /// Loads configuration values into the runtime.
    ///
    /// Configuration values are world-level defaults declared in `config{}` blocks
    /// that can be overridden by scenarios. They are frozen after loading and remain
    /// immutable throughout execution.
    ///
    /// # Lifecycle
    ///
    /// Called during `build_runtime()` (stage 4: Scenario Application) after compiling
    /// the world but before warmup. Values are extracted from world defaults and merged
    /// with scenario overrides (when implemented).
    ///
    /// # Access
    ///
    /// Config values are accessible in all DSL phases via `load_config("path")` and
    /// the `LoadConfig` opcode. They behave like frozen parameters, not signals (no
    /// prev/current distinction).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In world DSL:
    /// config { physics.gravity: 9.81 }
    ///
    /// // In signal resolve:
    /// signal velocity {
    ///     resolve { prev - load_config("physics.gravity") * dt }
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// The caller (`build_runtime`) panics if any config value contains a non-literal
    /// expression (enforcing Fail Loudly).
    pub fn set_config_values(&mut self, values: std::collections::HashMap<continuum_foundation::Path, Value>) {
        self.bytecode_executor.set_config_values(values);
    }

    /// Loads constant values into the runtime.
    ///
    /// Constant values are world-level immutable globals declared in `const{}` blocks.
    /// Unlike config, constants are NOT scenario-overridable. They are frozen after
    /// loading and remain immutable throughout execution.
    ///
    /// # Lifecycle
    ///
    /// Called during `build_runtime()` (stage 4: Scenario Application) after compiling
    /// the world but before warmup. Values are extracted from world const declarations.
    ///
    /// # Access
    ///
    /// Const values are accessible in all DSL phases via `load_const("path")` and
    /// the `LoadConst` opcode. They behave like frozen parameters, not signals (no
    /// prev/current distinction).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // In world DSL:
    /// const { physics.stefan_boltzmann: 5.67e-8 }
    ///
    /// // In signal resolve:
    /// signal radiation {
    ///     resolve { 
    ///         load_const("physics.stefan_boltzmann") * temp^4 
    ///     }
    /// }
    /// ```
    ///
    /// # Panics
    ///
    /// The caller (`build_runtime`) panics if any const value contains a non-literal
    /// expression (enforcing Fail Loudly).
    pub fn set_const_values(&mut self, values: std::collections::HashMap<continuum_foundation::Path, Value>) {
        self.bytecode_executor.set_const_values(values);
    }

    /// Stores signal type information for zero value initialization.
    ///
    /// Signal types are used by the bytecode executor to create correct zero values
    /// when no inputs have been accumulated for a signal.
    pub fn set_signal_types(&mut self, types: std::collections::HashMap<crate::types::SignalId, continuum_cdsl::foundation::Type>) {
        self.bytecode_executor.set_signal_types(types);
    }

    /// Add a breakpoint for a signal
    pub fn add_breakpoint(&mut self, signal: SignalId) {
        info!(%signal, "breakpoint added");
        self.breakpoints.insert(signal);
    }

    /// Remove a breakpoint for a signal
    pub fn remove_breakpoint(&mut self, signal: &SignalId) {
        info!(%signal, "breakpoint removed");
        self.breakpoints.remove(signal);
    }

    /// Clear all breakpoints
    pub fn clear_breakpoints(&mut self) {
        info!("all breakpoints cleared");
        self.breakpoints.clear();
    }

    /// Add a mapping for an impulse ID to its bytecode block index
    pub fn add_impulse_mapping(&mut self, id: ImpulseId, idx: usize) {
        self.impulse_map.insert(id, idx);
    }

    /// Inject an impulse by its ID
    pub fn inject_impulse_by_id(&mut self, id: &ImpulseId, payload: Value) -> Result<()> {
        let idx = self
            .impulse_map
            .get(id)
            .ok_or_else(|| Error::Generic(format!("Impulse '{}' not found", id)))?;
        self.inject_impulse(*idx, payload);
        Ok(())
    }

    /// Check if a signal has a breakpoint
    pub fn has_breakpoint(&self, signal: &SignalId) -> bool {
        self.breakpoints.contains(signal)
    }

    /// Register a resolver function, returns its index
    pub fn register_resolver(&mut self, resolver: crate::executor::phases::ResolverFn) -> usize {
        self.phase_executor.register_resolver(resolver)
    }

    /// Register a collect operator, returns its index
    pub fn register_collect_op(&mut self, op: crate::executor::phases::CollectFn) -> usize {
        self.phase_executor.register_collect_op(op)
    }

    /// Register a fracture function, returns its index
    pub fn register_fracture(&mut self, fracture: crate::executor::phases::FractureFn) -> usize {
        self.phase_executor.register_fracture(fracture)
    }

    /// Register a measure operator, returns its index
    pub fn register_measure_op(&mut self, op: crate::executor::phases::MeasureFn) -> usize {
        self.phase_executor.register_measure_op(op)
    }

    /// Register a chronicle handler, returns its index
    pub fn register_chronicle(&mut self, handler: crate::executor::phases::ChronicleFn) -> usize {
        self.phase_executor.register_chronicle(handler)
    }

    /// Register an impulse handler, returns its index
    pub fn register_impulse(&mut self, handler: crate::executor::phases::ImpulseFn) -> usize {
        self.phase_executor.register_impulse(handler)
    }

    /// Inject an impulse to be applied in the next tick's Collect phase
    pub fn inject_impulse(&mut self, handler_idx: usize, payload: Value) {
        tracing::debug!(handler_idx, ?payload, "impulse injected");
        self.pending_impulses.push((handler_idx, payload));
    }

    /// Register a warmup function for a signal
    pub fn register_warmup(
        &mut self,
        signal: SignalId,
        warmup_fn: crate::executor::warmup::WarmupFn,
        config: WarmupConfig,
    ) {
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
    pub fn init_member_instances(&mut self, instance_count: usize) {
        tracing::debug!(count = instance_count, "member instances initialized");
        self.member_signals.init_instances(instance_count);
    }

    /// Register the instance count for a specific entity.
    pub fn register_entity_count(&mut self, entity_id: &str, count: usize) {
        tracing::debug!(
            entity = entity_id,
            count,
            "entity instance count registered"
        );
        self.member_signals.register_entity_count(entity_id, count);
    }

    /// Register a scalar member resolver function
    pub fn register_member_resolver(&mut self, _signal_name: String, resolver: ScalarResolverFn) {
        tracing::debug!(signal = %_signal_name, "scalar member resolver registered");
        self.phase_executor
            .register_member_resolver(crate::executor::phases::MemberResolver::Scalar(resolver));
    }

    /// Register a Vec3 member resolver function
    pub fn register_vec3_member_resolver(
        &mut self,
        _signal_name: String,
        resolver: Vec3ResolverFn,
    ) {
        tracing::debug!(signal = %_signal_name, "vec3 member resolver registered");
        self.phase_executor
            .register_member_resolver(crate::executor::phases::MemberResolver::Vec3(resolver));
    }

    /// Register an aggregate resolver
    pub fn register_aggregate_resolver(
        &mut self,
        _signal_id: SignalId,
        resolver: AggregateResolverFn,
    ) {
        self.phase_executor.register_aggregate_resolver(resolver);
    }

    /// Set a member signal value for an instance
    pub fn set_member_signal(
        &mut self,
        signal_name: &str,
        instance_idx: usize,
        value: Value,
    ) -> std::result::Result<(), String> {
        self.member_signals
            .set_current(signal_name, instance_idx, value)
    }

    /// Commit all member initial values (swap buffers)
    pub fn commit_member_initials(&mut self) {
        self.member_signals.advance_tick();
    }

    /// Get current simulation tick
    pub fn tick(&self) -> u64 {
        self.tick
    }

    /// Get accumulated simulation time in seconds
    pub fn sim_time(&self) -> f64 {
        self.sim_time
    }

    /// Get current simulation era
    pub fn era(&self) -> &EraId {
        &self.current_era
    }

    /// Get value of a signal by ID
    pub fn get_signal(&self, id: &SignalId) -> Option<&Value> {
        self.signals.get_resolved(id)
    }

    /// Check if warmup has been executed
    pub fn is_warmup_complete(&self) -> bool {
        self.warmup_executor.is_complete()
    }

    /// Drain field samples (clears the buffer)
    pub fn drain_fields(&mut self) -> IndexMap<FieldId, Vec<FieldSample>> {
        self.field_buffer.drain()
    }

    /// Get access to the event buffer
    pub fn event_buffer(&self) -> &EventBuffer {
        &self.event_buffer
    }

    /// Drain the event buffer
    pub fn drain_events(&mut self) -> Vec<EmittedEventRecord> {
        self.event_buffer.drain()
    }

    /// Get context for the current tick state
    pub fn tick_context(&self) -> TickContext {
        let dt = self
            .eras
            .get(&self.current_era)
            .unwrap_or_else(|| panic!("current era '{}' missing from runtime", self.current_era))
            .dt;
        TickContext {
            tick: self.tick,
            sim_time: self.sim_time,
            dt,
            era: self.current_era.clone(),
        }
    }

    /// Get access to the signals storage
    pub fn signals(&self) -> &SignalStorage {
        &self.signals
    }

    /// Get access to the entities storage
    pub fn entities(&self) -> &EntityStorage {
        &self.entities
    }

    /// Get access to the entities storage mutably
    pub fn entities_mut(&mut self) -> &mut EntityStorage {
        &mut self.entities
    }

    /// Get access to the member signals buffer
    pub fn member_signals(&self) -> &MemberSignalBuffer {
        &self.member_signals
    }

    /// Get access to the assertion checker
    pub fn assertion_checker(&self) -> &AssertionChecker {
        &self.assertion_checker
    }

    /// Get mutable access to the assertion checker
    pub fn assertion_checker_mut(&mut self) -> &mut AssertionChecker {
        &mut self.assertion_checker
    }

    /// Execute warmup phase
    #[instrument(skip_all, name = "warmup")]
    pub fn execute_warmup(&mut self) -> Result<WarmupResult> {
        self.warmup_executor
            .execute(&mut self.signals, &self.entities, self.sim_time)
    }

    /// Get current phase
    pub fn phase(&self) -> crate::types::TickPhase {
        self.current_phase
    }

    // ============================================================================
    // Phase Execution Handlers (Internal)
    // ============================================================================

    /// Determine execution mode (bytecode vs procedural)
    #[inline(always)]
    fn has_bytecode(&self) -> bool {
        !self.bytecode_blocks.is_empty()
    }

    /// Execute Configure phase: Initialize tick context and run Configure DAG
    fn execute_configure_phase(&mut self, dt: Dt, strata_states: &IndexMap<StratumId, StratumState>) -> Result<()> {
        trace!("phase: configure");
        self.active_tick_ctx = Some(TickContext {
            tick: self.tick,
            sim_time: self.sim_time,
            dt,
            era: self.current_era.clone(),
        });
        
        // Execute Configure phase DAG (initial blocks)
        if self.has_bytecode() {
            self.bytecode_executor.execute_configure(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.bytecode_blocks,
                &mut self.signals,
                &self.entities,
                &mut self.member_signals,
                &mut self.input_channels,
            )?;
            
            // Initialize any remaining uninitialized signals with zero/default values
            // This ensures all signals have a prev value before the first Resolve phase
            use crate::dag::NodeKind;
            let era_dags = self.dags.get_era(&self.current_era).unwrap();
            for dag in era_dags.for_phase(Phase::Resolve) {
                for level in &dag.levels {
                    for node in &level.nodes {
                        if let NodeKind::SignalResolve { signal, .. } = &node.kind {
                            // Initialize signal to zero if not already initialized
                            if !self.signals.has(signal) {
                                self.signals.init(signal.clone(), Value::Scalar(0.0));
                                trace!("initialized uninitialized signal '{}' to 0.0", signal);
                            }
                        }
                    }
                }
            }
        }
        
        // Commit member signal initial values (copy current to previous buffer)
        // This must happen after Configure phase so that prev is available in Resolve
        self.commit_member_initials();
        
        self.current_phase = crate::types::TickPhase::Simulation(Phase::Collect);
        Ok(())
    }

    /// Execute Collect phase: Accumulate inputs, impulses
    fn execute_collect_phase(&mut self, dt: Dt, strata_states: &IndexMap<StratumId, StratumState>) -> Result<()> {
        trace!("phase: collect");
        if !self.has_bytecode() {
            self.phase_executor.execute_collect(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.signals,
                &self.entities,
                &mut self.input_channels,
                &mut self.pending_impulses,
            )?;
        } else {
            self.bytecode_executor.execute_collect(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.bytecode_blocks,
                &self.signals,
                &self.entities,
                &self.member_signals,
                &mut self.input_channels,
            )?;
        }
        // Apply pending impulses (procedural for now)
        let impulses = std::mem::take(&mut self.pending_impulses);
        for (handler_idx, payload) in impulses {
            let handler = &self.phase_executor.impulse_handlers[handler_idx];
            let mut ctx = ImpulseContext {
                signals: &self.signals,
                entities: &self.entities,
                channels: &mut self.input_channels,
                dt,
                sim_time: self.sim_time,
            };
            handler(&mut ctx, &payload);
        }
        self.current_phase = crate::types::TickPhase::Simulation(Phase::Resolve);
        Ok(())
    }

    /// Execute Resolve phase: Resolve signals, check breakpoints
    fn execute_resolve_phase(&mut self, dt: Dt, strata_states: &IndexMap<StratumId, StratumState>) -> Result<Option<SignalId>> {
        trace!("phase: resolve");
        let signal = if !self.has_bytecode() {
            self.phase_executor.execute_resolve(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &mut self.signals,
                &self.entities,
                &mut self.member_signals,
                &mut self.input_channels,
                &mut self.assertion_checker,
                &self.breakpoints,
            )?
        } else {
            self.bytecode_executor.execute_resolve(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.bytecode_blocks,
                &mut self.signals,
                &self.entities,
                &mut self.member_signals,
                &mut self.input_channels,
                &mut self.assertion_checker,
                &self.breakpoints,
            )?
        };
        self.current_phase = crate::types::TickPhase::Simulation(Phase::Fracture);
        Ok(signal)
    }

    /// Execute Fracture phase: Detect tension and queue fracture signals
    fn execute_fracture_phase(&mut self, dt: Dt, strata_states: &IndexMap<StratumId, StratumState>) -> Result<()> {
        trace!("phase: fracture");
        if !self.has_bytecode() {
            self.phase_executor.execute_fracture(
                &self.current_era,
                dt,
                self.sim_time,
                &self.dags,
                &self.signals,
                &self.entities,
                &mut self.fracture_queue,
            )?;
        } else {
            self.bytecode_executor.execute_fracture(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.bytecode_blocks,
                &self.signals,
                &mut self.entities,
                &self.member_signals,
                &mut self.fracture_queue,
            )?;
        }
        self.current_phase = crate::types::TickPhase::Simulation(Phase::Measure);
        Ok(())
    }

    /// Execute Measure phase: Emit fields and chronicle events
    fn execute_measure_phase(&mut self, dt: Dt, strata_states: &IndexMap<StratumId, StratumState>) -> Result<()> {
        trace!("phase: measure");
        if !self.has_bytecode() {
            self.phase_executor.execute_measure(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.signals,
                &self.entities,
                &mut self.field_buffer,
            )?;

            self.phase_executor.execute_chronicles(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.signals,
                &self.entities,
                &mut self.event_buffer,
            )?;
        } else {
            self.bytecode_executor.execute_measure(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.bytecode_blocks,
                &self.signals,
                &self.entities,
                &self.member_signals,
                &mut self.field_buffer,
            )?;

            self.bytecode_executor.execute_chronicles(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.bytecode_blocks,
                &self.signals,
                &self.entities,
                &self.member_signals,
                &mut self.event_buffer,
            )?;
        }
        self.current_phase = crate::types::TickPhase::EraTransition;
        Ok(())
    }

    /// Execute EraTransition phase: Check and apply era transitions
    fn execute_era_transition_phase(&mut self) -> Result<()> {
        trace!("phase: era transition");
        self.check_era_transition()?;
        self.current_phase = crate::types::TickPhase::PostTick;
        Ok(())
    }

    /// Execute PostTick phase: Finalize tick, advance time
    fn execute_post_tick_phase(&mut self, dt: Dt) -> Result<()> {
        trace!("phase: post tick");
        self.signals.advance_tick();
        self.entities.advance_tick();
        self.member_signals.advance_tick();
        self.fracture_queue.drain_into(&mut self.input_channels);
        self.sim_time += dt.seconds();
        self.tick += 1;
        self.validate_determinism()?;
        self.current_phase = crate::types::TickPhase::Simulation(Phase::Configure);
        Ok(())
    }

    // ============================================================================
    // Phase Orchestration
    // ============================================================================

    /// Execute a single phase of the simulation.
    pub fn execute_step(&mut self) -> Result<crate::types::StepResult> {
        // Get era config (dt and strata states)
        let (dt, strata_states) = {
            let era_config = self
                .eras
                .get(&self.current_era)
                .ok_or_else(|| Error::EraNotFound(self.current_era.clone()))?;
            (era_config.dt, era_config.strata.clone())
        };

        // Dispatch to appropriate phase handler
        match self.current_phase {
            // Reject lifecycle phases (should not call execute_step during these)
            crate::types::TickPhase::Simulation(Phase::CollectConfig)
            | crate::types::TickPhase::Simulation(Phase::Initialize)
            | crate::types::TickPhase::Simulation(Phase::WarmUp) => {
                return Err(Error::PhaseViolation {
                    operation: "execute_step".to_string(),
                    phase: match self.current_phase {
                        crate::types::TickPhase::Simulation(phase) => phase,
                        _ => Phase::Configure,
                    },
                });
            }
            
            // Main simulation phases
            crate::types::TickPhase::Simulation(Phase::Configure) => {
                self.execute_configure_phase(dt, &strata_states)?;
            }
            crate::types::TickPhase::Simulation(Phase::Collect) => {
                self.execute_collect_phase(dt, &strata_states)?;
            }
            crate::types::TickPhase::Simulation(Phase::Resolve) => {
                if let Some(signal) = self.execute_resolve_phase(dt, &strata_states)? {
                    return Ok(crate::types::StepResult::Breakpoint { signal });
                }
            }
            crate::types::TickPhase::Simulation(Phase::Fracture) => {
                self.execute_fracture_phase(dt, &strata_states)?;
            }
            crate::types::TickPhase::Simulation(Phase::Measure) => {
                self.execute_measure_phase(dt, &strata_states)?;
            }
            crate::types::TickPhase::Simulation(Phase::Assert) => {
                trace!("phase: assert");
                self.current_phase = crate::types::TickPhase::EraTransition;
            }
            
            // Tick finalization phases
            crate::types::TickPhase::EraTransition => {
                self.execute_era_transition_phase()?;
            }
            crate::types::TickPhase::PostTick => {
                self.execute_post_tick_phase(dt)?;
            }
        }

        // Check if we completed a full tick
        if self.current_phase == crate::types::TickPhase::START {
            let ctx = self
                .active_tick_ctx
                .take()
                .expect("tick context missing at end of tick");
            Ok(crate::types::StepResult::TickCompleted(ctx))
        } else {
            Ok(crate::types::StepResult::Continue)
        }
    }

    /// Execute a single tick
    #[instrument(level = "debug", skip(self), fields(tick = self.tick, era = %self.current_era))]
    pub fn execute_tick(&mut self) -> Result<TickContext> {
        loop {
            match self.execute_step()? {
                crate::types::StepResult::TickCompleted(ctx) => return Ok(ctx),
                crate::types::StepResult::Breakpoint { signal } => {
                    return Err(Error::Generic(format!(
                        "Hit breakpoint on signal '{}' during tick",
                        signal
                    )));
                }
                crate::types::StepResult::Continue => continue,
            }
        }
    }

    /// Execute until a breakpoint or tick completed
    pub fn execute_until_breakpoint(&mut self) -> Result<crate::types::StepResult> {
        loop {
            let result = self.execute_step()?;
            match result {
                crate::types::StepResult::Breakpoint { .. } => return Ok(result),
                crate::types::StepResult::TickCompleted(..) => return Ok(result),
                crate::types::StepResult::Continue => continue,
            }
        }
    }

    fn check_era_transition(&mut self) -> Result<()> {
        let era_config = self.eras.get(&self.current_era).unwrap();
        if let Some(ref transition) = era_config.transition
            && let Some(next_era) = transition(&self.signals, &self.entities, self.sim_time)
        {
            if !self.eras.contains_key(&next_era) {
                error!(era = %next_era, "transition to unknown era");
                return Err(Error::EraNotFound(next_era));
            }
            debug!(from = %self.current_era, to = %next_era, "era transition");
            self.current_era = next_era;
        }
        Ok(())
    }

    // ========================================================================
    // Checkpoint Methods
    // ========================================================================

    /// Enable checkpoint writer with specified queue depth.
    pub fn enable_checkpointing(&mut self, queue_depth: usize) {
        info!(queue_depth, "enabling checkpoint writer");
        self.checkpoint_writer = Some(crate::checkpoint::CheckpointWriter::new(queue_depth));
    }

    /// Set the world IR hash for checkpoint validation.
    pub fn set_world_ir_hash(&mut self, hash: [u8; 32]) {
        self.world_ir_hash = Some(hash);
    }

    /// Set the initial seed for determinism.
    pub fn set_initial_seed(&mut self, seed: u64) {
        self.initial_seed = seed;
    }

    /// Validate determinism if policy is Strict
    pub fn validate_determinism(&self) -> Result<()> {
        if self.policy.determinism != DeterminismPolicy::Strict {
            return Ok(());
        }
        // TODO: Implement actual determinism checks (hash state etc)
        trace!("Strict determinism check performed (placeholder)");
        Ok(())
    }

    /// Request a checkpoint write (non-blocking).
    ///
    /// If checkpointing is not enabled, returns an error.
    /// If the queue is full, drops the checkpoint and returns an error.
    pub fn request_checkpoint(&self, path: &std::path::Path) -> Result<()> {
        let writer = self
            .checkpoint_writer
            .as_ref()
            .ok_or_else(|| Error::Checkpoint("checkpointing not enabled".to_string()))?;

        // Extract member signal data
        let member_signals = crate::checkpoint::MemberSignalData::from_buffer(&self.member_signals)
            .map_err(|e| Error::Checkpoint(e.to_string()))?;

        // Build era configs for validation
        let era_configs = self
            .eras
            .iter()
            .map(|(id, cfg)| {
                (
                    id.clone(),
                    crate::checkpoint::EraConfigSnapshot {
                        dt: cfg.dt.0,
                        strata_count: cfg.strata.len(),
                    },
                )
            })
            .collect();

        let world_ir_hash = self
            .world_ir_hash
            .ok_or_else(|| Error::Checkpoint("world IR hash missing".to_string()))?;
        let stratum_states = self
            .eras
            .get(&self.current_era)
            .ok_or_else(|| Error::Checkpoint("current era not found".to_string()))?
            .strata
            .iter()
            .map(|(id, state)| {
                let is_gated = matches!(state, StratumState::Gated);
                (
                    id.clone(),
                    crate::checkpoint::StratumState {
                        cadence_counter: 0,
                        is_gated,
                    },
                )
            })
            .collect();

        // Build checkpoint
        let checkpoint = crate::checkpoint::Checkpoint {
            header: crate::checkpoint::CheckpointHeader {
                version: crate::checkpoint::CHECKPOINT_VERSION,
                world_ir_hash,
                tick: self.tick,
                sim_time: self.sim_time,
                seed: self.initial_seed,
                current_era: self.current_era.clone(),
                created_at: std::time::SystemTime::now(),
                world_git_hash: None,
            },
            state: crate::checkpoint::CheckpointState {
                signals: self.signals.clone(),
                entities: self.entities.clone(),
                member_signals,
                era_configs,
                stratum_states,
            },
        };

        writer
            .request_checkpoint(
                path.to_owned(),
                checkpoint,
                crate::checkpoint::DEFAULT_COMPRESSION_LEVEL,
            )
            .map_err(|e| Error::Checkpoint(e.to_string()))
    }

    /// Load a checkpoint and replace runtime state (validation optional).
    ///
    /// If `force` is false, validates world IR hash.
    /// Returns error if validation fails or deserialization fails.
    pub fn load_checkpoint(&mut self, path: &std::path::Path, force: bool) -> Result<()> {
        info!(path = %path.display(), "loading checkpoint");

        let checkpoint = crate::checkpoint::load_checkpoint(path)
            .map_err(|e| Error::Checkpoint(e.to_string()))?;

        // Validate world IR hash
        if !force {
            if let Some(current_hash) = self.world_ir_hash {
                if checkpoint.header.world_ir_hash != current_hash {
                    return Err(Error::Checkpoint(
                        "World IR mismatch: checkpoint hash does not match current world"
                            .to_string(),
                    ));
                }
            }
        }

        // Restore state
        self.signals = checkpoint.state.signals;
        self.entities = checkpoint.state.entities;

        // Restore member signals
        checkpoint
            .state
            .member_signals
            .restore_into_buffer(&mut self.member_signals)
            .map_err(|e| Error::Checkpoint(e.to_string()))?;

        self.tick = checkpoint.header.tick;
        self.sim_time = checkpoint.header.sim_time;
        self.current_era = checkpoint.header.current_era;
        self.initial_seed = checkpoint.header.seed;

        info!(
            tick = checkpoint.header.tick,
            sim_time = checkpoint.header.sim_time,
            "checkpoint loaded successfully"
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dag::{DagBuilder, DagNode, EraDags, NodeId, NodeKind};
    use crate::executor::EmittedEvent;
    use crate::types::{Phase, StratumId};
    use crate::FaultPolicy;
    use std::collections::HashSet;

    fn create_minimal_runtime(era_id: EraId) -> Runtime {
        let mut eras = IndexMap::new();
        eras.insert(
            era_id.clone(),
            EraConfig {
                dt: Dt(1.0),
                strata: IndexMap::new(),
                transition: None,
            },
        );
        Runtime::new(
            era_id,
            eras,
            DagSet::default(),
            Vec::new(),
            WorldPolicy::default(),
        )
    }

    #[test]
    fn test_runtime_creation() {
        let era_id: EraId = "default".into();
        let mut eras = IndexMap::new();
        eras.insert(
            era_id.clone(),
            EraConfig {
                dt: Dt(1.0),
                strata: IndexMap::new(),
                transition: None,
            },
        );
        let runtime = Runtime::new(
            era_id,
            eras,
            DagSet::default(),
            Vec::new(),
            WorldPolicy::default(),
        );
        assert_eq!(runtime.tick(), 0);
        assert_eq!(runtime.sim_time(), 0.0);
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

        let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

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
        let drained = runtime.drain_fields();
        let samples = drained.get(&field_id).unwrap();
        assert_eq!(samples.len(), 1);
        assert_eq!(samples[0].value.as_scalar(), Some(110.0));

        // Execute another tick
        runtime.execute_tick().unwrap();

        let drained = runtime.drain_fields();
        let samples = drained.get(&field_id).unwrap();
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

        let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

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

        let mut policy = WorldPolicy::default();
        policy.faults = FaultPolicy::Fatal;
        let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), policy);

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
            transition: Some(Box::new(move |signals, _entities, _sim_time| {
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

        let mut runtime = Runtime::new(
            era_a.clone(),
            eras,
            dags,
            Vec::new(),
            WorldPolicy::default(),
        );

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

        let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

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

        let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

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

        let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

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

    #[test]
    fn test_chronicle_event_emission() {
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "default".into();
        let signal_id: SignalId = "temperature".into();

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

        // Build DAG for Measure phase with chronicle
        let mut measure_builder = DagBuilder::new(Phase::Measure, stratum_id.clone());
        measure_builder.add_node(DagNode {
            id: NodeId("temp_chronicle".to_string()),
            reads: [signal_id.clone()].into_iter().collect(),
            writes: None,
            kind: NodeKind::ChronicleObserve { chronicle_idx: 0 },
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

        let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

        // Register resolver: temperature increments by 10 each tick
        runtime.register_resolver(Box::new(|ctx| {
            let prev = ctx.prev.as_scalar().unwrap_or(0.0);
            Value::Scalar(prev + 10.0)
        }));

        // Register chronicle: emit event when temperature > 100
        let signal_id_clone = signal_id.clone();
        runtime.register_chronicle(Box::new(move |ctx| {
            let temp = ctx
                .signals
                .get(&signal_id_clone)
                .unwrap()
                .as_scalar()
                .unwrap();
            if temp > 100.0 {
                vec![EmittedEvent {
                    chronicle_id: "test.chronicle".to_string(),
                    name: "high_temperature".to_string(),
                    fields: vec![("temp".to_string(), Value::Scalar(temp))],
                }]
            } else {
                vec![]
            }
        }));

        runtime.init_signal(signal_id.clone(), Value::Scalar(100.0));

        // Tick 1: temp = 110, should emit event
        runtime.execute_tick().unwrap();
        assert_eq!(runtime.get_signal(&signal_id), Some(&Value::Scalar(110.0)));

        // Check event buffer
        assert!(!runtime.event_buffer().is_empty());
        assert_eq!(runtime.event_buffer().len(), 1);

        let events = runtime.drain_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].name, "high_temperature");
        assert_eq!(events[0].fields.len(), 1);
        assert_eq!(events[0].fields[0].0, "temp");
        assert_eq!(events[0].fields[0].1.as_scalar(), Some(110.0));

        // After drain, buffer should be empty
        assert!(runtime.event_buffer().is_empty());

        // Tick 2: temp = 120, should emit another event
        runtime.execute_tick().unwrap();
        assert_eq!(runtime.event_buffer().len(), 1);
    }

    #[test]
    fn test_chronicle_no_emission_when_condition_false() {
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "default".into();
        let signal_id: SignalId = "pressure".into();

        // Build DAG for Resolve phase
        let mut resolve_builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
        resolve_builder.add_node(DagNode {
            id: NodeId("pressure_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_id.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_id.clone(),
                resolver_idx: 0,
            },
        });
        let resolve_dag = resolve_builder.build().unwrap();

        // Build DAG for Measure phase with chronicle
        let mut measure_builder = DagBuilder::new(Phase::Measure, stratum_id.clone());
        measure_builder.add_node(DagNode {
            id: NodeId("pressure_chronicle".to_string()),
            reads: [signal_id.clone()].into_iter().collect(),
            writes: None,
            kind: NodeKind::ChronicleObserve { chronicle_idx: 0 },
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

        let mut runtime = Runtime::new(era_id, eras, dags, Vec::new(), WorldPolicy::default());

        // Register resolver: pressure stays constant at 50
        runtime.register_resolver(Box::new(|_ctx| Value::Scalar(50.0)));

        // Register chronicle: emit event only when pressure > 100 (never true)
        let signal_id_clone = signal_id.clone();
        runtime.register_chronicle(Box::new(move |ctx| {
            let pressure = ctx
                .signals
                .get(&signal_id_clone)
                .unwrap()
                .as_scalar()
                .unwrap();
            if pressure > 100.0 {
                vec![EmittedEvent {
                    chronicle_id: "test.chronicle".to_string(),
                    name: "high_pressure".to_string(),
                    fields: vec![],
                }]
            } else {
                vec![]
            }
        }));

        runtime.init_signal(signal_id.clone(), Value::Scalar(0.0));

        // Execute ticks - no events should be emitted
        runtime.execute_tick().unwrap();
        assert!(runtime.event_buffer().is_empty());

        runtime.execute_tick().unwrap();
        assert!(runtime.event_buffer().is_empty());

        runtime.execute_tick().unwrap();
        assert!(runtime.event_buffer().is_empty());
    }
}
