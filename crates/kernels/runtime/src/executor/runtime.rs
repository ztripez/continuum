//! Runtime core — struct definition, construction, getters, and phase execution.
//!
//! The Runtime struct manages all simulation state and coordinates execution
//! through phases. This module contains the core orchestration logic.
//!
//! Related modules split out for maintainability:
//! - [`super::registration`] — registration and initialization methods
//! - [`super::checkpoint_ops`] — checkpoint serialization/deserialization

use indexmap::IndexMap;
use tracing::{debug, error, info, instrument, trace};

use crate::bytecode::CompiledBlock;
use crate::dag::DagSet;
use crate::error::{Error, Result};
use crate::storage::{
    EmittedEventRecord, EventBuffer, FieldBuffer, FieldSample, FractureQueue,
    InputChannels,
};
use crate::types::{
    Dt, EraId, FieldId, ImpulseId, Phase, SignalId, StratumId, StratumState,
    TickContext, Value, WarmupResult, WorldPolicy,
};
use crate::unified_storage::UnifiedStorage;
use crate::soa_storage::MemberSignalBuffer;
use crate::storage::EntityStorage;

use super::assertions::AssertionChecker;
use super::context::ImpulseContext;

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
    Box<dyn Fn(&MemberSignalBuffer, &EntityStorage, f64) -> Option<EraId> + Send + Sync>;

/// Function that resolves an aggregate value for a signal
pub type AggregateResolverFn = Box<
    dyn Fn(&MemberSignalBuffer, &EntityStorage, &MemberSignalBuffer, Dt, f64) -> Value + Send + Sync,
>;

/// The Continuum runtime orchestrates simulation execution.
pub struct Runtime {
    /// Unified simulation state storage (signals, member signals, entities).
    pub(crate) storage: UnifiedStorage,
    /// Channels for accumulating inputs before resolution
    pub(crate) input_channels: InputChannels,
    /// Buffer for emitted fields
    pub(crate) field_buffer: FieldBuffer,
    /// Buffer for emitted events
    pub(crate) event_buffer: EventBuffer,
    /// Queue for signals emitted during the fracture phase
    pub(crate) fracture_queue: FractureQueue,
    /// Current simulation tick
    pub(crate) tick: u64,
    /// Accumulated simulation time in seconds
    pub(crate) sim_time: f64,
    /// Current era
    pub(crate) current_era: EraId,
    /// Current internal phase within the tick
    pub(crate) current_phase: crate::types::TickPhase,
    /// Era configurations (IndexMap for deterministic iteration order)
    pub(crate) eras: IndexMap<EraId, EraConfig>,
    /// Execution DAGs
    pub(crate) dags: DagSet,
    /// Compiled bytecode blocks indexed by DAG node
    pub(crate) bytecode_blocks: Vec<CompiledBlock>,
    /// Bytecode phase executor
    pub(crate) bytecode_executor: crate::executor::bytecode::BytecodePhaseExecutor,
    /// Phase executor (procedural handlers)
    pub(crate) phase_executor: crate::executor::phases::PhaseExecutor,
    /// Warmup executor
    pub(crate) warmup_executor: crate::executor::warmup::WarmupExecutor,
    /// Assertion checker
    pub(crate) assertion_checker: AssertionChecker,
    /// Registered breakpoints (SignalId)
    pub(crate) breakpoints: std::collections::HashSet<SignalId>,
    /// Active tick context (frozen at start of tick)
    pub(crate) active_tick_ctx: Option<TickContext>,
    /// Pending impulses to apply in next Collect phase (handler_idx, payload)
    pub(crate) pending_impulses: Vec<(usize, Value)>,
    /// Background checkpoint writer (optional)
    pub(crate) checkpoint_writer: Option<crate::checkpoint::CheckpointWriter>,
    /// World IR hash for checkpoint validation (set once at initialization)
    pub(crate) world_ir_hash: Option<[u8; 32]>,
    /// Initial seed for determinism
    pub(crate) initial_seed: u64,
    /// Mapping from impulse ID to bytecode block index (for interactive injection)
    pub(crate) impulse_map: IndexMap<ImpulseId, usize>,
    /// Execution policy for the world
    pub(crate) policy: WorldPolicy,
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
            storage: UnifiedStorage::new(),
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
            impulse_map: IndexMap::new(),
            policy,
        }
    }

    /// Commit all member initial values (swap buffers).
    ///
    /// Called during Configure phase to copy current member signal values
    /// into the previous buffer so `prev` is available in Resolve.
    fn commit_member_initials(&mut self) {
        self.storage.member_signals.advance_tick();
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
    pub fn get_signal(&self, id: &SignalId) -> Option<Value> {
        self.storage.member_signals.get_global_or_prev(&id.to_string())
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

    /// Get access to the unified storage.
    pub fn storage(&self) -> &UnifiedStorage {
        &self.storage
    }

    /// Get mutable access to the unified storage.
    pub fn storage_mut(&mut self) -> &mut UnifiedStorage {
        &mut self.storage
    }

    /// Get access to the entities storage.
    pub fn entities(&self) -> &EntityStorage {
        &self.storage.entities
    }

    /// Get access to the entities storage mutably.
    pub fn entities_mut(&mut self) -> &mut EntityStorage {
        &mut self.storage.entities
    }

    /// Get access to the member signals buffer.
    pub fn member_signals(&self) -> &MemberSignalBuffer {
        &self.storage.member_signals
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
            .execute(&mut self.storage.member_signals, &self.storage.entities, self.sim_time)
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
    ///
    /// Configure phase is primarily engine-internal. It:
    /// 1. Creates the tick context (tick number, sim_time, dt, era)
    /// 2. Executes signal `:initial(...)` blocks for stateful signals
    /// 3. Commits member signal initial values (current → previous buffers)
    /// 4. Validates all Resolve DAG signals are initialized
    ///
    /// Most simulation logic should use other phases (Collect, Resolve, Fracture, Measure).
    /// See docs/execution/phases.md § 1 for details.
    fn execute_configure_phase(&mut self, dt: Dt, strata_states: &IndexMap<StratumId, StratumState>) -> Result<()> {
        trace!("phase: configure");
        self.active_tick_ctx = Some(TickContext {
            tick: self.tick,
            sim_time: self.sim_time,
            dt,
            era: self.current_era.clone(),
        });
        
        // Execute Configure phase DAG (signal :initial(...) blocks)
        if self.has_bytecode() {
            self.bytecode_executor.execute_configure(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.bytecode_blocks,
                &self.storage.entities,
                &mut self.storage.member_signals,
                &mut self.input_channels,
            )?;
            
            // Initialize any remaining uninitialized signals with zero/default values
            // This ensures all signals have a prev value before the first Resolve phase
            use crate::dag::NodeKind;
            let era_dags = self.dags.get_era(&self.current_era)
                .ok_or_else(|| Error::UnknownEra { era: self.current_era.clone() })?;
            for dag in era_dags.for_phase(Phase::Resolve) {
                for level in &dag.levels {
                    for node in &level.nodes {
                        if let NodeKind::SignalResolve { signal, entity: None, .. } = &node.kind {
                            // Global signals must be initialized before first Resolve phase
                            if !self.storage.member_signals.has_global(&signal.to_string()) {
                                panic!("Signal '{}' in DAG not initialized before Resolve phase. This indicates a compiler/loader bug.", signal);
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
                &self.storage.member_signals,
                &self.storage.entities,
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
                &self.storage.entities,
                &self.storage.member_signals,
                &mut self.input_channels,
            )?;
        }
        // Apply pending impulses (procedural for now)
        let impulses = std::mem::take(&mut self.pending_impulses);
        for (handler_idx, payload) in impulses {
            let handler = &self.phase_executor.impulse_handlers[handler_idx];
            let mut ctx = ImpulseContext {
                signals: &self.storage.member_signals,
                entities: &self.storage.entities,
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
                &self.storage.entities,
                &mut self.storage.member_signals,
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
                &self.storage.entities,
                &mut self.storage.member_signals,
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
                &self.storage.member_signals,
                &self.storage.entities,
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
                &mut self.storage.entities,
                &self.storage.member_signals,
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
                &self.storage.member_signals,
                &self.storage.entities,
                &mut self.field_buffer,
            )?;

            self.phase_executor.execute_chronicles(
                &self.current_era,
                self.tick,
                dt,
                self.sim_time,
                strata_states,
                &self.dags,
                &self.storage.member_signals,
                &self.storage.entities,
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
                &self.storage.entities,
                &self.storage.member_signals,
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
                &self.storage.entities,
                &self.storage.member_signals,
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
        self.storage.advance_tick();
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
        let era_config = self.eras.get(&self.current_era)
            .ok_or_else(|| Error::EraNotFound(self.current_era.clone()))?;
        if let Some(ref transition) = era_config.transition
            && let Some(next_era) = transition(&self.storage.member_signals, &self.storage.entities, self.sim_time)
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

}
