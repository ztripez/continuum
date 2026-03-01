//! Bytecode phase executor — orchestrates compiled DSL execution per phase.
//!
//! This module contains [`BytecodePhaseExecutor`], which drives the bytecode
//! VM through each simulation phase (Configure, Collect, Resolve, Fracture,
//! Measure, Chronicle).
//!
//! The VM execution context types ([`VMContext`], [`EntityContext`]) and the
//! [`ExecutionContext`] trait implementation live in [`super::bytecode_vm`].

use crate::bytecode::runtime::ExecutionError;
use crate::bytecode::{BytecodeExecutor, CompiledBlock};
use crate::dag::{DagSet, NodeKind};
use crate::error::{Error, Result};
use crate::executor::assertions::AssertionChecker;
use crate::executor::bytecode_vm::{EntityContext, VMContext};
use crate::soa_storage::MemberSignalBuffer;
use crate::storage::{EntityStorage, EventBuffer, FieldBuffer, FractureQueue, InputChannels};
use crate::types::{Dt, EraId, Phase, SignalId, StratumId, StratumState, Value};
use continuum_cdsl::foundation::Type;
use continuum_foundation::Path;
use indexmap::IndexMap;
use tracing::instrument;

/// Borrowed view of the immutable configuration fields from [`BytecodePhaseExecutor`].
///
/// This exists to solve a borrow-checker limitation: phase methods need to pass
/// immutable config references into [`VMContext`] while simultaneously borrowing
/// `&mut self.executor`. By splitting the struct into `(VMConfig, &mut BytecodeExecutor)`
/// via [`BytecodePhaseExecutor::parts`], Rust can see the borrows don't conflict.
pub(crate) struct VMConfig<'a> {
    /// World configuration values loaded from config {} blocks.
    pub config_values: &'a IndexMap<Path, Value>,
    /// Global simulation constants loaded from const {} blocks.
    pub const_values: &'a IndexMap<Path, Value>,
    /// Signal types for zero value initialization.
    pub signal_types: &'a IndexMap<SignalId, Type>,
    /// Spatial topologies for entity neighbor queries.
    pub topologies: &'a crate::topology::TopologyStorage,
}

/// Executes individual simulation phases using the bytecode VM.
pub struct BytecodePhaseExecutor {
    /// The underlying bytecode interpreter.
    executor: BytecodeExecutor,
    /// World configuration values loaded from config {} blocks.
    /// Frozen at world initialization, immutable during execution.
    config_values: IndexMap<Path, Value>,
    /// Global simulation constants loaded from const {} blocks.
    /// Frozen at world initialization, immutable during execution.
    const_values: IndexMap<Path, Value>,
    /// Signal types for zero value initialization.
    /// Used to create correct zero values for inputs accumulator.
    signal_types: IndexMap<SignalId, Type>,
    /// Spatial topologies for entity neighbor queries.
    /// Frozen in Configure phase, immutable during execution.
    topologies: crate::topology::TopologyStorage,
}

impl Default for BytecodePhaseExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl BytecodePhaseExecutor {
    /// Create a new bytecode phase executor.
    pub fn new() -> Self {
        Self {
            executor: BytecodeExecutor::new(),
            config_values: IndexMap::new(),
            const_values: IndexMap::new(),
            signal_types: IndexMap::new(),
            topologies: crate::topology::TopologyStorage::new(),
        }
    }

    /// Split self into immutable config references and a mutable executor.
    ///
    /// This is the classic Rust "split borrow" pattern. Phase methods need to
    /// pass `&self.config_values` (etc.) into `VMContext` while calling
    /// `self.executor.execute(&mut ctx)`. A naive `self.make_context()` borrows
    /// all of `self`, conflicting with `&mut self.executor`. Destructuring into
    /// `(VMConfig, &mut BytecodeExecutor)` proves to the borrow checker that the
    /// two borrows are disjoint.
    fn parts(&mut self) -> (VMConfig<'_>, &mut BytecodeExecutor) {
        (
            VMConfig {
                config_values: &self.config_values,
                const_values: &self.const_values,
                signal_types: &self.signal_types,
                topologies: &self.topologies,
            },
            &mut self.executor,
        )
    }

    /// Stores configuration values for access during execution.
    ///
    /// Config values are world-level defaults that can be overridden by scenarios.
    /// They are loaded once during `build_runtime()` and remain frozen throughout
    /// all execution phases.
    ///
    /// # Storage
    ///
    /// Values are stored in `self.config_values` HashMap and passed to all `VMContext`
    /// instances via reference. This allows `LoadConfig` opcodes to retrieve values
    /// without copying.
    ///
    /// # Immutability
    ///
    /// Once set, config values cannot be modified. Operators may read them via
    /// `ctx.load_config()` but cannot write to them (enforcing the frozen parameter
    /// model).
    pub fn set_config_values(&mut self, values: IndexMap<Path, Value>) {
        self.config_values = values;
    }

    /// Stores constant values for access during execution.
    ///
    /// Const values are world-level immutable globals that are NOT scenario-overridable.
    /// They are loaded once during `build_runtime()` and remain frozen throughout
    /// all execution phases.
    ///
    /// # Storage
    ///
    /// Values are stored in `self.const_values` HashMap and passed to all `VMContext`
    /// instances via reference. This allows `LoadConst` opcodes to retrieve values
    /// without copying.
    ///
    /// # Immutability
    ///
    /// Once set, const values cannot be modified. Operators may read them via
    /// `ctx.load_const()` but cannot write to them (enforcing the frozen parameter
    /// model).
    pub fn set_const_values(&mut self, values: IndexMap<Path, Value>) {
        self.const_values = values;
    }

    /// Stores signal type information for zero value initialization.
    ///
    /// Signal types are used to create correct zero values when no inputs
    /// have been accumulated for a signal (e.g., Vec3([0.0, 0.0, 0.0]) instead
    /// of Scalar(0.0) for vector signals).
    pub fn set_signal_types(&mut self, types: IndexMap<SignalId, Type>) {
        self.signal_types = types;
    }

    /// Initializes spatial topologies from entity topology expressions.
    ///
    /// Topologies are frozen at initialization (Configure phase) and remain
    /// immutable during execution. This ensures deterministic neighbor queries.
    ///
    /// # Parameters
    ///
    /// * `entities` - Entity declarations with optional topology expressions (from World.entities)
    ///
    /// # Panics
    ///
    /// Panics if topology construction fails (e.g., invalid subdivisions).
    pub fn initialize_topologies(
        &mut self,
        entities: &IndexMap<Path, continuum_cdsl::ast::Entity>,
    ) {
        use continuum_cdsl::ast::TopologyExpr;

        for (_path, entity) in entities {
            if let Some(ref topology_expr) = entity.topology {
                match topology_expr {
                    TopologyExpr::IcosahedronGrid { subdivisions, .. } => {
                        let topology =
                            crate::topology::icosahedron::IcosahedralTopology::new(*subdivisions);
                        self.topologies
                            .register(entity.id.clone(), std::sync::Arc::new(topology));
                    }
                }
            }
        }
    }

    /// Execute the Configure phase (signal :initial(...) blocks)
    ///
    /// Configure phase is engine-internal and executes signal initialization blocks.
    /// Most user DSL logic belongs in other phases (Collect, Resolve, Fracture, Measure).
    ///
    /// See docs/execution/phases.md § 1 for semantics and usage guidance.
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all, name = "configure")]
    pub fn execute_configure(
        &mut self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        sim_time: f64,
        strata_states: &IndexMap<StratumId, StratumState>,
        dags: &DagSet,
        compiled_blocks: &[CompiledBlock],
        entities: &EntityStorage,
        member_signals: &mut MemberSignalBuffer,
        input_channels: &mut InputChannels,
    ) -> Result<()> {
        let era_dags = dags
            .get_era(era)
            .ok_or_else(|| Error::UnknownEra { era: era.clone() })?;

        let (config, executor) = self.parts();
        let mut placeholder_field_buffer = FieldBuffer::default();
        for dag in era_dags.for_phase(Phase::Configure) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or_else(|| panic!("stratum {:?} not found in strata_states", dag.stratum));

            if !stratum_state.is_eligible(tick) {
                continue;
            }

            for level in &dag.levels {
                let mut level_results = Vec::new();

                for node in &level.nodes {
                    match &node.kind {
                        NodeKind::SignalResolve {
                            signal,
                            resolver_idx,
                            entity: None,
                        } => {
                            // Global signal (root entity, count=1)
                            let compiled = compiled_blocks.get(*resolver_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for signal configure {}",
                                        resolver_idx
                                    ),
                                }
                            })?;

                            let mut placeholder_fracture_queue = FractureQueue::default();
                            let mut placeholder_event_buffer = EventBuffer::default();
                            let mut ctx = VMContext::new(
                                &config,
                                Phase::Configure,
                                era,
                                dt,
                                sim_time,
                                entities,
                                member_signals,
                                input_channels,
                                &mut placeholder_fracture_queue,
                                &mut placeholder_field_buffer,
                                &mut placeholder_event_buffer,
                                Some(signal.clone()),
                                None,
                                None,
                                None,
                            );

                            let block_id = compiled.root;

                            // Diagnostic: dump bytecode structure (trace-level to avoid hot-path overhead)
                            tracing::trace!(
                                "Executing Configure block for signal '{}': block_id={:?}, slot_count={}, instructions={}",
                                signal,
                                block_id,
                                compiled.slot_count,
                                compiled.program.block(block_id).map(|b| b.instructions.len()).unwrap_or(0),
                            );
                            if let Some(block) = compiled
                                .program
                                .block(block_id)
                                .filter(|_| tracing::enabled!(tracing::Level::TRACE))
                            {
                                for (i, instr) in block.instructions.iter().enumerate() {
                                    tracing::trace!("  [{}] {:?}", i, instr);
                                }
                            }

                            let value = executor
                                .execute(compiled, &mut ctx)
                                .map_err(|e| Error::ExecutionFailure {
                                    message: format!("signal '{}': {}", signal, e),
                                })?
                                .ok_or_else(|| Error::ExecutionFailure {
                                    message: format!(
                                        "Signal configure for '{}' returned no value",
                                        signal
                                    ),
                                })?;

                            level_results.push((signal.clone(), value));
                        }
                        NodeKind::SignalResolve {
                            resolver_idx,
                            entity: Some(entity_ctx),
                            ..
                        } => {
                            // Member signal (child entity, count=N)
                            let member_signal = &entity_ctx.member_signal;
                            let compiled = compiled_blocks.get(*resolver_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for member signal configure {}",
                                        resolver_idx
                                    ),
                                }
                            })?;

                            // Get entity instance count
                            let full_signal_path = format!(
                                "{}.{}",
                                member_signal.entity_id, member_signal.signal_name
                            );
                            let instance_count =
                                member_signals.instance_count_for_signal(&full_signal_path);

                            tracing::debug!(
                                "Executing Configure for member signal '{}' ({} instances)",
                                full_signal_path,
                                instance_count
                            );

                            // Execute for each entity instance
                            for instance_idx in 0..instance_count {
                                let vm_entity_ctx = EntityContext {
                                    entity_id: &member_signal.entity_id,
                                    instance_index: instance_idx,
                                    target_member: &Path::from(full_signal_path.clone()),
                                };

                                let mut placeholder_fracture_queue = FractureQueue::default();
                                let mut placeholder_event_buffer = EventBuffer::default();
                                let mut ctx = VMContext::new(
                                    &config,
                                    Phase::Configure,
                                    era,
                                    dt,
                                    sim_time,
                                    entities,
                                    member_signals,
                                    input_channels,
                                    &mut placeholder_fracture_queue,
                                    &mut placeholder_field_buffer,
                                    &mut placeholder_event_buffer,
                                    None,
                                    None,
                                    None,
                                    Some(vm_entity_ctx),
                                );

                                let value = executor
                                    .execute(compiled, &mut ctx)
                                    .map_err(|e| Error::ExecutionFailure {
                                        message: format!(
                                            "member signal '{}' instance {}: {}",
                                            full_signal_path, instance_idx, e
                                        ),
                                    })?
                                    .ok_or_else(|| Error::ExecutionFailure {
                                        message: format!(
                                            "Member signal configure for '{}' instance {} returned no value",
                                            full_signal_path, instance_idx
                                        ),
                                    })?;

                                // Store in member signal buffer
                                member_signals
                                    .set_current(&full_signal_path, instance_idx, value)
                                    .map_err(|e| Error::ExecutionFailure {
                                        message: format!(
                                            "Failed to store member signal '{}' instance {}: {}",
                                            full_signal_path, instance_idx, e
                                        ),
                                    })?;
                            }
                        }
                        NodeKind::OperatorCollect { .. }
                        | NodeKind::OperatorMeasure { .. }
                        | NodeKind::FieldEmit { .. }
                        | NodeKind::Fracture { .. }
                        | NodeKind::ChronicleObserve { .. }
                        | NodeKind::PopulationAggregate { .. } => {
                            return Err(Error::ExecutionFailure {
                                message: format!(
                                    "Unexpected node kind in Configure phase: {:?}",
                                    node.kind
                                ),
                            });
                        }
                    }
                }

                // Commit results to global signal storage (sets current AND prev)
                // Using init_global() to set both previous and current for first-time initialization
                for (signal, value) in level_results {
                    member_signals
                        .init_global(&signal.to_string(), value)
                        .map_err(|e| Error::ExecutionFailure {
                            message: format!("failed to init global signal '{}': {}", signal, e),
                        })?;
                }
            }
        }

        Ok(())
    }

    /// Execute the Collect phase
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all, name = "collect")]
    pub fn execute_collect(
        &mut self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        sim_time: f64,
        strata_states: &IndexMap<StratumId, StratumState>,
        dags: &DagSet,
        compiled_blocks: &[CompiledBlock],
        entities: &EntityStorage,
        member_signals: &MemberSignalBuffer,
        input_channels: &mut InputChannels,
    ) -> Result<()> {
        let era_dags = dags
            .get_era(era)
            .ok_or_else(|| Error::UnknownEra { era: era.clone() })?;

        let (config, executor) = self.parts();
        let mut placeholder_field_buffer = FieldBuffer::default();
        for dag in era_dags.for_phase(Phase::Collect) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or_else(|| panic!("stratum {:?} not found in strata_states", dag.stratum));

            if !stratum_state.is_eligible(tick) {
                continue;
            }

            for level in &dag.levels {
                for node in &level.nodes {
                    if let NodeKind::OperatorCollect { operator_idx } = &node.kind {
                        let compiled = compiled_blocks.get(*operator_idx).ok_or_else(|| {
                            Error::ExecutionFailure {
                                message: format!(
                                    "Missing bytecode block for collect operator {}",
                                    operator_idx
                                ),
                            }
                        })?;
                        let mut placeholder_fracture_queue = FractureQueue::default();
                        let mut placeholder_event_buffer = EventBuffer::default();
                        let mut ctx = VMContext::new(
                            &config,
                            Phase::Collect,
                            era,
                            dt,
                            sim_time,
                            entities,
                            member_signals,
                            input_channels,
                            &mut placeholder_fracture_queue,
                            &mut placeholder_field_buffer,
                            &mut placeholder_event_buffer,
                            None,
                            None,
                            None,
                            None,
                        );

                        match executor.execute(compiled, &mut ctx) {
                            Ok(_) => {}
                            Err(ExecutionError::InvalidOperand { message })
                                if message.contains("no impulse payload is available") =>
                            {
                                // This is an impulse collect block executing without a scheduled impulse.
                                // Skip it silently - impulses only fire when explicitly triggered.
                                tracing::trace!(
                                    "Skipping impulse collect block (no scheduled impulse)"
                                );
                            }
                            Err(e) => {
                                return Err(Error::ExecutionFailure {
                                    message: e.to_string(),
                                });
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute the Resolve phase
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all, name = "resolve")]
    pub fn execute_resolve(
        &mut self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        sim_time: f64,
        strata_states: &IndexMap<StratumId, StratumState>,
        dags: &DagSet,
        compiled_blocks: &[CompiledBlock],
        entities: &EntityStorage,
        member_signals: &mut MemberSignalBuffer,
        input_channels: &mut InputChannels,
        assertion_checker: &mut AssertionChecker,
        breakpoints: &std::collections::HashSet<SignalId>,
    ) -> Result<Option<SignalId>> {
        let era_dags = dags
            .get_era(era)
            .ok_or_else(|| Error::UnknownEra { era: era.clone() })?;

        let (config, executor) = self.parts();
        let mut placeholder_field_buffer = FieldBuffer::default();
        for dag in era_dags.for_phase(Phase::Resolve) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or_else(|| panic!("stratum {:?} not found in strata_states", dag.stratum));

            if !stratum_state.is_eligible(tick) {
                continue;
            }

            for level in &dag.levels {
                let mut level_results = Vec::new();

                for node in &level.nodes {
                    match &node.kind {
                        NodeKind::SignalResolve {
                            signal,
                            resolver_idx,
                            entity: None,
                        } => {
                            // Global signal (root entity, count=1)
                            let compiled = compiled_blocks.get(*resolver_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for signal resolve {}",
                                        resolver_idx
                                    ),
                                }
                            })?;

                            let mut placeholder_fracture_queue = FractureQueue::default();
                            let mut placeholder_event_buffer = EventBuffer::default();
                            let mut ctx = VMContext::new(
                                &config,
                                Phase::Resolve,
                                era,
                                dt,
                                sim_time,
                                entities,
                                member_signals,
                                input_channels,
                                &mut placeholder_fracture_queue,
                                &mut placeholder_field_buffer,
                                &mut placeholder_event_buffer,
                                Some(signal.clone()),
                                None,
                                None,
                                None,
                            );

                            let block_id = compiled.root;

                            // Diagnostic: dump bytecode structure (trace-level to avoid hot-path overhead)
                            tracing::trace!(
                                "Executing Resolve block for signal '{}': block_id={:?}, slot_count={}, instructions={}",
                                signal,
                                block_id,
                                compiled.slot_count,
                                compiled.program.block(block_id).map(|b| b.instructions.len()).unwrap_or(0),
                            );
                            if let Some(block) = compiled
                                .program
                                .block(block_id)
                                .filter(|_| tracing::enabled!(tracing::Level::TRACE))
                            {
                                for (i, instr) in block.instructions.iter().enumerate() {
                                    tracing::trace!("  [{}] {:?}", i, instr);
                                }
                            }

                            let value = executor
                                .execute(compiled, &mut ctx)
                                .map_err(|e| Error::ExecutionFailure {
                                    message: format!("signal '{}': {}", signal, e),
                                })?
                                .ok_or_else(|| Error::ExecutionFailure {
                                    message: format!(
                                        "Signal resolve for '{}' returned no value",
                                        signal
                                    ),
                                })?;

                            level_results.push((signal.clone(), value));
                        }
                        NodeKind::SignalResolve {
                            resolver_idx,
                            entity: Some(entity_ctx),
                            ..
                        } => {
                            // Member signal (child entity, count=N)
                            let member_signal = &entity_ctx.member_signal;
                            let compiled = compiled_blocks.get(*resolver_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for member signal resolve {}",
                                        resolver_idx
                                    ),
                                }
                            })?;

                            // Construct full signal path
                            let full_signal_path = format!(
                                "{}.{}",
                                member_signal.entity_id, member_signal.signal_name
                            );

                            // Get instance count for this entity
                            let instance_count =
                                member_signals.instance_count_for_signal(&full_signal_path);

                            tracing::debug!(
                                "Resolving member signal '{}' for {} instances",
                                full_signal_path,
                                instance_count
                            );

                            // Execute bytecode for each entity instance
                            for instance_idx in 0..instance_count {
                                // Set up entity context for this instance
                                let vm_entity_ctx = EntityContext {
                                    entity_id: &member_signal.entity_id,
                                    instance_index: instance_idx,
                                    target_member: &Path::from(full_signal_path.clone()),
                                };

                                let mut placeholder_fracture_queue = FractureQueue::default();
                                let mut placeholder_event_buffer = EventBuffer::default();
                                let mut ctx = VMContext::new(
                                    &config,
                                    Phase::Resolve,
                                    era,
                                    dt,
                                    sim_time,
                                    entities,
                                    member_signals,
                                    input_channels,
                                    &mut placeholder_fracture_queue,
                                    &mut placeholder_field_buffer,
                                    &mut placeholder_event_buffer,
                                    None, // Member signals don't use target_signal
                                    None,
                                    None,
                                    Some(vm_entity_ctx),
                                );

                                // Execute bytecode with entity context
                                let value = executor
                                    .execute(compiled, &mut ctx)
                                    .map_err(|e| Error::ExecutionFailure {
                                        message: format!(
                                            "member signal '{}' instance {}: {}",
                                            full_signal_path, instance_idx, e
                                        ),
                                    })?
                                    .ok_or_else(|| Error::ExecutionFailure {
                                        message: format!(
                                            "Member signal resolve for '{}' instance {} returned no value",
                                            full_signal_path, instance_idx
                                        ),
                                    })?;

                                // Store result in member signal buffer at this instance
                                member_signals
                                    .set_current(&full_signal_path, instance_idx, value)
                                    .map_err(|e| Error::ExecutionFailure {
                                        message: format!(
                                            "Failed to store member signal '{}' instance {}: {}",
                                            full_signal_path, instance_idx, e
                                        ),
                                    })?;
                            }
                        }
                        NodeKind::OperatorCollect { .. }
                        | NodeKind::OperatorMeasure { .. }
                        | NodeKind::FieldEmit { .. }
                        | NodeKind::Fracture { .. }
                        | NodeKind::ChronicleObserve { .. }
                        | NodeKind::PopulationAggregate { .. } => {
                            return Err(Error::ExecutionFailure {
                                message: format!(
                                    "Unexpected node kind in Resolve phase: {:?}",
                                    node.kind
                                ),
                            });
                        }
                    }
                }

                // Commit results and run assertions
                for (signal, value) in level_results {
                    if breakpoints.contains(&signal) {
                        return Ok(Some(signal));
                    }

                    let prev = member_signals
                        .get_global_prev(&signal.to_string())
                        .ok_or_else(|| Error::ExecutionFailure {
                            message: format!(
                                "Signal '{}' has no previous value for history read",
                                signal
                            ),
                        })?;
                    member_signals
                        .set_global(&signal.to_string(), value.clone())
                        .map_err(|e| Error::ExecutionFailure {
                            message: format!("failed to set global signal '{}': {}", signal, e),
                        })?;

                    assertion_checker.check_signal(
                        &signal,
                        &value,
                        &prev,
                        member_signals,
                        entities,
                        dt,
                        sim_time,
                        tick,
                        &era.to_string(),
                    )?;
                }
            }
        }
        Ok(None)
    }

    /// Execute the Fracture phase
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all, name = "fracture")]
    pub fn execute_fracture(
        &mut self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        sim_time: f64,
        strata_states: &IndexMap<StratumId, StratumState>,
        dags: &DagSet,
        compiled_blocks: &[CompiledBlock],
        entities: &mut EntityStorage,
        member_signals: &MemberSignalBuffer,
        fracture_queue: &mut FractureQueue,
    ) -> Result<()> {
        let era_dags = dags
            .get_era(era)
            .ok_or_else(|| Error::UnknownEra { era: era.clone() })?;

        let (config, executor) = self.parts();
        let mut placeholder_field_buffer = FieldBuffer::default();
        for dag in era_dags.for_phase(Phase::Fracture) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or_else(|| panic!("stratum {:?} not found in strata_states", dag.stratum));

            if !stratum_state.is_eligible(tick) {
                continue;
            }

            for level in &dag.levels {
                for node in &level.nodes {
                    if let NodeKind::Fracture { fracture_idx } = &node.kind {
                        let compiled = compiled_blocks.get(*fracture_idx).ok_or_else(|| {
                            Error::ExecutionFailure {
                                message: format!(
                                    "Missing bytecode block for fracture {}",
                                    fracture_idx
                                ),
                            }
                        })?;
                        let mut placeholder_input_channels = InputChannels::default();
                        let mut placeholder_event_buffer = EventBuffer::default();
                        let mut ctx = VMContext::new(
                            &config,
                            Phase::Fracture,
                            era,
                            dt,
                            sim_time,
                            entities,
                            member_signals,
                            &mut placeholder_input_channels,
                            fracture_queue,
                            &mut placeholder_field_buffer,
                            &mut placeholder_event_buffer,
                            None, // Fracture nodes currently lack single-signal target context
                            None,
                            None,
                            None,
                        );
                        executor.execute(compiled, &mut ctx).map_err(|e| {
                            Error::ExecutionFailure {
                                message: e.to_string(),
                            }
                        })?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute the Measure phase
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all, name = "measure")]
    pub fn execute_measure(
        &mut self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        sim_time: f64,
        strata_states: &IndexMap<StratumId, StratumState>,
        dags: &DagSet,
        compiled_blocks: &[CompiledBlock],
        entities: &EntityStorage,
        member_signals: &MemberSignalBuffer,
        field_buffer: &mut FieldBuffer,
    ) -> Result<()> {
        let era_dags = dags
            .get_era(era)
            .ok_or_else(|| Error::UnknownEra { era: era.clone() })?;

        let (config, executor) = self.parts();
        for dag in era_dags.for_phase(Phase::Measure) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or_else(|| panic!("stratum {:?} not found in strata_states", dag.stratum));

            if !stratum_state.is_eligible(tick) {
                continue;
            }

            for level in &dag.levels {
                for node in &level.nodes {
                    match &node.kind {
                        NodeKind::OperatorMeasure { operator_idx } => {
                            let compiled = compiled_blocks.get(*operator_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for measure operator {}",
                                        operator_idx
                                    ),
                                }
                            })?;
                            let mut inner_input_channels = InputChannels::default();
                            let mut placeholder_fracture_queue = FractureQueue::default();
                            let mut placeholder_event_buffer = EventBuffer::default();
                            let mut ctx = VMContext::new(
                                &config,
                                Phase::Measure,
                                era,
                                dt,
                                sim_time,
                                entities,
                                member_signals,
                                &mut inner_input_channels,
                                &mut placeholder_fracture_queue,
                                field_buffer,
                                &mut placeholder_event_buffer,
                                None,
                                None,
                                None,
                                None,
                            );

                            executor.execute(compiled, &mut ctx).map_err(|e| {
                                Error::ExecutionFailure {
                                    message: e.to_string(),
                                }
                            })?;
                        }
                        NodeKind::FieldEmit {
                            field_idx,
                            field_id,
                        } => {
                            let compiled = compiled_blocks.get(*field_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for field emit {}",
                                        field_idx
                                    ),
                                }
                            })?;
                            let mut inner_input_channels = InputChannels::default();
                            let mut placeholder_fracture_queue = FractureQueue::default();
                            let mut placeholder_event_buffer = EventBuffer::default();
                            let mut ctx = VMContext::new(
                                &config,
                                Phase::Measure,
                                era,
                                dt,
                                sim_time,
                                entities,
                                member_signals,
                                &mut inner_input_channels,
                                &mut placeholder_fracture_queue,
                                field_buffer,
                                &mut placeholder_event_buffer,
                                None,
                                None,
                                None,
                                None,
                            );

                            let result = executor.execute(compiled, &mut ctx).map_err(|e| {
                                Error::ExecutionFailure {
                                    message: e.to_string(),
                                }
                            })?;

                            // Expression-body field measure blocks return a value
                            // but don't contain explicit EmitField opcodes. Capture
                            // the return value and emit it to the field buffer.
                            if let Some(value) = result {
                                if let Some(scalar) = value.as_scalar() {
                                    field_buffer.emit_scalar(field_id.clone(), scalar);
                                } else {
                                    field_buffer.emit(field_id.clone(), [0.0, 0.0, 0.0], value);
                                }
                            }
                        }
                        NodeKind::SignalResolve { .. }
                        | NodeKind::OperatorCollect { .. }
                        | NodeKind::Fracture { .. }
                        | NodeKind::ChronicleObserve { .. }
                        | NodeKind::PopulationAggregate { .. } => {
                            return Err(Error::ExecutionFailure {
                                message: format!(
                                    "Unexpected node kind in Measure phase: {:?}",
                                    node.kind
                                ),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute the Chronicle phase
    #[allow(clippy::too_many_arguments)]
    #[instrument(skip_all, name = "chronicle")]
    pub fn execute_chronicles(
        &mut self,
        era: &EraId,
        tick: u64,
        dt: Dt,
        sim_time: f64,
        strata_states: &IndexMap<StratumId, StratumState>,
        dags: &DagSet,
        compiled_blocks: &[CompiledBlock],
        entities: &EntityStorage,
        member_signals: &MemberSignalBuffer,
        event_buffer: &mut EventBuffer,
    ) -> Result<()> {
        let era_dags = dags
            .get_era(era)
            .ok_or_else(|| Error::UnknownEra { era: era.clone() })?;

        let (config, executor) = self.parts();
        let mut placeholder_field_buffer = FieldBuffer::default();
        for dag in era_dags.for_phase(Phase::Measure) {
            let stratum_state = strata_states
                .get(&dag.stratum)
                .copied()
                .unwrap_or_else(|| panic!("stratum {:?} not found in strata_states", dag.stratum));

            if !stratum_state.is_eligible(tick) {
                continue;
            }

            for level in &dag.levels {
                for node in &level.nodes {
                    match &node.kind {
                        NodeKind::ChronicleObserve { chronicle_idx } => {
                            let compiled =
                                compiled_blocks.get(*chronicle_idx).ok_or_else(|| {
                                    Error::ExecutionFailure {
                                        message: format!(
                                            "Missing bytecode block for chronicle {}",
                                            chronicle_idx
                                        ),
                                    }
                                })?;
                            let mut input_channels = InputChannels::default();
                            let mut placeholder_fracture_queue = FractureQueue::default();
                            // Chronicles receive the real event_buffer (not placeholder)
                            let mut ctx = VMContext::new(
                                &config,
                                Phase::Measure,
                                era,
                                dt,
                                sim_time,
                                entities,
                                member_signals,
                                &mut input_channels,
                                &mut placeholder_fracture_queue,
                                &mut placeholder_field_buffer,
                                event_buffer,
                                None,
                                None,
                                None,
                                None,
                            );

                            executor.execute(compiled, &mut ctx).map_err(|e| {
                                Error::ExecutionFailure {
                                    message: e.to_string(),
                                }
                            })?;
                        }
                        NodeKind::OperatorMeasure { .. } | NodeKind::FieldEmit { .. } => {
                            continue;
                        }
                        _ => {
                            return Err(Error::ExecutionFailure {
                                message: format!(
                                    "Unexpected node kind in chronicle phase {:?}",
                                    node.kind
                                ),
                            });
                        }
                    }
                }
            }
        }
        Ok(())
    }
}
