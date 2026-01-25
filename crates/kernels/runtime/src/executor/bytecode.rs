use crate::bytecode::runtime::{ExecutionContext, ExecutionError};
use crate::bytecode::{BytecodeExecutor, CompiledBlock};
use crate::dag::{DagSet, NodeKind};
use crate::error::{Error, Result};
use crate::executor::assertions::AssertionChecker;
use crate::reductions;
use crate::soa_storage::MemberSignalBuffer;
use crate::storage::{
    EntityStorage, EventBuffer, FieldBuffer, FractureQueue, InputChannels, SignalStorage,
};
use crate::types::{Dt, EraId, Phase, SignalId, StratumId, StratumState, Value};
use continuum_cdsl::foundation::{Shape, Type};
use continuum_foundation::{AggregateOp, EntityId, Mat2, Mat3, Mat4, Path, Quat};
use indexmap::IndexMap;
use tracing::instrument;

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
        }
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

    /// Execute the Configure phase (initial blocks)
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
        signals: &mut SignalStorage,
        entities: &EntityStorage,
        member_signals: &mut MemberSignalBuffer,
        input_channels: &mut InputChannels,
    ) -> Result<()> {
        let era_dags = dags.get_era(era).unwrap();

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
                        } => {
                            let compiled = compiled_blocks.get(*resolver_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for signal configure {}",
                                        resolver_idx
                                    ),
                                }
                            })?;

                            let mut placeholder_fracture_queue = FractureQueue::default();
                            let mut ctx = VMContext {
                                phase: Phase::Configure,
                                era,
                                dt,
                                sim_time,
                                signals,
                                entities,
                                member_signals,
                                channels: input_channels,
                                fracture_queue: &mut placeholder_fracture_queue,
                                field_buffer: &mut placeholder_field_buffer,
                                event_buffer: &mut EventBuffer::default(),
                                target_signal: Some(signal.clone()),
                                cached_inputs: None,
                                config_values: &self.config_values,
                                const_values: &self.const_values,
                                signal_types: &self.signal_types,
                                payload: None,
                                entity_context: None,
                            };

                            let block_id = compiled.root;

                            // Diagnostic: dump bytecode structure
                            tracing::debug!(
                                "Executing Configure block for signal '{}': block_id={:?}, slot_count={}, instructions={}",
                                signal,
                                block_id,
                                compiled.slot_count,
                                compiled.program.block(block_id).map(|b| b.instructions.len()).unwrap_or(0),
                            );
                            if let Some(block) = compiled.program.block(block_id) {
                                for (i, instr) in block.instructions.iter().enumerate() {
                                    tracing::debug!("  [{}] {:?}", i, instr);
                                }
                            }

                            let value = self
                                .executor
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
                        NodeKind::MemberSignalResolve {
                            member_signal,
                            kernel_idx,
                        } => {
                            let compiled = compiled_blocks.get(*kernel_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for member signal configure {}",
                                        kernel_idx
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
                                let entity_ctx = EntityContext {
                                    entity_id: &member_signal.entity_id,
                                    instance_index: instance_idx,
                                    target_member: &Path::from(full_signal_path.clone()),
                                };

                                let mut placeholder_fracture_queue = FractureQueue::default();
                                let mut ctx = VMContext {
                                    phase: Phase::Configure,
                                    era,
                                    dt,
                                    sim_time,
                                    signals,
                                    entities,
                                    member_signals,
                                    channels: input_channels,
                                    fracture_queue: &mut placeholder_fracture_queue,
                                    field_buffer: &mut placeholder_field_buffer,
                                    event_buffer: &mut EventBuffer::default(),
                                    target_signal: None,
                                    cached_inputs: None,
                                    config_values: &self.config_values,
                                    const_values: &self.const_values,
                                    signal_types: &self.signal_types,
                                    payload: None,
                                    entity_context: Some(entity_ctx),
                                };

                                let value = self
                                    .executor
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
                        _ => {}
                    }
                }

                // Commit results to signal storage (sets current AND prev)
                // Using init() to set both previous and current for first-time initialization
                for (signal, value) in level_results {
                    signals.init(signal, value);
                }
            }
        }

        Ok(())
    }

    /// Execute the Collect phase
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
        signals: &SignalStorage,
        entities: &EntityStorage,
        member_signals: &MemberSignalBuffer,
        input_channels: &mut InputChannels,
    ) -> Result<()> {
        let era_dags = dags.get_era(era).unwrap();

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
                        let mut ctx = VMContext {
                            phase: Phase::Collect,
                            era,
                            dt,
                            sim_time,
                            signals,
                            entities,
                            member_signals,
                            channels: input_channels,
                            fracture_queue: &mut placeholder_fracture_queue,
                            field_buffer: &mut placeholder_field_buffer,
                            event_buffer: &mut EventBuffer::default(),
                            target_signal: None,
                            cached_inputs: None,
                            config_values: &self.config_values,
                            const_values: &self.const_values,
                            signal_types: &self.signal_types,
                            payload: None,
                            entity_context: None,
                        };

                        match self.executor.execute(compiled, &mut ctx) {
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
        signals: &mut SignalStorage,
        entities: &EntityStorage,
        member_signals: &mut MemberSignalBuffer,
        input_channels: &mut InputChannels,
        assertion_checker: &mut AssertionChecker,
        breakpoints: &std::collections::HashSet<SignalId>,
    ) -> Result<Option<SignalId>> {
        let era_dags = dags.get_era(era).unwrap();

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
                        } => {
                            let compiled = compiled_blocks.get(*resolver_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for signal resolve {}",
                                        resolver_idx
                                    ),
                                }
                            })?;

                            let mut placeholder_fracture_queue = FractureQueue::default();
                            let mut ctx = VMContext {
                                phase: Phase::Resolve,
                                era,
                                dt,
                                sim_time,
                                signals,
                                entities,
                                member_signals,
                                channels: input_channels,
                                fracture_queue: &mut placeholder_fracture_queue,
                                field_buffer: &mut placeholder_field_buffer,
                                event_buffer: &mut EventBuffer::default(),
                                target_signal: Some(signal.clone()),
                                cached_inputs: None,
                                config_values: &self.config_values,
                                const_values: &self.const_values,
                                signal_types: &self.signal_types,
                                payload: None,
                                entity_context: None,
                            };

                            let block_id = compiled.root;

                            // Diagnostic: dump bytecode structure
                            tracing::debug!(
                                "Executing Resolve block for signal '{}': block_id={:?}, slot_count={}, instructions={}",
                                signal,
                                block_id,
                                compiled.slot_count,
                                compiled.program.block(block_id).map(|b| b.instructions.len()).unwrap_or(0),
                            );
                            if let Some(block) = compiled.program.block(block_id) {
                                for (i, instr) in block.instructions.iter().enumerate() {
                                    tracing::debug!("  [{}] {:?}", i, instr);
                                }
                            }

                            let value = self
                                .executor
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
                        NodeKind::MemberSignalResolve {
                            member_signal,
                            kernel_idx,
                        } => {
                            let compiled = compiled_blocks.get(*kernel_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for member signal resolve {}",
                                        kernel_idx
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
                                let entity_ctx = EntityContext {
                                    entity_id: &member_signal.entity_id,
                                    instance_index: instance_idx,
                                    target_member: &Path::from(full_signal_path.clone()),
                                };

                                let mut placeholder_fracture_queue = FractureQueue::default();
                                let mut ctx = VMContext {
                                    phase: Phase::Resolve,
                                    era,
                                    dt,
                                    sim_time,
                                    signals,
                                    entities,
                                    member_signals,
                                    channels: input_channels,
                                    fracture_queue: &mut placeholder_fracture_queue,
                                    field_buffer: &mut placeholder_field_buffer,
                                    event_buffer: &mut EventBuffer::default(),
                                    target_signal: None, // Member signals don't use target_signal
                                    cached_inputs: None,
                                    config_values: &self.config_values,
                                    const_values: &self.const_values,
                                    signal_types: &self.signal_types,
                                    payload: None,
                                    entity_context: Some(entity_ctx),
                                };

                                // Execute bytecode with entity context
                                let value = self
                                    .executor
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
                        _ => {}
                    }
                }

                // Commit results and run assertions
                for (signal, value) in level_results {
                    if breakpoints.contains(&signal) {
                        return Ok(Some(signal));
                    }

                    let prev = signals
                        .get_prev(&signal)
                        .ok_or_else(|| Error::ExecutionFailure {
                            message: format!(
                                "Signal '{}' has no previous value for history read",
                                signal
                            ),
                        })?
                        .clone();
                    signals.set_current(signal.clone(), value.clone());

                    assertion_checker.check_signal(
                        &signal,
                        &value,
                        &prev,
                        signals,
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
        signals: &SignalStorage,
        entities: &mut EntityStorage,
        member_signals: &MemberSignalBuffer,
        fracture_queue: &mut FractureQueue,
    ) -> Result<()> {
        let era_dags = dags.get_era(era).unwrap();

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
                        let mut ctx = VMContext {
                            phase: Phase::Fracture,
                            era,
                            dt,
                            sim_time,
                            signals,
                            entities,
                            member_signals,
                            channels: &mut placeholder_input_channels,
                            fracture_queue,
                            field_buffer: &mut placeholder_field_buffer,
                            event_buffer: &mut EventBuffer::default(),
                            target_signal: None, // TODO: Fracture nodes currently lack single-signal target context
                            cached_inputs: None,
                            config_values: &self.config_values,
                            const_values: &self.const_values,
                            signal_types: &self.signal_types,
                            payload: None,
                            entity_context: None,
                        };
                        self.executor.execute(compiled, &mut ctx).map_err(|e| {
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
        signals: &SignalStorage,
        entities: &EntityStorage,
        member_signals: &MemberSignalBuffer,
        field_buffer: &mut FieldBuffer,
    ) -> Result<()> {
        let era_dags = dags.get_era(era).unwrap();

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
                        NodeKind::OperatorMeasure { operator_idx }
                        | NodeKind::FieldEmit {
                            field_idx: operator_idx,
                        } => {
                            let compiled = compiled_blocks.get(*operator_idx).ok_or_else(|| {
                                Error::ExecutionFailure {
                                    message: format!(
                                        "Missing bytecode block for measure {}",
                                        operator_idx
                                    ),
                                }
                            })?;
                            let mut inner_input_channels = InputChannels::default();
                            let mut placeholder_fracture_queue = FractureQueue::default();
                            let mut ctx = VMContext {
                                phase: Phase::Measure,
                                era,
                                dt,
                                sim_time,
                                signals,
                                entities,
                                member_signals,
                                channels: &mut inner_input_channels,
                                fracture_queue: &mut placeholder_fracture_queue,
                                field_buffer,
                                event_buffer: &mut EventBuffer::default(),
                                target_signal: None,
                                cached_inputs: None,
                                config_values: &self.config_values,
                                const_values: &self.const_values,
                                signal_types: &self.signal_types,
                                payload: None,
                                entity_context: None,
                            };

                            self.executor.execute(compiled, &mut ctx).map_err(|e| {
                                Error::ExecutionFailure {
                                    message: e.to_string(),
                                }
                            })?;
                        }
                        _ => {}
                    }
                }
            }
        }
        Ok(())
    }

    /// Execute the Chronicle phase
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
        signals: &SignalStorage,
        entities: &EntityStorage,
        member_signals: &MemberSignalBuffer,
        event_buffer: &mut EventBuffer,
    ) -> Result<()> {
        let era_dags = dags.get_era(era).unwrap();

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
                            let mut ctx = VMContext {
                                phase: Phase::Measure,
                                era,
                                dt,
                                sim_time,
                                signals,
                                entities,
                                member_signals,
                                channels: &mut input_channels,
                                fracture_queue: &mut placeholder_fracture_queue,
                                field_buffer: &mut placeholder_field_buffer,
                                event_buffer,
                                target_signal: None,
                                cached_inputs: None,
                                config_values: &self.config_values,
                                const_values: &self.const_values,
                                signal_types: &self.signal_types,
                                payload: None,
                                entity_context: None,
                            };

                            self.executor.execute(compiled, &mut ctx).map_err(|e| {
                                Error::ExecutionFailure {
                                    message: e.to_string(),
                                }
                            })?;
                            let _ = event_buffer;
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

/// Create a zero value for a given Type.
///
/// Returns the additive identity for the type:
/// - Scalar → 0.0
/// - Vec2 → [0.0, 0.0]
/// - Vec3 → [0.0, 0.0, 0.0]
/// - Vec4 → [0.0, 0.0, 0.0, 0.0]
/// - Bool → false
/// - Quat → identity quaternion (1.0, 0.0, 0.0, 0.0)
/// - Mat2/3/4 → zero matrices
///
/// For User types and other complex types, panics (not supported for inputs).
fn zero_value_for_type(ty: &Type) -> Value {
    match ty {
        Type::Kernel(kt) => match &kt.shape {
            Shape::Scalar => Value::Scalar(0.0),
            Shape::Vector { dim } => match dim {
                2 => Value::Vec2([0.0, 0.0]),
                3 => Value::Vec3([0.0, 0.0, 0.0]),
                4 => Value::Vec4([0.0, 0.0, 0.0, 0.0]),
                _ => panic!(
                    "Unsupported vector dimension for inputs zero value: {}",
                    dim
                ),
            },
            Shape::Matrix { rows, cols } => match (rows, cols) {
                (2, 2) => Value::Mat2(Mat2([0.0; 4])),
                (3, 3) => Value::Mat3(Mat3([0.0; 9])),
                (4, 4) => Value::Mat4(Mat4([0.0; 16])),
                _ => panic!(
                    "Unsupported matrix dimensions for inputs zero value: {}x{}",
                    rows, cols
                ),
            },
            _ => panic!("Unsupported shape for inputs zero value: {:?}", kt.shape),
        },
        Type::Bool => Value::Boolean(false),
        Type::User(_) | Type::String | Type::Unit | Type::Seq(_) => {
            panic!("Unsupported type for inputs zero value: {:?}", ty)
        }
    }
}

/// Implementation of the VM execution context that bridges to runtime storage.
pub struct VMContext<'a> {
    pub phase: Phase,
    pub era: &'a EraId,
    pub dt: Dt,
    pub sim_time: f64,
    pub signals: &'a SignalStorage,
    pub entities: &'a EntityStorage,
    pub member_signals: &'a MemberSignalBuffer,
    pub channels: &'a mut InputChannels,
    pub fracture_queue: &'a mut FractureQueue,
    pub field_buffer: &'a mut FieldBuffer,
    pub event_buffer: &'a mut EventBuffer,
    pub target_signal: Option<SignalId>,
    pub cached_inputs: Option<Value>,
    /// World configuration values loaded from config {} blocks
    pub config_values: &'a IndexMap<Path, Value>,
    /// Global simulation constants loaded from const {} blocks
    pub const_values: &'a IndexMap<Path, Value>,
    /// Signal types for zero value initialization
    pub signal_types: &'a IndexMap<SignalId, Type>,
    /// Current impulse payload (only valid when executing impulse collect blocks)
    pub payload: Option<&'a Value>,
    /// Current entity context for member signal execution
    /// When Some, enables LoadSelf/LoadOther opcodes for member signal access
    pub entity_context: Option<EntityContext<'a>>,
}

/// Entity execution context for member signal bytecode.
///
/// When executing member signal resolve/initial blocks, this provides:
/// - Entity ID (e.g., "hydrology.cell")
/// - Instance index (0..N for N instances)
/// - Access to member signal buffer for self/other member reads
#[derive(Debug, Clone, Copy)]
pub struct EntityContext<'a> {
    /// The entity ID (e.g., "hydrology.cell")
    pub entity_id: &'a EntityId,
    /// The instance index being executed (0..instance_count)
    pub instance_index: usize,
    /// The member signal path being resolved (e.g., "hydrology.cell.temperature")
    pub target_member: &'a Path,
}

impl<'a> ExecutionContext for VMContext<'a> {
    fn phase(&self) -> Phase {
        self.phase
    }

    fn load_signal(&self, path: &Path) -> std::result::Result<Value, ExecutionError> {
        self.signals
            .get(&SignalId::from(path.clone()))
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Signal not found: {}", path),
            })
    }

    fn load_field(&self, _path: &Path) -> std::result::Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadField not yet supported in VM".to_string(),
            phase: self.phase,
        })
    }

    fn load_config(&self, path: &Path) -> std::result::Result<Value, ExecutionError> {
        self.config_values
            .get(path)
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Config value not found: {}", path),
            })
    }

    fn load_const(&self, path: &Path) -> std::result::Result<Value, ExecutionError> {
        self.const_values
            .get(path)
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Const value not found: {}", path),
            })
    }

    fn load_prev(&self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Resolve && self.phase != Phase::Configure {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadPrev".to_string(),
                phase: self.phase,
            });
        }

        // Check if we're in entity context (member signal)
        if let Some(entity_ctx) = &self.entity_context {
            // Load from member signal buffer
            let full_path = entity_ctx.target_member.to_string();
            self.member_signals
                .get_previous(&full_path, entity_ctx.instance_index)
                .ok_or_else(|| ExecutionError::InvalidOperand {
                    message: format!(
                        "Member signal '{}' instance {} has no previous value",
                        full_path, entity_ctx.instance_index
                    ),
                })
        } else {
            // Load from global signal storage
            let signal =
                self.target_signal
                    .as_ref()
                    .ok_or_else(|| ExecutionError::InvalidOpcode {
                        opcode: "LoadPrev requires target signal or entity context".to_string(),
                        phase: self.phase,
                    })?;
            self.signals
                .get_prev(signal)
                .cloned()
                .ok_or_else(|| ExecutionError::InvalidOperand {
                    message: format!("Signal '{}' has no previous value", signal),
                })
        }
    }

    fn load_current(&self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Fracture && self.phase != Phase::Measure {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadCurrent".to_string(),
                phase: self.phase,
            });
        }

        // Check if we're in entity context (member signal)
        if let Some(entity_ctx) = &self.entity_context {
            // Load from member signal buffer
            let full_path = entity_ctx.target_member.to_string();
            self.member_signals
                .get_current(&full_path, entity_ctx.instance_index)
                .ok_or_else(|| ExecutionError::InvalidOperand {
                    message: format!(
                        "Member signal '{}' instance {} has no current value",
                        full_path, entity_ctx.instance_index
                    ),
                })
        } else {
            // Load from global signal storage
            let signal =
                self.target_signal
                    .as_ref()
                    .ok_or_else(|| ExecutionError::InvalidOpcode {
                        opcode: "LoadCurrent requires target signal or entity context".to_string(),
                        phase: self.phase,
                    })?;
            self.signals
                .get(signal)
                .cloned()
                .ok_or_else(|| ExecutionError::InvalidOperand {
                    message: format!("Signal '{}' has no resolved value", signal),
                })
        }
    }

    fn load_inputs(&mut self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Resolve {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadInputs".to_string(),
                phase: self.phase,
            });
        }

        // Check if we're in entity context (member signal)
        if let Some(entity_ctx) = &self.entity_context {
            // Member signals don't currently support input accumulation
            panic!(
                "Member signal input accumulation not implemented. Signal '{}' instance {} attempted to use LoadInputs. \
                This feature requires per-instance input channels.",
                entity_ctx.target_member,
                entity_ctx.instance_index
            );
        }

        // Global signal - load from input channels
        let signal = self
            .target_signal
            .as_ref()
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: "LoadInputs requires target signal or entity context".to_string(),
                phase: self.phase,
            })?;

        if let Some(cached) = &self.cached_inputs {
            return Ok(cached.clone());
        }

        let value = self.channels.drain_sum(signal);

        // Create the appropriate Value based on the signal type
        let result = if value == 0.0 {
            // If no inputs were accumulated, return typed zero
            if let Some(ty) = self.signal_types.get(signal) {
                zero_value_for_type(ty)
            } else {
                Value::Scalar(0.0)
            }
        } else {
            Value::Scalar(value)
        };

        self.cached_inputs = Some(result.clone());
        Ok(result)
    }

    fn load_dt(&self) -> std::result::Result<Value, ExecutionError> {
        Ok(Value::Scalar(self.dt.0))
    }

    fn load_self(&self) -> std::result::Result<Value, ExecutionError> {
        let _entity_ctx = self
            .entity_context
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: "LoadSelf requires entity context (member signal execution)".to_string(),
                phase: self.phase,
            })?;

        // LoadSelf returns a special marker value that signals to FieldAccess
        // that member signal access should be used.
        // We use a Map with a special "__entity_instance__" key set to 1.
        // This is a sentinel value that FieldAccess will recognize.
        Ok(Value::Map(std::sync::Arc::new(vec![(
            "__entity_instance__".to_string(),
            Value::Integer(1),
        )])))
    }

    fn load_other(&self) -> std::result::Result<Value, ExecutionError> {
        let entity_ctx = self
            .entity_context
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: "LoadOther requires entity context (member signal execution)".to_string(),
                phase: self.phase,
            })?;

        // LoadOther is used for accessing other member signals of the same entity instance
        // The path is determined by the opcode operand (handled by the opcode handler)
        // This stub should not be called directly - the opcode handler resolves the path
        Err(ExecutionError::InvalidOperand {
            message: format!(
                "LoadOther requires path operand (entity: {}, instance: {})",
                entity_ctx.entity_id, entity_ctx.instance_index
            ),
        })
    }

    fn load_payload(&self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Collect {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadPayload".to_string(),
                phase: self.phase,
            });
        }
        self.payload
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: "LoadPayload called but no impulse payload is available. This block should only execute when an impulse is triggered.".to_string(),
            })
    }

    fn find_nearest(
        &self,
        seq: &[Value],
        position: Value,
    ) -> std::result::Result<Value, ExecutionError> {
        let pos = position
            .as_vec3()
            .ok_or_else(|| ExecutionError::TypeMismatch {
                expected: "Vec3".to_string(),
                found: format!("{:?}", position),
            })?;

        let mut nearest = None;
        let mut min_dist_sq = f64::MAX;

        for instance in seq {
            if let Some(inst_map) = instance.as_map() {
                let inst_pos_val = inst_map
                    .iter()
                    .find(|(name, _)| name == "position")
                    .map(|(_, v)| v);

                if let Some(Value::Vec3(inst_pos)) = inst_pos_val {
                    let dx = inst_pos[0] - pos[0];
                    let dy = inst_pos[1] - pos[1];
                    let dz = inst_pos[2] - pos[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    if dist_sq < min_dist_sq {
                        min_dist_sq = dist_sq;
                        nearest = Some(instance.clone());
                    }
                }
            }
        }

        nearest.ok_or_else(|| ExecutionError::InvalidOperand {
            message: "No instances with 'position' field found in sequence for nearest lookup"
                .to_string(),
        })
    }

    fn filter_within(
        &self,
        seq: &[Value],
        position: Value,
        radius: Value,
    ) -> std::result::Result<Vec<Value>, ExecutionError> {
        let pos = position
            .as_vec3()
            .ok_or_else(|| ExecutionError::TypeMismatch {
                expected: "Vec3".to_string(),
                found: format!("{:?}", position),
            })?;
        let r = radius
            .as_scalar()
            .ok_or_else(|| ExecutionError::TypeMismatch {
                expected: "Scalar".to_string(),
                found: format!("{:?}", radius),
            })?;
        let r_sq = r * r;

        let mut filtered = Vec::new();
        for instance in seq {
            if let Some(inst_map) = instance.as_map() {
                let inst_pos_val = inst_map
                    .iter()
                    .find(|(name, _)| name == "position")
                    .map(|(_, v)| v);

                if let Some(Value::Vec3(inst_pos)) = inst_pos_val {
                    let dx = inst_pos[0] - pos[0];
                    let dy = inst_pos[1] - pos[1];
                    let dz = inst_pos[2] - pos[2];
                    let dist_sq = dx * dx + dy * dy + dz * dz;
                    if dist_sq <= r_sq {
                        filtered.push(instance.clone());
                    }
                }
            }
        }
        Ok(filtered)
    }

    fn find_neighbors(
        &self,
        _entity: &continuum_foundation::EntityId,
        _instance: Value,
    ) -> std::result::Result<Vec<Value>, ExecutionError> {
        // TODO: Implement actual topology lookup
        // For now, return empty sequence
        Ok(vec![])
    }

    fn emit_signal(
        &mut self,
        path: &Path,
        value: Value,
    ) -> std::result::Result<(), ExecutionError> {
        let val = value
            .as_scalar()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: "Only scalar signals supported for VM emit for now".to_string(),
            })?;

        let signal_id = SignalId::from(path.clone());

        match self.phase {
            Phase::Collect => {
                // Collect phase: inputs for THIS tick's resolve
                self.channels.accumulate(&signal_id, val);
            }
            Phase::Fracture => {
                // Fracture phase: queue for NEXT tick's resolve (alpha model)
                self.fracture_queue.queue(signal_id, val);
            }
            _ => {
                return Err(ExecutionError::InvalidOpcode {
                    opcode: "Emit signal only allowed in Collect or Fracture phase".to_string(),
                    phase: self.phase,
                });
            }
        }
        Ok(())
    }

    fn emit_field(
        &mut self,
        path: &Path,
        _pos: Value,
        value: Value,
    ) -> std::result::Result<(), ExecutionError> {
        if self.phase != Phase::Measure {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "Emit field only allowed in Measure phase".to_string(),
                phase: self.phase,
            });
        }
        let val = value
            .as_scalar()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: "Only scalar fields supported for VM emit for now".to_string(),
            })?;
        self.field_buffer
            .emit_scalar(crate::types::FieldId::from(path.clone()), val);
        Ok(())
    }

    fn emit_event(
        &mut self,
        chronicle_id: String,
        name: String,
        fields: Vec<(String, Value)>,
    ) -> std::result::Result<(), ExecutionError> {
        if self.phase != Phase::Measure {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "Emit event only allowed in Measure phase".to_string(),
                phase: self.phase,
            });
        }
        self.event_buffer.emit(chronicle_id, name, fields);
        Ok(())
    }

    fn spawn(
        &mut self,
        _entity: &EntityId,
        _data: Value,
    ) -> std::result::Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "Spawn not yet supported in VM".to_string(),
            phase: self.phase,
        })
    }

    fn destroy(
        &mut self,
        _entity: &EntityId,
        _instance: Value,
    ) -> std::result::Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "Destroy not yet supported in VM".to_string(),
            phase: self.phase,
        })
    }

    fn iter_entity(&self, entity: &EntityId) -> std::result::Result<Vec<Value>, ExecutionError> {
        // Read from MemberSignalBuffer (the single source of truth for entity state)
        let entity_str = entity.to_string();
        let instance_count = self.member_signals.instance_count_for_entity(&entity_str);

        if instance_count == 0 {
            return Err(ExecutionError::InvalidOperand {
                message: format!("Entity {} has no instances", entity),
            });
        }

        // Get all member signal names for this entity
        let signal_names = self.member_signals.signals_for_entity(&entity_str);

        if signal_names.is_empty() {
            return Err(ExecutionError::InvalidOperand {
                message: format!("Entity {} has no member signals", entity),
            });
        }

        // Build a Value::Map for each instance containing all member signals
        let mut values = Vec::with_capacity(instance_count);
        for idx in 0..instance_count {
            let mut fields = Vec::new();
            for signal in &signal_names {
                // Extract member name from full path (e.g., "hydrology.cell.temperature" -> "temperature")
                let member_name = signal.rsplit('.').next().unwrap_or(signal).to_string();

                if let Some(value) = self.member_signals.get_current(signal, idx) {
                    fields.push((member_name, value));
                }
            }
            values.push(Value::map(fields));
        }
        Ok(values)
    }

    fn reduce_aggregate(
        &self,
        op: AggregateOp,
        values: Vec<Value>,
    ) -> std::result::Result<Value, ExecutionError> {
        if values.is_empty() {
            return Err(ExecutionError::InvalidOperand {
                message: "Aggregate reduction requires at least one value".to_string(),
            });
        }

        match op {
            AggregateOp::Sum => {
                if let Some(v) = values
                    .iter()
                    .map(Value::as_scalar)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Scalar(reductions::sum(&v)))
                } else if let Some(v) = values
                    .iter()
                    .map(Value::as_vec3)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Vec3(reductions::sum_vec3(&v)))
                } else {
                    Err(ExecutionError::TypeMismatch {
                        expected: "Scalar or Vec3 values".to_string(),
                        found: format!("{values:?}"),
                    })
                }
            }
            AggregateOp::Product => {
                if let Some(v) = values
                    .iter()
                    .map(Value::as_scalar)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Scalar(reductions::product(&v)))
                } else {
                    Err(ExecutionError::TypeMismatch {
                        expected: "Scalar values".to_string(),
                        found: format!("{values:?}"),
                    })
                }
            }
            AggregateOp::Max => {
                if let Some(v) = values
                    .iter()
                    .map(Value::as_scalar)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Scalar(reductions::max(&v)))
                } else {
                    Err(ExecutionError::TypeMismatch {
                        expected: "Scalar values".to_string(),
                        found: format!("{values:?}"),
                    })
                }
            }
            AggregateOp::Min => {
                if let Some(v) = values
                    .iter()
                    .map(Value::as_scalar)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Scalar(reductions::min(&v)))
                } else {
                    Err(ExecutionError::TypeMismatch {
                        expected: "Scalar values".to_string(),
                        found: format!("{values:?}"),
                    })
                }
            }
            AggregateOp::Mean => {
                if let Some(v) = values
                    .iter()
                    .map(Value::as_scalar)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Scalar(reductions::mean(&v)))
                } else {
                    Err(ExecutionError::TypeMismatch {
                        expected: "Scalar values".to_string(),
                        found: format!("{values:?}"),
                    })
                }
            }
            AggregateOp::Count => {
                let count = values
                    .iter()
                    .map(Value::as_bool)
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| ExecutionError::TypeMismatch {
                        expected: "Boolean values".to_string(),
                        found: format!("{values:?}"),
                    })?
                    .iter()
                    .filter(|&&value| value)
                    .count();
                Ok(Value::Scalar(count as f64))
            }
            AggregateOp::Any => {
                let any = values
                    .iter()
                    .map(Value::as_bool)
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| ExecutionError::TypeMismatch {
                        expected: "Boolean values".to_string(),
                        found: format!("{values:?}"),
                    })?
                    .iter()
                    .any(|&value| value);
                Ok(Value::Boolean(any))
            }
            AggregateOp::All => {
                let all = values
                    .iter()
                    .map(Value::as_bool)
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| ExecutionError::TypeMismatch {
                        expected: "Boolean values".to_string(),
                        found: format!("{values:?}"),
                    })?
                    .iter()
                    .all(|&value| value);
                Ok(Value::Boolean(all))
            }
            AggregateOp::None => {
                let any = values
                    .iter()
                    .map(Value::as_bool)
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| ExecutionError::TypeMismatch {
                        expected: "Boolean values".to_string(),
                        found: format!("{values:?}"),
                    })?
                    .iter()
                    .any(|&value| value);
                Ok(Value::Boolean(!any))
            }
            AggregateOp::First => Ok(values[0].clone()),
            AggregateOp::Map => Err(ExecutionError::InvalidOperand {
                message: "Aggregate Map is not a runtime reduction".to_string(),
            }),
        }
    }

    fn call_kernel(
        &self,
        kernel: &continuum_kernel_types::KernelId,
        args: &[Value],
    ) -> std::result::Result<Value, ExecutionError> {
        continuum_kernel_registry::eval_in_namespace(
            kernel.namespace.as_ref(),
            kernel.name.as_ref(),
            args,
            self.dt.0,
        )
        .ok_or_else(|| ExecutionError::KernelCallFailed {
            message: format!("Kernel not found: {}", kernel.qualified_name()),
        })
    }

    fn trigger_assertion_fault(
        &mut self,
        severity: Option<&str>,
        message: Option<&str>,
    ) -> std::result::Result<(), ExecutionError> {
        // Default values are intentional: assertions may omit severity/message metadata.
        // DSL syntax: `assert { condition }` uses defaults.
        // Explicit: `assert { condition } severity("warn") message("custom")` overrides.
        // The .unwrap_or() here provides documented fallback behavior, not silent error hiding.
        Err(ExecutionError::AssertionFailed {
            severity: severity.unwrap_or("error").to_string(),
            message: message.unwrap_or("assertion failed").to_string(),
        })
    }

    fn load_member_signal(&self, member_name: &str) -> std::result::Result<Value, ExecutionError> {
        let entity_ctx = self
            .entity_context
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: format!(
                    "load_member_signal('{}') requires entity context",
                    member_name
                ),
                phase: self.phase,
            })?;

        // Construct the full member signal path: entity_id.member_name
        let full_path = format!("{}.{}", entity_ctx.entity_id, member_name);

        // Load the current value from the member signal buffer
        self.member_signals
            .get_current(&full_path, entity_ctx.instance_index)
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!(
                    "Member signal '{}' at instance {} not found or not initialized",
                    full_path, entity_ctx.instance_index
                ),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Compiler;
    use crate::dag::{DagBuilder, DagNode, DagSet, EraDags, NodeId};
    use crate::executor::{EraConfig, Runtime};
    use crate::WorldPolicy;
    use continuum_cdsl::ast::{Execution, ExecutionBody, ExprKind, Stmt, TypedExpr};
    use continuum_cdsl::foundation::{Shape, Span, Type, Unit};
    use continuum_functions as _;
    use continuum_kernel_types::KernelId;
    use std::collections::HashSet;

    fn make_span() -> Span {
        Span::new(0, 0, 0, 0)
    }

    fn make_scalar_literal(value: f64) -> TypedExpr {
        TypedExpr::new(
            ExprKind::Literal { value, unit: None },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            make_span(),
        )
    }

    #[test]
    fn test_bytecode_integration_resolve_simple() {
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "default".into();
        let signal_id: SignalId = "counter".into();

        // 1. Compile a resolve block: prev + 1.0
        let mut compiler = Compiler::new();
        let execution = Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Expr(TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("maths", "add"),
                    args: vec![
                        TypedExpr::new(
                            ExprKind::Prev,
                            Type::kernel(Shape::Scalar, Unit::seconds(), None),
                            make_span(),
                        ),
                        make_scalar_literal(1.0),
                    ],
                },
                Type::kernel(Shape::Scalar, Unit::seconds(), None),
                make_span(),
            )),
            reads: vec![],
            temporal_reads: vec![Path::from("counter")],
            emits: vec![],
            span: make_span(),
        };

        let compiled = compiler.compile_execution(&execution).unwrap();
        let blocks = vec![compiled];

        // 2. Setup DAG
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

        // 3. Setup Runtime
        let mut eras = IndexMap::new();
        let mut strata = IndexMap::new();
        strata.insert(stratum_id, StratumState::Active);
        eras.insert(
            era_id.clone(),
            EraConfig {
                dt: Dt(1.0),
                strata,
                transition: None,
            },
        );

        let mut runtime = Runtime::new(era_id, eras, dags, blocks, WorldPolicy::default());
        runtime.init_signal(signal_id.clone(), Value::Scalar(10.0));

        // 4. Execute tick
        runtime.execute_tick().unwrap();

        // 5. Verify result: 10.0 + 1.0 = 11.0
        let val = runtime.get_signal(&signal_id).unwrap();
        assert_eq!(val.as_scalar(), Some(11.0));
    }

    #[test]
    fn test_bytecode_integration_collect_emit() {
        let era_id: EraId = "test".into();
        let stratum_id: StratumId = "default".into();
        let signal_id: SignalId = "accumulator".into();

        // 1. Compile a collect block: emit(accumulator, 5.0)
        let mut compiler = Compiler::new();
        let execution = Execution {
            name: "collect".to_string(),
            phase: Phase::Collect,
            body: ExecutionBody::Statements(vec![Stmt::SignalAssign {
                target: Path::from("accumulator"),
                value: make_scalar_literal(5.0),
                span: make_span(),
            }]),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![Path::from("accumulator")],
            span: make_span(),
        };

        let compiled = compiler.compile_execution(&execution).unwrap();

        // 2. Compile a resolve block: prev + inputs
        let resolve_execution = Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Expr(TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("maths", "add"),
                    args: vec![
                        TypedExpr::new(
                            ExprKind::Prev,
                            Type::kernel(Shape::Scalar, Unit::seconds(), None),
                            make_span(),
                        ),
                        TypedExpr::new(
                            ExprKind::Inputs,
                            Type::kernel(Shape::Scalar, Unit::seconds(), None),
                            make_span(),
                        ),
                    ],
                },
                Type::kernel(Shape::Scalar, Unit::seconds(), None),
                make_span(),
            )),
            reads: vec![],
            temporal_reads: vec![Path::from("accumulator")],
            emits: vec![],
            span: make_span(),
        };
        let compiled_resolve = compiler.compile_execution(&resolve_execution).unwrap();

        let blocks = vec![compiled, compiled_resolve];

        // 3. Setup DAG
        let mut collect_builder = DagBuilder::new(Phase::Collect, stratum_id.clone());
        collect_builder.add_node(DagNode {
            id: NodeId("acc_collect".to_string()),
            reads: [signal_id.clone()].into_iter().collect(),
            writes: None,
            kind: NodeKind::OperatorCollect { operator_idx: 0 },
        });
        let collect_dag = collect_builder.build().unwrap();

        let mut resolve_builder = DagBuilder::new(Phase::Resolve, stratum_id.clone());
        resolve_builder.add_node(DagNode {
            id: NodeId("acc_resolve".to_string()),
            reads: HashSet::new(),
            writes: Some(signal_id.clone()),
            kind: NodeKind::SignalResolve {
                signal: signal_id.clone(),
                resolver_idx: 1,
            },
        });
        let resolve_dag = resolve_builder.build().unwrap();

        let mut era_dags = EraDags::default();
        era_dags.insert(collect_dag);
        era_dags.insert(resolve_dag);
        let mut dags = DagSet::default();
        dags.insert_era(era_id.clone(), era_dags);

        // 4. Setup Runtime
        let mut eras = IndexMap::new();
        let mut strata = IndexMap::new();
        strata.insert(stratum_id, StratumState::Active);
        eras.insert(
            era_id.clone(),
            EraConfig {
                dt: Dt(1.0),
                strata,
                transition: None,
            },
        );

        let mut runtime = Runtime::new(era_id, eras, dags, blocks, WorldPolicy::default());
        runtime.init_signal(signal_id.clone(), Value::Scalar(100.0));

        // 5. Execute tick
        runtime.execute_tick().unwrap();

        // 6. Verify result: 100.0 + 5.0 = 105.0
        let val = runtime.get_signal(&signal_id).unwrap();
        assert_eq!(val.as_scalar(), Some(105.0));
    }
}
