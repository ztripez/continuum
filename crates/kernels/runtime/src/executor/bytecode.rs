use crate::bytecode::runtime::{ExecutionContext, ExecutionError};
use crate::bytecode::{BytecodeExecutor, CompiledBlock};
use crate::dag::{DagSet, NodeKind};
use crate::error::{Error, Result};
use crate::executor::assertions::AssertionChecker;
use crate::reductions;
use crate::soa_storage::MemberSignalBuffer;
use crate::storage::{EntityStorage, EventBuffer, FieldBuffer, InputChannels, SignalStorage};
use crate::types::{Dt, EraId, Phase, SignalId, StratumId, StratumState, Value};
use continuum_cdsl::ast::AggregateOp;
use continuum_foundation::{EntityId, Path};
use indexmap::IndexMap;
use tracing::instrument;

/// Executes individual simulation phases using the bytecode VM.
pub struct BytecodePhaseExecutor {
    /// The underlying bytecode interpreter.
    executor: BytecodeExecutor,
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
        }
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
                        let mut ctx = VMContext {
                            phase: Phase::Collect,
                            era,
                            dt,
                            sim_time,
                            signals,
                            entities,
                            member_signals,
                            channels: input_channels,
                            field_buffer: &mut placeholder_field_buffer,
                            target_signal: None,
                        };
                        let block_id = compiled.root;
                        self.executor
                            .execute_block(block_id, &compiled.program, &mut ctx)
                            .map_err(|e| Error::ExecutionFailure {
                                message: e.to_string(),
                            })?;
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

                            let mut ctx = VMContext {
                                phase: Phase::Resolve,
                                era,
                                dt,
                                sim_time,
                                signals,
                                entities,
                                member_signals,
                                channels: input_channels,
                                field_buffer: &mut placeholder_field_buffer,
                                target_signal: Some(signal.clone()),
                            };

                            let block_id = compiled.root;
                            let value = self
                                .executor
                                .execute_block(block_id, &compiled.program, &mut ctx)
                                .map_err(|e| Error::ExecutionFailure {
                                    message: e.to_string(),
                                })?
                                .ok_or_else(|| Error::ExecutionFailure {
                                    message: format!(
                                        "Signal resolve for '{}' returned no value",
                                        signal
                                    ),
                                })?;

                            level_results.push((signal.clone(), value));
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
        input_channels: &mut InputChannels,
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
                        let mut ctx = VMContext {
                            phase: Phase::Fracture,
                            era,
                            dt,
                            sim_time,
                            signals,
                            entities,
                            member_signals,
                            channels: input_channels,
                            field_buffer: &mut placeholder_field_buffer,
                            target_signal: None, // TODO: Fracture nodes currently lack single-signal target context
                        };
                        let block_id = compiled.root;
                        self.executor
                            .execute_block(block_id, &compiled.program, &mut ctx)
                            .map_err(|e| Error::ExecutionFailure {
                                message: e.to_string(),
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
                            let mut ctx = VMContext {
                                phase: Phase::Measure,
                                era,
                                dt,
                                sim_time,
                                signals,
                                entities,
                                member_signals,
                                channels: &mut inner_input_channels,
                                field_buffer,
                                target_signal: None,
                            };

                            let block_id = compiled.root;
                            self.executor
                                .execute_block(block_id, &compiled.program, &mut ctx)
                                .map_err(|e| Error::ExecutionFailure {
                                    message: e.to_string(),
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
                            let mut ctx = VMContext {
                                phase: Phase::Measure,
                                era,
                                dt,
                                sim_time,
                                signals,
                                entities,
                                member_signals,
                                channels: &mut input_channels,
                                field_buffer: &mut placeholder_field_buffer,
                                target_signal: None,
                            };

                            let block_id = compiled.root;
                            self.executor
                                .execute_block(block_id, &compiled.program, &mut ctx)
                                .map_err(|e| Error::ExecutionFailure {
                                    message: e.to_string(),
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
    pub field_buffer: &'a mut FieldBuffer,
    pub target_signal: Option<SignalId>,
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

    fn load_config(&self, _path: &Path) -> std::result::Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadConfig not yet supported in VM".to_string(),
            phase: self.phase,
        })
    }

    fn load_const(&self, _path: &Path) -> std::result::Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadConst not yet supported in VM".to_string(),
            phase: self.phase,
        })
    }

    fn load_prev(&self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Resolve {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadPrev".to_string(),
                phase: self.phase,
            });
        }
        let signal = self
            .target_signal
            .as_ref()
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: "LoadPrev requires target signal context".to_string(),
                phase: self.phase,
            })?;
        self.signals
            .get_prev(signal)
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Signal '{}' has no previous value", signal),
            })
    }

    fn load_current(&self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Fracture && self.phase != Phase::Measure {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadCurrent".to_string(),
                phase: self.phase,
            });
        }
        let signal = self
            .target_signal
            .as_ref()
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: "LoadCurrent requires target signal context".to_string(),
                phase: self.phase,
            })?;
        self.signals
            .get(signal)
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Signal '{}' has no resolved value", signal),
            })
    }

    fn load_inputs(&mut self) -> std::result::Result<Value, ExecutionError> {
        if self.phase != Phase::Resolve {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "LoadInputs".to_string(),
                phase: self.phase,
            });
        }
        let signal = self
            .target_signal
            .as_ref()
            .ok_or_else(|| ExecutionError::InvalidOpcode {
                opcode: "LoadInputs requires target signal context".to_string(),
                phase: self.phase,
            })?;
        // Use peek_sum instead of drain_sum to preserve "One Truth" for multiple reads in one block
        let value = self.channels.peek_sum(signal);
        Ok(Value::Scalar(value))
    }

    fn load_dt(&self) -> std::result::Result<Value, ExecutionError> {
        Ok(Value::Scalar(self.dt.0))
    }

    fn load_self(&self) -> std::result::Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadSelf requires entity context".to_string(),
            phase: self.phase,
        })
    }

    fn load_other(&self) -> std::result::Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadOther requires entity context".to_string(),
            phase: self.phase,
        })
    }

    fn load_payload(&self) -> std::result::Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadPayload requires impulse context".to_string(),
            phase: self.phase,
        })
    }

    fn emit_signal(
        &mut self,
        path: &Path,
        value: Value,
    ) -> std::result::Result<(), ExecutionError> {
        if self.phase != Phase::Collect {
            return Err(ExecutionError::InvalidOpcode {
                opcode: "Emit signal only allowed in Collect phase".to_string(),
                phase: self.phase,
            });
        }
        let val = value
            .as_scalar()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: "Only scalar signals supported for VM emit for now".to_string(),
            })?;
        self.channels.accumulate(&SignalId::from(path.clone()), val);
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
        let instances = self.entities.get_current_instances(entity).ok_or_else(|| {
            ExecutionError::InvalidOperand {
                message: format!("Missing current instances for entity {}", entity),
            }
        })?;

        let mut values = Vec::with_capacity(instances.count());
        for (_id, data) in instances.iter() {
            let fields = data
                .fields
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect();
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
                if let Some(values) = values
                    .iter()
                    .map(Value::as_scalar)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Scalar(reductions::sum(&values)))
                } else if let Some(values) = values
                    .iter()
                    .map(Value::as_vec3)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Vec3(reductions::sum_vec3(&values)))
                } else {
                    Err(ExecutionError::TypeMismatch {
                        expected: "Scalar or Vec3 values".to_string(),
                        found: format!("{values:?}"),
                    })
                }
            }
            AggregateOp::Max => {
                if let Some(values) = values
                    .iter()
                    .map(Value::as_scalar)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Scalar(reductions::max(&values)))
                } else {
                    Err(ExecutionError::TypeMismatch {
                        expected: "Scalar values".to_string(),
                        found: format!("{values:?}"),
                    })
                }
            }
            AggregateOp::Min => {
                if let Some(values) = values
                    .iter()
                    .map(Value::as_scalar)
                    .collect::<Option<Vec<_>>>()
                {
                    Ok(Value::Scalar(reductions::min(&values)))
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bytecode::Compiler;
    use crate::dag::{DagBuilder, DagNode, DagSet, EraDags, NodeId};
    use crate::executor::{EraConfig, Runtime};
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

        let mut runtime = Runtime::new(era_id, eras, dags, blocks);
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

        let mut runtime = Runtime::new(era_id, eras, dags, blocks);
        runtime.init_signal(signal_id.clone(), Value::Scalar(100.0));

        // 5. Execute tick
        runtime.execute_tick().unwrap();

        // 6. Verify result: 100.0 + 5.0 = 105.0
        let val = runtime.get_signal(&signal_id).unwrap();
        assert_eq!(val.as_scalar(), Some(105.0));
    }
}
