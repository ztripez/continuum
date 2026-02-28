//! Read-only execution context for evaluating era transition conditions.
//!
//! [`TransitionEvalContext`] implements [`ExecutionContext`] with only the
//! capabilities needed for transition condition evaluation:
//! - `load_signal` — read resolved signal values
//! - `load_config` — read frozen configuration values
//! - `load_const` — read frozen constant values
//! - `load_dt` — read current era time step
//! - `call_kernel` — evaluate pure kernel operations (compare, logic, maths)
//!
//! All mutating operations (`emit_signal`, `emit_field`, `spawn`, `destroy`)
//! return errors — transition conditions are strictly read-only.

use crate::bytecode::runtime::{ExecutionContext, ExecutionError};
use crate::storage::SignalStorage;
use crate::types::{Dt, Phase, SignalId, Value};
use continuum_foundation::{AggregateOp, EntityId, Path};
use indexmap::IndexMap;

/// Lightweight execution context for era transition condition evaluation.
///
/// Transition conditions are boolean expressions evaluated after the Resolve
/// phase to determine if an era transition should fire. They are pure
/// read-only observations over resolved signals, configs, and constants.
///
/// This context deliberately forbids all side effects. Any attempt to emit
/// signals, fields, events, or modify entities will fail loudly.
pub struct TransitionEvalContext<'a> {
    /// Current era time step
    dt: Dt,
    /// Resolved signal storage (current tick values)
    signals: &'a SignalStorage,
    /// Frozen configuration values
    config_values: &'a IndexMap<Path, Value>,
    /// Frozen constant values
    const_values: &'a IndexMap<Path, Value>,
}

impl<'a> TransitionEvalContext<'a> {
    /// Creates a new transition evaluation context.
    ///
    /// # Parameters
    /// - `dt`: Current era time step
    /// - `signals`: Resolved signal storage from the current tick
    /// - `config_values`: Frozen world config values
    /// - `const_values`: Frozen world constant values
    pub fn new(
        dt: Dt,
        signals: &'a SignalStorage,
        config_values: &'a IndexMap<Path, Value>,
        const_values: &'a IndexMap<Path, Value>,
    ) -> Self {
        Self {
            dt,
            signals,
            config_values,
            const_values,
        }
    }
}

impl ExecutionContext for TransitionEvalContext<'_> {
    fn phase(&self) -> Phase {
        // Transition evaluation happens after Resolve, conceptually its own phase.
        // Use Resolve to allow signal reads.
        Phase::Resolve
    }

    fn load_signal(&self, path: &Path) -> Result<Value, ExecutionError> {
        self.signals
            .get(&SignalId::from(path.clone()))
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Signal not found during transition eval: {path}"),
            })
    }

    fn load_field(&self, path: &Path) -> Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: format!(
                "LoadField({path}) forbidden in transition conditions — fields are observer-only"
            ),
            phase: Phase::Resolve,
        })
    }

    fn load_config(&self, path: &Path) -> Result<Value, ExecutionError> {
        self.config_values
            .get(path)
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Config value not found during transition eval: {path}"),
            })
    }

    fn load_const(&self, path: &Path) -> Result<Value, ExecutionError> {
        self.const_values
            .get(path)
            .cloned()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: format!("Const value not found during transition eval: {path}"),
            })
    }

    fn load_prev(&self) -> Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadPrev forbidden in transition conditions — no target signal".to_string(),
            phase: Phase::Resolve,
        })
    }

    fn load_current(&self) -> Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadCurrent forbidden in transition conditions — no target signal".to_string(),
            phase: Phase::Resolve,
        })
    }

    fn load_inputs(&mut self) -> Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadInputs forbidden in transition conditions".to_string(),
            phase: Phase::Resolve,
        })
    }

    fn load_dt(&self) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(self.dt.0))
    }

    fn load_self(&self) -> Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadSelf forbidden in transition conditions — no entity context".to_string(),
            phase: Phase::Resolve,
        })
    }

    fn load_other(&self) -> Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadOther forbidden in transition conditions — no entity context".to_string(),
            phase: Phase::Resolve,
        })
    }

    fn load_payload(&self) -> Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "LoadPayload forbidden in transition conditions — no impulse context"
                .to_string(),
            phase: Phase::Resolve,
        })
    }

    fn find_nearest(&self, _seq: &[Value], _position: Value) -> Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "FindNearest forbidden in transition conditions".to_string(),
            phase: Phase::Resolve,
        })
    }

    fn filter_within(
        &self,
        _seq: &[Value],
        _position: Value,
        _radius: Value,
    ) -> Result<Vec<Value>, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "FilterWithin forbidden in transition conditions".to_string(),
            phase: Phase::Resolve,
        })
    }

    fn find_neighbors(
        &self,
        _entity: &EntityId,
        _instance: Value,
    ) -> Result<Vec<Value>, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "FindNeighbors forbidden in transition conditions".to_string(),
            phase: Phase::Resolve,
        })
    }

    fn emit_signal(&mut self, target: &Path, _value: Value) -> Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: format!("EmitSignal({target}) forbidden in transition conditions — read-only"),
            phase: Phase::Resolve,
        })
    }

    fn emit_member_signal(
        &mut self,
        entity: &EntityId,
        _instance_idx: u32,
        member_path: &Path,
        _value: Value,
    ) -> Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: format!(
                "EmitMemberSignal({entity}.{member_path}) forbidden in transition conditions — read-only"
            ),
            phase: Phase::Resolve,
        })
    }

    fn emit_field(
        &mut self,
        target: &Path,
        _position: Value,
        _value: Value,
    ) -> Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: format!("EmitField({target}) forbidden in transition conditions — read-only"),
            phase: Phase::Resolve,
        })
    }

    fn emit_event(
        &mut self,
        chronicle_id: String,
        _name: String,
        _fields: Vec<(String, Value)>,
    ) -> Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: format!(
                "EmitEvent({chronicle_id}) forbidden in transition conditions — read-only"
            ),
            phase: Phase::Resolve,
        })
    }

    fn spawn(&mut self, entity: &EntityId, _data: Value) -> Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: format!("Spawn({entity}) forbidden in transition conditions — read-only"),
            phase: Phase::Resolve,
        })
    }

    fn destroy(&mut self, entity: &EntityId, _instance: Value) -> Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: format!("Destroy({entity}) forbidden in transition conditions — read-only"),
            phase: Phase::Resolve,
        })
    }

    fn iter_entity(&self, entity: &EntityId) -> Result<Vec<Value>, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: format!(
                "IterEntity({entity}) forbidden in transition conditions — no entity context"
            ),
            phase: Phase::Resolve,
        })
    }

    fn reduce_aggregate(
        &self,
        _op: AggregateOp,
        _values: Vec<Value>,
    ) -> Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "ReduceAggregate forbidden in transition conditions".to_string(),
            phase: Phase::Resolve,
        })
    }

    fn call_kernel(
        &self,
        kernel: &continuum_kernel_types::KernelId,
        args: &[Value],
    ) -> Result<Value, ExecutionError> {
        continuum_kernel_registry::eval_in_namespace(
            kernel.namespace.as_ref(),
            kernel.name.as_ref(),
            args,
            self.dt.0,
        )
        .ok_or_else(|| ExecutionError::KernelCallFailed {
            message: format!(
                "Kernel not found during transition eval: {}",
                kernel.qualified_name()
            ),
        })
    }

    fn trigger_assertion_fault(
        &mut self,
        _severity: Option<&str>,
        _message: Option<&str>,
    ) -> Result<(), ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: "AssertionFault forbidden in transition conditions".to_string(),
            phase: Phase::Resolve,
        })
    }

    fn load_member_signal(&self, member_name: &str) -> Result<Value, ExecutionError> {
        Err(ExecutionError::InvalidOpcode {
            opcode: format!(
                "LoadMemberSignal({member_name}) forbidden in transition conditions — no entity context"
            ),
            phase: Phase::Resolve,
        })
    }
}
