use super::*;
use crate::bytecode::operand::Operand;
use crate::bytecode::program::BytecodeBlock;
use crate::bytecode::runtime::{ExecutionContext, ExecutionError};
use continuum_cdsl::ast::AggregateOp;
use continuum_foundation::{EntityId, Path, Phase, Value};
use continuum_kernel_types::KernelId;

struct TestContext;

impl ExecutionContext for TestContext {
    fn phase(&self) -> Phase {
        Phase::Resolve
    }

    fn load_signal(&self, _path: &Path) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(1.0))
    }

    fn load_field(&self, _path: &Path) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(2.0))
    }

    fn load_config(&self, _path: &Path) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(3.0))
    }

    fn load_const(&self, _path: &Path) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(4.0))
    }

    fn load_prev(&self) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(5.0))
    }

    fn load_current(&self) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(6.0))
    }

    fn load_inputs(&mut self) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(7.0))
    }

    fn load_dt(&self) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(0.1))
    }

    fn load_self(&self) -> Result<Value, ExecutionError> {
        Ok(Value::map(vec![("mass".to_string(), Value::Scalar(9.0))]))
    }

    fn load_other(&self) -> Result<Value, ExecutionError> {
        Ok(Value::map(vec![("mass".to_string(), Value::Scalar(8.0))]))
    }

    fn load_payload(&self) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(11.0))
    }

    fn emit_signal(&mut self, _target: &Path, _value: Value) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn emit_field(
        &mut self,
        _target: &Path,
        _position: Value,
        _value: Value,
    ) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn emit_event(
        &mut self,
        _chronicle_id: String,
        _name: String,
        _fields: Vec<(String, Value)>,
    ) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn spawn(&mut self, _entity: &EntityId, _value: Value) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn destroy(&mut self, _entity: &EntityId, _instance: Value) -> Result<(), ExecutionError> {
        Ok(())
    }

    fn iter_entity(&self, _entity: &EntityId) -> Result<Vec<Value>, ExecutionError> {
        Ok(vec![Value::Scalar(1.0), Value::Scalar(2.0)])
    }

    fn reduce_aggregate(
        &self,
        _op: AggregateOp,
        values: Vec<Value>,
    ) -> Result<Value, ExecutionError> {
        values
            .into_iter()
            .try_fold(Value::Scalar(0.0), |acc, val| match (acc, val) {
                (Value::Scalar(a), Value::Scalar(b)) => Ok(Value::Scalar(a + b)),
                (left, right) => Err(ExecutionError::TypeMismatch {
                    expected: "Scalar reduction".to_string(),
                    found: format!("{left:?} + {right:?}"),
                }),
            })
    }

    fn call_kernel(&self, _kernel: &KernelId, args: &[Value]) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(args.len() as f64))
    }

    fn find_nearest(&self, _seq: &[Value], _position: Value) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(0.0))
    }

    fn filter_within(
        &self,
        _seq: &[Value],
        _position: Value,
        _radius: Value,
    ) -> Result<Vec<Value>, ExecutionError> {
        Ok(vec![])
    }
}

#[test]
fn test_execute_literal() {
    let program = BytecodeProgram::from_blocks(vec![BytecodeBlock {
        instructions: vec![
            Instruction::new(
                OpcodeKind::PushLiteral,
                vec![Operand::Literal(Value::Scalar(42.0))],
            ),
            Instruction::new(OpcodeKind::Return, vec![]),
        ],
        returns_value: true,
    }]);
    let block = CompiledBlock {
        phase: Phase::Resolve,
        program,
        root: BlockId::new(0),
        slot_count: 0,
        reads: vec![],
        temporal_reads: vec![],
        emits: vec![],
    };

    let mut executor = BytecodeExecutor::new();
    let mut ctx = TestContext;
    let result = executor.execute(&block, &mut ctx).unwrap();
    assert_eq!(result, Some(Value::Scalar(42.0)));
}
