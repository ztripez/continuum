//! Tests for Assert opcode execution.
//!
//! Validates that assertions work correctly including:
//! - Success path (condition true)
//! - Failure path (condition false)
//! - Type validation (condition must be Bool)
//! - Severity and message propagation

use super::*;
use crate::bytecode::operand::Operand;
use crate::bytecode::program::BytecodeBlock;
use crate::bytecode::runtime::{ExecutionContext, ExecutionError};
use continuum_cdsl::ast::AggregateOp;
use continuum_foundation::{EntityId, Path, Phase, Value};

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

    fn load_member_signal(&self, _member_name: &str) -> Result<Value, ExecutionError> {
        Ok(Value::Scalar(12.0))
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
            .next()
            .ok_or_else(|| ExecutionError::InvalidOperand {
                message: "reduce_aggregate with empty values".to_string(),
            })
    }

    fn call_kernel(
        &self,
        _kernel: &continuum_kernel_types::KernelId,
        args: &[Value],
    ) -> Result<Value, ExecutionError> {
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

    fn trigger_assertion_fault(
        &mut self,
        severity: Option<&str>,
        message: Option<&str>,
    ) -> Result<(), ExecutionError> {
        Err(ExecutionError::AssertionFailed {
            severity: severity.unwrap_or("error").to_string(),
            message: message.unwrap_or("assertion failed").to_string(),
        })
    }
}

#[test]
fn test_assert_success_continues_execution() {
    // Assert with true condition should not block execution
    let program = BytecodeProgram::from_blocks(vec![BytecodeBlock {
        instructions: vec![
            Instruction::new(
                OpcodeKind::PushLiteral,
                vec![Operand::Literal(Value::Boolean(true))],
            ),
            Instruction::new(OpcodeKind::Assert, vec![]),
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

    assert_eq!(
        result,
        Some(Value::Scalar(42.0)),
        "Assert with true condition should not block execution"
    );
}

#[test]
fn test_assert_failure_triggers_fault() {
    // Assert with false condition should trigger assertion fault
    let program = BytecodeProgram::from_blocks(vec![BytecodeBlock {
        instructions: vec![
            Instruction::new(
                OpcodeKind::PushLiteral,
                vec![Operand::Literal(Value::Boolean(false))],
            ),
            Instruction::new(OpcodeKind::Assert, vec![]),
        ],
        returns_value: false,
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
    let result = executor.execute(&block, &mut ctx);

    assert!(result.is_err(), "Expected assertion failure");
    match result.unwrap_err() {
        ExecutionError::AssertionFailed { severity, message } => {
            assert_eq!(severity, "error", "Default severity should be 'error'");
            assert_eq!(
                message, "assertion failed",
                "Default message should be 'assertion failed'"
            );
        }
        other => panic!("Expected AssertionFailed, got {:?}", other),
    }
}

#[test]
fn test_assert_with_custom_severity_and_message() {
    // Assert with explicit severity and message
    let program = BytecodeProgram::from_blocks(vec![BytecodeBlock {
        instructions: vec![
            Instruction::new(
                OpcodeKind::PushLiteral,
                vec![Operand::Literal(Value::Boolean(false))],
            ),
            Instruction::new(
                OpcodeKind::Assert,
                vec![
                    Operand::String("fatal".to_string()),
                    Operand::String("temperature out of bounds".to_string()),
                ],
            ),
        ],
        returns_value: false,
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
    let result = executor.execute(&block, &mut ctx);

    assert!(result.is_err(), "Expected assertion failure");
    match result.unwrap_err() {
        ExecutionError::AssertionFailed { severity, message } => {
            assert_eq!(severity, "fatal", "Severity should be 'fatal'");
            assert_eq!(
                message, "temperature out of bounds",
                "Message should be custom message"
            );
        }
        other => panic!("Expected AssertionFailed, got {:?}", other),
    }
}

#[test]
fn test_assert_with_severity_only() {
    // Assert with severity but no message
    let program = BytecodeProgram::from_blocks(vec![BytecodeBlock {
        instructions: vec![
            Instruction::new(
                OpcodeKind::PushLiteral,
                vec![Operand::Literal(Value::Boolean(false))],
            ),
            Instruction::new(
                OpcodeKind::Assert,
                vec![Operand::String("warning".to_string())],
            ),
        ],
        returns_value: false,
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
    let result = executor.execute(&block, &mut ctx);

    assert!(result.is_err(), "Expected assertion failure");
    match result.unwrap_err() {
        ExecutionError::AssertionFailed { severity, message } => {
            assert_eq!(severity, "warning", "Severity should be 'warning'");
            assert_eq!(
                message, "assertion failed",
                "Message should be default when not provided"
            );
        }
        other => panic!("Expected AssertionFailed, got {:?}", other),
    }
}

#[test]
fn test_assert_type_mismatch_non_bool() {
    // Assert with non-Bool condition should fail with type mismatch
    let program = BytecodeProgram::from_blocks(vec![BytecodeBlock {
        instructions: vec![
            Instruction::new(
                OpcodeKind::PushLiteral,
                vec![Operand::Literal(Value::Scalar(1.0))],
            ),
            Instruction::new(OpcodeKind::Assert, vec![]),
        ],
        returns_value: false,
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
    let result = executor.execute(&block, &mut ctx);

    assert!(result.is_err(), "Expected type mismatch error");
    assert!(
        matches!(result.unwrap_err(), ExecutionError::TypeMismatch { .. }),
        "Expected TypeMismatch error"
    );
}

#[test]
fn test_assert_with_message_only() {
    // Assert with message but no severity (using 2 operands, first empty/default)
    // This tests that the handler correctly extracts both operands
    let program = BytecodeProgram::from_blocks(vec![BytecodeBlock {
        instructions: vec![
            Instruction::new(
                OpcodeKind::PushLiteral,
                vec![Operand::Literal(Value::Boolean(false))],
            ),
            Instruction::new(
                OpcodeKind::Assert,
                vec![
                    Operand::String("".to_string()), // Empty severity (will default)
                    Operand::String("custom failure message".to_string()),
                ],
            ),
        ],
        returns_value: false,
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
    let result = executor.execute(&block, &mut ctx);

    assert!(result.is_err(), "Expected assertion failure");
    match result.unwrap_err() {
        ExecutionError::AssertionFailed { severity, message } => {
            // Empty string is passed, so it becomes the severity
            assert_eq!(severity, "", "Empty severity string should be preserved");
            assert_eq!(message, "custom failure message");
        }
        other => panic!("Expected AssertionFailed, got {:?}", other),
    }
}

#[test]
fn test_assert_with_too_many_operands() {
    // Assert with 3+ operands is rejected by the executor's operand count validation.
    // The Assert opcode metadata specifies max=2, so instructions with more operands
    // are considered malformed and fail validation before execution.
    let program = BytecodeProgram::from_blocks(vec![BytecodeBlock {
        instructions: vec![
            Instruction::new(
                OpcodeKind::PushLiteral,
                vec![Operand::Literal(Value::Boolean(false))],
            ),
            Instruction::new(
                OpcodeKind::Assert,
                vec![
                    Operand::String("error".to_string()),
                    Operand::String("test message".to_string()),
                    Operand::String("extra operand 1".to_string()),
                    Operand::String("extra operand 2".to_string()),
                ],
            ),
        ],
        returns_value: false,
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
    let result = executor.execute(&block, &mut ctx);

    // Should fail with InvalidOperand error due to operand count mismatch
    assert!(result.is_err(), "Expected operand count error");
    assert!(
        matches!(result.unwrap_err(), ExecutionError::InvalidOperand { .. }),
        "Expected InvalidOperand error for too many operands"
    );
}

#[test]
fn test_assert_with_empty_stack() {
    // Assert with empty stack should fail with StackUnderflow
    let program = BytecodeProgram::from_blocks(vec![BytecodeBlock {
        instructions: vec![
            // No PushLiteral - stack is empty
            Instruction::new(OpcodeKind::Assert, vec![]),
        ],
        returns_value: false,
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
    let result = executor.execute(&block, &mut ctx);

    assert!(result.is_err(), "Expected stack underflow error");
    assert!(
        matches!(result.unwrap_err(), ExecutionError::StackUnderflow),
        "Expected StackUnderflow error"
    );
}
