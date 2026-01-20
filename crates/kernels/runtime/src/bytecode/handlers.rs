//! Opcode handlers for bytecode execution.

use continuum_foundation::Value;

use crate::bytecode::opcode::Instruction;
use crate::bytecode::operand::{
    field_access, operand_aggregate_op, operand_block, operand_config_path, operand_const_path,
    operand_entity, operand_field_path, operand_literal, operand_signal_path, operand_slot,
    operand_string, operand_usize,
};
use crate::bytecode::program::BytecodeProgram;
use crate::bytecode::runtime::{ExecutionContext, ExecutionError, ExecutionRuntime};

/// Functional interface for an opcode execution handler.
///
/// Handlers are responsible for reading instruction operands, interacting with
/// the evaluation stack and slot storage via [`ExecutionRuntime`], and performing
/// simulation side effects or state lookups via [`ExecutionContext`].
pub type Handler = fn(
    &Instruction,
    &mut dyn ExecutionRuntime,
    &BytecodeProgram,
    &mut dyn ExecutionContext,
) -> Result<(), ExecutionError>;

fn handle_load_with_path(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    ctx: &mut dyn ExecutionContext,
    load: fn(&dyn ExecutionContext, &continuum_foundation::Path) -> Result<Value, ExecutionError>,
    decode: fn(
        &crate::bytecode::operand::Operand,
    ) -> Result<continuum_foundation::Path, ExecutionError>,
) -> Result<(), ExecutionError> {
    let path = decode(&instruction.operands[0])?;
    let value = load(ctx, &path)?;
    runtime.push(value)?;
    Ok(())
}

fn handle_load_ctx(
    runtime: &mut dyn ExecutionRuntime,
    ctx: &mut dyn ExecutionContext,
    load: fn(&dyn ExecutionContext) -> Result<Value, ExecutionError>,
) -> Result<(), ExecutionError> {
    let value = load(ctx)?;
    runtime.push(value)?;
    Ok(())
}

fn handle_emit_value(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    ctx: &mut dyn ExecutionContext,
    emit: fn(
        &mut dyn ExecutionContext,
        &continuum_foundation::Path,
        Value,
    ) -> Result<(), ExecutionError>,
) -> Result<(), ExecutionError> {
    let target = operand_signal_path(&instruction.operands[0])?;
    let value = runtime.pop()?;
    emit(ctx, &target, value)
}

pub(crate) fn handle_noop(
    _instruction: &Instruction,
    _runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    Ok(())
}

pub(crate) fn handle_push_literal(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let value = operand_literal(&instruction.operands[0])?;
    runtime.push(value)?;
    Ok(())
}

pub(crate) fn handle_load(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let slot = operand_slot(&instruction.operands[0])?;
    let value = runtime.load_slot(slot)?;
    runtime.push(value)?;
    Ok(())
}

pub(crate) fn handle_store(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let slot = operand_slot(&instruction.operands[0])?;
    let value = runtime.pop()?;
    runtime.store_slot(slot, value)?;
    Ok(())
}

pub(crate) fn handle_dup(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let value = runtime.pop()?;
    runtime.push(value.clone())?;
    runtime.push(value)?;
    Ok(())
}

pub(crate) fn handle_pop(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    runtime.pop()?;
    Ok(())
}

pub(crate) fn handle_build_vector(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let count = operand_usize(&instruction.operands[0])?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(runtime.pop()?);
    }
    values.reverse();
    let vector = match values.as_slice() {
        [Value::Scalar(x), Value::Scalar(y)] => Value::Vec2([*x, *y]),
        [Value::Scalar(x), Value::Scalar(y), Value::Scalar(z)] => Value::Vec3([*x, *y, *z]),
        [Value::Scalar(x), Value::Scalar(y), Value::Scalar(z), Value::Scalar(w)] => {
            Value::Vec4([*x, *y, *z, *w])
        }
        [..] if count >= 2 && count <= 4 => {
            return Err(ExecutionError::TypeMismatch {
                expected: "Scalar vector components".to_string(),
                found: format!("{values:?}"),
            })
        }
        _ => return Err(ExecutionError::UnsupportedVectorSize { size: count }),
    };
    runtime.push(vector)?;
    Ok(())
}

pub(crate) fn handle_build_struct(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let mut fields = Vec::with_capacity(instruction.operands.len());
    for operand in instruction.operands.iter().rev() {
        let name = operand_string(operand)?;
        let value = runtime.pop()?;
        fields.push((name, value));
    }
    fields.reverse();
    runtime.push(Value::map(fields))?;
    Ok(())
}

pub(crate) fn handle_call_kernel(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let kernel = instruction
        .kernel
        .ok_or_else(|| ExecutionError::InvalidOperand {
            message: "CallKernel missing kernel id".to_string(),
        })?;
    let arg_count = operand_usize(&instruction.operands[0])?;
    let mut args = Vec::with_capacity(arg_count);
    for _ in 0..arg_count {
        args.push(runtime.pop()?);
    }
    args.reverse();
    let result = ctx.call_kernel(&kernel, &args)?;
    runtime.push(result)?;
    Ok(())
}

pub(crate) fn handle_aggregate(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let entity = operand_entity(&instruction.operands[0])?;
    let binding_slot = operand_slot(&instruction.operands[1])?;
    let block_id = operand_block(&instruction.operands[2])?;
    let op = operand_aggregate_op(&instruction.operands[3])?;

    let instances = ctx.iter_entity(&entity)?;
    let mut values = Vec::with_capacity(instances.len());
    for instance in instances {
        runtime.store_slot(binding_slot, instance)?;
        let value = runtime
            .execute_block(block_id, program, ctx)?
            .ok_or(ExecutionError::MissingReturn)?;
        values.push(value);
    }
    let reduced = ctx.reduce_aggregate(op, values)?;
    runtime.push(reduced)?;
    Ok(())
}

pub(crate) fn handle_fold(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let entity = operand_entity(&instruction.operands[0])?;
    let acc_slot = operand_slot(&instruction.operands[1])?;
    let elem_slot = operand_slot(&instruction.operands[2])?;
    let block_id = operand_block(&instruction.operands[3])?;

    let instances = ctx.iter_entity(&entity)?;
    for instance in instances {
        runtime.store_slot(elem_slot, instance)?;
        let next_acc = runtime
            .execute_block(block_id, program, ctx)?
            .ok_or(ExecutionError::MissingReturn)?;
        runtime.store_slot(acc_slot, next_acc)?;
    }
    let acc_value = runtime.load_slot(acc_slot)?;
    runtime.push(acc_value)?;
    Ok(())
}

pub(crate) fn handle_field_access(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let field = operand_string(&instruction.operands[0])?;
    let object = runtime.pop()?;
    let value = field_access(&object, &field)?;
    runtime.push(value)?;
    Ok(())
}

pub(crate) fn handle_load_signal(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_with_path(
        instruction,
        runtime,
        ctx,
        ExecutionContext::load_signal,
        operand_signal_path,
    )
}

pub(crate) fn handle_load_field(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_with_path(
        instruction,
        runtime,
        ctx,
        ExecutionContext::load_field,
        operand_field_path,
    )
}

pub(crate) fn handle_load_config(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_with_path(
        instruction,
        runtime,
        ctx,
        ExecutionContext::load_config,
        operand_config_path,
    )
}

pub(crate) fn handle_load_const(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_with_path(
        instruction,
        runtime,
        ctx,
        ExecutionContext::load_const,
        operand_const_path,
    )
}

pub(crate) fn handle_load_prev(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, ExecutionContext::load_prev)
}

pub(crate) fn handle_load_current(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, ExecutionContext::load_current)
}

pub(crate) fn handle_load_inputs(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, ExecutionContext::load_inputs)
}

pub(crate) fn handle_load_dt(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, ExecutionContext::load_dt)
}

pub(crate) fn handle_load_self(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, ExecutionContext::load_self)
}

pub(crate) fn handle_load_other(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, ExecutionContext::load_other)
}

pub(crate) fn handle_load_payload(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, ExecutionContext::load_payload)
}

pub(crate) fn handle_emit(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_emit_value(instruction, runtime, ctx, ExecutionContext::emit_signal)
}

pub(crate) fn handle_emit_field(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let target = operand_field_path(&instruction.operands[0])?;
    let value = runtime.pop()?;
    let position = runtime.pop()?;
    ctx.emit_field(&target, position, value)
}

pub(crate) fn handle_spawn(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let entity = operand_entity(&instruction.operands[0])?;
    let value = runtime.pop()?;
    ctx.spawn(&entity, value)
}

pub(crate) fn handle_destroy(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let entity = operand_entity(&instruction.operands[0])?;
    let value = runtime.pop()?;
    ctx.destroy(&entity, value)
}
