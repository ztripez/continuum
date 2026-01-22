//! Opcode handlers for bytecode execution.

use continuum_foundation::{AggregateOp, Value};

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

/// Shared logic for opcodes that load simulation state via a path.
///
/// This helper handles the common pattern of:
/// 1. Decoding a [`Path`] from the first instruction operand.
/// 2. Calling a domain-specific load function from the [`ExecutionContext`].
/// 3. Pushing the resulting [`Value`] onto the stack.
fn handle_load_with_path<F>(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    ctx: &mut dyn ExecutionContext,
    load: F,
    decode: fn(
        &crate::bytecode::operand::Operand,
    ) -> Result<continuum_foundation::Path, ExecutionError>,
) -> Result<(), ExecutionError>
where
    F: Fn(&dyn ExecutionContext, &continuum_foundation::Path) -> Result<Value, ExecutionError>,
{
    let path = decode(&instruction.operands[0])?;
    let value = load(ctx, &path)?;
    runtime.push(value)?;
    Ok(())
}

/// Shared logic for opcodes that load context-dependent values (prev, dt, self, etc.).
///
/// This helper executes a load function from the [`ExecutionContext`] and pushes
/// the result onto the stack. It is used for opcodes that do not require operands.
fn handle_load_ctx<F>(
    runtime: &mut dyn ExecutionRuntime,
    ctx: &mut dyn ExecutionContext,
    load: F,
) -> Result<(), ExecutionError>
where
    F: Fn(&mut dyn ExecutionContext) -> Result<Value, ExecutionError>,
{
    let value = load(ctx)?;
    runtime.push(value)?;
    Ok(())
}

/// Shared logic for opcodes that emit values to a signal path.
///
/// Decodes the target signal path from operand[0], pops the value to emit from
/// the top of the stack, and dispatches the emission via the [`ExecutionContext`].
fn handle_emit_value<F>(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    ctx: &mut dyn ExecutionContext,
    emit: F,
) -> Result<(), ExecutionError>
where
    F: Fn(
        &mut dyn ExecutionContext,
        &continuum_foundation::Path,
        Value,
    ) -> Result<(), ExecutionError>,
{
    let target = operand_signal_path(&instruction.operands[0])?;
    let value = runtime.pop()?;
    emit(ctx, &target, value)
}

/// No-op handler for instructions that serve as structural markers (e.g., Let, Return).
pub(crate) fn handle_noop(
    _instruction: &Instruction,
    _runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    Ok(())
}

/// Pushes a literal value onto the evaluation stack.
///
/// # Operands
/// - 0: The literal value to push ([`Operand::Literal`]).
///
/// # Stack
/// - [ ] → [Value]
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

/// Loads a value from a specific VM slot and pushes it onto the stack.
///
/// Used for reading let-bound variables and temporary computation results.
///
/// # Operands
/// - 0: The slot index to load from ([`Operand::Slot`]).
///
/// # Stack
/// - [ ] → [Value]
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

/// Pops the top stack value and stores it into a VM slot.
///
/// # Operands
/// - 0: The target slot index ([`Operand::Slot`]).
///
/// # Stack
/// - [Value] → [ ]
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

/// Duplicates the top value on the stack.
///
/// # Stack
/// - [Value] → [Value, Value]
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

/// Discards the top value from the stack.
///
/// # Stack
/// - [Value] → [ ]
pub(crate) fn handle_pop(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    _ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    runtime.pop()?;
    Ok(())
}

/// Constructs a vector (Vec2, Vec3, or Vec4) from stack values.
///
/// Pops $N$ scalar values from the stack and pushes a single vector value.
/// Components must be pushed in order (x, y, z, w); this handler pops them
/// in reverse order to maintain correct orientation.
///
/// # Operands
/// - 0: The number of components $N \in [2, 4]$ ([`Operand::Literal`] integer).
///
/// # Stack
/// - [v1, ..., vN] → [Vector]
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
            });
        }
        _ => return Err(ExecutionError::UnsupportedVectorSize { size: count }),
    };
    runtime.push(vector)?;
    Ok(())
}

/// Constructs a structured Map from stack values and operand field names.
///
/// Pops one value per field name provided in operands. Values must be pushed
/// onto the stack in the same order as field names; this handler pops them
/// in reverse order to build the map.
///
/// # Operands
/// - Variable: List of field names ([`Operand::String`]).
///
/// # Stack
/// - [v1, ..., vN] → [Map]
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

/// Dispatches a call to an engine kernel primitive.
///
/// # Operands
/// - 0: Argument count $N$ ([`Operand::Literal`] integer).
///
/// # Stack
/// - [arg1, ..., argN] → [Result]
pub(crate) fn handle_call_kernel(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let kernel = instruction
        .kernel
        .as_ref()
        .ok_or_else(|| ExecutionError::InvalidOperand {
            message: "CallKernel missing kernel id".to_string(),
        })?
        .clone();
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

/// Loads all instances of an entity type as a sequence.
///
/// # Operands
/// - 0: Entity type ID ([`Operand::Entity`]).
///
/// # Stack
/// - [ ] → [Seq]
pub(crate) fn handle_load_entity(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let entity = operand_entity(&instruction.operands[0])?;
    let instances = ctx.iter_entity(&entity)?;
    runtime.push(Value::Seq(std::sync::Arc::new(instances)))?;
    Ok(())
}

/// Filters a sequence using a predicate block.
///
/// # Operands
/// - 0: VM slot to bind the current instance to ([`Operand::Slot`]).
/// - 1: ID of the predicate bytecode block ([`Operand::Block`]).
///
/// # Stack
/// - [Seq] → [Filtered Seq]
pub(crate) fn handle_filter(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let binding_slot = operand_slot(&instruction.operands[0])?;
    let block_id = operand_block(&instruction.operands[1])?;
    let seq_value = runtime.pop()?;
    let seq = seq_value
        .as_seq()
        .ok_or_else(|| ExecutionError::TypeMismatch {
            expected: "Seq".to_string(),
            found: format!("{seq_value:?}"),
        })?;

    let mut filtered = Vec::new();
    for instance in seq {
        runtime.store_slot(binding_slot, instance.clone())?;
        let result = runtime
            .execute_block(block_id, program, ctx)?
            .ok_or(ExecutionError::MissingReturn)?;
        let is_match = result
            .as_bool()
            .ok_or_else(|| ExecutionError::TypeMismatch {
                expected: "Boolean".to_string(),
                found: format!("{:?}", result),
            })?;
        if is_match {
            filtered.push(instance.clone());
        }
    }
    runtime.push(Value::Seq(std::sync::Arc::new(filtered)))?;
    Ok(())
}

/// Finds the instance in a sequence nearest to a position.
///
/// # Stack
/// - [Seq, Position] → [Nearest Instance]
pub(crate) fn handle_nearest(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let position = runtime.pop()?;
    let seq_value = runtime.pop()?;
    let seq = seq_value
        .as_seq()
        .ok_or_else(|| ExecutionError::TypeMismatch {
            expected: "Seq".to_string(),
            found: format!("{seq_value:?}"),
        })?;

    if seq.is_empty() {
        return Err(ExecutionError::InvalidOperand {
            message: "Nearest lookup on empty sequence".to_string(),
        });
    }

    let nearest = ctx.find_nearest(seq, position)?;
    runtime.push(nearest)?;
    Ok(())
}

/// Filters a sequence to instances within a radius of a position.
///
/// # Stack
/// - [Seq, Position, Radius] → [Filtered Seq]
pub(crate) fn handle_within(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let radius = runtime.pop()?;
    let position = runtime.pop()?;
    let seq_value = runtime.pop()?;
    let seq = seq_value
        .as_seq()
        .ok_or_else(|| ExecutionError::TypeMismatch {
            expected: "Seq".to_string(),
            found: format!("{seq_value:?}"),
        })?;

    let filtered = ctx.filter_within(seq, position, radius)?;
    runtime.push(Value::Seq(std::sync::Arc::new(filtered)))?;
    Ok(())
}

/// Performs a distributed aggregate operation (sum, map, count, etc.) over a sequence.
///
/// # Operands
/// 0. VM slot to bind the current instance to ([`Operand::Slot`]).
/// 1. ID of the bytecode block to execute ([`Operand::Block`]).
/// 2. Aggregate operation kind ([`Operand::AggregateOp`]).
///
/// # Stack
/// - [Seq] → [Result]
pub(crate) fn handle_aggregate(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let binding_slot = operand_slot(&instruction.operands[0])?;
    let block_id = operand_block(&instruction.operands[1])?;
    let op = operand_aggregate_op(&instruction.operands[2])?;
    let seq_value = runtime.pop()?;
    let seq = seq_value
        .as_seq()
        .ok_or_else(|| ExecutionError::TypeMismatch {
            expected: "Seq".to_string(),
            found: format!("{seq_value:?}"),
        })?;

    if matches!(op, AggregateOp::Map) {
        let mut results = Vec::with_capacity(seq.len());
        for instance in seq {
            runtime.store_slot(binding_slot, instance.clone())?;
            let value = runtime
                .execute_block(block_id, program, ctx)?
                .ok_or(ExecutionError::MissingReturn)?;
            results.push(value);
        }
        runtime.push(Value::Seq(std::sync::Arc::new(results)))?;
    } else {
        let mut values = Vec::with_capacity(seq.len());
        for instance in seq {
            runtime.store_slot(binding_slot, instance.clone())?;
            let value = runtime
                .execute_block(block_id, program, ctx)?
                .ok_or(ExecutionError::MissingReturn)?;
            values.push(value);
        }
        let reduced = ctx.reduce_aggregate(op, values)?;
        runtime.push(reduced)?;
    }
    Ok(())
}

/// Performs a stateful fold operation over a sequence.
///
/// # Operands
/// 0. VM slot for the accumulator ([`Operand::Slot`]).
/// 1. VM slot for the current instance ([`Operand::Slot`]).
/// 2. ID of the bytecode block to execute ([`Operand::Block`]).
///
/// # Stack
/// - [Seq, InitialValue] → [Result]
pub(crate) fn handle_fold(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let acc_slot = operand_slot(&instruction.operands[0])?;
    let elem_slot = operand_slot(&instruction.operands[1])?;
    let block_id = operand_block(&instruction.operands[2])?;
    let init_value = runtime.pop()?;
    let seq_value = runtime.pop()?;
    let seq = seq_value
        .as_seq()
        .ok_or_else(|| ExecutionError::TypeMismatch {
            expected: "Seq".to_string(),
            found: format!("{seq_value:?}"),
        })?;

    runtime.store_slot(acc_slot, init_value)?;
    for instance in seq {
        runtime.store_slot(elem_slot, instance.clone())?;
        let next_acc = runtime
            .execute_block(block_id, program, ctx)?
            .ok_or(ExecutionError::MissingReturn)?;
        runtime.store_slot(acc_slot, next_acc)?;
    }
    let acc_value = runtime.load_slot(acc_slot)?;
    runtime.push(acc_value)?;
    Ok(())
}

/// Accesses a named field on a Map value or a component on a vector.
///
/// # Operands
/// - 0: The name of the field or component ([`Operand::String`]).
///
/// # Stack
/// - [Object] → [Value]
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

/// Loads a resolved signal value by its path.
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
        |ctx, path| ctx.load_signal(path),
        operand_signal_path,
    )
}

/// Loads a value from a spatial field by its path.
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
        |ctx, path| ctx.load_field(path),
        operand_field_path,
    )
}

/// Loads a world configuration value by its path.
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
        |ctx, path| ctx.load_config(path),
        operand_config_path,
    )
}

/// Loads a global simulation constant by its path.
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
        |ctx, path| ctx.load_const(path),
        operand_const_path,
    )
}

/// Loads the value of the current signal from the previous tick.
pub(crate) fn handle_load_prev(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, |ctx| ctx.load_prev())
}

/// Loads the just-resolved value of the current signal.
pub(crate) fn handle_load_current(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, |ctx| ctx.load_current())
}

/// Loads the accumulated inputs for the current signal.
pub(crate) fn handle_load_inputs(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, |ctx| ctx.load_inputs())
}

/// Loads the current time step (dt).
pub(crate) fn handle_load_dt(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, |ctx| ctx.load_dt())
}

/// Loads the identity of the current entity instance.
///
/// # Stack
/// - [ ] → [EntityId]
pub(crate) fn handle_load_self(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, |ctx| ctx.load_self())
}

/// Loads the identity of the "other" entity instance (if any).
///
/// # Stack
/// - [ ] → [EntityId]
pub(crate) fn handle_load_other(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, |ctx| ctx.load_other())
}

/// Loads the payload data from the triggering impulse.
///
/// # Stack
/// - [ ] → [Value]
pub(crate) fn handle_load_payload(
    _instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_load_ctx(runtime, ctx, |ctx| ctx.load_payload())
}

/// Emits a value to a signal.
///
/// # Operands
/// - 0: The target signal path ([`Operand::Signal`]).
///
/// # Stack
/// - [Value] → [ ]
pub(crate) fn handle_emit(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    handle_emit_value(instruction, runtime, ctx, |ctx, path, value| {
        ctx.emit_signal(path, value)
    })
}

/// Emits a value to a spatial field at a given position.
///
/// Pops [Value, Position] from the stack. Target field is in operand[0].
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

/// Spawns a new instance of an entity type.
///
/// Pops initial value from stack. Entity type is in operand[0].
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

/// Marks an entity instance for destruction.
///
/// Pops instance ID from stack. Entity type is in operand[0].
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

/// Validates an assertion condition and triggers a fault if it fails.
///
/// # Operands
///
/// - operand[0]: Optional severity level string ("warn", "error", "fatal")
/// - operand[1]: Optional custom message string
///
/// # Stack
///
/// - Pops: Bool condition value
///
/// # Phase Restrictions
///
/// Only valid in Resolve and Fracture phases (enforced at compile time).
/// Assertions in Measure phase violate the observer boundary.
///
/// # Behavior
///
/// If the condition is `true`, execution continues normally.
/// If the condition is `false`, triggers an assertion fault via
/// [`ExecutionContext::trigger_assertion_fault`].
///
/// Fault handling is policy-driven:
/// - 'warn': Log and continue
/// - 'error': Halt tick, continue simulation (default)
/// - 'fatal': Halt simulation immediately
pub(crate) fn handle_assert(
    instruction: &Instruction,
    runtime: &mut dyn ExecutionRuntime,
    _program: &BytecodeProgram,
    ctx: &mut dyn ExecutionContext,
) -> Result<(), ExecutionError> {
    let condition = runtime.pop()?;
    let is_true = condition
        .as_bool()
        .ok_or_else(|| ExecutionError::TypeMismatch {
            expected: "Bool".into(),
            found: format!("{:?}", condition),
        })?;

    if !is_true {
        let severity = instruction
            .operands
            .get(0)
            .and_then(|op| operand_string(op).ok());
        let message = instruction
            .operands
            .get(1)
            .and_then(|op| operand_string(op).ok());
        return ctx.trigger_assertion_fault(severity.as_deref(), message.as_deref());
    }
    Ok(())
}
