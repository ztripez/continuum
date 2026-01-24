//! Bytecode executor integrated with runtime DAG.
//!
//! This module implements the bytecode interpreter that executes compiled bytecode
//! within the deterministic DAG runtime.
//!
//! # Design Principles
//!
//! 1. **Deterministic** - Iteration order explicit and stable
//! 2. **Fail loudly** - Invalid opcodes/indices are assertion failures
//! 3. **Phase-safe** - Respects observer boundaries and capability constraints
//! 4. **No silent fallbacks** - All errors are explicit
//!
//! # Execution Model
//!
//! The executor runs bytecode within a stack-based VM:
//! - Value stack for computation
//! - Slot array for locals, signals, temporaries
//! - Entity iteration state
//! - Kernel call dispatch

use continuum_foundation::Value;

use crate::bytecode::compiler::CompiledBlock;
use crate::bytecode::opcode::{Instruction, OpcodeKind};
use crate::bytecode::operand::{BlockId, Slot};
use crate::bytecode::program::{BytecodeBlock, BytecodeProgram};
use crate::bytecode::registry::handler_for;
use crate::bytecode::runtime::{ExecutionContext, ExecutionError, ExecutionRuntime};

/// Executes compiled bytecode blocks within the deterministic runtime.
///
/// The executor maintains its own evaluation stack and slot storage, which are
/// reset for each execution run. It dispatches instructions to their respective
/// handlers registered in the opcode registry.
///
/// # Execution Safety
/// - **Stack Limits**: Enforces a maximum stack depth to prevent recursion-based overflows.
/// - **Phase Validation**: Ensures that the compiled block's phase matches the
///   active runtime phase.
/// - **Deterministic Dispatch**: Uses table-driven handler lookup for O(1) execution
///   without complex branch logic.
pub struct BytecodeExecutor {
    /// Evaluation stack for computation results and intermediate values.
    stack: Vec<Value>,
    /// Slot storage for locals, signals, and temporaries allocated during compilation.
    slots: Vec<Option<Value>>,
    /// Jump target for control flow (None = continue normally, Some(offset) = jump).
    jump_target: Option<i32>,
}

/// Maximum depth of the VM evaluation stack.
const MAX_STACK_DEPTH: usize = 1024;

impl BytecodeExecutor {
    /// Creates a new bytecode executor with pre-allocated capacity for the stack and slots.
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(256),
            slots: Vec::with_capacity(256),
            jump_target: None,
        }
    }

    /// Executes a compiled bytecode block in the provided context.
    ///
    /// This is the main entry point for running simulation logic via bytecode.
    ///
    /// # Parameters
    /// - `block`: The compiled bytecode and metadata.
    /// - `ctx`: The runtime context providing state access and side-effect dispatch.
    ///
    /// # Returns
    /// - `Ok(Some(Value))` if the block returned a value.
    /// - `Ok(None)` if the block finished without a return value.
    /// - `Err(ExecutionError)` if a runtime violation occurred.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The block phase does not match the context phase.
    /// - A stack underflow or overflow occurs.
    /// - A handler returns a domain error.
    pub fn execute(
        &mut self,
        block: &CompiledBlock,
        ctx: &mut dyn ExecutionContext,
    ) -> Result<Option<Value>, ExecutionError> {
        if block.phase != ctx.phase() {
            return Err(ExecutionError::InvalidOpcode {
                opcode: format!("PhaseMismatch({:?} != {:?})", block.phase, ctx.phase()),
                phase: ctx.phase(),
            });
        }
        self.stack.clear();
        self.slots.clear();
        self.slots.resize(block.slot_count as usize, None);
        self.execute_block(block.root, &block.program, ctx)
    }

    /// Executes a specific block from a bytecode program.
    ///
    /// Used for the root entry point and for sub-computations (aggregates/folds).
    ///
    /// # Parameters
    /// - `block_id`: The identifier of the block to execute.
    /// - `program`: The complete bytecode program containing the block.
    /// - `ctx`: The runtime context for simulation interface.
    ///
    /// # Returns
    /// The block's return value, or `None` if it does not yield a result.
    ///
    /// # Errors
    /// Returns [`ExecutionError`] if the block ID is invalid or a runtime
    /// violation occurs during execution.
    pub fn execute_block(
        &mut self,
        block_id: BlockId,
        program: &BytecodeProgram,
        ctx: &mut dyn ExecutionContext,
    ) -> Result<Option<Value>, ExecutionError> {
        let block = program
            .block(block_id)
            .ok_or(ExecutionError::InvalidBlock {
                block: block_id.id(),
            })?;
        let base = self.stack.len();

        let mut ip: isize = 0; // Instruction pointer
        while ip >= 0 && (ip as usize) < block.instructions.len() {
            let instruction = &block.instructions[ip as usize];
            self.validate_instruction(instruction)?;

            if instruction.kind == OpcodeKind::Return {
                return self.handle_return(block, base);
            }

            // Execute handler
            handler_for(instruction.kind)(instruction, self, program, ctx)?;

            // Check for jump request
            if let Some(offset) = self.jump_target.take() {
                let target = ip + offset as isize;
                if target < 0 || target as usize >= block.instructions.len() {
                    return Err(ExecutionError::InvalidJump {
                        offset,
                        position: ip as usize,
                        block_size: block.instructions.len(),
                    });
                }
                ip = target;
            } else {
                ip += 1;
            }
        }

        self.stack.truncate(base);
        Err(ExecutionError::MissingReturn)
    }

    /// Validates instruction operands against the static opcode metadata.
    ///
    /// # Errors
    /// Returns [`ExecutionError::InvalidOperand`] if the operand count does
    /// not match the opcode's static specification.
    fn validate_instruction(&self, instruction: &Instruction) -> Result<(), ExecutionError> {
        let meta = instruction.kind.metadata();
        if !meta.operand_count.matches(instruction.operands.len()) {
            return Err(ExecutionError::InvalidOperand {
                message: format!(
                    "Opcode {name:?} expected {expected:?} operands, got {actual}",
                    name = instruction.kind,
                    expected = meta.operand_count,
                    actual = instruction.operands.len()
                ),
            });
        }
        Ok(())
    }

    /// Internal helper to push a value onto the evaluation stack.
    ///
    /// # Errors
    /// Returns [`ExecutionError::StackOverflow`] if the stack depth exceeds
    /// the hard limit of [`MAX_STACK_DEPTH`].
    fn push(&mut self, value: Value) -> Result<(), ExecutionError> {
        if self.stack.len() >= MAX_STACK_DEPTH {
            return Err(ExecutionError::StackOverflow);
        }
        self.stack.push(value);
        Ok(())
    }

    /// Internal helper to pop a value from the evaluation stack.
    ///
    /// # Errors
    /// Returns [`ExecutionError::StackUnderflow`] if the stack is empty.
    fn pop(&mut self) -> Result<Value, ExecutionError> {
        self.stack.pop().ok_or(ExecutionError::StackUnderflow)
    }

    /// Internal helper to load a value from a VM slot.
    ///
    /// # Errors
    /// Returns an error if the slot index is out of bounds or if the slot
    /// has not been initialized with a value.
    fn load_slot(&self, slot: Slot) -> Result<Value, ExecutionError> {
        let slot_value = self
            .slots
            .get(slot.id() as usize)
            .ok_or(ExecutionError::InvalidSlot {
                slot: slot.id(),
                max: self.slots.len() as u32,
            })?;
        slot_value
            .clone()
            .ok_or(ExecutionError::UninitializedSlot { slot: slot.id() })
    }

    /// Internal helper to store a value into a VM slot.
    ///
    /// # Errors
    /// Returns [`ExecutionError::InvalidSlot`] if the slot index is out of bounds.
    fn store_slot(&mut self, slot: Slot, value: Value) -> Result<(), ExecutionError> {
        if slot.id() as usize >= self.slots.len() {
            return Err(ExecutionError::InvalidSlot {
                slot: slot.id(),
                max: self.slots.len() as u32,
            });
        }
        self.slots[slot.id() as usize] = Some(value);
        Ok(())
    }

    /// Handles the return sequence for a block, validating stack state.
    ///
    /// # Parameters
    /// - `block`: The block finishing execution.
    /// - `base`: The stack depth at the start of block execution.
    ///
    /// # Returns
    /// The block's result value (if any).
    ///
    /// # Errors
    /// Returns [`ExecutionError`] if the stack depth is inconsistent with the
    /// block's `returns_value` property.
    fn handle_return(
        &mut self,
        block: &BytecodeBlock,
        base: usize,
    ) -> Result<Option<Value>, ExecutionError> {
        if block.returns_value {
            if self.stack.len() == base {
                return Err(ExecutionError::MissingReturn);
            }
            if self.stack.len() != base + 1 {
                return Err(ExecutionError::UnexpectedStackDepth {
                    expected: base + 1,
                    found: self.stack.len(),
                });
            }
            let value = self.pop()?;
            self.stack.truncate(base);
            Ok(Some(value))
        } else {
            if self.stack.len() != base {
                return Err(ExecutionError::UnexpectedStackDepth {
                    expected: base,
                    found: self.stack.len(),
                });
            }
            self.stack.truncate(base);
            Ok(None)
        }
    }
}

impl ExecutionRuntime for BytecodeExecutor {
    fn push(&mut self, value: Value) -> Result<(), ExecutionError> {
        self.push(value)
    }

    fn pop(&mut self) -> Result<Value, ExecutionError> {
        self.pop()
    }

    fn load_slot(&self, slot: Slot) -> Result<Value, ExecutionError> {
        self.load_slot(slot)
    }

    fn store_slot(&mut self, slot: Slot, value: Value) -> Result<(), ExecutionError> {
        self.store_slot(slot, value)
    }

    fn execute_block(
        &mut self,
        block_id: BlockId,
        program: &BytecodeProgram,
        ctx: &mut dyn ExecutionContext,
    ) -> Result<Option<Value>, ExecutionError> {
        self.execute_block(block_id, program, ctx)
    }

    fn jump(&mut self, offset: i32) -> Result<(), ExecutionError> {
        self.jump_target = Some(offset);
        Ok(())
    }
}

impl Default for BytecodeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod assert_tests;
