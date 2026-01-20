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

use continuum_foundation::{Phase, Value};

use crate::bytecode::compiler::CompiledBlock;
use crate::bytecode::opcode::{Instruction, OpcodeKind};
use crate::bytecode::operand::{BlockId, Slot};
use crate::bytecode::program::{BytecodeBlock, BytecodeProgram};
use crate::bytecode::registry::handler_for;
use crate::bytecode::runtime::{ExecutionContext, ExecutionError, ExecutionRuntime};

/// Executes compiled bytecode blocks within the runtime.
pub struct BytecodeExecutor {
    /// Value stack
    stack: Vec<Value>,
    /// Slot storage
    slots: Vec<Option<Value>>,
}

impl BytecodeExecutor {
    pub fn new() -> Self {
        Self {
            stack: Vec::with_capacity(256),
            slots: Vec::with_capacity(256),
        }
    }

    /// Execute a compiled bytecode block
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Stack underflow/overflow
    /// - Invalid slot access
    /// - Kernel call fails
    /// - Type mismatch
    /// - Invalid opcode or operand counts
    /// - Missing or unexpected return value
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

    fn execute_block(
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

        for instruction in &block.instructions {
            self.validate_instruction(instruction)?;
            if instruction.kind == OpcodeKind::Return {
                return self.handle_return(block, base);
            }
            handler_for(instruction.kind)(instruction, self, program, ctx)?;
        }

        self.stack.truncate(base);
        Err(ExecutionError::MissingReturn)
    }

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

    fn push(&mut self, value: Value) -> Result<(), ExecutionError> {
        if self.stack.len() >= 1024 {
            return Err(ExecutionError::StackOverflow);
        }
        self.stack.push(value);
        Ok(())
    }

    fn pop(&mut self) -> Result<Value, ExecutionError> {
        self.stack.pop().ok_or(ExecutionError::StackUnderflow)
    }

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
}

impl Default for BytecodeExecutor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests;
