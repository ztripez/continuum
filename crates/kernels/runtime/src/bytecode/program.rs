//! Bytecode program and block structures.

use crate::bytecode::opcode::Instruction;
use crate::bytecode::operand::BlockId;

/// A complete bytecode program consisting of one or more execution blocks.
///
/// Programs are the primary artifact produced by the compiler and executed by
/// the VM. They represent a self-contained computation (like a signal resolve
/// or an operator effect) and include all nested blocks required for control
/// flow operations like `Aggregate` or `Fold`.
#[derive(Debug, Clone)]
pub struct BytecodeProgram {
    /// Ordered list of blocks in the program.
    ///
    /// By convention, the root block is at index 0 (ID 0).
    blocks: Vec<BytecodeBlock>,
}

impl BytecodeProgram {
    /// Creates a new, empty bytecode program.
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Appends an execution block to the program and returns its assigned ID.
    ///
    /// Identifiers are assigned sequentially, matching the block's index in
    /// the internal storage.
    pub fn add_block(&mut self, block: BytecodeBlock) -> BlockId {
        let id = BlockId::new(self.blocks.len() as u32);
        self.blocks.push(block);
        id
    }

    /// Creates a program from a pre-defined list of blocks.
    pub fn from_blocks(blocks: Vec<BytecodeBlock>) -> Self {
        Self { blocks }
    }

    /// Retrieves a reference to a block by its identifier.
    ///
    /// # Returns
    ///
    /// `Some(&BytecodeBlock)` if the ID is valid, or `None` if it is out of range.
    pub fn block(&self, id: BlockId) -> Option<&BytecodeBlock> {
        self.blocks.get(id.id() as usize)
    }

    /// Returns a slice of all blocks contained in the program.
    ///
    /// Used by the compiler during validation to perform static analysis
    /// across the entire program's instruction space.
    pub(crate) fn blocks(&self) -> &[BytecodeBlock] {
        &self.blocks
    }
}

/// A linear sequence of instructions representing a unit of execution.
///
/// Blocks are the basic executable entities in the VM. Each block manages
/// its own instruction sequence and specifies whether it yields a result.
///
/// Every block should conclude with a [`OpcodeKind::Return`] instruction to
/// ensure proper stack cleanup and control flow return.
#[derive(Debug, Clone)]
pub struct BytecodeBlock {
    /// The sequential list of instructions to execute.
    pub instructions: Vec<Instruction>,
    /// Indicates if this block is expected to push a final result onto the stack.
    ///
    /// If `true`, the executor will validate that exactly one value remains
    /// on the stack when the block returns.
    pub returns_value: bool,
}

impl BytecodeBlock {
    /// Creates a new, empty bytecode block.
    ///
    /// # Parameters
    /// - `returns_value`: Whether the block's computation produces a result.
    pub fn new(returns_value: bool) -> Self {
        Self {
            instructions: Vec::new(),
            returns_value,
        }
    }
}
