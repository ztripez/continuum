//! Bytecode program and block structures.

use crate::bytecode::opcode::Instruction;
use crate::bytecode::operand::BlockId;

/// Bytecode program with nested blocks.
#[derive(Debug, Clone)]
pub struct BytecodeProgram {
    blocks: Vec<BytecodeBlock>,
}

impl BytecodeProgram {
    /// Create an empty program
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Add a block to the program and return its block id
    ///
    /// Block ids are assigned in insertion order to preserve determinism.
    pub fn add_block(&mut self, block: BytecodeBlock) -> BlockId {
        let id = BlockId::new(self.blocks.len() as u32);
        self.blocks.push(block);
        id
    }

    /// Create a program from an explicit block list
    pub fn from_blocks(blocks: Vec<BytecodeBlock>) -> Self {
        Self { blocks }
    }

    /// Get a block by id
    ///
    /// Returns `None` when the block id is out of range.
    pub fn block(&self, id: BlockId) -> Option<&BytecodeBlock> {
        self.blocks.get(id.id() as usize)
    }

    pub(crate) fn blocks(&self) -> &[BytecodeBlock] {
        &self.blocks
    }
}

/// Bytecode block containing a linear sequence of instructions.
///
/// Blocks must end with a [`OpcodeKind::Return`] instruction.
#[derive(Debug, Clone)]
pub struct BytecodeBlock {
    /// Instruction sequence
    pub instructions: Vec<Instruction>,
    /// Whether this block must return a value
    pub returns_value: bool,
}

impl BytecodeBlock {
    /// Create an empty block
    pub fn new(returns_value: bool) -> Self {
        Self {
            instructions: Vec::new(),
            returns_value,
        }
    }
}
