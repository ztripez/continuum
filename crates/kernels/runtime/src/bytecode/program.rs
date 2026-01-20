//! Bytecode program and block structures.

use crate::bytecode::opcode::Instruction;
use crate::bytecode::operand::BlockId;

/// A complete bytecode program consisting of one or more execution blocks.
///
/// Programs are the unit of exchange between the compiler and the executor.
/// They contain all the instructions required to execute a simulation fragment,
/// including any nested blocks for sub-computations.
#[derive(Debug, Clone)]
pub struct BytecodeProgram {
    /// List of blocks in the program. The root block is typically the first one.
    blocks: Vec<BytecodeBlock>,
}

impl BytecodeProgram {
    /// Create a new, empty bytecode program.
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Appends a block to the program and returns its unique identifier.
    ///
    /// Identifiers are assigned sequentially to ensure deterministic program structure.
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
    /// Returns `None` if the identifier is invalid or out of range.
    pub fn block(&self, id: BlockId) -> Option<&BytecodeBlock> {
        self.blocks.get(id.id() as usize)
    }

    /// Returns a slice of all blocks in the program.
    pub(crate) fn blocks(&self) -> &[BytecodeBlock] {
        &self.blocks
    }
}

/// A sequential list of instructions within a program.
///
/// Blocks are the basic unit of execution in the VM. Each block must end
/// with a `Return` instruction to yield control back to the caller or executor.
#[derive(Debug, Clone)]
pub struct BytecodeBlock {
    /// The linear sequence of instructions to execute.
    pub instructions: Vec<Instruction>,
    /// Whether this block is expected to leave a value on the stack upon return.
    pub returns_value: bool,
}

impl BytecodeBlock {
    /// Creates a new, empty bytecode block.
    pub fn new(returns_value: bool) -> Self {
        Self {
            instructions: Vec::new(),
            returns_value,
        }
    }
}
