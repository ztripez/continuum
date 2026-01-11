//! SSA IR type definitions.
//!
//! Defines the core types for Static Single Assignment intermediate representation.

use std::fmt;

use continuum_foundation::SignalId;

use crate::{AggregateOpIr, BinaryOpIr, DtRobustOperator, IntegrationMethod, UnaryOpIr};

/// Virtual register identifier.
///
/// Each VReg represents a unique value in SSA form. Values are assigned
/// exactly once and can be used multiple times.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VReg(pub u32);

impl fmt::Display for VReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "%{}", self.0)
    }
}

/// Basic block identifier.
///
/// SSA IR is organized into basic blocks, each containing a sequence of
/// instructions followed by a terminator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

impl fmt::Display for BlockId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "block{}", self.0)
    }
}

/// Complete SSA function representing a signal resolve expression.
///
/// An SSA function consists of one or more basic blocks. Execution starts
/// at block 0 and ends when a `Return` terminator is reached.
#[derive(Debug, Clone)]
pub struct SsaFunction {
    /// Basic blocks in the function.
    pub blocks: Vec<SsaBlock>,
    /// Total number of virtual registers used.
    pub vreg_count: u32,
}

impl SsaFunction {
    /// Create a new SSA function with a single entry block.
    pub fn new() -> Self {
        Self {
            blocks: vec![SsaBlock::new(BlockId(0))],
            vreg_count: 0,
        }
    }

    /// Allocate a new virtual register.
    pub fn alloc_vreg(&mut self) -> VReg {
        let reg = VReg(self.vreg_count);
        self.vreg_count += 1;
        reg
    }

    /// Allocate a new basic block.
    pub fn alloc_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        self.blocks.push(SsaBlock::new(id));
        id
    }

    /// Get a mutable reference to a block.
    pub fn block_mut(&mut self, id: BlockId) -> &mut SsaBlock {
        &mut self.blocks[id.0 as usize]
    }

    /// Get a reference to a block.
    pub fn block(&self, id: BlockId) -> &SsaBlock {
        &self.blocks[id.0 as usize]
    }

    /// Pretty-print the SSA function.
    pub fn pretty_print(&self) -> String {
        let mut out = String::new();
        for block in &self.blocks {
            out.push_str(&block.pretty_print());
            out.push('\n');
        }
        out
    }
}

impl Default for SsaFunction {
    fn default() -> Self {
        Self::new()
    }
}

/// A basic block in SSA form.
///
/// Contains a sequence of instructions followed by a terminator that controls
/// the flow to subsequent blocks.
#[derive(Debug, Clone)]
pub struct SsaBlock {
    /// Block identifier.
    pub id: BlockId,
    /// Instructions in this block (excluding terminator).
    pub instructions: Vec<SsaInstruction>,
    /// Block terminator (controls flow to next block).
    pub terminator: Option<Terminator>,
}

impl SsaBlock {
    /// Create a new empty block.
    pub fn new(id: BlockId) -> Self {
        Self {
            id,
            instructions: Vec::new(),
            terminator: None,
        }
    }

    /// Add an instruction to this block.
    pub fn push(&mut self, inst: SsaInstruction) {
        self.instructions.push(inst);
    }

    /// Set the block terminator.
    pub fn terminate(&mut self, term: Terminator) {
        self.terminator = Some(term);
    }

    /// Pretty-print this block.
    pub fn pretty_print(&self) -> String {
        let mut out = format!("{}:\n", self.id);
        for inst in &self.instructions {
            out.push_str(&format!("  {}\n", inst.pretty_print()));
        }
        if let Some(term) = &self.terminator {
            out.push_str(&format!("  {}\n", term.pretty_print()));
        }
        out
    }
}

/// An SSA instruction that produces a value.
///
/// Each instruction assigns its result to exactly one virtual register.
/// Instructions are pure and have no side effects.
#[derive(Debug, Clone)]
pub enum SsaInstruction {
    /// Load the previous value of the current signal.
    LoadPrev { dst: VReg },

    /// Load the raw dt (time step) value.
    LoadDt { dst: VReg },

    /// Load the accumulated simulation time in seconds.
    LoadSimTime { dst: VReg },

    /// Load the collected/accumulated value from Collect phase.
    LoadCollected { dst: VReg },

    /// Load a signal value by ID.
    LoadSignal { dst: VReg, signal: SignalId },

    /// Load a constant value.
    LoadConst { dst: VReg, value: f64 },

    /// Load a named constant (const.*).
    LoadNamedConst { dst: VReg, name: String },

    /// Load a config value (config.*).
    LoadConfig { dst: VReg, name: String },

    /// Load a local variable.
    LoadLocal { dst: VReg, name: String },

    /// Binary operation.
    BinOp {
        dst: VReg,
        op: BinaryOpIr,
        lhs: VReg,
        rhs: VReg,
    },

    /// Unary operation.
    UnaryOp { dst: VReg, op: UnaryOpIr, operand: VReg },

    /// User-defined function call.
    Call {
        dst: VReg,
        function: String,
        args: Vec<VReg>,
    },

    /// Kernel function call (engine-provided).
    KernelCall {
        dst: VReg,
        function: String,
        args: Vec<VReg>,
    },

    /// dt-robust operator call.
    DtRobustCall {
        dst: VReg,
        operator: DtRobustOperator,
        args: Vec<VReg>,
        method: IntegrationMethod,
    },

    /// Field access on a value.
    FieldAccess { dst: VReg, object: VReg, field: String },

    /// Phi node for control flow merge.
    ///
    /// Selects a value based on which predecessor block was executed.
    Phi { dst: VReg, arms: Vec<(BlockId, VReg)> },

    /// Store to a local variable.
    StoreLocal { name: String, value: VReg },

    // === Entity operations ===

    /// Access current entity instance field.
    SelfField { dst: VReg, field: String },

    /// Aggregate operation over entity instances.
    Aggregate {
        dst: VReg,
        op: AggregateOpIr,
        entity: String,
        /// Block ID containing the body expression for each instance.
        body_block: BlockId,
    },
}

impl SsaInstruction {
    /// Get the destination register of this instruction.
    pub fn dst(&self) -> Option<VReg> {
        match self {
            SsaInstruction::LoadPrev { dst }
            | SsaInstruction::LoadDt { dst }
            | SsaInstruction::LoadSimTime { dst }
            | SsaInstruction::LoadCollected { dst }
            | SsaInstruction::LoadSignal { dst, .. }
            | SsaInstruction::LoadConst { dst, .. }
            | SsaInstruction::LoadNamedConst { dst, .. }
            | SsaInstruction::LoadConfig { dst, .. }
            | SsaInstruction::LoadLocal { dst, .. }
            | SsaInstruction::BinOp { dst, .. }
            | SsaInstruction::UnaryOp { dst, .. }
            | SsaInstruction::Call { dst, .. }
            | SsaInstruction::KernelCall { dst, .. }
            | SsaInstruction::DtRobustCall { dst, .. }
            | SsaInstruction::FieldAccess { dst, .. }
            | SsaInstruction::Phi { dst, .. }
            | SsaInstruction::SelfField { dst, .. }
            | SsaInstruction::Aggregate { dst, .. } => Some(*dst),
            SsaInstruction::StoreLocal { .. } => None,
        }
    }

    /// Get all registers used by this instruction.
    pub fn uses(&self) -> Vec<VReg> {
        match self {
            SsaInstruction::LoadPrev { .. }
            | SsaInstruction::LoadDt { .. }
            | SsaInstruction::LoadSimTime { .. }
            | SsaInstruction::LoadCollected { .. }
            | SsaInstruction::LoadSignal { .. }
            | SsaInstruction::LoadConst { .. }
            | SsaInstruction::LoadNamedConst { .. }
            | SsaInstruction::LoadConfig { .. }
            | SsaInstruction::LoadLocal { .. }
            | SsaInstruction::SelfField { .. } => vec![],

            SsaInstruction::BinOp { lhs, rhs, .. } => vec![*lhs, *rhs],

            SsaInstruction::UnaryOp { operand, .. } => vec![*operand],

            SsaInstruction::Call { args, .. }
            | SsaInstruction::KernelCall { args, .. }
            | SsaInstruction::DtRobustCall { args, .. } => args.clone(),

            SsaInstruction::FieldAccess { object, .. } => vec![*object],

            SsaInstruction::Phi { arms, .. } => arms.iter().map(|(_, v)| *v).collect(),

            SsaInstruction::StoreLocal { value, .. } => vec![*value],

            SsaInstruction::Aggregate { .. } => vec![], // Body is in separate block
        }
    }

    /// Pretty-print this instruction.
    pub fn pretty_print(&self) -> String {
        match self {
            SsaInstruction::LoadPrev { dst } => format!("{} = LoadPrev", dst),
            SsaInstruction::LoadDt { dst } => format!("{} = LoadDt", dst),
            SsaInstruction::LoadSimTime { dst } => format!("{} = LoadSimTime", dst),
            SsaInstruction::LoadCollected { dst } => format!("{} = LoadCollected", dst),
            SsaInstruction::LoadSignal { dst, signal } => {
                format!("{} = LoadSignal({})", dst, signal.0)
            }
            SsaInstruction::LoadConst { dst, value } => format!("{} = LoadConst({})", dst, value),
            SsaInstruction::LoadNamedConst { dst, name } => {
                format!("{} = LoadNamedConst({})", dst, name)
            }
            SsaInstruction::LoadConfig { dst, name } => format!("{} = LoadConfig({})", dst, name),
            SsaInstruction::LoadLocal { dst, name } => format!("{} = LoadLocal({})", dst, name),
            SsaInstruction::BinOp { dst, op, lhs, rhs } => {
                format!("{} = {:?}({}, {})", dst, op, lhs, rhs)
            }
            SsaInstruction::UnaryOp { dst, op, operand } => {
                format!("{} = {:?}({})", dst, op, operand)
            }
            SsaInstruction::Call { dst, function, args } => {
                let args_str: Vec<_> = args.iter().map(|a| a.to_string()).collect();
                format!("{} = Call({}, [{}])", dst, function, args_str.join(", "))
            }
            SsaInstruction::KernelCall { dst, function, args } => {
                let args_str: Vec<_> = args.iter().map(|a| a.to_string()).collect();
                format!(
                    "{} = KernelCall({}, [{}])",
                    dst,
                    function,
                    args_str.join(", ")
                )
            }
            SsaInstruction::DtRobustCall {
                dst,
                operator,
                args,
                method,
            } => {
                let args_str: Vec<_> = args.iter().map(|a| a.to_string()).collect();
                format!(
                    "{} = DtRobust({:?}, [{}], {:?})",
                    dst,
                    operator,
                    args_str.join(", "),
                    method
                )
            }
            SsaInstruction::FieldAccess { dst, object, field } => {
                format!("{} = FieldAccess({}, {})", dst, object, field)
            }
            SsaInstruction::Phi { dst, arms } => {
                let arms_str: Vec<_> = arms
                    .iter()
                    .map(|(b, v)| format!("{}: {}", b, v))
                    .collect();
                format!("{} = Phi([{}])", dst, arms_str.join(", "))
            }
            SsaInstruction::StoreLocal { name, value } => {
                format!("StoreLocal({}, {})", name, value)
            }
            SsaInstruction::SelfField { dst, field } => format!("{} = SelfField({})", dst, field),
            SsaInstruction::Aggregate {
                dst,
                op,
                entity,
                body_block,
            } => {
                format!("{} = Aggregate({:?}, {}, {})", dst, op, entity, body_block)
            }
        }
    }
}

/// Block terminator that controls flow to subsequent blocks.
#[derive(Debug, Clone)]
pub enum Terminator {
    /// Return a value from the function.
    Return(VReg),

    /// Unconditional jump to a block.
    Jump(BlockId),

    /// Conditional branch.
    Branch {
        cond: VReg,
        then_block: BlockId,
        else_block: BlockId,
    },
}

impl Terminator {
    /// Get all registers used by this terminator.
    pub fn uses(&self) -> Vec<VReg> {
        match self {
            Terminator::Return(v) => vec![*v],
            Terminator::Jump(_) => vec![],
            Terminator::Branch { cond, .. } => vec![*cond],
        }
    }

    /// Get successor blocks.
    pub fn successors(&self) -> Vec<BlockId> {
        match self {
            Terminator::Return(_) => vec![],
            Terminator::Jump(b) => vec![*b],
            Terminator::Branch {
                then_block,
                else_block,
                ..
            } => vec![*then_block, *else_block],
        }
    }

    /// Pretty-print this terminator.
    pub fn pretty_print(&self) -> String {
        match self {
            Terminator::Return(v) => format!("Return({})", v),
            Terminator::Jump(b) => format!("Jump({})", b),
            Terminator::Branch {
                cond,
                then_block,
                else_block,
            } => format!("Branch({}, {}, {})", cond, then_block, else_block),
        }
    }
}
