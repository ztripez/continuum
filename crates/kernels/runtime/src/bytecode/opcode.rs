//! Opcode definitions and metadata for the Continuum bytecode VM.
//!
//! This module defines the instruction kinds for the bytecode VM. Opcodes are designed
//! to be **data, not behavior**: opcode semantics are defined by metadata tables and
//! kernel dispatch, not by large match statements in the executor.
//!
//! # Design Rules
//!
//! 1. **No hard-coded behavior** - Opcodes must not embed domain logic
//! 2. **No large match arms** - Use metadata tables and handler registries
//! 3. **Explicit ordering** - Never rely on enum discriminant order
//! 4. **Fail loudly** - Invalid opcodes/operands are assertion failures
//!
//! # Opcode Categories
//!
//! - **Stack** - Push/pop/load/store operations
//! - **Kernel** - Kernel calls (maths.*, vector.*, logic.*, compare.*)
//! - **Control** - Let bindings, aggregates, folds
//! - **Temporal** - Prev/current/dt access
//! - **Effect** - Emit, spawn, destroy (phase-restricted)
//! - **Observation** - Field reads (observer only)

use continuum_foundation::Phase;
use continuum_kernel_types::KernelId;
use serde::{Deserialize, Serialize};

use super::operand::Operand;
use super::registry::metadata_for;
use continuum_cdsl::ast::expr::AggregateOp;

/// Bytecode instruction kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
pub enum OpcodeKind {
    // === Stack Operations ===
    /// Push literal operand[0] onto the stack.
    PushLiteral,
    /// Load slot operand[0] and push the value.
    Load,
    /// Pop value and store into slot operand[0].
    Store,
    /// Duplicate the top stack value.
    Dup,
    /// Discard the top stack value.
    Pop,

    // === Constructors ===
    /// Pop N scalar values (operand[0]) and build a Vec2/Vec3/Vec4.
    BuildVector,
    /// Consume values for field names (operands), building a map.
    BuildStruct,

    // === Kernel Calls ===
    /// Call kernel with arg count operand[0].
    CallKernel,

    // === Control Flow ===
    /// Declare a local slot binding (operand[0]).
    Let,
    /// End a local binding scope.
    EndLet,
    /// Aggregate over entity operand[0] using block operand[2] and op operand[3].
    Aggregate,
    /// Fold over entity operand[0] using accumulator/element slots and block operand[3].
    Fold,
    /// Pop object and access field operand[0].
    FieldAccess,

    // === Temporal ===
    /// Load signal value by path operand[0].
    LoadSignal,
    /// Load config value by path operand[0].
    LoadConfig,
    /// Load const value by path operand[0].
    LoadConst,
    /// Load previous tick value for the current signal.
    LoadPrev,
    /// Load current resolved value for the current signal.
    LoadCurrent,
    /// Load accumulated inputs for the current signal.
    LoadInputs,
    /// Load time step (dt).
    LoadDt,
    /// Load the current entity instance (self).
    LoadSelf,
    /// Load the current "other" entity instance.
    LoadOther,
    /// Load the impulse payload value.
    LoadPayload,

    // === Effect ===
    /// Emit a value to signal path operand[0].
    Emit,
    /// Emit a value to field path operand[0] with position.
    EmitField,
    /// Spawn a new entity instance of operand[0].
    Spawn,
    /// Destroy an entity instance of operand[0].
    Destroy,

    // === Observation ===
    /// Load a field value by path operand[0].
    LoadField,

    // === Structural ===
    /// Return from the current block.
    Return,
}

/// Bytecode instruction.
///
/// Instructions consist of a kind and a list of operands. The meaning of each
/// operand is defined by the opcode metadata table and executor handlers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    /// Instruction kind
    pub kind: OpcodeKind,
    /// Operand list
    pub operands: Vec<Operand>,
    /// Optional kernel id (only valid for CallKernel)
    pub kernel: Option<KernelId>,
}

impl Instruction {
    /// Create a new instruction without a kernel
    ///
    /// The operand ordering must match the opcode metadata for the specified kind.
    pub fn new(kind: OpcodeKind, operands: Vec<Operand>) -> Self {
        Self {
            kind,
            operands,
            kernel: None,
        }
    }

    /// Create a kernel call instruction
    ///
    /// The operands must include the argument count as a literal integer.
    pub fn kernel_call(kernel: KernelId, operands: Vec<Operand>) -> Self {
        Self {
            kind: OpcodeKind::CallKernel,
            operands,
            kernel: Some(kernel),
        }
    }
}

/// Metadata for an opcode.
///
/// This table-driven approach avoids hard-coding opcode behavior in match statements.
#[derive(Debug, Clone)]
pub struct OpcodeMetadata {
    /// Operand ordering follows the opcode kind docs and is positional.
    /// Operand count specification
    pub operand_count: OperandCount,
    /// Whether this opcode has side effects
    pub has_effect: bool,
    /// Phase restrictions (None = any phase)
    pub allowed_phases: Option<&'static [Phase]>,
}

/// Operand count specification for an opcode.
#[derive(Debug, Clone, Copy)]
pub enum OperandCount {
    /// Fixed operand count
    Fixed(usize),
    /// Variable operand count with optional bounds
    Variable { min: usize, max: Option<usize> },
}

impl OperandCount {
    /// Check if the given operand length is valid for this count.
    ///
    /// For `Variable`, `min` is inclusive and `max` is optional/inclusive.
    pub fn matches(self, len: usize) -> bool {
        match self {
            OperandCount::Fixed(count) => len == count,
            OperandCount::Variable { min, max } => len >= min && max.map_or(true, |max| len <= max),
        }
    }
}

impl OpcodeKind {
    /// Get metadata for this opcode kind.
    ///
    /// # Panics
    ///
    /// Panics if no metadata entry exists for the opcode kind.
    pub fn metadata(self) -> &'static OpcodeMetadata {
        metadata_for(self)
    }

    /// Check if this opcode is valid in the given phase.
    pub fn is_valid_in_phase(self, phase: Phase) -> bool {
        match self.metadata().allowed_phases {
            None => true,
            Some(allowed) => allowed.contains(&phase),
        }
    }

    /// Check if this opcode has side effects.
    pub fn has_effect(self) -> bool {
        self.metadata().has_effect
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_foundation::Path;

    #[test]
    fn test_opcode_metadata() {
        let meta = OpcodeKind::PushLiteral.metadata();
        assert!(matches!(meta.operand_count, OperandCount::Fixed(1)));
    }

    #[test]
    fn test_phase_restrictions() {
        assert!(OpcodeKind::Emit.is_valid_in_phase(Phase::Collect));
        assert!(!OpcodeKind::Emit.is_valid_in_phase(Phase::Resolve));
        assert!(!OpcodeKind::Emit.is_valid_in_phase(Phase::Measure));
    }

    #[test]
    fn test_instruction_builder() {
        let instr = Instruction::new(
            OpcodeKind::LoadSignal,
            vec![Operand::Signal(Path::from_str("force"))],
        );
        assert_eq!(instr.kind, OpcodeKind::LoadSignal);
        assert!(instr.kernel.is_none());
    }

    #[test]
    fn test_kernel_instruction_builder() {
        let kernel = KernelId::new("maths", "add");
        let instr = Instruction::kernel_call(kernel, vec![]);
        assert_eq!(instr.kind, OpcodeKind::CallKernel);
        assert!(instr.kernel.is_some());
    }

    #[test]
    fn test_aggregate_operands() {
        let instr = Instruction::new(
            OpcodeKind::Aggregate,
            vec![
                Operand::Entity(continuum_foundation::EntityId::new("plate")),
                Operand::Slot(super::super::operand::Slot::new(0)),
                Operand::Block(super::super::operand::BlockId::new(1)),
                Operand::AggregateOp(AggregateOp::Sum),
            ],
        );
        assert_eq!(instr.operands.len(), 4);
    }
}
