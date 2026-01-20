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
///
/// Variants represent the atomic operations supported by the VM. Opcodes are
/// categorized by their purpose and potential side effects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
pub enum OpcodeKind {
    // === Stack Operations ===
    /// Pushes the literal value from operand[0] onto the evaluation stack.
    PushLiteral,
    /// Loads a value from the slot index specified in operand[0] and pushes it.
    Load,
    /// Pops the top stack value and stores it in the slot index specified in operand[0].
    Store,
    /// Duplicates the current top value on the evaluation stack.
    Dup,
    /// Discards the top value from the evaluation stack.
    Pop,

    // === Constructors ===
    /// Pops the number of scalar values specified in operand[0] and constructs a vector value.
    ///
    /// Supports Vec2, Vec3, and Vec4 results.
    BuildVector,
    /// Constructs a Map value from the evaluation stack using field names provided in operands.
    ///
    /// Pops one value per field name in reverse order.
    BuildStruct,

    // === Kernel Calls ===
    /// Dispatches a call to an engine kernel (maths, physics, etc.).
    ///
    /// Argument count is provided in operand[0]. Arguments are popped from the stack.
    CallKernel,

    // === Control Flow ===
    /// Marks the start of a local binding scope for the slot in operand[0].
    ///
    /// This is used for lifecycle tracking and is usually a no-op in the executor.
    Let,
    /// Marks the end of a local binding scope.
    EndLet,
    /// Iterates over entities and reduces results using an aggregate operation.
    ///
    /// Operands: [EntityId, BindingSlot, BlockId, AggregateOp]
    Aggregate,
    /// Iterates over entities and performs a stateful reduction (fold).
    ///
    /// Operands: [EntityId, AccSlot, ElemSlot, BlockId]
    Fold,
    /// Pops a Map/Struct and accesses the field named in operand[0].
    FieldAccess,

    // === Temporal ===
    /// Loads the current value of a signal by its path in operand[0].
    LoadSignal,
    /// Loads a configuration value by its path in operand[0].
    LoadConfig,
    /// Loads a constant value by its path in operand[0].
    LoadConst,
    /// Loads the value of the current signal from the previous tick.
    LoadPrev,
    /// Loads the current resolved value of the current signal.
    ///
    /// Only valid in phases where signal resolution has completed (e.g., Fracture, Measure).
    LoadCurrent,
    /// Loads the accumulated inputs for the current signal.
    ///
    /// Only valid during the Resolve phase.
    LoadInputs,
    /// Loads the current time step (dt).
    LoadDt,
    /// Loads the identity (Value::Entity) of the current entity instance.
    LoadSelf,
    /// Loads the identity of the "other" entity (e.g., in collision contexts).
    LoadOther,
    /// Loads the payload data associated with an impulse.
    LoadPayload,

    // === Effect ===
    /// Emits a value to the signal path in operand[0].
    ///
    /// Only valid in the Collect phase.
    Emit,
    /// Emits a value to a spatial field at a given position.
    ///
    /// Pops [Value, Position] and writes to field path in operand[0]. Only valid in Measure phase.
    EmitField,
    /// Spawns a new instance of the entity type in operand[0].
    ///
    /// Only valid in the Fracture phase.
    Spawn,
    /// Marks the entity instance (popped from stack) of type in operand[0] for destruction.
    ///
    /// Only valid in the Fracture phase.
    Destroy,

    // === Observation ===
    /// Loads a value from a spatial field by its path in operand[0].
    ///
    /// Only valid in the Measure phase (observer boundary).
    LoadField,

    // === Structural ===
    /// Terminates the current block execution and returns the top stack value (if any).
    Return,
}

/// A single bytecode instruction.
///
/// Each instruction pairs an [`OpcodeKind`] with a list of [`Operand`]s required
/// for its execution. CallKernel instructions also carry a [`KernelId`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    /// The type of operation to perform.
    pub kind: OpcodeKind,
    /// Positional operands for the instruction.
    pub operands: Vec<Operand>,
    /// Target kernel identifier (only populated for CallKernel).
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
