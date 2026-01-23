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

/// Bytecode instruction kind.
///
/// Variants represent the atomic operations supported by the VM. Opcodes are
/// categorized by their purpose and potential side effects.
///
/// Each variant corresponds to a runtime [`super::handlers::Handler`] registered
/// in the [`super::registry::opcode_specs`] table.
///
/// # Instruction Reference
///
/// | Opcode | Category | Operands | Description | Allowed Phases |
/// |--------|----------|----------|-------------|----------------|
/// | `PushLiteral` | Stack | `Value` | Pushes a constant value onto the evaluation stack. | All |
/// | `Load` | Stack | `Slot` | Pushes the value from the specified local slot onto the stack. | All |
/// | `Store` | Stack | `Slot` | Pops the top stack value and stores it in the specified local slot. | All |
/// | `Dup` | Stack | - | Duplicates the current top value on the evaluation stack. | All |
/// | `Pop` | Stack | - | Discards the top value from the evaluation stack. | All |
/// | `BuildVector` | Constructor | `dim: u8` | Pops `dim` scalar values and pushes a `VecN` result. | All |
/// | `BuildStruct` | Constructor | `fields: Vec<String>` | Pops values for each field and pushes a `Map`. | All |
/// | `CallKernel` | Kernel | `arg_count: u8`, `KernelId` | Dispatches a call to an engine-provided kernel. | All (per kernel purity) |
/// | `Let` | Control | `Slot` | Marks the start of a local binding scope. | All |
/// | `EndLet` | Control | - | Marks the end of a local binding scope. | All |
/// | `Aggregate` | Control | `EntityId, Slot, BlockId, Op` | Iterates over entity instances, executing a block for each. | All |
/// | `Fold` | Control | `EntityId, Acc, Elem, BlockId` | Stateful reduction over entity instances. | All |
/// | `FieldAccess` | Control | `field: String` | Pops a Map/Struct and accesses the named field. | All |
/// | `LoadSignal` | Temporal | `path: Path` | Loads the current value of a signal by its path. | `Resolve`, `Fracture`, `Measure` |
/// | `LoadPrev` | Temporal | - | Loads the previous tick value of the current signal. | `Resolve` |
/// | `LoadCurrent` | Temporal | - | Loads the current resolved value of the current signal. | `Fracture`, `Measure` |
/// | `LoadDt` | Temporal | - | Loads the current timestep (dt) in seconds. | All |
/// | `Emit` | Effect | `path: Path` | Accumulates a value to a signal. | `Collect` |
/// | `EmitField` | Effect | `path: Path` | Writes a value to a spatial field at a given position. | `Measure` |
/// | `Spawn` | Effect | `entity: EntityId` | Creates a new instance of the specified entity type. | `Fracture` |
/// | `Destroy` | Effect | `entity: EntityId` | Marks an entity instance for destruction. | `Fracture` |
/// | `LoadField` | Observation | `path: Path` | Loads a value from a spatial field by its path. | `Measure` |
/// | `Return` | Structural | - | Terminates block execution and returns the top value. | All |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Ord, PartialOrd)]
pub enum OpcodeKind {
    // === Stack Operations ===
    /// Pushes the literal value from operand[0] onto the evaluation stack.
    ///
    /// The operand must be an [`Operand::Literal`].
    PushLiteral,
    /// Loads a value from the slot index specified in operand[0] and pushes it.
    ///
    /// The operand must be an [`Operand::Slot`].
    Load,
    /// Pops the top stack value and stores it in the slot index specified in operand[0].
    ///
    /// The operand must be an [`Operand::Slot`].
    Store,
    /// Duplicates the current top value on the evaluation stack.
    Dup,
    /// Discards the top value from the evaluation stack.
    Pop,

    // === Constructors ===
    /// Pops the number of scalar values specified in operand[0] and constructs a vector value.
    ///
    /// Supports Vec2, Vec3, and Vec4 results. Operand[0] must be an [`Operand::Literal`] integer.
    BuildVector,
    /// Constructs a Map value from the evaluation stack using field names provided in operands.
    ///
    /// Pops one value per field name in reverse order. Operands must be [`Operand::String`].
    BuildStruct,

    // === Kernel Calls ===
    /// Dispatches a call to an engine kernel (maths, physics, etc.).
    ///
    /// Argument count is provided in operand[0]. Arguments are popped from the stack.
    /// The instruction must also carry a [`KernelId`].
    CallKernel,

    // === Control Flow ===
    /// Marks the start of a local binding scope for the slot in operand[0].
    ///
    /// This is used for lifecycle tracking and is usually a no-op in the executor.
    Let,
    /// Marks the end of a local binding scope.
    EndLet,
    /// Loads all instances of an entity type onto the stack as a sequence (Seq).
    ///
    /// Operands: [EntityId].
    LoadEntity,
    /// Filters a sequence using a predicate block.
    ///
    /// Pops: [Seq].
    /// Operands: [BindingSlot, BlockId].
    /// Pushes: [Filtered Seq].
    Filter,
    /// Finds the instance in a sequence nearest to a position.
    ///
    /// Pops: [Seq, Position].
    /// Pushes: [Nearest Instance].
    Nearest,
    /// Filters a sequence to instances within a radius of a position.
    ///
    /// Pops: [Seq, Position, Radius].
    /// Pushes: [Filtered Seq].
    Within,
    /// Iterates over a sequence and reduces results using an aggregate operation.
    ///
    /// Pops: [Seq].
    /// Operands: [BindingSlot, BlockId, AggregateOp].
    /// Executes the specified block for each entity instance.
    Aggregate,
    /// Iterates over a sequence and performs a stateful reduction (fold).
    ///
    /// Pops: [Seq, InitialValue].
    /// Operands: [AccSlot, ElemSlot, BlockId].
    /// Maintains an accumulator value in `AccSlot` across iterations.
    Fold,
    /// Pops a Map/Struct and accesses the field named in operand[0].
    ///
    /// The operand must be an [`Operand::String`].
    FieldAccess,

    // === Temporal ===
    /// Loads the current value of a signal by its path in operand[0].
    ///
    /// Valid in Resolve, Fracture, and Measure phases.
    LoadSignal,
    /// Loads a configuration value by its path in operand[0].
    LoadConfig,
    /// Loads a constant value by its path in operand[0].
    LoadConst,
    /// Loads the value of the current signal from the previous tick.
    ///
    /// Only valid for signals that requested history.
    LoadPrev,
    /// Loads the current resolved value of the current signal.
    ///
    /// Only valid in phases where signal resolution has completed (e.g., Fracture, Measure).
    LoadCurrent,
    /// Loads the accumulated inputs for the current signal.
    ///
    /// Only valid during the Resolve phase of a signal kernel.
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
    /// Only valid in the Measure phase (preserving the observer boundary).
    LoadField,

    // === Validation ===
    /// Evaluates a runtime assertion.
    ///
    /// Pops a Bool value from the stack. If false, triggers a fault with optional severity and message.
    /// Operands: [Severity (String), Message (String)] (both optional).
    Assert,

    // === Structural ===
    /// Terminates the current block execution and returns the top stack value (if any).
    Return,
}

/// A single bytecode instruction.
///
/// Each instruction pairs an [`OpcodeKind`] with a list of [`Operand`]s required
/// for its execution. Instructions that dispatch to engine kernels ([`OpcodeKind::CallKernel`])
/// also carry a [`KernelId`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Instruction {
    /// The type of operation to perform.
    pub kind: OpcodeKind,
    /// Positional operands for the instruction. Ordering must match the opcode specification.
    pub operands: Vec<Operand>,
    /// Target kernel identifier (only populated for [`OpcodeKind::CallKernel`]).
    pub kernel: Option<KernelId>,
}

impl Instruction {
    /// Create a new instruction without a kernel ID.
    ///
    /// The operand ordering and count must match the [`OpcodeMetadata`] for the specified kind.
    pub fn new(kind: OpcodeKind, operands: Vec<Operand>) -> Self {
        Self {
            kind,
            operands,
            kernel: None,
        }
    }

    /// Create a kernel call instruction.
    ///
    /// # Parameters
    /// - `kernel`: The unique identifier of the engine kernel to invoke.
    /// - `operands`: Must include the argument count as the first operand.
    pub fn kernel_call(kernel: KernelId, operands: Vec<Operand>) -> Self {
        Self {
            kind: OpcodeKind::CallKernel,
            operands,
            kernel: Some(kernel),
        }
    }
}

/// Metadata describing the static properties and constraints of an opcode.
///
/// This metadata is used by the compiler for validation and by the executor
/// to ensure architectural invariants (like phase boundaries) are respected.
#[derive(Debug, Clone)]
pub struct OpcodeMetadata {
    /// The number of operands expected by this opcode.
    pub operand_count: OperandCount,
    /// Whether this opcode performs a side effect (e.g., emission, spawning).
    pub has_effect: bool,
    /// Optional list of simulation phases where this opcode is permitted.
    /// If `None`, the opcode is valid in all phases.
    pub allowed_phases: Option<&'static [Phase]>,
}

/// Specification for the number of operands an opcode expects.
#[derive(Debug, Clone, Copy)]
pub enum OperandCount {
    /// A fixed number of operands is required.
    Fixed(usize),
    /// A variable number of operands within an optional range.
    Variable {
        /// Minimum number of operands (inclusive).
        min: usize,
        /// Maximum number of operands (inclusive), or `None` if unbounded.
        max: Option<usize>,
    },
}

impl OperandCount {
    /// Check if the given operand count is valid according to this specification.
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
    /// Retrieves metadata for this opcode kind from the global registry.
    ///
    /// # Panics
    ///
    /// Panics if no metadata entry exists for the opcode kind (indicates a registry bug).
    pub fn metadata(self) -> &'static OpcodeMetadata {
        metadata_for(self)
    }

    /// Validates if this opcode is permitted to execute in the specified simulation phase.
    ///
    /// Used to enforce the absolute observer boundary and effect isolation.
    pub fn is_valid_in_phase(self, phase: Phase) -> bool {
        match self.metadata().allowed_phases {
            None => true,
            Some(allowed) => allowed.contains(&phase),
        }
    }

    /// Returns `true` if this opcode produces a simulation side effect.
    pub fn has_effect(self) -> bool {
        self.metadata().has_effect
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl::ast::AggregateOp;
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
            vec![Operand::Signal(Path::from_path_str("force"))],
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
