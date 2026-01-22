//! Runtime traits and errors for bytecode execution.

use continuum_foundation::{AggregateOp, EntityId, Path, Phase, Value};
use continuum_kernel_types::KernelId;

use crate::bytecode::operand::{BlockId, Slot};
use crate::bytecode::program::BytecodeProgram;

/// Bytecode execution error.
///
/// These errors represent runtime violations during bytecode interpretation,
/// such as stack mismatches, phase boundary violations, or invalid operand data.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ExecutionError {
    /// Stack underflow: attempted to pop from an empty evaluation stack.
    #[error("Stack underflow: tried to pop from empty stack")]
    StackUnderflow,

    /// Stack overflow: evaluation stack exceeded its fixed or configured limit.
    #[error("Stack overflow: stack size limit exceeded")]
    StackOverflow,

    /// Invalid slot access: requested a VM slot index that is out of range for the current block.
    #[error("Invalid slot access: slot {slot} out of bounds (max {max})")]
    InvalidSlot {
        /// The requested slot index.
        slot: u32,
        /// The maximum valid slot index for the current block.
        max: u32,
    },

    /// Uninitialized slot access: attempted to read from a slot that has not been written to.
    #[error("Uninitialized slot access: slot {slot}")]
    UninitializedSlot {
        /// The index of the uninitialized slot.
        slot: u32,
    },

    /// Invalid opcode for phase: the instruction is forbidden in the current simulation phase.
    #[error("Opcode {opcode} not valid in phase {phase:?}")]
    InvalidOpcode {
        /// The name or kind of the forbidden opcode.
        opcode: String,
        /// The current simulation phase where it was attempted.
        phase: Phase,
    },

    /// Invalid operand: the operand data does not match the opcode's expectation.
    #[error("Invalid operand: {message}")]
    InvalidOperand {
        /// Detailed description of the operand mismatch.
        message: String,
    },

    /// Missing return value: a sub-block (e.g., aggregate body) finished without leaving a result.
    #[error("Expected return value from block")]
    MissingReturn,

    /// Unexpected stack depth: the stack was not empty or at the expected depth after execution.
    #[error("Unexpected stack depth: expected {expected}, found {found}")]
    UnexpectedStackDepth {
        /// The expected number of items on the stack.
        expected: usize,
        /// The actual number of items found.
        found: usize,
    },

    /// Invalid block reference: the block ID does not exist in the program.
    #[error("Invalid block reference: {block}")]
    InvalidBlock {
        /// The invalid block identifier index.
        block: u32,
    },

    /// Kernel call failed: the engine kernel returned an error during execution.
    #[error("Kernel call failed: {message}")]
    KernelCallFailed {
        /// Error message returned by the kernel.
        message: String,
    },

    /// Type mismatch: an operation received a [`Value`] of an unexpected variant.
    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch {
        /// Description of the expected value type.
        expected: String,
        /// Description of the actual value type received.
        found: String,
    },

    /// Unsupported vector size: attempted to build a vector with more than 4 components.
    #[error("Unsupported vector size: {size}")]
    UnsupportedVectorSize {
        /// The requested vector dimension.
        size: usize,
    },
}

/// Execution context required by the bytecode executor to interface with the simulation.
///
/// This trait provides the executor with access to authoritative simulation state,
/// engine services, and side-effect dispatchers. It abstracts the underlying DAG
/// runtime from the bytecode interpreter.
///
/// Implementations must ensure that all lookups and effects adhere to the current
/// simulation phase and observer boundaries.
pub trait ExecutionContext {
    /// Returns the current simulation phase.
    fn phase(&self) -> Phase;

    /// Loads a resolved signal value by its canonical path.
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is unavailable, the path is invalid,
    /// or the current phase does not allow signal reads (e.g., Collect phase).
    fn load_signal(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Loads a value from a spatial field by its canonical path.
    ///
    /// This is an observer-only operation and must preserve the observer boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if the field is unavailable or if called from a kernel phase.
    fn load_field(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Loads a world configuration parameter by its canonical path.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration value is missing.
    fn load_config(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Loads a global simulation constant by its path.
    ///
    /// # Errors
    ///
    /// Returns an error if the constant is missing.
    fn load_const(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Loads the value of the current signal as it was at the end of the previous tick.
    ///
    /// # Errors
    ///
    /// Returns an error if history is unavailable or not requested for this signal.
    fn load_prev(&self) -> Result<Value, ExecutionError>;

    /// Loads the resolved value of the current signal for the current tick.
    ///
    /// # Errors
    ///
    /// Returns an error if signal resolution has not yet occurred (e.g., in Collect phase).
    fn load_current(&self) -> Result<Value, ExecutionError>;

    /// Loads the accumulated inputs for the current signal.
    ///
    /// Used during the Resolve phase to compute the authoritative signal value.
    ///
    /// # Errors
    ///
    /// Returns an error if called outside of the Resolve phase.
    fn load_inputs(&mut self) -> Result<Value, ExecutionError>;

    /// Returns the time step (delta time) for the current tick.
    fn load_dt(&self) -> Result<Value, ExecutionError>;

    /// Returns the identity ([`Value::Entity`]) of the current entity instance.
    ///
    /// # Errors
    ///
    /// Returns an error if execution is not occurring within an entity context.
    fn load_self(&self) -> Result<Value, ExecutionError>;

    /// Returns the identity of the "other" entity instance in a dual-entity context.
    ///
    /// # Errors
    ///
    /// Returns an error if no "other" entity is present (e.g., not in an interaction block).
    fn load_other(&self) -> Result<Value, ExecutionError>;

    /// Returns the payload data associated with the current impulse.
    ///
    /// # Errors
    ///
    /// Returns an error if execution was not triggered by an impulse.
    fn load_payload(&self) -> Result<Value, ExecutionError>;

    /// Finds the instance in a sequence nearest to a position.
    fn find_nearest(&self, seq: &[Value], position: Value) -> Result<Value, ExecutionError>;

    /// Filters a sequence to instances within a radius of a position.
    fn filter_within(
        &self,
        seq: &[Value],
        position: Value,
        radius: Value,
    ) -> Result<Vec<Value>, ExecutionError>;

    /// Emits a value to a signal.
    fn emit_signal(&mut self, target: &Path, value: Value) -> Result<(), ExecutionError>;

    /// Emits a value to a spatial field at a given position.
    fn emit_field(
        &mut self,
        target: &Path,
        position: Value,
        value: Value,
    ) -> Result<(), ExecutionError>;

    /// Emits an event to a chronicle.
    fn emit_event(
        &mut self,
        chronicle_id: String,
        name: String,
        fields: Vec<(String, Value)>,
    ) -> Result<(), ExecutionError>;

    /// Spawns a new instance of an entity type.
    fn spawn(&mut self, entity: &EntityId, data: Value) -> Result<(), ExecutionError>;

    /// Marks an entity instance for destruction.
    fn destroy(&mut self, entity: &EntityId, instance: Value) -> Result<(), ExecutionError>;

    /// Returns a sequence of all instances of an entity type.
    fn iter_entity(&self, entity: &EntityId) -> Result<Vec<Value>, ExecutionError>;

    /// Performs a reduction operation over a collection of values.
    fn reduce_aggregate(
        &self,
        op: AggregateOp,
        values: Vec<Value>,
    ) -> Result<Value, ExecutionError>;

    /// Dispatches a call to an engine kernel primitive.
    fn call_kernel(&self, kernel: &KernelId, args: &[Value]) -> Result<Value, ExecutionError>;
}

/// Runtime interface provided by the executor to opcode handlers.
///
/// This trait abstracts the internal state of the VM (stack, slots), allowing
/// handlers to be implemented as pure functions without direct access to the executor.
pub trait ExecutionRuntime {
    /// Pushes a [`Value`] onto the evaluation stack.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionError::StackOverflow`] if the stack limit is reached.
    fn push(&mut self, value: Value) -> Result<(), ExecutionError>;

    /// Pops the top [`Value`] from the evaluation stack.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionError::StackUnderflow`] if the stack is empty.
    fn pop(&mut self) -> Result<Value, ExecutionError>;

    /// Loads a [`Value`] from the specified storage slot.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionError::InvalidSlot`] or [`ExecutionError::UninitializedSlot`] if the access is invalid.
    fn load_slot(&self, slot: Slot) -> Result<Value, ExecutionError>;

    /// Stores a [`Value`] into the specified storage slot.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutionError::InvalidSlot`] if the slot index is out of bounds.
    fn store_slot(&mut self, slot: Slot, value: Value) -> Result<(), ExecutionError>;

    /// Executes a nested bytecode block within the same program context.
    ///
    /// This is used for sub-computations like those in `Aggregate` or `Fold`.
    ///
    /// # Returns
    ///
    /// The value returned by the block, or `None` if the block does not return a value.
    fn execute_block(
        &mut self,
        block_id: BlockId,
        program: &BytecodeProgram,
        ctx: &mut dyn ExecutionContext,
    ) -> Result<Option<Value>, ExecutionError>;
}
