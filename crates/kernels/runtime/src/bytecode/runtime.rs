//! Runtime traits and errors for bytecode execution.

use continuum_cdsl::ast::expr::AggregateOp;
use continuum_foundation::{EntityId, Path, Phase, Value};
use continuum_kernel_types::KernelId;

use crate::bytecode::operand::{BlockId, Slot};
use crate::bytecode::program::BytecodeProgram;

/// Bytecode execution error.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ExecutionError {
    /// Stack underflow
    #[error("Stack underflow: tried to pop from empty stack")]
    StackUnderflow,

    /// Stack overflow
    #[error("Stack overflow: stack size limit exceeded")]
    StackOverflow,

    /// Invalid slot access
    #[error("Invalid slot access: slot {slot} out of bounds (max {max})")]
    InvalidSlot { slot: u32, max: u32 },

    /// Uninitialized slot access
    #[error("Uninitialized slot access: slot {slot}")]
    UninitializedSlot { slot: u32 },

    /// Invalid opcode for phase
    #[error("Opcode {opcode} not valid in phase {phase:?}")]
    InvalidOpcode { opcode: String, phase: Phase },

    /// Invalid operand
    #[error("Invalid operand: {message}")]
    InvalidOperand { message: String },

    /// Missing return value
    #[error("Expected return value from block")]
    MissingReturn,

    /// Unexpected stack depth
    #[error("Unexpected stack depth: expected {expected}, found {found}")]
    UnexpectedStackDepth { expected: usize, found: usize },

    /// Invalid block reference
    #[error("Invalid block reference: {block}")]
    InvalidBlock { block: u32 },

    /// Kernel call failed
    #[error("Kernel call failed: {message}")]
    KernelCallFailed { message: String },

    /// Type mismatch
    #[error("Type mismatch: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    /// Unsupported vector size
    #[error("Unsupported vector size: {size}")]
    UnsupportedVectorSize { size: usize },
}

/// Execution context required by the bytecode executor.
///
/// This trait provides the executor with access to the simulation state and
/// engine services required to resolve signals, load configurations, and
/// perform side effects (emissions, spawning).
pub trait ExecutionContext {
    /// Returns the current simulation phase.
    fn phase(&self) -> Phase;

    /// Loads a signal value by its canonical path.
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is unavailable, the path is invalid,
    /// or the current phase does not allow signal reads.
    fn load_signal(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Loads a value from a spatial field by its canonical path.
    ///
    /// # Errors
    ///
    /// Returns an error if the field is unavailable or if read from a kernel phase.
    fn load_field(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Loads a configuration parameter by its canonical path.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration value is missing.
    fn load_config(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Loads a global constant by its canonical path.
    ///
    /// # Errors
    ///
    /// Returns an error if the constant is missing.
    fn load_const(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Loads the value of the current signal as it was at the end of the previous tick.
    ///
    /// # Errors
    ///
    /// Returns an error if history is unavailable or not requested by the signal definition.
    fn load_prev(&self) -> Result<Value, ExecutionError>;

    /// Loads the resolved value of the current signal for the current tick.
    ///
    /// # Errors
    ///
    /// Returns an error if signal resolution has not yet occurred (e.g., in Collect phase).
    fn load_current(&self) -> Result<Value, ExecutionError>;

    /// Loads the accumulated inputs for the current signal.
    ///
    /// # Errors
    ///
    /// Returns an error if called outside of the Resolve phase.
    fn load_inputs(&self) -> Result<Value, ExecutionError>;

    /// Returns the time step (delta time) for the current tick.
    fn load_dt(&self) -> Result<Value, ExecutionError>;

    /// Returns the identity of the current entity instance (`self`).
    ///
    /// # Errors
    ///
    /// Returns an error if execution is not occurring within an entity context.
    fn load_self(&self) -> Result<Value, ExecutionError>;

    /// Returns the identity of the "other" entity instance in a dual-entity context.
    ///
    /// # Errors
    ///
    /// Returns an error if no "other" entity is present.
    fn load_other(&self) -> Result<Value, ExecutionError>;

    /// Returns the payload data associated with the current impulse.
    ///
    /// # Errors
    ///
    /// Returns an error if execution was not triggered by an impulse.
    fn load_payload(&self) -> Result<Value, ExecutionError>;

    /// Emits a value to a signal.
    ///
    /// # Errors
    ///
    /// Returns an error if the signal is read-only or if called outside of the Collect phase.
    fn emit_signal(&mut self, target: &Path, value: Value) -> Result<(), ExecutionError>;

    /// Emits a value to a spatial field at a specific position.
    ///
    /// # Errors
    ///
    /// Returns an error if called outside of the Measure phase.
    fn emit_field(
        &mut self,
        target: &Path,
        position: Value,
        value: Value,
    ) -> Result<(), ExecutionError>;

    /// Spawns a new entity instance of the specified type with initial data.
    ///
    /// # Errors
    ///
    /// Returns an error if called outside of the Fracture phase.
    fn spawn(&mut self, entity: &EntityId, value: Value) -> Result<(), ExecutionError>;

    /// Marks an entity instance for destruction.
    ///
    /// # Errors
    ///
    /// Returns an error if called outside of the Fracture phase.
    fn destroy(&mut self, entity: &EntityId, instance: Value) -> Result<(), ExecutionError>;

    /// Returns all instances of an entity type in deterministic order.
    ///
    /// # Errors
    ///
    /// Returns an error if the entity type is unknown.
    fn iter_entity(&self, entity: &EntityId) -> Result<Vec<Value>, ExecutionError>;

    /// Reduces a collection of values using a specified aggregate operation.
    ///
    /// # Errors
    ///
    /// Returns an error if the values cannot be reduced (e.g., type mismatch).
    fn reduce_aggregate(
        &self,
        op: AggregateOp,
        values: Vec<Value>,
    ) -> Result<Value, ExecutionError>;

    /// Dispatches a call to an engine kernel.
    ///
    /// # Errors
    ///
    /// Returns an error if the kernel name is unknown or arguments are invalid.
    fn call_kernel(&self, kernel: &KernelId, args: &[Value]) -> Result<Value, ExecutionError>;
}

/// Runtime interface provided by the executor to opcode handlers.
///
/// This trait abstracts the internal state of the VM (stack, slots), allowing
/// handlers to be implemented as pure functions without direct access to the executor.
pub trait ExecutionRuntime {
    /// Pushes a value onto the evaluation stack.
    ///
    /// # Errors
    ///
    /// Returns `ExecutionError::StackOverflow` if the stack limit is reached.
    fn push(&mut self, value: Value) -> Result<(), ExecutionError>;

    /// Pops the top value from the evaluation stack.
    ///
    /// # Errors
    ///
    /// Returns `ExecutionError::StackUnderflow` if the stack is empty.
    fn pop(&mut self) -> Result<Value, ExecutionError>;

    /// Loads a value from the specified storage slot.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot index is invalid or the slot is uninitialized.
    fn load_slot(&self, slot: Slot) -> Result<Value, ExecutionError>;

    /// Stores a value into the specified storage slot.
    ///
    /// # Errors
    ///
    /// Returns an error if the slot index is invalid.
    fn store_slot(&mut self, slot: Slot, value: Value) -> Result<(), ExecutionError>;

    /// Executes a nested bytecode block and returns its result.
    ///
    /// This is used for sub-computations like those in `Aggregate` or `Fold`.
    fn execute_block(
        &mut self,
        block_id: BlockId,
        program: &BytecodeProgram,
        ctx: &mut dyn ExecutionContext,
    ) -> Result<Option<Value>, ExecutionError>;
}
