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
pub trait ExecutionContext {
    /// Current phase
    fn phase(&self) -> Phase;

    /// Load a signal value by path
    ///
    /// # Errors
    ///
    /// Returns an error when the value is unavailable or invalid.
    fn load_signal(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Load a field value by path
    ///
    /// # Errors
    ///
    /// Returns an error when the value is unavailable or invalid.
    fn load_field(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Load a config value by path
    ///
    /// # Errors
    ///
    /// Returns an error when the value is unavailable or invalid.
    fn load_config(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Load a const value by path
    ///
    /// # Errors
    ///
    /// Returns an error when the value is unavailable or invalid.
    fn load_const(&self, path: &Path) -> Result<Value, ExecutionError>;

    /// Load previous tick value for current signal
    ///
    /// # Errors
    ///
    /// Returns an error when prev is unavailable for the phase.
    fn load_prev(&self) -> Result<Value, ExecutionError>;

    /// Load current resolved value for current signal
    ///
    /// # Errors
    ///
    /// Returns an error when current is unavailable for the phase.
    fn load_current(&self) -> Result<Value, ExecutionError>;

    /// Load accumulated inputs for current signal
    ///
    /// # Errors
    ///
    /// Returns an error when inputs are unavailable for the phase.
    fn load_inputs(&self) -> Result<Value, ExecutionError>;

    /// Load time step
    ///
    /// # Errors
    ///
    /// Returns an error when dt is unavailable for the phase.
    fn load_dt(&self) -> Result<Value, ExecutionError>;

    /// Load self entity instance
    ///
    /// The value should match the instance shape returned by `iter_entity`.
    ///
    /// # Errors
    ///
    /// Returns an error when the instance is unavailable.
    fn load_self(&self) -> Result<Value, ExecutionError>;

    /// Load other entity instance
    ///
    /// The value should match the instance shape returned by `iter_entity`.
    ///
    /// # Errors
    ///
    /// Returns an error when the instance is unavailable.
    fn load_other(&self) -> Result<Value, ExecutionError>;

    /// Load impulse payload
    ///
    /// # Errors
    ///
    /// Returns an error when the payload is unavailable.
    fn load_payload(&self) -> Result<Value, ExecutionError>;

    /// Emit a value to a signal
    ///
    /// # Errors
    ///
    /// Returns an error when emission is invalid for the phase.
    fn emit_signal(&mut self, target: &Path, value: Value) -> Result<(), ExecutionError>;

    /// Emit a value to a field at a position
    ///
    /// Implementations define the expected position value shape (e.g., Vec2/Vec3).
    ///
    /// # Errors
    ///
    /// Returns an error when emission is invalid for the phase.
    fn emit_field(
        &mut self,
        target: &Path,
        position: Value,
        value: Value,
    ) -> Result<(), ExecutionError>;

    /// Spawn a new entity instance
    ///
    /// # Errors
    ///
    /// Returns an error when spawning is invalid for the phase.
    fn spawn(&mut self, entity: &EntityId, value: Value) -> Result<(), ExecutionError>;

    /// Destroy an entity instance
    ///
    /// # Errors
    ///
    /// Returns an error when destruction is invalid for the phase.
    fn destroy(&mut self, entity: &EntityId, instance: Value) -> Result<(), ExecutionError>;

    /// Iterate entity instances in deterministic order
    ///
    /// # Errors
    ///
    /// Returns an error when the entity instances are unavailable.
    fn iter_entity(&self, entity: &EntityId) -> Result<Vec<Value>, ExecutionError>;

    /// Reduce aggregate values
    ///
    /// # Errors
    ///
    /// Returns an error when reduction is invalid for the values provided.
    fn reduce_aggregate(
        &self,
        op: AggregateOp,
        values: Vec<Value>,
    ) -> Result<Value, ExecutionError>;

    /// Call a kernel operation
    ///
    /// # Errors
    ///
    /// Returns an error when kernel dispatch fails.
    fn call_kernel(&self, kernel: &KernelId, args: &[Value]) -> Result<Value, ExecutionError>;
}

/// Runtime interface required by opcode handlers.
///
/// This trait abstracts the stack and slot operations of the bytecode VM,
/// allowing handlers to operate without knowledge of the underlying executor implementation.
pub trait ExecutionRuntime {
    /// Push a value onto the evaluation stack.
    fn push(&mut self, value: Value) -> Result<(), ExecutionError>;

    /// Pop a value from the evaluation stack.
    fn pop(&mut self) -> Result<Value, ExecutionError>;

    /// Load a value from a specific slot.
    fn load_slot(&self, slot: Slot) -> Result<Value, ExecutionError>;

    /// Store a value into a specific slot.
    fn store_slot(&mut self, slot: Slot, value: Value) -> Result<(), ExecutionError>;

    /// Execute a nested bytecode block.
    fn execute_block(
        &mut self,
        block_id: BlockId,
        program: &BytecodeProgram,
        ctx: &mut dyn ExecutionContext,
    ) -> Result<Option<Value>, ExecutionError>;
}
