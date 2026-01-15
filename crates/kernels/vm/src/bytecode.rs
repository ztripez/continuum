//! Bytecode instruction set
//!
//! Flat, cache-friendly instruction encoding for stack-based execution.
//! Each instruction operates on an implicit operand stack.

use continuum_kernel_registry::Value;
use serde::{Deserialize, Serialize};

/// Slot identifier for local variables (let bindings)
pub type SlotId = u16;

/// Kernel function identifier (index into function table)
pub type KernelId = u16;

/// Reduction operation for aggregates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ReductionOp {
    Sum,
    Product,
    Min,
    Max,
    Mean,
    Count,
    Any,
    All,
    None,
}

/// Bytecode instruction
///
/// Stack-based: operands are popped from stack, results pushed back.
/// All instructions are designed to be simple enough for future GPU translation.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Op {
    // === Literals and loads ===
    /// Push a literal value onto the stack (index into literals table)
    Literal(u16),

    /// Push the previous value of current signal
    LoadPrev,

    /// Push dt (time step)
    LoadDt,

    /// Push accumulated simulation time in seconds
    LoadSimTime,

    /// Push sum of inputs for current signal
    LoadInputs,

    /// Push value of inputs component by component index (for vector signals)
    LoadInputsComponent(u16),

    /// Push value of signal by index (resolved at compile time)
    LoadSignal(u16),

    /// Push value of signal component by signal index and component index
    LoadSignalComponent(u16, u16),

    /// Push value of prev component by component index (for vector signals)
    LoadPrevComponent(u16),

    /// Push value of constant by index
    LoadConst(u16),

    /// Push value of config by index
    LoadConfig(u16),

    /// Push value from local slot
    LoadLocal(SlotId),

    /// Store top of stack to local slot (does not pop)
    StoreLocal(SlotId),

    // === Entity access ===
    /// Push value of a member signal of the current instance (component index)
    LoadSelfField(u16),

    /// Push value of a field from a specific entity instance
    /// (entity_idx, instance_idx, component_idx)
    LoadEntityField(u16, u16, u16),

    /// Push value of another instance in the same entity set (for n-body)
    /// (component_idx)
    LoadOtherField(u16),

    // === Entity Iteration / Aggregation ===
    /// Aggregate over an entity set (entity_idx, reduction_op, sub_chunk_idx)
    Aggregate(u16, ReductionOp, u16),

    /// Filter an entity set (entity_idx, pred_chunk_idx, body_chunk_idx)
    Filter(u16, u16, u16),

    /// Find first matching instance (entity_idx, pred_chunk_idx, component_idx)
    FindFirstField(u16, u16, u16),

    /// Find nearest instance and load field (entity_idx, component_idx)
    /// Expects position on stack.
    LoadNearestField(u16, u16),

    /// Iterate over instances within radius and aggregate
    /// (entity_idx, reduction_op, body_chunk_idx)
    /// Expects position and radius on stack.
    WithinAggregate(u16, ReductionOp, u16),

    /// Iterate over all unique pairs and execute body
    /// (entity_idx, body_chunk_idx)
    Pairs(u16, u16),

    // === Impulse ===
    /// Push the impulse payload
    LoadPayload,

    /// Push a field from the impulse payload (component index)
    LoadPayloadField(u16),

    /// Emit a signal from an impulse (signal_idx)
    /// Pops value from stack.
    EmitSignal(u16),

    // === Arithmetic ===
    /// Add top two stack values (pop b, pop a, push a + b)
    Add,
    /// Subtract top two stack values (pop b, pop a, push a - b)
    Sub,
    /// Multiply top two stack values (pop b, pop a, push a * b)
    Mul,
    /// Divide top two stack values (pop b, pop a, push a / b)
    Div,
    /// Raise a to power of b (pop b, pop a, push a^b)
    Pow,
    /// Negate top of stack (pop a, push -a)
    Neg,

    // === Comparison (push 1.0 for true, 0.0 for false) ===
    /// Check equality (pop b, pop a, push a == b)
    Eq,
    /// Check inequality (pop b, pop a, push a != b)
    Ne,
    /// Check less than (pop b, pop a, push a < b)
    Lt,
    /// Check less than or equal (pop b, pop a, push a <= b)
    Le,
    /// Check greater than (pop b, pop a, push a > b)
    Gt,
    /// Check greater than or equal (pop b, pop a, push a >= b)
    Ge,

    // === Logical ===
    /// Logical AND (pop b, pop a, push 1.0 if a != 0 and b != 0, else 0.0)
    And,
    /// Logical OR (pop b, pop a, push 1.0 if a != 0 or b != 0, else 0.0)
    Or,
    /// Logical NOT (pop a, push 1.0 if a == 0, else 0.0)
    Not,

    // === Control flow ===
    /// Jump forward by offset if top of stack is zero (consumes top)
    JumpIfZero(u16),

    /// Jump forward by offset unconditionally
    Jump(u16),

    // === Function calls ===
    /// Call kernel function with N arguments (pops N, pushes 1)
    Call {
        /// The kernel function identifier
        kernel: KernelId,
        /// Number of arguments to pop
        arity: u8,
    },

    // === Stack manipulation ===
    /// Duplicate top of stack
    Dup,

    /// Pop and discard top of stack
    Pop,
}

/// A compiled bytecode chunk
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BytecodeChunk {
    /// The instruction sequence
    pub ops: Vec<Op>,

    /// Signal name table
    pub signals: Vec<String>,

    /// Component name table
    pub components: Vec<String>,

    /// Constant name table
    pub constants: Vec<String>,

    /// Config name table
    pub configs: Vec<String>,

    /// Literal value table
    pub literals: Vec<Value>,

    /// Kernel function name table
    pub kernels: Vec<String>,

    /// Entity ID table
    pub entities: Vec<String>,

    /// Instance ID table
    pub instances: Vec<String>,

    /// Sub-chunks (for loops, filters, aggregates)
    pub sub_chunks: Vec<BytecodeChunk>,

    /// Number of local slots needed
    pub local_count: u16,
}

impl BytecodeChunk {
    /// Create a new, empty bytecode chunk.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an entity reference, returning its index
    pub fn add_entity(&mut self, id: &str) -> u16 {
        if let Some(idx) = self.entities.iter().position(|s| s == id) {
            return idx as u16;
        }
        let idx = self.entities.len() as u16;
        self.entities.push(id.to_string());
        idx
    }

    /// Add an instance reference, returning its index
    pub fn add_instance(&mut self, id: &str) -> u16 {
        if let Some(idx) = self.instances.iter().position(|s| s == id) {
            return idx as u16;
        }
        let idx = self.instances.len() as u16;
        self.instances.push(id.to_string());
        idx
    }

    /// Add a sub-chunk, returning its index
    pub fn add_sub_chunk(&mut self, chunk: BytecodeChunk) -> u16 {
        let idx = self.sub_chunks.len() as u16;
        self.sub_chunks.push(chunk);
        idx
    }

    /// Add a signal reference, returning its index
    pub fn add_signal(&mut self, name: &str) -> u16 {
        if let Some(idx) = self.signals.iter().position(|s| s == name) {
            return idx as u16;
        }
        let idx = self.signals.len() as u16;
        self.signals.push(name.to_string());
        idx
    }

    /// Add a component reference, returning its index
    pub fn add_component(&mut self, name: &str) -> u16 {
        if let Some(idx) = self.components.iter().position(|s| s == name) {
            return idx as u16;
        }
        let idx = self.components.len() as u16;
        self.components.push(name.to_string());
        idx
    }

    /// Add a constant reference, returning its index
    pub fn add_constant(&mut self, name: &str) -> u16 {
        if let Some(idx) = self.constants.iter().position(|s| s == name) {
            return idx as u16;
        }
        let idx = self.constants.len() as u16;
        self.constants.push(name.to_string());
        idx
    }

    /// Add a config reference, returning its index
    pub fn add_config(&mut self, name: &str) -> u16 {
        if let Some(idx) = self.configs.iter().position(|s| s == name) {
            return idx as u16;
        }
        let idx = self.configs.len() as u16;
        self.configs.push(name.to_string());
        idx
    }

    /// Add a literal value, returning its index
    pub fn add_literal(&mut self, value: Value) -> u16 {
        if let Some(idx) = self.literals.iter().position(|v| v == &value) {
            return idx as u16;
        }
        let idx = self.literals.len() as u16;
        self.literals.push(value);
        idx
    }

    /// Add a kernel function reference, returning its index
    pub fn add_kernel(&mut self, name: &str) -> u16 {
        if let Some(idx) = self.kernels.iter().position(|s| s == name) {
            return idx as u16;
        }
        let idx = self.kernels.len() as u16;
        self.kernels.push(name.to_string());
        idx
    }

    /// Emit an instruction
    pub fn emit(&mut self, op: Op) {
        self.ops.push(op);
    }

    /// Current instruction offset (for jump patching)
    pub fn offset(&self) -> usize {
        self.ops.len()
    }

    /// Patch a jump instruction at the given offset with a new target.
    ///
    /// This is used during bytecode generation to backpatch forward jumps
    /// once the target offset is known (e.g., for if-else branches).
    ///
    /// # Panics
    ///
    /// Panics if the instruction at `offset` is not a jump instruction
    /// (`JumpIfZero` or `Jump`).
    pub fn patch_jump(&mut self, offset: usize, target: u16) {
        match &mut self.ops[offset] {
            Op::JumpIfZero(t) | Op::Jump(t) => *t = target,
            _ => panic!("attempted to patch non-jump instruction"),
        }
    }
}
