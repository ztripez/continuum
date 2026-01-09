//! Bytecode instruction set
//!
//! Flat, cache-friendly instruction encoding for stack-based execution.
//! Each instruction operates on an implicit operand stack.

/// Slot identifier for local variables (let bindings)
pub type SlotId = u16;

/// Kernel function identifier (index into function table)
pub type KernelId = u16;

/// Bytecode instruction
///
/// Stack-based: operands are popped from stack, results pushed back.
/// All instructions are designed to be simple enough for future GPU translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Op {
    // === Literals and loads ===
    /// Push a literal f64 onto the stack
    Const(f64),

    /// Push the previous value of current signal
    LoadPrev,

    /// Push dt (time step)
    LoadDt,

    /// Push sum of inputs for current signal
    LoadInputs,

    /// Push value of signal by index (resolved at compile time)
    LoadSignal(u16),

    /// Push value of signal component by signal index and component index
    LoadSignalComponent(u16, u16),

    /// Push value of constant by index
    LoadConst(u16),

    /// Push value of config by index
    LoadConfig(u16),

    /// Push value from local slot
    LoadLocal(SlotId),

    /// Store top of stack to local slot (does not pop)
    StoreLocal(SlotId),

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
#[derive(Debug, Clone, Default)]
pub struct BytecodeChunk {
    /// The instruction sequence
    pub ops: Vec<Op>,

    /// Signal name table (indices referenced by LoadSignal)
    pub signals: Vec<String>,

    /// Component name table (indices referenced by LoadSignalComponent)
    pub components: Vec<String>,

    /// Constant name table (indices referenced by LoadConst)
    pub constants: Vec<String>,

    /// Config name table (indices referenced by LoadConfig)
    pub configs: Vec<String>,

    /// Kernel function name table (indices referenced by Call)
    pub kernels: Vec<String>,

    /// Number of local slots needed
    pub local_count: u16,
}

impl BytecodeChunk {
    /// Create a new, empty bytecode chunk.
    pub fn new() -> Self {
        Self::default()
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
