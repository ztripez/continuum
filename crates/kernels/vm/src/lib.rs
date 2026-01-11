//! Continuum VM - Stack-based bytecode virtual machine
//!
//! Compiles expression IR to flat bytecode for efficient execution.
//! Designed for future extensibility to GPU compute via Naga.

mod bytecode;
pub mod compiler;
mod executor;

pub use bytecode::{BytecodeChunk, Op, SlotId};
pub use compiler::compile_expr;
pub use executor::{ExecutionContext, execute};
