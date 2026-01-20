//! Bytecode virtual machine for Continuum DSL execution.
//!
//! This module defines the bytecode representation of DSL execution blocks,
//! the compiler from typed IR to bytecode, and the executor that runs bytecode
//! within the deterministic DAG runtime.
//!
//! # Design Principles
//!
//! 1. **No hard-coded behavior** - Opcodes are data, not match arms with logic
//! 2. **Deterministic by construction** - All iteration orders explicit
//! 3. **Phase-aware** - Bytecode respects observer boundaries and capability constraints
//! 4. **One Truth** - IR → bytecode conversion lives in one place only
//!
//! # Architecture
//!
//! - [`opcode`] - Opcode definitions and metadata tables
//! - [`compiler`] - IR → bytecode compilation pass
//! - [`executor`] - Bytecode interpreter integrated with runtime DAG
//! - [`operand`] - Operand encoding (slots, indices, IDs, temporal markers)
//!
//! # Execution Model
//!
//! Bytecode executes within a stack-based VM with explicit slots for:
//! - Signal values (resolved or previous)
//! - Local let bindings
//! - Temporary computation results
//! - Entity iteration state
//!
//! All operations are explicit: no implicit coercion, no silent fallbacks.

pub mod compiler;
pub mod executor;
pub mod handlers;
pub mod opcode;
pub mod operand;
pub mod program;
pub mod registry;
pub mod runtime;

pub use compiler::{CompiledBlock, Compiler};
pub use executor::BytecodeExecutor;
pub use opcode::{Instruction, OpcodeKind, OpcodeMetadata};
pub use operand::{BlockId, Operand, Slot};
pub use program::{BytecodeBlock, BytecodeProgram};
pub use runtime::{ExecutionContext, ExecutionError, ExecutionRuntime};
