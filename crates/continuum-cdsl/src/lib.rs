//! # Continuum DSL Compiler
//!
//! Complete compiler pipeline for Continuum DSL (CDSL):
//!
//! ```text
//! Source Files
//!     ↓
//! Lexer (logos) → Tokens
//!     ↓
//! Parser (chumsky) → Untyped AST
//!     ↓
//! Name Resolution → Resolved AST
//!     ↓
//! Type Resolution → Typed AST
//!     ↓
//! Validation Passes
//!     ↓
//! Bytecode Compiler → Bytecode
//!     ↓
//! CompiledWorld (ready for runtime)
//! ```
//!
//! ## Architecture
//!
//! - **Foundation**: Shared types (Path, Unit, Type, Shape, etc.)
//! - **Lexer**: Token stream from source text
//! - **Parser**: Untyped AST construction
//! - **Analyzer**: Multi-pass resolution and validation
//! - **Bytecode**: Compilation to executable bytecode
//! - **VM**: Bytecode executor
//!
//! ## Module Organization
//!
//! - `foundation`: Core types used throughout compiler
//! - `span`: Source location tracking
//! - `ast`: AST node definitions
//! - `lexer`: Tokenization
//! - `parser`: Parsing to AST
//! - `resolve`: Name and type resolution
//! - `validate`: Validation passes
//! - `bytecode`: Bytecode structures and compiler
//! - `vm`: Bytecode executor
//! - `compile`: Public compilation API
//!
//! ## Usage
//!
//! ```rust,ignore
//! use continuum_cdsl::compile;
//! use std::collections::HashMap;
//! use std::path::PathBuf;
//!
//! let mut sources = HashMap::new();
//! sources.insert(PathBuf::from("world.cdsl"), source_code);
//!
//! let result = compile(&sources);
//! match result {
//!     Ok(compiled) => {
//!         // Ready for runtime
//!     }
//!     Err(diagnostics) => {
//!         // Handle errors
//!     }
//! }
//! ```

#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

// Compiler foundation module
pub mod foundation;

/// Source location tracking (Span, SourceMap)
///
/// Phase 2 will implement span tracking for error reporting.
pub mod span {
    //! Source code location tracking for diagnostics.
}

/// AST node definitions
///
/// Phase 3-4: AST implementation
/// - Phase 3.1: Node<I> wrapper and Role system ✅
/// - Phase 3.2: Scoping, Assertion, Execution structs (TODO)
/// - Phase 4: ExprKind and TypedExpr (TODO)
pub mod ast;

/// Lexer (logos-based tokenization)
///
/// Phase 10 will implement the lexer.
pub mod lexer {
    //! Tokenization of CDSL source code.
}

/// Parser (chumsky-based parsing)
///
/// Phase 10 will implement the parser.
pub mod parser {
    //! Parsing CDSL tokens into untyped AST.
}

/// Name and type resolution
///
/// Phase 11 will implement resolution passes.
pub mod resolve {
    //! Name resolution and type resolution passes.
}

/// Validation passes
///
/// Phase 12 will implement validation:
/// - Type validation
/// - Effect validation
/// - Capability validation
/// - Structure validation (cycles, collisions)
pub mod validate {
    //! Multi-pass validation and linting.
}

/// Bytecode structures and compiler
///
/// Phase 14 will implement bytecode:
/// - Opcodes
/// - BytecodeChunk
/// - Compiler from TypedExpr to Bytecode
pub mod bytecode {
    //! Bytecode compilation and structures.
}

/// Bytecode VM executor
///
/// Phase 14 will implement the VM.
pub mod vm {
    //! Bytecode execution engine.
}

/// Error types and diagnostics
///
/// Phase 9 will implement the CompileError system.
pub mod error {
    //! Compilation errors and diagnostics.
}

/// Public compilation API
///
/// Phase 15 will expose the public interface.
pub mod compile {
    //! High-level compilation API.
}

// Placeholder version info
/// Compiler version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
