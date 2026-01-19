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
/// Provides tokenization of CDSL source code.
pub mod lexer;

/// Parser (chumsky-based parsing)
///
/// Implements recursive descent parsing with chumsky combinators.
pub mod parser;

/// Desugaring pass
///
/// Phase 10: Converts syntax sugar (operators, if-expressions) to kernel calls.
/// Transforms Binary/Unary/If nodes into Call nodes before type resolution.
pub mod desugar;

/// Name and type resolution
///
/// Phase 11: Resolution passes
/// - Name resolution: Validates Path references (Phase 11)
/// - Type resolution: Resolves TypeExpr to Type (Phase 11)
pub mod resolve;

/// Validation passes
///
/// Phase 12: Validation complete ✅
/// - Type validation (resolve/validation.rs)
/// - Effect validation (resolve/effects.rs)
/// - Capability validation (resolve/capabilities.rs)
/// - Structure validation (resolve/structure.rs) - cycles and collisions
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
/// Provides structured error reporting for all compiler phases.
pub mod error;

/// Public compilation API
///
/// Phase 15 will expose the public interface.
pub mod compile {
    //! High-level compilation API.
}

// Placeholder version info
/// Compiler version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
