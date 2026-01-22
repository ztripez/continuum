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
//! Desugar → Normalized AST
//!     ↓
//! Name Resolution → Resolved AST
//!     ↓
//! Type Resolution → Typed AST
//!     ↓
//! Validation Passes → Validated AST
//!   ├─ Type validation
//!   ├─ Effect validation
//!   ├─ Capability validation
//!   └─ Structure validation
//!     ↓
//! Stratum Resolution → Stratum-assigned AST
//!     ↓
//! Era Resolution → Era-validated AST
//!     ↓
//! Uses Validation → Safety-validated AST
//!     ↓
//! Execution Block Compilation → Compiled Execution DAG
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

/// Parser (hand-written recursive descent)
///
/// Implements recursive descent parsing without combinator overhead.
pub mod parser;

/// Old chumsky-based parser (temporary, for comparison)
#[allow(dead_code)]
mod parser_chumsky;

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
///
/// Phase 12: Validation passes
/// - Type validation (resolve/validation.rs)
/// - Effect validation (resolve/effects.rs)
/// - Capability validation (resolve/capabilities.rs)
/// - Structure validation (resolve/structure.rs)
///
/// Phase 12.5: Execution prerequisites
/// - Stratum resolution (resolve/strata.rs) - assigns strata to nodes
/// - Era resolution (resolve/eras.rs) - validates eras and transitions
/// - Uses validation (resolve/uses.rs) - enforces dangerous function declarations
pub mod resolve;

/// Validation passes (re-export for backward compatibility)
///
/// Validation modules now live under `resolve::` but are re-exported here.
pub mod validate {
    //! Multi-pass validation and linting.
    //!
    //! Note: Core validation modules are now in `resolve::validation`,
    //! `resolve::effects`, `resolve::capabilities`, and `resolve::structure`.
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
/// Phase 15 exposes the high-level interface for compiling entire worlds
/// from the filesystem and serializing the results.
pub mod compile;

pub use compile::{compile, deserialize_world, format_errors, serialize_world};

// Placeholder version info
/// Compiler version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
