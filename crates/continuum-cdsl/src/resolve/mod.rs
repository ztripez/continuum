//! Name and type resolution passes
//!
//! This module implements the resolution phases of the compiler pipeline:
//!
//! ```text
//! Parse → Desugar → Name Resolution → Type Resolution → Validation
//!                       ^^^^^^^^           ^^^^^^^^
//!                      resolve/names    resolve/types
//! ```
//!
//! # Name Resolution (`names`)
//!
//! Validates that all Path references in expressions refer to declared symbols:
//! - Signal paths resolve to signal declarations
//! - Field paths resolve to field declarations
//! - Config/Const paths resolve to config/const blocks
//! - Struct type paths resolve to type declarations
//! - Local variables are in scope (from Let bindings)
//!
//! This pass builds a symbol table and validates references but does NOT resolve types.
//!
//! # Type Resolution (`types`)
//!
//! Resolves TypeExpr → Type and infers types where not explicit:
//! - Resolve unit expressions to Unit
//! - Infer types from context (bidirectional type inference)
//! - Resolve kernel calls to kernel signatures
//! - Validate type compatibility
//!
//! This pass consumes `type_expr` and populates `output` on Node<I>.
//!
//! # Pipeline Integration
//!
//! These passes operate on parsed AST and prepare nodes for validation:
//!
//! ```text
//! Parser output:
//!   Node<I> { type_expr: Some(_), execution_blocks: [...], output: None }
//!
//! After name resolution:
//!   - All Path references validated
//!   - Symbol table built
//!   - Scope rules enforced
//!   (No change to Node<I> fields - this is a validation-only pass)
//!
//! After type resolution:
//!   Node<I> { type_expr: None, execution_blocks: [...], output: Some(Type) }
//! ```

pub mod names;
pub mod types;
