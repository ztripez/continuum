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
//! Resolves TypeExpr → Type:
//! - Resolve unit expressions to Unit
//! - Resolve user type names to TypeIds
//! - Convert AST type syntax to semantic types
//!
//! This pass consumes `type_expr` and populates `output` on Node<I>.
//!
//! # Validation (`validation`)
//!
//! Validates typed expressions for semantic correctness:
//! - Type compatibility checks
//! - Kernel call signature validation
//! - Bounds checking
//! - Unit consistency validation
//!
//! This pass populates `validation_errors` on Node<I>.
//!
//! # Effect Validation (`effects`)
//!
//! Validates kernel purity restrictions based on execution phase:
//! - Pure kernels allowed in all phases
//! - Effect kernels (emit, spawn, destroy) only in effect-allowed phases (Collect, Fracture)
//! - Validates phase purity rules for kernel calls
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

pub mod effects;
pub mod names;
pub mod types;
pub mod validation;
