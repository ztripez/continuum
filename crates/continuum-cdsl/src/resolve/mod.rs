//! Resolution and validation passes for CDSL compilation
//!
//! This module implements the resolution phases of the compiler pipeline:
//!
//! ```text
//! Parse → Desugar → Name Resolution → Type Resolution → Validation →
//!                       ^^^^^^^^           ^^^^^^^^        ^^^^^^^^^^
//!                      resolve/names    resolve/types   validation/
//!                                                        effects/
//!                                                        capabilities
//!
//! → Stratum Resolution → Era Resolution → Uses Validation → Block Compilation
//!      ^^^^^^^^^           ^^^^^^^^^         ^^^^^
//!   resolve/strata      resolve/eras     resolve/uses
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
//! # Capability Validation (`capabilities`)
//!
//! Validates that expressions only access capabilities available in their execution context:
//! - Prev/Current/Inputs access controlled by phase
//! - Emit only in Collect phase and impulse handlers
//! - Payload only in impulse handlers
//! - Validates capability access rules for all expression kinds
//!
//! # Stratum Resolution (`strata`)
//!
//! Resolves stratum assignments from node attributes:
//! - Extracts `:stratum(name)` attributes and validates against world's stratum declarations
//! - Populates `node.stratum` field
//! - Provides default stratum if not specified (when world has single stratum)
//! - Resolves cadence from `:stride(N)` or `:cadence(N)` attributes
//! - Validates cadence is positive integer, defaults to 1
//!
//! This pass is part of Phase 12.5 execution prerequisites.
//!
//! # Era Resolution (`eras`)
//!
//! Validates era declarations and transition graphs:
//! - Validates `dt` expressions have time units
//! - Validates stratum references in `strata_policy`
//! - Validates transition target eras exist
//! - Validates transition conditions are Bool-typed
//! - Detects cycles in era transition graphs (warns, does not error)
//!
//! This pass runs after stratum resolution and is part of Phase 12.5 execution prerequisites.
//!
//! # Uses Validation (`uses`)
//!
//! Validates dangerous function usage declarations:
//! - Enforces `: uses(maths.clamping)` for clamp/saturate/wrap functions
//! - Enforces `: uses(dt.raw)` for raw dt access
//! - Validates both TypedExpr (compiled blocks) and Expr (warmup/when/observe)
//! - Emits helpful error messages with hints for missing declarations
//!
//! This pass runs after era resolution and before execution block compilation.
//!
//! # Structure Validation (`structure`)
//!
//! Validates structural coherence of the AST:
//! - Entity references
//! - Member definitions
//! - Structural constraints
//!
//! # Pipeline Integration
//!
//! These passes operate on parsed AST and prepare nodes for execution:
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
//!
//! After stratum resolution:
//!   Node<I> { stratum: Some(StratumId), ... }
//!
//! After all passes:
//!   Ready for execution block compilation and DAG construction
//! ```

pub mod blocks;
pub mod capabilities;
pub mod effects;
pub mod eras;
pub mod names;
pub mod strata;
pub mod structure;
pub mod types;
pub mod uses;
pub mod validation;
