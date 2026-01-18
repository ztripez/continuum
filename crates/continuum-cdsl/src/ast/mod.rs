//! Abstract Syntax Tree (AST) for Continuum DSL
//!
//! This module defines the unified AST structure used throughout the compiler
//! pipeline. Unlike traditional compilers with separate AST→IR passes, Continuum
//! uses a single `Node<I>` structure that flows through all compilation phases.
//!
//! # Architecture
//!
//! The AST is built on three core concepts:
//!
//! 1. **Unified Node** - `Node<I>` is the single structure for all primitives
//!    (signals, fields, operators, etc). The generic parameter `I` distinguishes
//!    global (`I = ()`) from per-entity (`I = EntityId`) nodes.
//!
//! 2. **Role Composition** - Each node has a `RoleData` that determines what it
//!    is (Signal, Field, Operator, etc) and carries role-specific data. This makes
//!    invalid combinations unrepresentable at compile time.
//!
//! 3. **Compile-Time Registry** - Role capabilities (which phases can execute,
//!    what context is available) are defined in a static `ROLE_REGISTRY` array,
//!    providing zero-cost role metadata lookup.
//!
//! # Compilation Flow
//!
//! ```text
//! Parser → Node<I> (untyped)
//!    ↓
//! Type Resolution → Node<I> (typed, output set)
//!    ↓
//! Validation → Node<I> (errors recorded)
//!    ↓
//! Compilation → Node<I> (executions added)
//!    ↓
//! DAG Builder → Execution Graph
//! ```
//!
//! Fields are cleared after each phase consumes them (e.g., `type_expr` cleared
//! after type resolution), making the pipeline state explicit.
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::ast::{Node, RoleData, EntityId};
//! use continuum_cdsl::foundation::Path;
//!
//! // Create a global signal node
//! let signal = Node::new(
//!     Path::from_str("world.temperature"),
//!     span,
//!     RoleData::Signal,
//!     (), // global index
//! );
//!
//! // Create a per-entity member node
//! let member = Node::new(
//!     Path::from_str("plate.velocity"),
//!     span,
//!     RoleData::Signal,
//!     EntityId(Path::from_str("plate")), // per-entity index
//! );
//! ```

mod block;
mod capability;
mod declaration;
mod expr;
mod kernel;
mod node;
mod pipeline;
mod role;
mod untyped;
mod warmup;

pub use block::*;
pub use capability::*;
pub use declaration::*;
pub use expr::*;
pub use kernel::*;
pub use node::*;
pub use pipeline::*;
pub use role::*;
pub use warmup::*;

// Re-export KernelId from kernel-types (single source of truth)
pub use continuum_kernel_types::KernelId;

// Re-export untyped AST types explicitly to avoid ExprKind ambiguity
pub use untyped::{BinaryOp, Expr, TypeExpr, UnaryOp, UnitExpr};

// ExprKind from untyped module is public but not re-exported at top level
// to avoid name collision with expr::ExprKind. Access it as untyped::ExprKind.
pub use untyped::ExprKind as UntypedKind;
