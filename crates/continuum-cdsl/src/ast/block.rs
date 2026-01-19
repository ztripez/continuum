//! Block bodies and statements
//!
//! This module defines block body types used in execution blocks.
//! These types are shared between declaration.rs and node.rs.

use crate::ast::TypedExpr;
use crate::ast::untyped::Expr;
use crate::foundation::{Path, Span};

/// Statement in a block body.
///
/// Statements appear in blocks with effect capabilities (Collect, Apply, Emit).
/// Blocks in pure phases (Resolve, Measure, Assert) use expression bodies.
///
/// The type parameter `E` represents the expression type (typically [`Expr`]
/// for untyped AST or [`TypedExpr`] for compiled IR).
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt<E = Expr> {
    /// Let binding: `let x = expr`
    ///
    /// Introduces a local variable visible in subsequent statements.
    /// Unlike `let...in` expressions, this doesn't have a body scope.
    Let {
        /// Variable name
        name: String,
        /// Value expression
        value: E,
        /// Source location
        span: Span,
    },

    /// Signal assignment: `signal.path <- expr`
    ///
    /// Emits a value to a signal's input accumulator.
    /// Valid in statement blocks: collect blocks, fracture emit blocks.
    SignalAssign {
        /// Target signal path
        target: Path,
        /// Value to emit
        value: E,
        /// Source location
        span: Span,
    },

    /// Field assignment: `field.path <- position, value`
    ///
    /// Emits a positioned sample to a field.
    /// Valid only in Measure phase with Emit capability.
    FieldAssign {
        /// Target field path
        target: Path,
        /// Position expression (Vec2/Vec3 or other spatial coordinate)
        position: E,
        /// Value expression (the field data at this position)
        value: E,
        /// Source location
        span: Span,
    },

    /// Expression statement
    ///
    /// An expression evaluated for its side effects (usually a function call).
    Expr(E),
}

/// Alias for a compiled, typed statement.
pub type TypedStmt = Stmt<TypedExpr>;

/// Block body - either single expression or statement list.
///
/// The body kind is determined by the block's phase capabilities:
/// - Pure phases (Resolve, Measure): Expression or TypedExpression
/// - Effect phases (Collect, Fracture): Statements or TypedStatements
///
/// # Lifecycle
///
/// 1. Parser produces `Expression(Expr)` or `Statements(Vec<Stmt<Expr>>)`.
/// 2. Expression typing pass converts `Expression` to `TypedExpression`.
/// 3. Statement compilation pass (Phase 12.5-S) converts `Statements` to `TypedStatements`.
/// 4. Execution block compilation expects either `TypedExpression` or `TypedStatements`.
#[derive(Debug, Clone, PartialEq)]
pub enum BlockBody {
    /// Single untyped expression (pure phases, from parser)
    Expression(Expr),

    /// Single typed expression (pure phases, after type resolution)
    TypedExpression(TypedExpr),

    /// Untyped statement list (effect phases, from parser)
    Statements(Vec<Stmt<Expr>>),

    /// Typed statement list (effect phases, after statement compilation)
    TypedStatements(Vec<TypedStmt>),
}
