//! Block bodies and statements
//!
//! This module defines block body types used in execution blocks.
//! These types are shared between declaration.rs and node.rs.

use crate::ast::TypedExpr;
use crate::ast::untyped::Expr;
use crate::foundation::{Path, Span};

/// A single simulation statement within a procedural block body.
///
/// Statements represent effectful operations or local bindings. They are exclusively
/// permitted in blocks with effect capabilities (e.g., [`Phase::Collect`][crate::foundation::Phase::Collect]
/// or [`Phase::Fracture`][crate::foundation::Phase::Fracture]). Pure blocks
/// (e.g., [`Phase::Resolve`][crate::foundation::Phase::Resolve] or
/// [`Phase::Measure`][crate::foundation::Phase::Measure] with expression bodies)
/// may not contain statements.
///
/// The type parameter `E` defines the expression representation used within the statement,
/// allowing the same structure to represent both untyped parser output ([`Expr`][crate::ast::untyped::Expr])
/// and compiled IR ([`TypedExpr`][crate::ast::TypedExpr]).
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

/// A compiled and type-validated simulation statement.
///
/// `TypedStmt` is the IR representation of a statement after it has passed through
/// [`compile_statements`][crate::resolve::blocks::compile_statements]. It contains
/// [`TypedExpr`][crate::ast::TypedExpr] nodes which include fully resolved type metadata
/// and source span information.
pub type TypedStmt = Stmt<TypedExpr>;

/// Block body - either single expression or statement list.
///
/// The body kind is determined by the block's phase capabilities:
/// - Pure phases (Resolve, Measure, Assert): Expression or TypedExpression
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

    /// A list of compiled, type-validated statements.
    ///
    /// This variant represents the final IR state of an effectful execution block.
    /// It is produced during the statement compilation pass and is used during
    /// Phase 13 DAG construction to extract side effects ([`Execution::emits`][crate::ast::Execution::emits]).
    TypedStatements(Vec<TypedStmt>),
}
