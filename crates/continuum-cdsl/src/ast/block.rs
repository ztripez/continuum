//! Block bodies and statements
//!
//! This module defines block body types used in execution blocks.
//! These types are shared between declaration.rs and node.rs.

use crate::ast::untyped::Expr;
use crate::foundation::{Path, Span};

/// Statement in a block body.
///
/// Statements appear in blocks with effect capabilities (Collect, Apply, Emit).
/// Blocks in pure phases (Resolve, Measure, Assert) use expression bodies.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Let binding: `let x = expr`
    ///
    /// Introduces a local variable visible in subsequent statements.
    /// Unlike `let...in` expressions, this doesn't have a body scope.
    Let {
        /// Variable name
        name: String,
        /// Value expression
        value: Expr,
        /// Source location
        span: Span,
    },

    /// Signal assignment: `signal.path <- expr`
    ///
    /// Emits a value to a signal's input accumulator.
    /// Valid only in blocks with Emit capability (Collect, Apply, WarmUp).
    SignalAssign {
        /// Target signal path
        target: Path,
        /// Value to emit
        value: Expr,
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
        position: Expr,
        /// Value expression (the field data at this position)
        value: Expr,
        /// Source location
        span: Span,
    },

    /// Expression statement
    ///
    /// An expression evaluated for its side effects (usually a function call).
    Expr(Expr),
}

/// Block body - either single expression or statement list.
///
/// The body kind is determined by the block's phase capabilities:
/// - Pure phases (Resolve, Iterate, Assert): Expression
/// - Effect phases (Collect, Apply, Emit): Statements
#[derive(Debug, Clone, PartialEq)]
pub enum BlockBody {
    /// Single expression (pure phases)
    ///
    /// Used in: resolve, iterate, measure (when simple), assert
    Expression(Expr),

    /// Statement list (effect phases)
    ///
    /// Used in: collect, apply, emit, when
    Statements(Vec<Stmt>),
}
