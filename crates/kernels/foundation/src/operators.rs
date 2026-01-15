//! Operators for expressions in the Continuum DSL.
//!
//! These operator enums are used consistently across the AST, IR, and VM
//! to avoid duplication and 1:1 conversion boilerplate.

use serde::{Deserialize, Serialize};

/// Binary operators for arithmetic, comparison, and logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    /// Addition: `a + b`.
    Add,
    /// Subtraction: `a - b`.
    Sub,
    /// Multiplication: `a * b`.
    Mul,
    /// Division: `a / b`.
    Div,
    /// Exponentiation: `a ^ b` or `a ** b`.
    Pow,
    /// Equality: `a == b`.
    Eq,
    /// Inequality: `a != b`.
    Ne,
    /// Less than: `a < b`.
    Lt,
    /// Less than or equal: `a <= b`.
    Le,
    /// Greater than: `a > b`.
    Gt,
    /// Greater than or equal: `a >= b`.
    Ge,
    /// Logical and: `a and b` (also accepts `a && b`).
    And,
    /// Logical or: `a or b` (also accepts `a || b`).
    Or,
}

/// Unary operators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    /// Numeric negation: `-x`.
    Neg,
    /// Logical not: `not x` (also accepts `!x`).
    Not,
}

/// Aggregate operations over entity instances.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AggregateOp {
    /// Sum of values: `agg.sum(entity.moon, self.mass)`
    Sum,
    /// Product of values: `agg.product(entity.layer, self.transmittance)`
    Product,
    /// Minimum value: `agg.min(entity.moon, self.orbit_radius)`
    Min,
    /// Maximum value: `agg.max(entity.star, self.luminosity)`
    Max,
    /// Average value: `agg.mean(entity.plate, self.age)`
    Mean,
    /// Count of instances: `agg.count(entity.moon)`
    Count,
    /// Any instance matches predicate: `agg.any(entity.moon, self.mass > 1e22)`
    Any,
    /// All instances match predicate: `agg.all(entity.star, self.luminosity > 0)`
    All,
    /// No instance matches predicate: `agg.none(entity.plate, self.age < 0)`
    None,
}
