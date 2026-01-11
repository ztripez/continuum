//! Expression types for the Continuum DSL AST.
//!
//! This module defines expression nodes that represent computations and data flow
//! in the DSL. Expressions appear in resolve blocks, assertions, function bodies,
//! and anywhere values are computed.

use super::{Path, Spanned};

/// A function call argument, which can be positional or named.
///
/// Named arguments use the syntax `name: value`, e.g., `method: rk4`.
/// Positional arguments are just bare expressions.
///
/// # Examples
///
/// ```text
/// integrate(prev, rate, method: rk4)  // 'prev' and 'rate' positional, 'method' named
/// decay(value, tau)                    // both positional
/// relax(current, target, method: exp) // 'method' is named
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CallArg {
    /// Optional parameter name for named arguments.
    /// `None` for positional arguments.
    pub name: Option<String>,
    /// The argument value expression.
    pub value: Spanned<Expr>,
}

impl CallArg {
    /// Creates a positional (unnamed) argument.
    pub fn positional(value: Spanned<Expr>) -> Self {
        Self { name: None, value }
    }

    /// Creates a named argument.
    pub fn named(name: String, value: Spanned<Expr>) -> Self {
        Self {
            name: Some(name),
            value,
        }
    }

    /// Returns true if this is a named argument.
    pub fn is_named(&self) -> bool {
        self.name.is_some()
    }
}

/// Expression node representing computations and data flow.
///
/// Expressions form the core of DSL computation. They appear in resolve
/// blocks, assertions, function bodies, and anywhere values are computed.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Literal value: `42`, `3.14`, `"hello"`, `true`.
    Literal(Literal),

    /// Literal with unit annotation: `1000 <yr>`, `288.15 <K>`.
    LiteralWithUnit {
        /// The numeric or string value.
        value: Literal,
        /// Unit string (e.g., "yr", "K", "m/s").
        unit: String,
    },

    /// Unqualified path reference: `foo.bar.baz`.
    Path(Path),

    /// Previous tick's value of the current signal: `prev`.
    Prev,

    /// Previous tick's value of a specific field: `prev.temperature`.
    PrevField(String),

    /// Raw (unscaled) time step for dt-robust expressions: `dt_raw`.
    DtRaw,

    /// Accumulated simulation time in seconds: `sim_time`.
    SimTime,

    /// Impulse payload in apply blocks: `payload`.
    Payload,

    /// Field access on impulse payload: `payload.magnitude`.
    PayloadField(String),

    /// Explicit signal reference: `signal.terra.temperature`.
    SignalRef(Path),

    /// Constant reference: `const.physics.G`.
    ConstRef(Path),

    /// Config value reference: `config.terra.thermal.tau`.
    ConfigRef(Path),

    /// Field reference (observation): `field.terra.surface.temp`.
    FieldRef(Path),

    /// Binary operation: `a + b`, `x * y`, `p && q`.
    Binary {
        /// The operator.
        op: BinaryOp,
        /// Left operand.
        left: Box<Spanned<Expr>>,
        /// Right operand.
        right: Box<Spanned<Expr>>,
    },

    /// Unary operation: `-x`, `!flag`.
    Unary {
        /// The operator.
        op: UnaryOp,
        /// Operand expression.
        operand: Box<Spanned<Expr>>,
    },

    /// Function call: `lerp(a, b, t)`, `sin(angle)`, `integrate(prev, rate, method: rk4)`.
    ///
    /// Arguments can be positional or named. Named arguments use `name: value` syntax
    /// and must come after all positional arguments.
    Call {
        /// Function being called.
        function: Box<Spanned<Expr>>,
        /// Call arguments (positional and/or named).
        args: Vec<CallArg>,
    },

    /// Method call on an object: `vec.normalize()`, `signal.clamp(0, 1)`.
    ///
    /// Arguments can be positional or named, same as function calls.
    MethodCall {
        /// Object receiving the method call.
        object: Box<Spanned<Expr>>,
        /// Method name.
        method: String,
        /// Method arguments (positional and/or named).
        args: Vec<CallArg>,
    },

    /// Field access on a struct: `orbital.semi_major`, `state.position`.
    FieldAccess {
        /// Object to access.
        object: Box<Spanned<Expr>>,
        /// Field name.
        field: String,
    },

    /// Local binding: `let x = expr in body`.
    Let {
        /// Variable name.
        name: String,
        /// Value to bind.
        value: Box<Spanned<Expr>>,
        /// Body where binding is visible.
        body: Box<Spanned<Expr>>,
    },

    /// Conditional expression: `if cond { then } else { else }`.
    If {
        /// Condition to test.
        condition: Box<Spanned<Expr>>,
        /// Expression if true.
        then_branch: Box<Spanned<Expr>>,
        /// Optional expression if false.
        else_branch: Option<Box<Spanned<Expr>>>,
    },

    /// For loop over a sequence: `for x in items { body }`.
    For {
        /// Loop variable name.
        var: String,
        /// Sequence to iterate.
        iter: Box<Spanned<Expr>>,
        /// Loop body.
        body: Box<Spanned<Expr>>,
    },

    /// Block of sequential expressions: `{ expr1; expr2; result }`.
    Block(Vec<Spanned<Expr>>),

    /// Emit a value to a signal: `emit(signal.terra.temp, value)`.
    EmitSignal {
        /// Target signal path.
        target: Path,
        /// Value to emit.
        value: Box<Spanned<Expr>>,
    },

    /// Emit a positioned sample to a field: `emit_field(field.surface, pos, value)`.
    EmitField {
        /// Target field path.
        target: Path,
        /// Position for the sample.
        position: Box<Spanned<Expr>>,
        /// Value at that position.
        value: Box<Spanned<Expr>>,
    },

    /// Struct literal: `{ x: 1.0, y: 2.0 }`.
    Struct(Vec<(String, Spanned<Expr>)>),

    /// Accumulated inputs from Collect phase: `collected`.
    Collected,

    /// Mathematical constant: `pi`, `tau`, `e`.
    MathConst(MathConst),

    /// Map function over sequence: `map(items, fn(x) { x * 2 })`.
    Map {
        /// Sequence to map over.
        sequence: Box<Spanned<Expr>>,
        /// Function to apply.
        function: Box<Spanned<Expr>>,
    },

    /// Fold/reduce sequence: `fold(items, 0, fn(acc, x) { acc + x })`.
    Fold {
        /// Sequence to fold.
        sequence: Box<Spanned<Expr>>,
        /// Initial accumulator value.
        init: Box<Spanned<Expr>>,
        /// Folding function.
        function: Box<Spanned<Expr>>,
    },

    // === Entity expressions ===

    /// Reference to current entity instance field: `self.mass`.
    SelfField(String),

    /// Reference to an entity type: `entity.stellar.moon`.
    EntityRef(Path),

    /// Access entity instance by ID: `entity.moon["luna"]`.
    EntityAccess {
        /// Entity type path.
        entity: Path,
        /// Instance identifier expression.
        instance: Box<Spanned<Expr>>,
    },

    /// Aggregate operation over entity instances: `sum(entity.moon, self.mass)`.
    Aggregate {
        /// Aggregation operator.
        op: AggregateOp,
        /// Entity type to aggregate over.
        entity: Path,
        /// Expression evaluated per instance.
        body: Box<Spanned<Expr>>,
    },

    /// Other instances excluding current: `other(entity.moon)`.
    /// Used for N-body interactions where you need all instances except self.
    Other(Path),

    /// Pairwise iteration: `pairs(entity.moon)`.
    /// Generates all unique (i,j) combinations where i < j.
    Pairs(Path),

    /// Filter entity instances by predicate: `filter(entity.moon, self.mass > 1e20)`.
    Filter {
        /// Entity type to filter.
        entity: Path,
        /// Predicate expression.
        predicate: Box<Spanned<Expr>>,
    },

    /// First instance matching predicate: `first(entity.plate, self.type == Continental)`.
    First {
        /// Entity type to search.
        entity: Path,
        /// Predicate to match.
        predicate: Box<Spanned<Expr>>,
    },

    /// Nearest instance to position: `nearest(entity.plate, position)`.
    Nearest {
        /// Entity type to search.
        entity: Path,
        /// Position to measure from.
        position: Box<Spanned<Expr>>,
    },

    /// All instances within radius: `within(entity.moon, pos, 1e9)`.
    Within {
        /// Entity type to search.
        entity: Path,
        /// Center position.
        position: Box<Spanned<Expr>>,
        /// Search radius.
        radius: Box<Spanned<Expr>>,
    },
}

/// Literal value in the DSL.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    /// Integer literal: `42`, `-7`, `1000000`.
    Integer(i64),
    /// Floating-point literal: `3.14`, `1e-6`, `2.998e8`.
    Float(f64),
    /// String literal: `"hello"`, `"temperature"`.
    String(String),
    /// Boolean literal: `true`, `false`.
    Bool(bool),
}

/// Mathematical constants available in the DSL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MathConst {
    /// Pi (3.14159...): ratio of circumference to diameter.
    Pi,
    /// Tau (6.28318...): ratio of circumference to radius (2*pi).
    Tau,
    /// Euler's number (2.71828...): base of natural logarithm.
    E,
    /// Imaginary unit for complex numbers.
    I,
    /// Golden ratio (1.61803...): (1 + sqrt(5)) / 2.
    Phi,
}

/// Binary operators for arithmetic, comparison, and logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    /// Numeric negation: `-x`.
    Neg,
    /// Logical not: `not x` (also accepts `!x`).
    Not,
}

/// Aggregate operations over entity instances
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregateOp {
    /// Sum of values: `sum(entity.moon, self.mass)`
    Sum,
    /// Product of values: `product(entity.layer, self.transmittance)`
    Product,
    /// Minimum value: `min(entity.moon, self.orbit_radius)`
    Min,
    /// Maximum value: `max(entity.star, self.luminosity)`
    Max,
    /// Average value: `mean(entity.plate, self.age)`
    Mean,
    /// Count of instances: `count(entity.moon)`
    Count,
    /// Any instance matches predicate: `any(entity.moon, self.mass > 1e22)`
    Any,
    /// All instances match predicate: `all(entity.star, self.luminosity > 0)`
    All,
    /// No instance matches predicate: `none(entity.plate, self.age < 0)`
    None,
}
