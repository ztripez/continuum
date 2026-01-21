//! Untyped AST for parser output
//!
//! This module defines the untyped AST structures that the parser produces.
//! These are then type-checked and transformed into the typed AST ([`TypedExpr`])
//! by the type resolution pass.
//!
//! # Design Principles
//!
//! ## Parser Simplicity
//!
//! The parser produces simple, untyped structures:
//! - No type information (that's added by type resolution)
//! - No capability checking (that's enforced during type checking)
//! - Just syntactic structure + source locations
//!
//! ## Bidirectional Type Inference
//!
//! Untyped expressions support bidirectional type inference:
//! - Literals without units infer from context: `10.0` → `Scalar<m>` if context requires meters
//! - Struct literals infer field types from struct definition
//! - Vector literals infer element type from context
//!
//! ## Error Recovery
//!
//! Parser errors are captured in the AST as [`ParseError`] nodes, allowing
//! the parser to continue and report multiple errors in a single pass.
//!
//! # Compilation Flow
//!
//! ```text
//! Parser → Expr (untyped)
//!    ↓
//! Type Resolution → TypedExpr (with Type)
//!    ↓
//! Validation → TypedExpr (validated)
//!    ↓
//! DAG Builder → Execution Graph
//! ```
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::ast::{Expr, ExprKind as UntypedKind};
//!
//! // Parser produces untyped expression
//! let expr = Expr {
//!     kind: UntypedKind::Literal {
//!         value: 100.0,
//!         unit: None,  // Unit will be inferred
//!     },
//!     span,
//! };
//!
//! // Type resolution infers Scalar<m> from context
//! // and produces TypedExpr with ty: Scalar<m>
//! ```

use crate::foundation::{Path, Span};

use super::expr::AggregateOp;
use crate::foundation::EntityId;
use continuum_kernel_types::KernelId;

/// Untyped expression from parser
///
/// Represents an expression as parsed from source code, before type
/// resolution. The type checker transforms `Expr` → [`TypedExpr`](super::TypedExpr)
/// by inferring types and validating constraints.
///
/// # Type Inference
///
/// Untyped expressions support bidirectional type inference:
/// - **Checking mode** - Context provides expected type (e.g., function argument)
/// - **Synthesis mode** - Expression synthesizes its type (e.g., literal with unit)
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::ast::{Expr, UnitExpr};
///
/// // Literal with unit syntax (synthesis mode)
/// let with_unit = Expr::literal(
///     100.0,
///     Some(UnitExpr::Base("m".to_string())),
///     span
/// );
///
/// // Literal without unit (checking mode - infers from context)
/// let without_unit = Expr::literal(100.0, None, span);
///
/// // Binary operator (desugars to kernel call)
/// let add = Expr::binary(BinaryOp::Add, a, b, span);
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Expr {
    /// Expression kind (what kind of expression this is)
    pub kind: ExprKind,

    /// Source location for error messages
    pub span: Span,
}

impl Expr {
    /// Attempt to interpret this expression as a static [`Path`].
    ///
    /// Succeeds if the expression is a [`Local`] identifier or a chain of
    /// [`FieldAccess`] operations on a path.
    pub fn as_path(&self) -> Option<Path> {
        match &self.kind {
            ExprKind::Local(name) => Some(Path::from(name.as_str())),
            ExprKind::FieldAccess { object, field } => {
                let mut path = object.as_path()?;
                path.segments.push(field.clone());
                Some(path)
            }
            _ => None,
        }
    }

    /// Create a new untyped expression
    ///
    /// # Parameters
    ///
    /// - `kind`: The expression kind (literal, operator, binding form, etc.)
    /// - `span`: Source location for error messages
    ///
    /// # Returns
    ///
    /// An untyped expression wrapping the kind with source location.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let expr = Expr::new(ExprKind::Prev, span);
    /// ```
    pub fn new(kind: ExprKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Create a literal expression
    ///
    /// # Parameters
    ///
    /// - `value`: Numeric value of the literal
    /// - `unit`: Optional unit syntax (None = infer from context during type resolution)
    /// - `span`: Source location
    ///
    /// # Returns
    ///
    /// An untyped expression representing a numeric literal with optional unit syntax.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use continuum_cdsl::ast::{Expr, UnitExpr};
    ///
    /// // With unit syntax
    /// let distance = Expr::literal(
    ///     100.0,
    ///     Some(UnitExpr::Base("m".to_string())),
    ///     span
    /// );
    ///
    /// // Without unit (infer from context)
    /// let number = Expr::literal(42.0, None, span);
    /// ```
    pub fn literal(value: f64, unit: Option<UnitExpr>, span: Span) -> Self {
        Self::new(ExprKind::Literal { value, unit }, span)
    }

    /// Create a local variable reference
    ///
    /// # Parameters
    ///
    /// - `name`: Variable name to reference
    /// - `span`: Source location
    ///
    /// # Returns
    ///
    /// An untyped expression referencing a let-bound variable.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let var_ref = Expr::local("x", span);
    /// ```
    pub fn local(name: impl Into<String>, span: Span) -> Self {
        Self::new(ExprKind::Local(name.into()), span)
    }

    /// Create a binary operator expression
    ///
    /// Binary operators are preserved as syntax in the untyped AST and
    /// desugar to kernel calls (built-in runtime functions) during type resolution.
    ///
    /// # Parameters
    ///
    /// - `op`: Binary operator (Add, Mul, Lt, And, etc.)
    /// - `left`: Left operand expression
    /// - `right`: Right operand expression
    /// - `span`: Source location
    ///
    /// # Returns
    ///
    /// An untyped binary operator expression that will desugar during type resolution:
    /// - `a + b` → `maths.add(a, b)`
    /// - `a * b` → `maths.mul(a, b)`
    /// - `a < b` → `compare.lt(a, b)`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let sum = Expr::binary(BinaryOp::Add, a, b, span);
    /// ```
    pub fn binary(op: BinaryOp, left: Expr, right: Expr, span: Span) -> Self {
        Self::new(
            ExprKind::Binary {
                op,
                left: Box::new(left),
                right: Box::new(right),
            },
            span,
        )
    }

    /// Create a unary operator expression
    ///
    /// Unary operators are preserved as syntax in the untyped AST and
    /// desugar to kernel calls (built-in runtime functions) during type resolution.
    ///
    /// # Parameters
    ///
    /// - `op`: Unary operator (Neg or Not)
    /// - `operand`: Operand expression
    /// - `span`: Source location
    ///
    /// # Returns
    ///
    /// An untyped unary operator expression that will desugar during type resolution:
    /// - `-x` → `maths.neg(x)`
    /// - `!x` → `logic.not(x)`
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let neg = Expr::unary(UnaryOp::Neg, x, span);
    /// ```
    pub fn unary(op: UnaryOp, operand: Expr, span: Span) -> Self {
        Self::new(
            ExprKind::Unary {
                op,
                operand: Box::new(operand),
            },
            span,
        )
    }
}

/// Untyped expression kinds
///
/// These are the syntactic forms the parser recognizes. During type resolution,
/// operators are desugared to **kernel calls** (built-in runtime functions provided
/// by the engine, organized in namespaces like `maths.*`, `vector.*`, `logic.*`).
///
/// | Syntax | Desugars to Kernel |
/// |--------|-------------|
/// | `a + b` | `maths.add(a, b)` |
/// | `a * b` | `maths.mul(a, b)` |
/// | `-x` | `maths.neg(x)` |
/// | `a < b` | `compare.lt(a, b)` |
/// | `a && b` | `logic.and(a, b)` |
/// | `if c { t } else { e }` | `logic.select(c, t, e)` |
///
/// # Binding Forms
///
/// `Let`, `Aggregate`, and `Fold` introduce variable bindings and are preserved
/// as special forms (not desugared to calls).
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ExprKind {
    // === Literals ===
    /// Numeric literal with optional unit syntax
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// 100.0<m>     // value: 100.0, unit: Some(UnitExpr::Base("m"))
    /// 3.14         // value: 3.14, unit: None (infer from context)
    /// 0.0<>        // value: 0.0, unit: Some(UnitExpr::Dimensionless)
    /// ```
    ///
    /// **Important:** The unit is stored as syntactic [`UnitExpr`], not resolved
    /// [`Unit`](crate::foundation::Unit). Unit resolution happens during type checking.
    Literal {
        /// Numeric value
        value: f64,
        /// Optional unit syntax (None = infer from context during type resolution)
        unit: Option<UnitExpr>,
    },

    /// Boolean literal
    BoolLiteral(bool),

    /// String literal
    StringLiteral(String),

    /// Vector literal

    ///
    /// # Examples
    ///
    /// ```cdsl
    /// [1.0, 2.0, 3.0]
    /// [x, y]
    /// ```
    Vector(Vec<Expr>),

    // === References ===
    /// Local let-bound variable
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// let x = 10.0
    /// x  // Local("x")
    /// ```
    Local(String),

    /// Signal reference
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// velocity
    /// plate.mass
    /// ```
    Signal(Path),

    /// Field reference (observer-only)
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// field(temperature)
    /// ```
    Field(Path),

    /// Config value reference
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// config.initial_temp
    /// ```
    Config(Path),

    /// Const value reference
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// const.BOLTZMANN
    /// ```
    Const(Path),

    // === Context values ===
    /// Previous tick value
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// prev
    /// ```
    Prev,

    /// Just-resolved current value
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// current
    /// ```
    Current,

    /// Accumulated inputs
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// inputs
    /// ```
    Inputs,

    /// Time step
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// dt
    /// ```
    Dt,

    /// Current entity instance
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// self
    /// ```
    Self_,

    /// Other entity instance (n-body)
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// other
    /// ```
    Other,

    /// Impulse payload
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// payload
    /// ```
    Payload,

    // === Operators (desugar to kernel calls) ===
    /// Binary operator
    ///
    /// Desugars to kernel call during type resolution.
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// a + b    // Binary(Add, a, b) → maths.add(a, b)
    /// a * b    // Binary(Mul, a, b) → maths.mul(a, b)
    /// a < b    // Binary(Lt, a, b) → compare.lt(a, b)
    /// ```
    Binary {
        /// Binary operator
        op: BinaryOp,
        /// Left operand
        left: Box<Expr>,
        /// Right operand
        right: Box<Expr>,
    },

    /// Unary operator
    ///
    /// Desugars to kernel call during type resolution.
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// -x     // Unary(Neg, x) → maths.neg(x)
    /// !flag  // Unary(Not, flag) → logic.not(flag)
    /// ```
    Unary {
        /// Unary operator
        op: UnaryOp,
        /// Operand
        operand: Box<Expr>,
    },

    /// If-then-else expression
    ///
    /// Desugars to `logic.select` during type resolution (eager evaluation).
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// if cond { a } else { b }
    /// // If(cond, a, b) → logic.select(cond, a, b)
    /// ```
    ///
    /// **Important:** Both branches evaluate eagerly. This is not short-circuiting.
    If {
        /// Condition expression
        condition: Box<Expr>,
        /// Then branch
        then_branch: Box<Expr>,
        /// Else branch
        else_branch: Box<Expr>,
    },

    // === Binding forms (preserved, not desugared) ===
    /// Local variable binding
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// let x = 10.0
    /// let y = x * 2.0
    /// ```
    Let {
        /// Variable name
        name: String,
        /// Value to bind
        value: Box<Expr>,
        /// Body expression (can reference name)
        body: Box<Expr>,
    },

    /// Aggregate operation over entity instances
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// sum(plates, |p| p.mass)
    /// max(bodies, |b| b.temperature)
    /// count(stars, |s| s.active)
    /// ```
    Aggregate {
        /// Aggregate operation
        op: AggregateOp,
        /// Entity to iterate over
        entity: EntityId,
        /// Binding name for loop variable
        binding: String,
        /// Body expression (can reference binding)
        body: Box<Expr>,
    },

    /// Custom fold/reduction over entity instances
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// fold(plates, 0.0<kg>, |total, p| total + p.mass)
    /// ```
    Fold {
        /// Entity to iterate over
        entity: EntityId,
        /// Initial accumulator value
        init: Box<Expr>,
        /// Accumulator binding name
        acc: String,
        /// Element binding name
        elem: String,
        /// Body expression (can reference acc and elem)
        body: Box<Expr>,
    },

    // === Calls and construction ===
    /// Function/kernel call
    ///
    /// Parser produces this for explicit calls like `sin(x)`, `dot(a, b)`.
    /// Binary/unary operators are desugared to calls during type resolution.
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// sin(angle)
    /// dot(velocity, normal)
    /// emit(force, impulse)
    /// ```
    Call {
        /// Function/kernel being called (as a path)
        func: Path,
        /// Argument expressions
        args: Vec<Expr>,
    },

    /// Kernel call (desugared from operators)
    ///
    /// This variant preserves kernel identity from desugaring until type resolution.
    /// Operators desugar to KernelCall instead of generic Call to maintain the
    /// distinction between user function calls and kernel calls.
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// a + b    // desugars to KernelCall { kernel: maths.add, args: [a, b] }
    /// -x       // desugars to KernelCall { kernel: maths.neg, args: [x] }
    /// if c{t}else{e}  // desugars to KernelCall { kernel: logic.select, args: [c,t,e] }
    /// ```
    KernelCall {
        /// Kernel being called (preserves namespace + name identity)
        kernel: KernelId,
        /// Argument expressions
        args: Vec<Expr>,
    },

    /// User-defined struct construction
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// OrbitalElements {
    ///     semi_major: 1.5e11<m>,
    ///     eccentricity: 0.017<>,
    ///     inclination: 0.0<rad>
    /// }
    /// ```
    ///
    /// **Strict rules:**
    /// - All fields required
    /// - No extra fields
    /// - No shorthand (`{x}` forbidden, must write `{x: x}`)
    Struct {
        /// Type being constructed (as a path)
        ty: Path,
        /// Field name-value pairs
        fields: Vec<(String, Expr)>,
    },

    /// Field access on user types, vectors, or context values
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// orbit.semi_major
    /// velocity.x
    /// prev.temperature
    /// payload.data
    /// ```
    FieldAccess {
        /// Object being accessed
        object: Box<Expr>,
        /// Field name
        field: String,
    },

    /// Parse error placeholder
    ///
    /// Allows parser to continue after encountering an error and report
    /// multiple errors in a single pass.
    ///
    /// # String Format
    ///
    /// The string contains a human-readable error message describing what
    /// the parser expected but didn't find. It should be non-empty and
    /// suitable for display in diagnostic output.
    ///
    /// # Examples
    ///
    /// ```text
    /// let x = 10.0 +   // Missing right operand
    /// ```
    ///
    /// Parser produces:
    /// ```rust,ignore
    /// Binary {
    ///     op: Add,
    ///     left: literal(10.0),
    ///     right: ParseError("expected expression"),
    /// }
    /// ```
    ///
    /// The error message is preserved in the AST and can be reported later
    /// during type checking or validation.
    ParseError(String),
}

/// Binary operators
///
/// All binary operators desugar to kernel calls during type resolution.
///
/// # Kernel Mapping
///
/// | Operator | Kernel |
/// |----------|--------|
/// | `+` | `maths.add` |
/// | `-` | `maths.sub` |
/// | `*` | `maths.mul` |
/// | `/` | `maths.div` |
/// | `%` | `maths.mod` |
/// | `**` | `maths.pow` |
/// | `==` | `compare.eq` |
/// | `!=` | `compare.ne` |
/// | `<` | `compare.lt` |
/// | `<=` | `compare.le` |
/// | `>` | `compare.gt` |
/// | `>=` | `compare.ge` |
/// | `&&` | `logic.and` |
/// | `\|\|` | `logic.or` |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum BinaryOp {
    // Arithmetic
    /// Addition: `a + b` → `maths.add(a, b)`
    Add,
    /// Subtraction: `a - b` → `maths.sub(a, b)`
    Sub,
    /// Multiplication: `a * b` → `maths.mul(a, b)`
    Mul,
    /// Division: `a / b` → `maths.div(a, b)`
    Div,
    /// Modulo: `a % b` → `maths.mod(a, b)`
    Mod,
    /// Power: `a ** b` → `maths.pow(a, b)`
    Pow,

    // Comparison
    /// Equal: `a == b` → `compare.eq(a, b)`
    Eq,
    /// Not equal: `a != b` → `compare.ne(a, b)`
    Ne,
    /// Less than: `a < b` → `compare.lt(a, b)`
    Lt,
    /// Less than or equal: `a <= b` → `compare.le(a, b)`
    Le,
    /// Greater than: `a > b` → `compare.gt(a, b)`
    Gt,
    /// Greater than or equal: `a >= b` → `compare.ge(a, b)`
    Ge,

    // Logical
    /// Logical AND: `a && b` → `logic.and(a, b)`
    And,
    /// Logical OR: `a || b` → `logic.or(a, b)`
    Or,
}

impl BinaryOp {
    /// Get the kernel that implements this operator
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::ast::{BinaryOp, KernelId};
    ///
    /// assert_eq!(BinaryOp::Add.kernel(), KernelId::new("maths", "add"));
    /// assert_eq!(BinaryOp::Lt.kernel(), KernelId::new("compare", "lt"));
    /// assert_eq!(BinaryOp::And.kernel(), KernelId::new("logic", "and"));
    /// ```
    pub fn kernel(self) -> KernelId {
        match self {
            Self::Add => KernelId::new("maths", "add"),
            Self::Sub => KernelId::new("maths", "sub"),
            Self::Mul => KernelId::new("maths", "mul"),
            Self::Div => KernelId::new("maths", "div"),
            Self::Mod => KernelId::new("maths", "mod"),
            Self::Pow => KernelId::new("maths", "pow"),
            Self::Eq => KernelId::new("compare", "eq"),
            Self::Ne => KernelId::new("compare", "ne"),
            Self::Lt => KernelId::new("compare", "lt"),
            Self::Le => KernelId::new("compare", "le"),
            Self::Gt => KernelId::new("compare", "gt"),
            Self::Ge => KernelId::new("compare", "ge"),
            Self::And => KernelId::new("logic", "and"),
            Self::Or => KernelId::new("logic", "or"),
        }
    }
}

/// Unary operators
///
/// All unary operators desugar to kernel calls during type resolution.
///
/// # Kernel Mapping
///
/// | Operator | Kernel |
/// |----------|--------|
/// | `-` | `maths.neg` |
/// | `!` | `logic.not` |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum UnaryOp {
    /// Negation: `-x` → `maths.neg(x)`
    Neg,
    /// Logical NOT: `!x` → `logic.not(x)`
    Not,
}

impl UnaryOp {
    /// Get the kernel that implements this operator
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::ast::{UnaryOp, KernelId};
    ///
    /// assert_eq!(UnaryOp::Neg.kernel(), KernelId::new("maths", "neg"));
    /// assert_eq!(UnaryOp::Not.kernel(), KernelId::new("logic", "not"));
    /// ```
    pub fn kernel(self) -> KernelId {
        match self {
            Self::Neg => KernelId::new("maths", "neg"),
            Self::Not => KernelId::new("logic", "not"),
        }
    }
}

/// Type expression from source (before type resolution)
///
/// Represents a type as written in source code. Type resolution converts
/// these to actual [`Type`](crate::foundation::Type) values.
///
/// # Examples
///
/// ```cdsl
/// Scalar<m/s>           // KernelType with unit
/// Vector<3, N>          // KernelType with dimension and unit
/// Matrix<3, 3, kg*m^2>  // KernelType with dimensions and unit
/// OrbitalElements       // UserType
/// Bool                  // Bool
/// ```
///
/// # Type Syntax
///
/// - **Scalar types:** `Scalar<unit>` where unit is optional (defaults to dimensionless)
/// - **Vector types:** `Vector<dim, unit>` where dim is a literal integer
/// - **Matrix types:** `Matrix<rows, cols, unit>`
/// - **User types:** Just the type name (e.g., `OrbitalElements`)
/// - **Bool:** The keyword `Bool`
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum TypeExpr {
    /// Scalar type: `Scalar<m/s>` or `Scalar<>` or just `Scalar`
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// Scalar<m>      // meters
    /// Scalar<m/s>    // meters per second
    /// Scalar<>       // dimensionless
    /// Scalar         // dimensionless (unit omitted)
    /// ```
    Scalar {
        /// Unit expression (None = dimensionless)
        unit: Option<UnitExpr>,
    },

    /// Vector type: `Vector<dim, unit>`
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// Vector<3, m>       // 3D position vector
    /// Vector<2, m/s>     // 2D velocity vector
    /// Vector<4, N>       // 4D force vector
    /// ```
    Vector {
        /// Dimension (number of components)
        dim: u8,
        /// Unit expression (None = dimensionless)
        unit: Option<UnitExpr>,
    },

    /// Matrix type: `Matrix<rows, cols, unit>`
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// Matrix<3, 3, kg*m^2>   // 3x3 inertia tensor
    /// Matrix<2, 2, <>        // 2x2 dimensionless matrix
    /// ```
    Matrix {
        /// Number of rows
        rows: u8,
        /// Number of columns
        cols: u8,
        /// Unit expression (None = dimensionless)
        unit: Option<UnitExpr>,
    },

    /// User-defined type reference
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// OrbitalElements
    /// Plate
    /// CollisionData
    /// ```
    User(Path),

    /// Boolean type
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// Bool
    /// ```
    Bool,
}

/// Unit expression from source (before unit resolution)
///
/// Represents a unit as written in source code. Unit resolution converts
/// these to actual [`Unit`](crate::foundation::Unit) values during type checking.
///
/// # Operand Ordering
///
/// - **Multiply(lhs, rhs)**: Represents `lhs * rhs` (e.g., `kg * m`)
/// - **Divide(numerator, denominator)**: Represents `numerator / denominator` (e.g., `m / s`)
/// - **Power(base, exponent)**: Represents `base ^ exponent` (e.g., `m ^ 2`)
///
/// Exponents are stored as `i8` to support both positive (`m^2`) and negative (`s^-1`) powers.
///
/// # Examples
///
/// ```cdsl
/// m           // Base("m")
/// m/s         // Divide(Base("m"), Base("s"))
/// kg*m/s^2    // Divide(Multiply(Base("kg"), Base("m")), Power(Base("s"), 2))
/// <>          // Dimensionless
/// K           // Base("K")
/// ```
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum UnitExpr {
    /// Base unit: `m`, `kg`, `s`, `K`, etc.
    ///
    /// String contains the base unit symbol (e.g., "m", "kg", "s").
    Base(String),

    /// Dimensionless: `<>`
    ///
    /// Represents explicitly dimensionless quantities.
    Dimensionless,

    /// Multiplication: `kg*m`
    ///
    /// Represents `lhs * rhs` where both operands are unit expressions.
    Multiply(Box<UnitExpr>, Box<UnitExpr>),

    /// Division: `m/s`
    ///
    /// Represents `numerator / denominator` where both operands are unit expressions.
    Divide(Box<UnitExpr>, Box<UnitExpr>),

    /// Power: `m^2`
    ///
    /// Represents `base ^ exponent` where base is a unit expression and
    /// exponent is an integer (positive or negative).
    Power(Box<UnitExpr>, i8),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_span() -> Span {
        Span::new(0, 0, 0, 0)
    }

    #[test]
    fn expr_literal() {
        let expr = Expr::literal(42.0, None, make_span());
        match expr.kind {
            ExprKind::Literal { value, unit } => {
                assert_eq!(value, 42.0);
                assert_eq!(unit, None);
            }
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn expr_local() {
        let expr = Expr::local("x", make_span());
        match expr.kind {
            ExprKind::Local(name) => assert_eq!(name, "x"),
            _ => panic!("expected local"),
        }
    }

    #[test]
    fn expr_binary() {
        let left = Expr::literal(1.0, None, make_span());
        let right = Expr::literal(2.0, None, make_span());
        let expr = Expr::binary(BinaryOp::Add, left, right, make_span());

        match expr.kind {
            ExprKind::Binary { op, .. } => assert_eq!(op, BinaryOp::Add),
            _ => panic!("expected binary"),
        }
    }

    #[test]
    fn expr_unary() {
        let operand = Expr::literal(42.0, None, make_span());
        let expr = Expr::unary(UnaryOp::Neg, operand, make_span());

        match expr.kind {
            ExprKind::Unary { op, .. } => assert_eq!(op, UnaryOp::Neg),
            _ => panic!("expected unary"),
        }
    }

    #[test]
    fn binary_op_kernels_complete() {
        // Test all 14 binary operators
        let cases = [
            (BinaryOp::Add, KernelId::new("maths", "add")),
            (BinaryOp::Sub, KernelId::new("maths", "sub")),
            (BinaryOp::Mul, KernelId::new("maths", "mul")),
            (BinaryOp::Div, KernelId::new("maths", "div")),
            (BinaryOp::Mod, KernelId::new("maths", "mod")),
            (BinaryOp::Pow, KernelId::new("maths", "pow")),
            (BinaryOp::Eq, KernelId::new("compare", "eq")),
            (BinaryOp::Ne, KernelId::new("compare", "ne")),
            (BinaryOp::Lt, KernelId::new("compare", "lt")),
            (BinaryOp::Le, KernelId::new("compare", "le")),
            (BinaryOp::Gt, KernelId::new("compare", "gt")),
            (BinaryOp::Ge, KernelId::new("compare", "ge")),
            (BinaryOp::And, KernelId::new("logic", "and")),
            (BinaryOp::Or, KernelId::new("logic", "or")),
        ];
        for (op, expected) in cases {
            assert_eq!(op.kernel(), expected, "kernel mismatch for {:?}", op);
        }
    }

    #[test]
    fn unary_op_kernels() {
        assert_eq!(UnaryOp::Neg.kernel(), KernelId::new("maths", "neg"));
        assert_eq!(UnaryOp::Not.kernel(), KernelId::new("logic", "not"));
    }

    #[test]
    fn type_expr_scalar() {
        let ty = TypeExpr::Scalar { unit: None };
        match ty {
            TypeExpr::Scalar { unit } => assert_eq!(unit, None),
            _ => panic!("expected scalar"),
        }
    }

    #[test]
    fn type_expr_vector() {
        let ty = TypeExpr::Vector { dim: 3, unit: None };
        match ty {
            TypeExpr::Vector { dim, unit } => {
                assert_eq!(dim, 3);
                assert_eq!(unit, None);
            }
            _ => panic!("expected vector"),
        }
    }

    #[test]
    fn type_expr_user() {
        let path = Path::from_path_str("OrbitalElements");
        let ty = TypeExpr::User(path.clone());
        match ty {
            TypeExpr::User(p) => assert_eq!(p, path),
            _ => panic!("expected user type"),
        }
    }

    #[test]
    fn unit_expr_base() {
        let unit = UnitExpr::Base("m".to_string());
        match unit {
            UnitExpr::Base(s) => assert_eq!(s, "m"),
            _ => panic!("expected base unit"),
        }
    }

    #[test]
    fn unit_expr_multiply() {
        let kg = Box::new(UnitExpr::Base("kg".to_string()));
        let m = Box::new(UnitExpr::Base("m".to_string()));
        let unit = UnitExpr::Multiply(kg, m);
        match unit {
            UnitExpr::Multiply(_, _) => {}
            _ => panic!("expected multiply"),
        }
    }

    #[test]
    fn unit_expr_divide() {
        let m = Box::new(UnitExpr::Base("m".to_string()));
        let s = Box::new(UnitExpr::Base("s".to_string()));
        let unit = UnitExpr::Divide(m, s);
        match unit {
            UnitExpr::Divide(_, _) => {}
            _ => panic!("expected divide"),
        }
    }

    #[test]
    fn unit_expr_dimensionless() {
        let unit = UnitExpr::Dimensionless;
        assert!(matches!(unit, UnitExpr::Dimensionless));
    }

    #[test]
    fn unit_expr_power() {
        let base = Box::new(UnitExpr::Base("m".to_string()));
        let unit = UnitExpr::Power(base, 2);
        match unit {
            UnitExpr::Power(_, exp) => assert_eq!(exp, 2),
            _ => panic!("expected power"),
        }
    }

    // === ExprKind Variant Coverage Tests ===

    #[test]
    fn expr_kind_vector() {
        let elem = Expr::literal(1.0, None, make_span());
        let expr = Expr::new(ExprKind::Vector(vec![elem.clone(), elem]), make_span());
        match expr.kind {
            ExprKind::Vector(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected vector"),
        }
    }

    #[test]
    fn expr_kind_signal() {
        let path = Path::from_path_str("velocity");
        let expr = Expr::new(ExprKind::Signal(path.clone()), make_span());
        match expr.kind {
            ExprKind::Signal(p) => assert_eq!(p, path),
            _ => panic!("expected signal"),
        }
    }

    #[test]
    fn expr_kind_field() {
        let path = Path::from_path_str("temperature");
        let expr = Expr::new(ExprKind::Field(path.clone()), make_span());
        match expr.kind {
            ExprKind::Field(p) => assert_eq!(p, path),
            _ => panic!("expected field"),
        }
    }

    #[test]
    fn expr_kind_config() {
        let path = Path::from_path_str("initial_temp");
        let expr = Expr::new(ExprKind::Config(path.clone()), make_span());
        match expr.kind {
            ExprKind::Config(p) => assert_eq!(p, path),
            _ => panic!("expected config"),
        }
    }

    #[test]
    fn expr_kind_const() {
        let path = Path::from_path_str("BOLTZMANN");
        let expr = Expr::new(ExprKind::Const(path.clone()), make_span());
        match expr.kind {
            ExprKind::Const(p) => assert_eq!(p, path),
            _ => panic!("expected const"),
        }
    }

    #[test]
    fn expr_kind_prev() {
        let expr = Expr::new(ExprKind::Prev, make_span());
        assert!(matches!(expr.kind, ExprKind::Prev));
    }

    #[test]
    fn expr_kind_current() {
        let expr = Expr::new(ExprKind::Current, make_span());
        assert!(matches!(expr.kind, ExprKind::Current));
    }

    #[test]
    fn expr_kind_inputs() {
        let expr = Expr::new(ExprKind::Inputs, make_span());
        assert!(matches!(expr.kind, ExprKind::Inputs));
    }

    #[test]
    fn expr_kind_dt() {
        let expr = Expr::new(ExprKind::Dt, make_span());
        assert!(matches!(expr.kind, ExprKind::Dt));
    }

    #[test]
    fn expr_kind_self() {
        let expr = Expr::new(ExprKind::Self_, make_span());
        assert!(matches!(expr.kind, ExprKind::Self_));
    }

    #[test]
    fn expr_kind_other() {
        let expr = Expr::new(ExprKind::Other, make_span());
        assert!(matches!(expr.kind, ExprKind::Other));
    }

    #[test]
    fn expr_kind_payload() {
        let expr = Expr::new(ExprKind::Payload, make_span());
        assert!(matches!(expr.kind, ExprKind::Payload));
    }

    #[test]
    fn expr_kind_if() {
        let cond = Expr::literal(1.0, None, make_span());
        let then_br = Expr::literal(2.0, None, make_span());
        let else_br = Expr::literal(3.0, None, make_span());
        let expr = Expr::new(
            ExprKind::If {
                condition: Box::new(cond),
                then_branch: Box::new(then_br),
                else_branch: Box::new(else_br),
            },
            make_span(),
        );
        match expr.kind {
            ExprKind::If { .. } => {}
            _ => panic!("expected if"),
        }
    }

    #[test]
    fn expr_kind_let() {
        let value = Expr::literal(10.0, None, make_span());
        let body = Expr::local("x", make_span());
        let expr = Expr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(value),
                body: Box::new(body),
            },
            make_span(),
        );
        match expr.kind {
            ExprKind::Let { name, .. } => assert_eq!(name, "x"),
            _ => panic!("expected let"),
        }
    }

    #[test]
    fn expr_kind_aggregate() {
        let body = Expr::local("p", make_span());
        let expr = Expr::new(
            ExprKind::Aggregate {
                op: AggregateOp::Sum,
                entity: EntityId::new("plate"),
                binding: "p".to_string(),
                body: Box::new(body),
            },
            make_span(),
        );
        match expr.kind {
            ExprKind::Aggregate { op, binding, .. } => {
                assert_eq!(op, AggregateOp::Sum);
                assert_eq!(binding, "p");
            }
            _ => panic!("expected aggregate"),
        }
    }

    #[test]
    fn expr_kind_fold() {
        let init = Expr::literal(0.0, None, make_span());
        let body = Expr::local("acc", make_span());
        let expr = Expr::new(
            ExprKind::Fold {
                entity: EntityId::new("plate"),
                init: Box::new(init),
                acc: "acc".to_string(),
                elem: "elem".to_string(),
                body: Box::new(body),
            },
            make_span(),
        );
        match expr.kind {
            ExprKind::Fold { acc, elem, .. } => {
                assert_eq!(acc, "acc");
                assert_eq!(elem, "elem");
            }
            _ => panic!("expected fold"),
        }
    }

    #[test]
    fn expr_kind_call() {
        let arg = Expr::literal(1.0, None, make_span());
        let expr = Expr::new(
            ExprKind::Call {
                func: Path::from_path_str("sin"),
                args: vec![arg],
            },
            make_span(),
        );
        match expr.kind {
            ExprKind::Call { func, args } => {
                assert_eq!(func, Path::from_path_str("sin"));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected call"),
        }
    }

    #[test]
    fn expr_kind_struct() {
        let field_val = Expr::literal(1.5e11, None, make_span());
        let expr = Expr::new(
            ExprKind::Struct {
                ty: Path::from_path_str("Orbit"),
                fields: vec![("semi_major".to_string(), field_val)],
            },
            make_span(),
        );
        match expr.kind {
            ExprKind::Struct { ty, fields } => {
                assert_eq!(ty, Path::from_path_str("Orbit"));
                assert_eq!(fields.len(), 1);
                assert_eq!(fields[0].0, "semi_major");
            }
            _ => panic!("expected struct"),
        }
    }

    #[test]
    fn expr_kind_field_access() {
        let object = Expr::local("orbit", make_span());
        let expr = Expr::new(
            ExprKind::FieldAccess {
                object: Box::new(object),
                field: "semi_major".to_string(),
            },
            make_span(),
        );
        match expr.kind {
            ExprKind::FieldAccess { field, .. } => {
                assert_eq!(field, "semi_major");
            }
            _ => panic!("expected field access"),
        }
    }

    #[test]
    fn expr_kind_parse_error() {
        let expr = Expr::new(
            ExprKind::ParseError("expected expression".to_string()),
            make_span(),
        );
        match expr.kind {
            ExprKind::ParseError(msg) => {
                assert_eq!(msg, "expected expression");
            }
            _ => panic!("expected parse error"),
        }
    }

    // === TypeExpr Missing Variants ===

    #[test]
    fn type_expr_matrix() {
        let ty = TypeExpr::Matrix {
            rows: 3,
            cols: 3,
            unit: None,
        };
        match ty {
            TypeExpr::Matrix { rows, cols, unit } => {
                assert_eq!(rows, 3);
                assert_eq!(cols, 3);
                assert_eq!(unit, None);
            }
            _ => panic!("expected matrix"),
        }
    }

    #[test]
    fn type_expr_bool() {
        let ty = TypeExpr::Bool;
        assert!(matches!(ty, TypeExpr::Bool));
    }

    // === Helper Method Operand Tests ===

    #[test]
    fn expr_binary_preserves_operands() {
        let span = make_span();
        let left = Expr::literal(1.0, None, span);
        let right = Expr::literal(2.0, None, span);
        let expr = Expr::binary(BinaryOp::Add, left.clone(), right.clone(), span);

        match expr.kind {
            ExprKind::Binary {
                op,
                left: l,
                right: r,
            } => {
                assert_eq!(op, BinaryOp::Add);
                assert_eq!(*l, left);
                assert_eq!(*r, right);
            }
            _ => panic!("expected binary"),
        }
    }

    #[test]
    fn expr_unary_preserves_operand() {
        let span = make_span();
        let operand = Expr::literal(42.0, None, span);
        let expr = Expr::unary(UnaryOp::Neg, operand.clone(), span);

        match expr.kind {
            ExprKind::Unary { op, operand: o } => {
                assert_eq!(op, UnaryOp::Neg);
                assert_eq!(*o, operand);
            }
            _ => panic!("expected unary"),
        }
    }
}
