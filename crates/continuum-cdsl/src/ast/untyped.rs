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

use crate::foundation::{Path, Span, Unit};

use super::expr::{AggregateOp, KernelId};
use super::node::EntityId;

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
/// use continuum_cdsl::ast::Expr;
///
/// // Literal with unit (synthesis mode)
/// let with_unit = Expr::literal(100.0, Some(Unit::meters()), span);
///
/// // Literal without unit (checking mode - infers from context)
/// let without_unit = Expr::literal(100.0, None, span);
///
/// // Binary operator (desugars to kernel call)
/// let add = Expr::binary(BinaryOp::Add, a, b, span);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    /// Expression kind (what kind of expression this is)
    pub kind: ExprKind,

    /// Source location for error messages
    pub span: Span,
}

impl Expr {
    /// Create a new untyped expression
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
    /// # Examples
    ///
    /// ```rust,ignore
    /// // With unit
    /// let distance = Expr::literal(100.0, Some(Unit::meters()), span);
    ///
    /// // Without unit (infer from context)
    /// let number = Expr::literal(42.0, None, span);
    /// ```
    pub fn literal(value: f64, unit: Option<Unit>, span: Span) -> Self {
        Self::new(ExprKind::Literal { value, unit }, span)
    }

    /// Create a local variable reference
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
    /// Binary operators desugar to kernel calls during type resolution:
    /// - `a + b` → `maths.add(a, b)`
    /// - `a * b` → `maths.mul(a, b)`
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
    /// Unary operators desugar to kernel calls during type resolution:
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
/// operators are desugared to kernel calls:
///
/// | Syntax | Desugars to |
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
#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    // === Literals ===
    /// Numeric literal with optional unit
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// 100.0<m>     // value: 100.0, unit: Some(Unit::meters())
    /// 3.14         // value: 3.14, unit: None (infer from context)
    /// 0.0<>        // value: 0.0, unit: Some(Unit::DIMENSIONLESS)
    /// ```
    Literal {
        /// Numeric value
        value: f64,
        /// Optional unit annotation (None = infer from context)
        unit: Option<Unit>,
    },

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
#[derive(Debug, Clone, PartialEq)]
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
/// these to actual [`Unit`] values.
///
/// # Examples
///
/// ```cdsl
/// m           // meters
/// m/s         // meters per second
/// kg*m/s^2    // Newtons
/// <>          // dimensionless
/// K           // Kelvin
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum UnitExpr {
    /// Base unit: `m`, `kg`, `s`, `K`, etc.
    Base(String),

    /// Dimensionless: `<>`
    Dimensionless,

    /// Multiplication: `kg*m`
    Multiply(Box<UnitExpr>, Box<UnitExpr>),

    /// Division: `m/s`
    Divide(Box<UnitExpr>, Box<UnitExpr>),

    /// Power: `m^2`
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
    fn binary_op_kernels() {
        assert_eq!(BinaryOp::Add.kernel(), KernelId::new("maths", "add"));
        assert_eq!(BinaryOp::Sub.kernel(), KernelId::new("maths", "sub"));
        assert_eq!(BinaryOp::Mul.kernel(), KernelId::new("maths", "mul"));
        assert_eq!(BinaryOp::Div.kernel(), KernelId::new("maths", "div"));
        assert_eq!(BinaryOp::Lt.kernel(), KernelId::new("compare", "lt"));
        assert_eq!(BinaryOp::Eq.kernel(), KernelId::new("compare", "eq"));
        assert_eq!(BinaryOp::And.kernel(), KernelId::new("logic", "and"));
        assert_eq!(BinaryOp::Or.kernel(), KernelId::new("logic", "or"));
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
        let path = Path::from_str("OrbitalElements");
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
}
