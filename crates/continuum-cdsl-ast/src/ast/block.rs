//! Block bodies and statements
//!
//! This module defines block body types used in execution blocks.
//! These types are shared between declaration.rs and node.rs.

use crate::ast::untyped::Expr;
use crate::ast::TypedExpr;
use crate::foundation::{Path, Span};

/// Trait for types that have a source span.
pub trait HasSpan {
    /// Returns the source span of this item.
    fn span(&self) -> Span;
}

impl HasSpan for Expr {
    fn span(&self) -> Span {
        self.span
    }
}

impl HasSpan for TypedExpr {
    fn span(&self) -> Span {
        self.span
    }
}

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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

    /// Expression statement: evaluate expression for side effects only.
    ///
    /// Represents an expression executed solely for its side effects, with the
    /// result value discarded. Typically used for function calls, kernel invocations,
    /// or other effectful operations where the return value is not needed.
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// debug("checkpoint reached")    // Function call for side effect
    /// physics::apply_force(entity)   // Kernel with side effect
    /// ```
    ///
    /// # Usage
    ///
    /// Valid in statement blocks (Collect, Fracture phases). Unlike other
    /// statement types that bind names or emit to signals/fields, this simply
    /// executes the expression and discards the result.
    ///
    /// The parser automatically wraps bare expressions in statement position
    /// with this variant. If the expression returns a non-unit type, the compiler
    /// may warn about unused values (depending on linting policy).
    Expr(E),

    /// Assertion with optional severity and message
    ///
    /// Represents an assertion condition with optional metadata for severity
    /// level and descriptive message. Assertions validate invariants and emit
    /// structured faults when conditions fail.
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// assert { x > 0 }                          // Basic assertion
    /// assert { x > 0 : fatal }                  // With severity
    /// assert { x > 0 : "must be positive" }     // With message
    /// assert { x > 0 : fatal, "must be positive" }  // Both
    /// ```
    ///
    /// # Syntax
    ///
    /// ```text
    /// assert { condition }
    /// assert { condition : severity }
    /// assert { condition : message }
    /// assert { condition : severity, message }
    /// ```
    ///
    /// Valid severity levels: `fatal`, `error`, `warn`
    Assert {
        /// Condition expression that must evaluate to Bool
        condition: E,
        /// Optional severity level ("fatal", "error", "warn")
        severity: Option<String>,
        /// Optional descriptive message
        message: Option<String>,
        /// Source location
        span: Span,
    },
}

impl<E: HasSpan> Stmt<E> {
    /// Returns the source span of this statement.
    pub fn span(&self) -> Span {
        match self {
            Stmt::Let { span, .. } => *span,
            Stmt::SignalAssign { span, .. } => *span,
            Stmt::FieldAssign { span, .. } => *span,
            Stmt::Expr(expr) => expr.span(),
            Stmt::Assert { span, .. } => *span,
        }
    }
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

impl BlockBody {
    /// Returns the source span of the block body content.
    ///
    /// For expressions, returns the expression's span.
    /// For statement lists, returns the span of the first statement.
    ///
    /// # Panics
    ///
    /// Panics if called on an empty statement list. Empty statement lists
    /// should be rejected during parsing or validation before this is called.
    pub fn span(&self) -> Span {
        match self {
            BlockBody::Expression(expr) => expr.span,
            BlockBody::TypedExpression(expr) => expr.span,
            BlockBody::Statements(stmts) => stmts
                .first()
                .expect("BlockBody::Statements must not be empty")
                .span(),
            BlockBody::TypedStatements(stmts) => stmts
                .first()
                .expect("BlockBody::TypedStatements must not be empty")
                .span(),
        }
    }
}

/// Visitor trait for traversing statement trees and their embedded expressions.
///
/// This trait provides a standardized way to walk statement nodes and visit
/// their child expressions. It eliminates duplicated statement traversal logic
/// across multiple compiler passes.
///
/// # Design
///
/// The visitor pattern is split into two responsibilities:
///
/// - [`visit_stmt`] — called once per statement node (override to collect statement-level data)
/// - [`visit_expr`] — called for each expression embedded in statements (override to process expressions)
///
/// The default implementation of [`visit_stmt`] delegates to [`walk_stmt`],
/// which handles the structural traversal by matching on statement variants
/// and calling [`visit_expr`] for embedded expressions.
///
/// # Usage
///
/// Implement this trait to define custom statement traversal logic:
///
/// ```rust,ignore
/// use continuum_cdsl_ast::{TypedStmt, TypedExpr, StatementVisitor};
///
/// struct MyVisitor {
///     // ... visitor state
/// }
///
/// impl StatementVisitor for MyVisitor {
///     fn visit_stmt(&mut self, stmt: &TypedStmt) {
///         // Custom statement-level logic here
///         self.walk_stmt(stmt); // Continue traversal
///     }
///
///     fn visit_expr(&mut self, expr: &TypedExpr) {
///         // Custom expression logic here
///     }
/// }
/// ```
///
/// # Example
///
/// Collecting all signal assignments from a statement list:
///
/// ```rust,ignore
/// struct SignalCollector {
///     signals: Vec<Path>,
/// }
///
/// impl StatementVisitor for SignalCollector {
///     fn visit_stmt(&mut self, stmt: &TypedStmt) {
///         if let TypedStmt::SignalAssign { target, .. } = stmt {
///             self.signals.push(target.clone());
///         }
///         self.walk_stmt(stmt); // Continue to expressions
///     }
///
///     fn visit_expr(&mut self, _expr: &TypedExpr) {
///         // Process embedded expressions if needed
///     }
/// }
/// ```
///
/// # Eliminated Duplication
///
/// This trait consolidates statement traversal logic previously duplicated across:
/// - `resolve/uses.rs` (lines 166-184)
/// - `resolve/integrators.rs` (lines 195-212)
/// - `resolve/dependencies.rs` (lines 111-158)
/// - `desugar.rs` (lines 266-306)
///
/// Before this trait, adding [`Stmt::Assert`] required updating match arms in 5 files.
/// Now, extending [`Stmt`] only requires updating [`walk_stmt`] once.
pub trait StatementVisitor {
    /// Visit a statement node.
    ///
    /// This method is called once for each statement in the tree.
    /// The default implementation delegates to [`walk_stmt`] to traverse
    /// the statement's structure and visit embedded expressions.
    ///
    /// Override this method to:
    /// - Collect statement-level data (e.g., target paths in assignments)
    /// - Perform statement-specific validation
    /// - Transform or analyze statement patterns
    ///
    /// # Default Behavior
    ///
    /// The default impl calls [`walk_stmt`], which handles the structural
    /// traversal by matching on statement variants and calling [`visit_expr`]
    /// for embedded expressions.
    #[inline]
    fn visit_stmt(&mut self, stmt: &TypedStmt) {
        self.walk_stmt(stmt);
    }

    /// Walk the structure of a statement, visiting embedded expressions.
    ///
    /// This method performs the structural traversal of a statement node:
    /// - Matches on [`Stmt`] variants
    /// - Calls [`visit_expr`] for each embedded expression
    ///
    /// Call this from your [`visit_stmt`] implementation to continue
    /// the traversal after performing statement-level processing.
    ///
    /// # Statement Traversal Rules
    ///
    /// - [`Stmt::Let`] — visits the `value` expression
    /// - [`Stmt::SignalAssign`] — visits the `value` expression  
    /// - [`Stmt::FieldAssign`] — visits `position` then `value` expressions
    /// - [`Stmt::Assert`] — visits the `condition` expression
    /// - [`Stmt::Expr`] — visits the wrapped expression
    #[inline]
    fn walk_stmt(&mut self, stmt: &TypedStmt) {
        match stmt {
            Stmt::Let { value, .. } => self.visit_expr(value),
            Stmt::SignalAssign { value, .. } => self.visit_expr(value),
            Stmt::FieldAssign {
                position, value, ..
            } => {
                self.visit_expr(position);
                self.visit_expr(value);
            }
            Stmt::Assert { condition, .. } => self.visit_expr(condition),
            Stmt::Expr(expr) => self.visit_expr(expr),
        }
    }

    /// Visit an expression embedded in a statement.
    ///
    /// This method is called for every expression encountered while
    /// traversing statement structures. Override this to process
    /// expressions within statements.
    ///
    /// # Note
    ///
    /// This method receives expressions but does not automatically
    /// recurse into their subexpressions. If you need to traverse
    /// the full expression tree, use [`TypedExpr::walk`] with an
    /// [`ExpressionVisitor`][crate::ast::ExpressionVisitor].
    fn visit_expr(&mut self, expr: &TypedExpr);
}
