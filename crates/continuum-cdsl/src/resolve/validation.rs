//! Type validation for typed CDSL expressions.
//!
//! Validates typed expressions for semantic correctness after type resolution.
//! This pass checks type compatibility, kernel signatures, bounds, and unit consistency.
//!
//! # What This Pass Does
//!
//! 1. **Type compatibility** - Verifies operand types match expected types
//! 2. **Kernel validation** - Checks kernel calls against signatures
//! 3. **Bounds checking** - Validates values satisfy min/max constraints
//! 4. **Unit validation** - Ensures unit consistency in operations
//! 5. **Struct validation** - Verifies struct field types match declarations
//!
//! # What This Pass Does NOT Do
//!
//! - **No type inference** - Types must already be assigned (by type resolution)
//! - **No name resolution** - Names must already be resolved
//! - **No code generation** - Validation only, no IR output
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Res → Type Res → Validation → Compilation
//!                                             ^^^^^^
//!                                          YOU ARE HERE
//! ```
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::resolve::validation::{validate_expr, ValidationContext};
//! use continuum_cdsl::ast::TypedExpr;
//!
//! let ctx = ValidationContext::new(type_table, kernel_registry);
//! let errors = validate_expr(&typed_expr, &ctx);
//! if errors.is_empty() {
//!     println!("Expression is valid!");
//! }
//! ```

use crate::ast::{ExprKind, KernelId, TypedExpr};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{Span, Type};

/// Validation context containing type tables and registries.
///
/// Provides access to type information needed during validation:
/// - User type definitions
/// - Kernel signatures
/// - Symbol tables
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::resolve::validation::ValidationContext;
///
/// let ctx = ValidationContext::new();
/// // Use ctx for validation...
/// ```
#[derive(Debug)]
pub struct ValidationContext {
    // Future: TypeTable, KernelRegistry, SymbolTable
}

impl ValidationContext {
    /// Create a new empty validation context.
    ///
    /// # Returns
    ///
    /// A new validation context ready for use.
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ValidationContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Validates a typed expression for semantic correctness.
///
/// Checks type compatibility, kernel signatures, bounds, and unit consistency.
/// Returns a list of validation errors (empty if expression is valid).
///
/// # Parameters
///
/// - `expr`: Typed expression to validate.
/// - `ctx`: Validation context with type tables and registries.
///
/// # Returns
///
/// Vector of validation errors. Empty if expression passes all checks.
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::resolve::validation::{validate_expr, ValidationContext};
///
/// let ctx = ValidationContext::new();
/// let errors = validate_expr(&typed_expr, &ctx);
/// for error in errors {
///     eprintln!("{}", error);
/// }
/// ```
pub fn validate_expr(expr: &TypedExpr, ctx: &ValidationContext) -> Vec<CompileError> {
    let mut errors = Vec::new();

    match &expr.expr {
        ExprKind::Literal { .. } => {
            // Literals are always valid
        }

        ExprKind::Call { kernel, args } => {
            // Validate kernel call
            errors.extend(validate_kernel_call(kernel, args, expr.span, ctx));
        }

        ExprKind::Vector(elements) => {
            // Validate all elements recursively
            for elem in elements {
                errors.extend(validate_expr(elem, ctx));
            }
            // TODO: Validate all elements have same type
        }

        ExprKind::Let { value, body, .. } => {
            errors.extend(validate_expr(value, ctx));
            errors.extend(validate_expr(body, ctx));
        }

        ExprKind::Aggregate { body, .. } => {
            errors.extend(validate_expr(body, ctx));
        }

        ExprKind::Fold { init, body, .. } => {
            errors.extend(validate_expr(init, ctx));
            errors.extend(validate_expr(body, ctx));
        }

        ExprKind::Struct { fields, .. } => {
            // Validate all field expressions
            for (_name, field_expr) in fields {
                errors.extend(validate_expr(field_expr, ctx));
            }
            // TODO: Validate field types match struct definition
        }

        ExprKind::FieldAccess { object, .. } => {
            errors.extend(validate_expr(object, ctx));
            // TODO: Validate field exists on object type
        }

        // References don't need validation
        ExprKind::Local(_)
        | ExprKind::Signal(_)
        | ExprKind::Field(_)
        | ExprKind::Config(_)
        | ExprKind::Const(_)
        | ExprKind::Prev
        | ExprKind::Current
        | ExprKind::Inputs
        | ExprKind::Dt
        | ExprKind::Self_
        | ExprKind::Other
        | ExprKind::Payload => {}
    }

    errors
}

/// Validates a kernel call expression.
///
/// Checks that:
/// - Kernel exists
/// - Argument count is correct
/// - Argument types satisfy kernel constraints
/// - Argument units satisfy kernel constraints
///
/// # Parameters
///
/// - `kernel`: Kernel identifier (namespace + name).
/// - `args`: Arguments to the kernel call.
/// - `span`: Source location for error reporting.
/// - `ctx`: Validation context with kernel registry.
///
/// # Returns
///
/// Vector of validation errors (empty if call is valid).
fn validate_kernel_call(
    _kernel: &KernelId,
    args: &[TypedExpr],
    _span: Span,
    ctx: &ValidationContext,
) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Recursively validate all arguments
    for arg in args {
        errors.extend(validate_expr(arg, ctx));
    }

    // TODO: Validate kernel exists
    // TODO: Validate argument count
    // TODO: Validate argument types
    // TODO: Validate argument shapes
    // TODO: Validate argument units

    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{Shape, Unit};

    fn test_span() -> Span {
        Span::new(0, 10, 20, 1)
    }

    #[test]
    fn test_validate_literal() {
        let ctx = ValidationContext::new();
        let expr = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&expr, &ctx);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_vector_elements() {
        let ctx = ValidationContext::new();

        let elem1 = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let elem2 = TypedExpr::new(
            ExprKind::Literal {
                value: 2.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let vec_expr = TypedExpr::new(
            ExprKind::Vector(vec![elem1, elem2]),
            Type::kernel(Shape::Vector { dim: 2 }, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&vec_expr, &ctx);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_let_binding() {
        let ctx = ValidationContext::new();

        let value = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let body = TypedExpr::new(
            ExprKind::Local("x".to_string()),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let let_expr = TypedExpr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(value),
                body: Box::new(body),
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&let_expr, &ctx);
        assert!(errors.is_empty());
    }
}
