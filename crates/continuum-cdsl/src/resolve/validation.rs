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
        ExprKind::Literal { value, .. } => {
            // Validate literal value against type bounds
            errors.extend(validate_literal_bounds(*value, &expr.ty, expr.span));
        }

        ExprKind::Call { kernel, args } => {
            // Validate kernel call
            errors.extend(validate_kernel_call(kernel, args, expr.span, ctx));
        }

        ExprKind::Vector(elements) => {
            // Validate all elements recursively
            for elem in elements {
                errors.extend(validate_expr(elem, ctx));

                // Seq types cannot appear in vector literals
                if elem.ty.is_seq() {
                    errors.push(CompileError::new(
                        ErrorKind::TypeMismatch,
                        elem.span,
                        "Seq types cannot be stored in vectors (must be immediately consumed by aggregate/fold)".to_string(),
                    ));
                }
            }

            // Validate all elements have same type
            if !elements.is_empty() {
                let first_ty = &elements[0].ty;
                for (i, elem) in elements.iter().enumerate().skip(1) {
                    if &elem.ty != first_ty {
                        errors.push(CompileError::new(
                            ErrorKind::TypeMismatch,
                            elem.span,
                            format!(
                                "vector element {} has type {:?}, expected {:?} (from element 0)",
                                i, elem.ty, first_ty
                            ),
                        ));
                    }
                }
            }
        }

        ExprKind::Let { value, body, .. } => {
            errors.extend(validate_expr(value, ctx));
            errors.extend(validate_expr(body, ctx));

            // Seq types cannot be stored in let bindings
            if value.ty.is_seq() {
                errors.push(CompileError::new(
                    ErrorKind::TypeMismatch,
                    value.span,
                    "Seq types cannot be stored in let bindings (must be immediately consumed by aggregate/fold)".to_string(),
                ));
            }
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
            for (name, field_expr) in fields {
                errors.extend(validate_expr(field_expr, ctx));

                // Seq types cannot appear in struct fields
                if field_expr.ty.is_seq() {
                    errors.push(CompileError::new(
                        ErrorKind::TypeMismatch,
                        field_expr.span,
                        format!(
                            "Seq types cannot be stored in struct field '{}' (must be immediately consumed by aggregate/fold)",
                            name
                        ),
                    ));
                }
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

/// Validates a literal value against type bounds.
///
/// Checks that the literal value satisfies any min/max constraints
/// specified in the type's bounds.
///
/// # Parameters
///
/// - `value`: The literal numeric value.
/// - `ty`: The type (must be kernel type with bounds).
/// - `span`: Source location for error reporting.
///
/// # Returns
///
/// Vector of validation errors (empty if value is within bounds).
fn validate_literal_bounds(value: f64, ty: &Type, span: Span) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Only kernel types have bounds
    if let Type::Kernel(kernel_ty) = ty {
        if let Some(bounds) = &kernel_ty.bounds {
            // Check minimum bound
            if let Some(min) = bounds.min {
                if value < min {
                    errors.push(CompileError::new(
                        ErrorKind::TypeMismatch,
                        span,
                        format!("literal value {} is below minimum bound {}", value, min),
                    ));
                }
            }

            // Check maximum bound
            if let Some(max) = bounds.max {
                if value > max {
                    errors.push(CompileError::new(
                        ErrorKind::TypeMismatch,
                        span,
                        format!("literal value {} exceeds maximum bound {}", value, max),
                    ));
                }
            }
        }
    }

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

    #[test]
    fn test_validate_vector_elements_type_mismatch() {
        let ctx = ValidationContext::new();

        // First element is Scalar
        let elem1 = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        // Second element is Bool (type mismatch!)
        let elem2 = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::Bool,
            test_span(),
        );

        let vec_expr = TypedExpr::new(
            ExprKind::Vector(vec![elem1, elem2]),
            Type::kernel(Shape::Vector { dim: 2 }, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&vec_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(errors[0].message.contains("element 1"));
    }

    #[test]
    fn test_validate_literal_below_minimum() {
        use crate::foundation::Bounds;

        let ctx = ValidationContext::new();

        // Type has min bound of 0.0
        let bounded_type = Type::kernel(
            Shape::Scalar,
            Unit::DIMENSIONLESS,
            Some(Bounds {
                min: Some(0.0),
                max: None,
            }),
        );

        // Literal value is -5.0 (below minimum!)
        let expr = TypedExpr::new(
            ExprKind::Literal {
                value: -5.0,
                unit: None,
            },
            bounded_type,
            test_span(),
        );

        let errors = validate_expr(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(errors[0].message.contains("below minimum"));
    }

    #[test]
    fn test_validate_literal_above_maximum() {
        use crate::foundation::Bounds;

        let ctx = ValidationContext::new();

        // Type has max bound of 100.0
        let bounded_type = Type::kernel(
            Shape::Scalar,
            Unit::DIMENSIONLESS,
            Some(Bounds {
                min: None,
                max: Some(100.0),
            }),
        );

        // Literal value is 200.0 (above maximum!)
        let expr = TypedExpr::new(
            ExprKind::Literal {
                value: 200.0,
                unit: None,
            },
            bounded_type,
            test_span(),
        );

        let errors = validate_expr(&expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(errors[0].message.contains("exceeds maximum"));
    }

    #[test]
    fn test_validate_literal_within_bounds() {
        use crate::foundation::Bounds;

        let ctx = ValidationContext::new();

        // Type has bounds [0.0, 100.0]
        let bounded_type = Type::kernel(
            Shape::Scalar,
            Unit::DIMENSIONLESS,
            Some(Bounds {
                min: Some(0.0),
                max: Some(100.0),
            }),
        );

        // Literal value is 50.0 (within bounds)
        let expr = TypedExpr::new(
            ExprKind::Literal {
                value: 50.0,
                unit: None,
            },
            bounded_type,
            test_span(),
        );

        let errors = validate_expr(&expr, &ctx);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_vector_unit_mismatch() {
        let ctx = ValidationContext::new();

        // First element has unit m
        let elem1 = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::meters()),
            },
            Type::kernel(Shape::Scalar, Unit::meters(), None),
            test_span(),
        );

        // Second element has unit s (unit mismatch!)
        let elem2 = TypedExpr::new(
            ExprKind::Literal {
                value: 2.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            test_span(),
        );

        let vec_expr = TypedExpr::new(
            ExprKind::Vector(vec![elem1, elem2]),
            Type::kernel(Shape::Vector { dim: 2 }, Unit::meters(), None),
            test_span(),
        );

        let errors = validate_expr(&vec_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(errors[0].message.contains("element 1"));
    }

    #[test]
    fn test_validate_empty_vector() {
        let ctx = ValidationContext::new();

        // Empty vector is valid (no type conflicts possible)
        let vec_expr = TypedExpr::new(
            ExprKind::Vector(vec![]),
            Type::kernel(Shape::Vector { dim: 0 }, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&vec_expr, &ctx);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_seq_in_let_binding_fails() {
        let ctx = ValidationContext::new();

        // Trying to store Seq<Scalar> in let binding (invalid!)
        let seq_value = TypedExpr::new(
            ExprKind::Local("some_seq".to_string()),
            Type::Seq(Box::new(Type::kernel(
                Shape::Scalar,
                Unit::DIMENSIONLESS,
                None,
            ))),
            test_span(),
        );

        let body = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let let_expr = TypedExpr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(seq_value),
                body: Box::new(body),
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&let_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(errors[0].message.contains("Seq types cannot be stored"));
    }

    #[test]
    fn test_validate_seq_in_vector_fails() {
        let ctx = ValidationContext::new();

        // Trying to create vector containing Seq (invalid!)
        let seq_elem = TypedExpr::new(
            ExprKind::Local("some_seq".to_string()),
            Type::Seq(Box::new(Type::kernel(
                Shape::Scalar,
                Unit::DIMENSIONLESS,
                None,
            ))),
            test_span(),
        );

        let vec_expr = TypedExpr::new(
            ExprKind::Vector(vec![seq_elem]),
            Type::kernel(Shape::Vector { dim: 1 }, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&vec_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(
            errors[0]
                .message
                .contains("Seq types cannot be stored in vectors")
        );
    }

    #[test]
    fn test_validate_seq_in_struct_field_fails() {
        let ctx = ValidationContext::new();

        // Trying to store Seq in struct field (invalid!)
        let seq_value = TypedExpr::new(
            ExprKind::Local("some_seq".to_string()),
            Type::Seq(Box::new(Type::kernel(
                Shape::Scalar,
                Unit::DIMENSIONLESS,
                None,
            ))),
            test_span(),
        );

        let struct_expr = TypedExpr::new(
            ExprKind::Struct {
                ty: continuum_foundation::TypeId::from("SomeType"),
                fields: vec![("bad_field".to_string(), seq_value)],
            },
            Type::user(continuum_foundation::TypeId::from("SomeType")),
            test_span(),
        );

        let errors = validate_expr(&struct_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(
            errors[0]
                .message
                .contains("Seq types cannot be stored in struct field")
        );
    }
}
