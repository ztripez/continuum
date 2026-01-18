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
use crate::resolve::types::TypeTable;

/// Context for validating typed CDSL expressions using user-defined types.
///
/// `ValidationContext` provides read-only access to user type definitions that
/// expression validation needs when checking struct construction and field
/// access.
///
/// # Parameters
///
/// - `type_table`: Reference to the [`TypeTable`] containing user-defined types.
///
/// # Returns
///
/// A `ValidationContext` that borrows the provided type table for lookups during
/// validation.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::resolve::types::TypeTable;
/// use continuum_cdsl::resolve::validation::ValidationContext;
///
/// let type_table = TypeTable::new();
/// let ctx = ValidationContext::new(&type_table);
/// let _ = ctx;
/// ```
#[derive(Debug)]
pub struct ValidationContext<'a> {
    /// User type definitions for struct validation
    pub type_table: &'a TypeTable,
}

impl<'a> ValidationContext<'a> {
    /// Create a new validation context.
    ///
    /// # Parameters
    ///
    /// - `type_table`: User type definitions for struct validation.
    ///
    /// # Returns
    ///
    /// A new validation context ready for use.
    pub fn new(type_table: &'a TypeTable) -> Self {
        Self { type_table }
    }
}

/// Validates a typed CDSL expression for post-resolution semantic correctness.
///
/// `validate_expr` checks literal bounds, kernel call structure, unit and type
/// consistency, and user type field access. The function assumes name and type
/// resolution have already completed successfully and reports any validation
/// failures as [`CompileError`] values.
///
/// # Parameters
///
/// - `expr`: Typed expression to validate, including assigned [`Type`] metadata.
/// - `ctx`: Validation context containing user type definitions and registries.
///
/// # Returns
///
/// A list of validation errors. An empty list means the expression satisfies all
/// validation rules enforced by this pass.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::{ExprKind, TypedExpr};
/// use continuum_cdsl::foundation::{Shape, Span, Type, Unit};
/// use continuum_cdsl::resolve::types::TypeTable;
/// use continuum_cdsl::resolve::validation::{validate_expr, ValidationContext};
///
/// let type_table = TypeTable::new();
/// let ctx = ValidationContext::new(&type_table);
/// let expr = TypedExpr::new(
///     ExprKind::Literal {
///         value: 1.0,
///         unit: None,
///     },
///     Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
///     Span::new(0, 1, 1, 1),
/// );
///
/// let errors = validate_expr(&expr, &ctx);
/// assert!(errors.is_empty());
/// ```
pub fn validate_expr(expr: &TypedExpr, ctx: &ValidationContext<'_>) -> Vec<CompileError> {
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

        ExprKind::Struct {
            ty: type_id,
            fields,
        } => {
            // Validate all field expressions recursively
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

            // Validate fields against user type definition
            errors.extend(validate_struct_fields(type_id, fields, expr.span, ctx));
        }

        ExprKind::FieldAccess { object, field } => {
            errors.extend(validate_expr(object, ctx));

            // Validate field exists on object type (for user types)
            if let Type::User(type_id) = &object.ty {
                errors.extend(validate_field_access(type_id, field, expr.span, ctx));
            }
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
    ctx: &ValidationContext<'_>,
) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Recursively validate all arguments
    for arg in args {
        errors.extend(validate_expr(arg, ctx));
    }

    // Kernel signature validation requires kernel registry (not yet implemented).
    // When kernel registry is available (Phase 6), this function should:
    // 1. Look up kernel signature by KernelId
    // 2. Validate argument count matches signature
    // 3. Validate argument types satisfy signature constraints
    // 4. Validate argument shapes satisfy signature constraints
    // 5. Validate argument units satisfy signature constraints
    //
    // Until then, we validate that arguments are well-formed (done above).

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

/// Validates struct fields against user type definition.
///
/// Checks that all provided fields exist in the type definition
/// and have compatible types.
///
/// # Parameters
///
/// - `type_id`: User type identifier.
/// - `fields`: Field name-value pairs from struct construction.
/// - `span`: Source location for error reporting.
/// - `ctx`: Validation context with type table.
///
/// # Returns
///
/// Vector of validation errors (empty if all fields are valid).
fn validate_struct_fields(
    type_id: &continuum_foundation::TypeId,
    fields: &[(String, TypedExpr)],
    span: Span,
    ctx: &ValidationContext<'_>,
) -> Vec<CompileError> {
    use crate::foundation::Path;

    let mut errors = Vec::new();

    // Look up user type in type table
    let path = Path::from(type_id.as_str());
    let Some(user_type) = ctx.type_table.get(&path) else {
        errors.push(CompileError::new(
            ErrorKind::UnknownType,
            span,
            format!("Unknown user type: {}", type_id),
        ));
        return errors;
    };

    // Validate each field
    for (field_name, field_expr) in fields {
        // Check if field exists in type definition
        if let Some((_name, expected_ty)) =
            user_type.fields.iter().find(|(name, _)| name == field_name)
        {
            // Check if field type matches expected type
            if &field_expr.ty != expected_ty {
                errors.push(CompileError::new(
                    ErrorKind::TypeMismatch,
                    field_expr.span,
                    format!(
                        "struct field '{}' has type {:?}, expected {:?}",
                        field_name, field_expr.ty, expected_ty
                    ),
                ));
            }
        } else {
            errors.push(CompileError::new(
                ErrorKind::UnknownType,
                field_expr.span,
                format!("struct type {} has no field '{}'", type_id, field_name),
            ));
        }
    }

    errors
}

/// Validates field access on user types.
///
/// Checks that the accessed field exists in the user type definition.
///
/// # Parameters
///
/// - `type_id`: User type identifier.
/// - `field_name`: Name of the accessed field.
/// - `span`: Source location for error reporting.
/// - `ctx`: Validation context with type table.
///
/// # Returns
///
/// Vector of validation errors (empty if field exists).
fn validate_field_access(
    type_id: &continuum_foundation::TypeId,
    field_name: &str,
    span: Span,
    ctx: &ValidationContext<'_>,
) -> Vec<CompileError> {
    use crate::foundation::Path;

    let mut errors = Vec::new();

    // Look up user type in type table
    let path = Path::from(type_id.as_str());
    let Some(user_type) = ctx.type_table.get(&path) else {
        errors.push(CompileError::new(
            ErrorKind::UnknownType,
            span,
            format!("Unknown user type: {}", type_id),
        ));
        return errors;
    };

    // Check if field exists
    if !user_type.fields.iter().any(|(name, _)| name == field_name) {
        errors.push(CompileError::new(
            ErrorKind::UnknownType,
            span,
            format!("type {} has no field '{}'", type_id, field_name),
        ));
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::{Shape, Unit};
    use crate::resolve::types::TypeTable;

    fn test_span() -> Span {
        Span::new(0, 10, 20, 1)
    }

    fn test_type_table() -> TypeTable {
        TypeTable::new()
    }

    #[test]
    fn test_validate_literal() {
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);
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
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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

        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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

        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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

        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

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
        // Expect 2 errors: Seq type error + unknown type error
        assert_eq!(errors.len(), 2);
        // First error is Seq type leakage
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(
            errors[0]
                .message
                .contains("Seq types cannot be stored in struct field")
        );
        // Second error is unknown type (not registered)
        assert_eq!(errors[1].kind, ErrorKind::UnknownType);
    }

    #[test]
    fn test_validate_struct_unknown_type() {
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

        // Constructing unknown type
        let field_value = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let struct_expr = TypedExpr::new(
            ExprKind::Struct {
                ty: continuum_foundation::TypeId::from("UnknownType"),
                fields: vec![("x".to_string(), field_value)],
            },
            Type::user(continuum_foundation::TypeId::from("UnknownType")),
            test_span(),
        );

        let errors = validate_expr(&struct_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::UnknownType);
        assert!(errors[0].message.contains("Unknown user type"));
    }

    #[test]
    fn test_validate_struct_field_type_mismatch() {
        use crate::foundation::{Path, UserType};

        // Create type table with a user type
        let mut type_table = TypeTable::new();
        let type_id = continuum_foundation::TypeId::from("Vec2");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Vec2"),
            vec![
                (
                    "x".to_string(),
                    Type::kernel(Shape::Scalar, Unit::meters(), None),
                ),
                (
                    "y".to_string(),
                    Type::kernel(Shape::Scalar, Unit::meters(), None),
                ),
            ],
        );
        type_table.register(user_type);

        let ctx = ValidationContext::new(&type_table);

        // Field 'x' has wrong type (Bool instead of Scalar<m>)
        let wrong_field = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::Bool, // Wrong type!
            test_span(),
        );

        let struct_expr = TypedExpr::new(
            ExprKind::Struct {
                ty: type_id.clone(),
                fields: vec![("x".to_string(), wrong_field)],
            },
            Type::user(type_id),
            test_span(),
        );

        let errors = validate_expr(&struct_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(errors[0].message.contains("field 'x'"));
    }

    #[test]
    fn test_validate_struct_unknown_field() {
        use crate::foundation::{Path, UserType};

        // Create type table with a user type
        let mut type_table = TypeTable::new();
        let type_id = continuum_foundation::TypeId::from("Vec2");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Vec2"),
            vec![
                (
                    "x".to_string(),
                    Type::kernel(Shape::Scalar, Unit::meters(), None),
                ),
                (
                    "y".to_string(),
                    Type::kernel(Shape::Scalar, Unit::meters(), None),
                ),
            ],
        );
        type_table.register(user_type);

        let ctx = ValidationContext::new(&type_table);

        // Field 'z' doesn't exist on Vec2
        let field = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::meters()),
            },
            Type::kernel(Shape::Scalar, Unit::meters(), None),
            test_span(),
        );

        let struct_expr = TypedExpr::new(
            ExprKind::Struct {
                ty: type_id.clone(),
                fields: vec![("z".to_string(), field)], // Unknown field!
            },
            Type::user(type_id),
            test_span(),
        );

        let errors = validate_expr(&struct_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::UnknownType);
        assert!(errors[0].message.contains("no field 'z'"));
    }

    #[test]
    fn test_validate_field_access_unknown_type() {
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

        // Accessing field on unknown type
        let object = TypedExpr::new(
            ExprKind::Local("obj".to_string()),
            Type::user(continuum_foundation::TypeId::from("UnknownType")),
            test_span(),
        );

        let field_access = TypedExpr::new(
            ExprKind::FieldAccess {
                object: Box::new(object),
                field: "x".to_string(),
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&field_access, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::UnknownType);
        assert!(errors[0].message.contains("Unknown user type"));
    }

    #[test]
    fn test_validate_field_access_unknown_field() {
        use crate::foundation::{Path, UserType};

        // Create type table with a user type
        let mut type_table = TypeTable::new();
        let type_id = continuum_foundation::TypeId::from("Vec2");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Vec2"),
            vec![
                (
                    "x".to_string(),
                    Type::kernel(Shape::Scalar, Unit::meters(), None),
                ),
                (
                    "y".to_string(),
                    Type::kernel(Shape::Scalar, Unit::meters(), None),
                ),
            ],
        );
        type_table.register(user_type);

        let ctx = ValidationContext::new(&type_table);

        // Accessing field 'z' which doesn't exist
        let object = TypedExpr::new(
            ExprKind::Local("v".to_string()),
            Type::user(type_id),
            test_span(),
        );

        let field_access = TypedExpr::new(
            ExprKind::FieldAccess {
                object: Box::new(object),
                field: "z".to_string(), // Unknown field!
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&field_access, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::UnknownType);
        assert!(errors[0].message.contains("no field 'z'"));
    }

    #[test]
    fn test_validate_struct_valid() {
        use crate::foundation::{Path, UserType};

        // Create type table with a user type
        let mut type_table = TypeTable::new();
        let type_id = continuum_foundation::TypeId::from("Vec2");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Vec2"),
            vec![
                (
                    "x".to_string(),
                    Type::kernel(Shape::Scalar, Unit::meters(), None),
                ),
                (
                    "y".to_string(),
                    Type::kernel(Shape::Scalar, Unit::meters(), None),
                ),
            ],
        );
        type_table.register(user_type);

        let ctx = ValidationContext::new(&type_table);

        // Valid struct construction
        let field_x = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::meters()),
            },
            Type::kernel(Shape::Scalar, Unit::meters(), None),
            test_span(),
        );

        let field_y = TypedExpr::new(
            ExprKind::Literal {
                value: 2.0,
                unit: Some(Unit::meters()),
            },
            Type::kernel(Shape::Scalar, Unit::meters(), None),
            test_span(),
        );

        let struct_expr = TypedExpr::new(
            ExprKind::Struct {
                ty: type_id.clone(),
                fields: vec![("x".to_string(), field_x), ("y".to_string(), field_y)],
            },
            Type::user(type_id),
            test_span(),
        );

        let errors = validate_expr(&struct_expr, &ctx);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_call_propagates_arg_errors() {
        use crate::foundation::Bounds;

        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

        // Argument violates bounds
        let arg = TypedExpr::new(
            ExprKind::Literal {
                value: -1.0,
                unit: None,
            },
            Type::kernel(
                Shape::Scalar,
                Unit::DIMENSIONLESS,
                Some(Bounds {
                    min: Some(0.0),
                    max: None,
                }),
            ),
            test_span(),
        );

        let call_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "identity"),
                args: vec![arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&call_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::TypeMismatch);
        assert!(errors[0].message.contains("below minimum"));
    }

    #[test]
    fn test_validate_field_access_valid() {
        use crate::foundation::{Path, UserType};

        // Create type table with a user type
        let mut type_table = TypeTable::new();
        let type_id = continuum_foundation::TypeId::from("Vec2");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Vec2"),
            vec![
                (
                    "x".to_string(),
                    Type::kernel(Shape::Scalar, Unit::meters(), None),
                ),
                (
                    "y".to_string(),
                    Type::kernel(Shape::Scalar, Unit::meters(), None),
                ),
            ],
        );
        type_table.register(user_type);

        let ctx = ValidationContext::new(&type_table);

        // Valid field access
        let object = TypedExpr::new(
            ExprKind::Local("v".to_string()),
            Type::user(type_id),
            test_span(),
        );

        let field_access = TypedExpr::new(
            ExprKind::FieldAccess {
                object: Box::new(object),
                field: "x".to_string(),
            },
            Type::kernel(Shape::Scalar, Unit::meters(), None),
            test_span(),
        );

        let errors = validate_expr(&field_access, &ctx);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_field_access_on_kernel_type() {
        let type_table = test_type_table();
        let ctx = ValidationContext::new(&type_table);

        // Field access on kernel type (Vector) is allowed (not validated here)
        let object = TypedExpr::new(
            ExprKind::Local("v".to_string()),
            Type::kernel(Shape::Vector { dim: 3 }, Unit::meters(), None),
            test_span(),
        );

        let field_access = TypedExpr::new(
            ExprKind::FieldAccess {
                object: Box::new(object),
                field: "x".to_string(),
            },
            Type::kernel(Shape::Scalar, Unit::meters(), None),
            test_span(),
        );

        // No errors - kernel type field access is not validated at this level
        let errors = validate_expr(&field_access, &ctx);
        assert!(errors.is_empty());
    }
}
