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

use crate::ast::{ExprKind, KernelId, KernelRegistry, TypedExpr};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{Span, Type};
use crate::resolve::types::TypeTable;

/// Context for validating typed CDSL expressions using user-defined types and kernel signatures.
///
/// `ValidationContext` provides read-only access to user type definitions and kernel
/// signatures that expression validation needs when checking struct construction, field
/// access, and kernel calls.
///
/// # Parameters
///
/// - `type_table`: Reference to the [`TypeTable`] containing user-defined types.
/// - `kernel_registry`: Reference to the [`KernelRegistry`] containing kernel signatures.
///
/// # Returns
///
/// A `ValidationContext` that borrows the provided registries for lookups during validation.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::KernelRegistry;
/// use continuum_cdsl::resolve::types::TypeTable;
/// use continuum_cdsl::resolve::validation::ValidationContext;
///
/// let type_table = TypeTable::new();
/// let kernel_registry = KernelRegistry::global();
/// let ctx = ValidationContext::new(&type_table, kernel_registry);
/// let _ = ctx;
/// ```
pub struct ValidationContext<'a> {
    /// User type definitions for struct validation
    pub type_table: &'a TypeTable,

    /// Kernel signatures for kernel call validation
    pub kernel_registry: &'a KernelRegistry,
}

impl<'a> ValidationContext<'a> {
    /// Create a new validation context.
    ///
    /// # Parameters
    ///
    /// - `type_table`: User type definitions for struct validation.
    /// - `kernel_registry`: Kernel signatures for kernel call validation.
    ///
    /// # Returns
    ///
    /// A new validation context ready for use.
    pub fn new(type_table: &'a TypeTable, kernel_registry: &'a KernelRegistry) -> Self {
        Self {
            type_table,
            kernel_registry,
        }
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
/// use continuum_cdsl::ast::{ExprKind, KernelRegistry, TypedExpr};
/// use continuum_cdsl::foundation::{Shape, Span, Type, Unit};
/// use continuum_cdsl::resolve::types::TypeTable;
/// use continuum_cdsl::resolve::validation::{validate_expr, ValidationContext};
///
/// let type_table = TypeTable::new();
/// let kernel_registry = KernelRegistry::global();
/// let ctx = ValidationContext::new(&type_table, kernel_registry);
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
    kernel: &KernelId,
    args: &[TypedExpr],
    span: Span,
    ctx: &ValidationContext<'_>,
) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Recursively validate all arguments
    for arg in args {
        errors.extend(validate_expr(arg, ctx));
    }

    // Look up kernel signature
    let Some(signature) = ctx.kernel_registry.get(kernel) else {
        errors.push(CompileError::new(
            ErrorKind::UnknownKernel,
            span,
            format!("Unknown kernel: {}", kernel.qualified_name()),
        ));
        return errors;
    };

    // Validate argument count
    if args.len() != signature.params.len() {
        errors.push(CompileError::new(
            ErrorKind::WrongArgCount,
            span,
            format!(
                "kernel {} expects {} arguments, got {}",
                kernel.qualified_name(),
                signature.params.len(),
                args.len()
            ),
        ));
        return errors;
    }

    // Validate each argument against its parameter constraint
    for (i, (arg, param)) in args.iter().zip(&signature.params).enumerate() {
        // Validate shape constraint
        errors.extend(validate_shape_constraint(
            &arg.ty,
            &param.shape,
            i,
            args,
            kernel,
            arg.span,
        ));

        // Validate unit constraint
        errors.extend(validate_unit_constraint(
            &arg.ty,
            &param.unit,
            i,
            args,
            kernel,
            arg.span,
        ));
    }

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

/// Validates an argument's shape against a parameter's shape constraint.
///
/// # Parameters
///
/// - `arg_type`: The argument's type.
/// - `constraint`: The parameter's shape constraint.
/// - `arg_index`: Index of this argument.
/// - `all_args`: All arguments (for SameAs/BroadcastWith constraints).
/// - `kernel`: Kernel identifier for error messages.
/// - `span`: Source location for error reporting.
///
/// # Returns
///
/// Vector of validation errors (empty if shape satisfies constraint).
fn validate_shape_constraint(
    arg_type: &Type,
    constraint: &crate::ast::ShapeConstraint,
    arg_index: usize,
    all_args: &[TypedExpr],
    kernel: &KernelId,
    span: Span,
) -> Vec<CompileError> {
    use crate::ast::ShapeConstraint;
    use crate::foundation::Shape;

    let mut errors = Vec::new();

    // Extract shape from type (only kernel types have shapes)
    let Some(arg_shape) = arg_type.as_kernel().map(|k| &k.shape) else {
        // Non-kernel types (Bool, User, Unit, Seq) fail shape constraints
        if !matches!(constraint, ShapeConstraint::Any) {
            errors.push(CompileError::new(
                ErrorKind::InvalidKernelShape,
                span,
                format!(
                    "kernel {} argument {} must be a kernel type, found {:?}",
                    kernel.qualified_name(),
                    arg_index,
                    arg_type
                ),
            ));
        }
        return errors;
    };

    match constraint {
        ShapeConstraint::Exact(expected) => {
            if arg_shape != expected {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelShape,
                    span,
                    format!(
                        "kernel {} argument {} must be {:?}, found {:?}",
                        kernel.qualified_name(),
                        arg_index,
                        expected,
                        arg_shape
                    ),
                ));
            }
        }

        ShapeConstraint::AnyScalar => {
            if !matches!(arg_shape, Shape::Scalar) {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelShape,
                    span,
                    format!(
                        "kernel {} argument {} must be Scalar, found {:?}",
                        kernel.qualified_name(),
                        arg_index,
                        arg_shape
                    ),
                ));
            }
        }

        ShapeConstraint::AnyVector => {
            if !matches!(arg_shape, Shape::Vector { .. }) {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelShape,
                    span,
                    format!(
                        "kernel {} argument {} must be Vector, found {:?}",
                        kernel.qualified_name(),
                        arg_index,
                        arg_shape
                    ),
                ));
            }
        }

        ShapeConstraint::AnyMatrix => {
            if !matches!(arg_shape, Shape::Matrix { .. }) {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelShape,
                    span,
                    format!(
                        "kernel {} argument {} must be Matrix, found {:?}",
                        kernel.qualified_name(),
                        arg_index,
                        arg_shape
                    ),
                ));
            }
        }

        ShapeConstraint::Any => {
            // Any shape is allowed
        }

        ShapeConstraint::SameAs(param_index) => {
            if *param_index < all_args.len() {
                if let Some(expected_shape) =
                    all_args[*param_index].ty.as_kernel().map(|k| &k.shape)
                {
                    if arg_shape != expected_shape {
                        errors.push(CompileError::new(
                            ErrorKind::InvalidKernelShape,
                            span,
                            format!(
                                "kernel {} argument {} must have same shape as argument {}, expected {:?}, found {:?}",
                                kernel.qualified_name(),
                                arg_index,
                                param_index,
                                expected_shape,
                                arg_shape
                            ),
                        ));
                    }
                }
            }
        }

        ShapeConstraint::VectorDim(dim_constraint) => {
            if let Shape::Vector { dim } = arg_shape {
                if !matches_dim_constraint(*dim, dim_constraint) {
                    errors.push(CompileError::new(
                        ErrorKind::InvalidKernelShape,
                        span,
                        format!(
                            "kernel {} argument {} vector dimension does not satisfy constraint {:?}",
                            kernel.qualified_name(),
                            arg_index,
                            dim_constraint
                        ),
                    ));
                }
            } else {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelShape,
                    span,
                    format!(
                        "kernel {} argument {} must be Vector, found {:?}",
                        kernel.qualified_name(),
                        arg_index,
                        arg_shape
                    ),
                ));
            }
        }

        ShapeConstraint::BroadcastWith(_) | ShapeConstraint::MatrixDims { .. } => {
            // TODO: Implement broadcast and matrix dimension validation
            // These are more complex and can be added later
        }
    }

    errors
}

/// Check if a dimension satisfies a dimension constraint.
fn matches_dim_constraint(dim: u8, constraint: &crate::ast::DimConstraint) -> bool {
    use crate::ast::DimConstraint;

    match constraint {
        DimConstraint::Exact(expected) => dim == *expected,
        DimConstraint::Any => true,
        DimConstraint::Var(_) => {
            // TODO: Track dimension variables across parameters
            // For now, accept any dimension for Var constraints
            true
        }
    }
}

/// Validates an argument's unit against a parameter's unit constraint.
///
/// # Parameters
///
/// - `arg_type`: The argument's type.
/// - `constraint`: The parameter's unit constraint.
/// - `arg_index`: Index of this argument.
/// - `all_args`: All arguments (for SameAs constraints).
/// - `kernel`: Kernel identifier for error messages.
/// - `span`: Source location for error reporting.
///
/// # Returns
///
/// Vector of validation errors (empty if unit satisfies constraint).
fn validate_unit_constraint(
    arg_type: &Type,
    constraint: &crate::ast::UnitConstraint,
    arg_index: usize,
    all_args: &[TypedExpr],
    kernel: &KernelId,
    span: Span,
) -> Vec<CompileError> {
    use crate::ast::UnitConstraint;

    let mut errors = Vec::new();

    // Extract unit from type (only kernel types have units)
    let Some(arg_unit) = arg_type.as_kernel().map(|k| &k.unit) else {
        // Non-kernel types don't have units
        if !matches!(constraint, UnitConstraint::Any) {
            errors.push(CompileError::new(
                ErrorKind::InvalidKernelUnit,
                span,
                format!(
                    "kernel {} argument {} must be a kernel type with units, found {:?}",
                    kernel.qualified_name(),
                    arg_index,
                    arg_type
                ),
            ));
        }
        return errors;
    };

    match constraint {
        UnitConstraint::Exact(expected) => {
            if arg_unit != expected {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "kernel {} argument {} must have unit {:?}, found {:?}",
                        kernel.qualified_name(),
                        arg_index,
                        expected,
                        arg_unit
                    ),
                ));
            }
        }

        UnitConstraint::Dimensionless => {
            if !arg_unit.is_dimensionless() {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "kernel {} argument {} must be dimensionless, found {:?}",
                        kernel.qualified_name(),
                        arg_index,
                        arg_unit
                    ),
                ));
            }
        }

        UnitConstraint::Angle => {
            // Check if the unit has angle dimension (angle exponent != 0)
            if arg_unit.dims().angle == 0 {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "kernel {} argument {} must be an angle unit, found {:?}",
                        kernel.qualified_name(),
                        arg_index,
                        arg_unit
                    ),
                ));
            }
        }

        UnitConstraint::Any => {
            // Any unit is allowed
        }

        UnitConstraint::SameAs(param_index) => {
            if *param_index < all_args.len() {
                if let Some(expected_unit) = all_args[*param_index].ty.as_kernel().map(|k| &k.unit)
                {
                    if arg_unit != expected_unit {
                        errors.push(CompileError::new(
                            ErrorKind::InvalidKernelUnit,
                            span,
                            format!(
                                "kernel {} argument {} must have same unit as argument {}, expected {:?}, found {:?}",
                                kernel.qualified_name(),
                                arg_index,
                                param_index,
                                expected_unit,
                                arg_unit
                            ),
                        ));
                    }
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
    use crate::resolve::types::TypeTable;

    fn test_span() -> Span {
        Span::new(0, 10, 20, 1)
    }

    fn test_type_table() -> TypeTable {
        TypeTable::new()
    }

    fn test_ctx(type_table: &TypeTable) -> ValidationContext<'_> {
        ValidationContext::new(type_table, KernelRegistry::global())
    }

    #[test]
    fn test_validate_literal() {
        let type_table = test_type_table();
        let ctx = test_ctx(&type_table);
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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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

        let ctx = test_ctx(&type_table);

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

        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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

        let ctx = test_ctx(&type_table);

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

        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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
                kernel: KernelId::new("maths", "abs"),
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

        let ctx = test_ctx(&type_table);

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
        let ctx = test_ctx(&type_table);

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

    // ============================================================================
    // Kernel Validation Tests
    // ============================================================================

    #[test]
    fn test_validate_kernel_unknown() {
        let type_table = test_type_table();
        let ctx = test_ctx(&type_table);

        let arg = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("unknown", "kernel"),
                args: vec![arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&call_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::UnknownKernel);
        assert!(errors[0].message.contains("unknown.kernel"));
    }

    #[test]
    fn test_validate_kernel_wrong_arg_count() {
        let type_table = test_type_table();
        let ctx = test_ctx(&type_table);

        // maths.abs expects 1 argument
        let call_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "abs"),
                args: vec![], // Wrong: should have 1 arg
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&call_expr, &ctx);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::WrongArgCount);
        assert!(errors[0].message.contains("expects 1 arguments, got 0"));
    }

    #[test]
    fn test_validate_kernel_shape_exact_mismatch() {
        let type_table = test_type_table();
        let ctx = test_ctx(&type_table);

        // vector.dot expects two Vector<3> arguments
        let vec2_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::kernel(Shape::Vector { dim: 2 }, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("vector", "dot"),
                args: vec![vec2_arg.clone(), vec2_arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&call_expr, &ctx);
        // Should have 2 errors (one for each argument shape mismatch)
        assert!(!errors.is_empty());
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidKernelShape)
        );
    }

    #[test]
    fn test_validate_kernel_shape_scalar_vs_vector() {
        let type_table = test_type_table();
        let ctx = test_ctx(&type_table);

        // vector.cross expects two Vector<3> arguments, but we give it scalars
        let scalar_arg1 = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let scalar_arg2 = TypedExpr::new(
            ExprKind::Literal {
                value: 2.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("vector", "cross"),
                args: vec![scalar_arg1, scalar_arg2],
            },
            Type::kernel(Shape::Vector { dim: 3 }, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&call_expr, &ctx);
        assert!(!errors.is_empty());
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidKernelShape)
        );
    }

    #[test]
    fn test_validate_kernel_shape_sameas_mismatch() {
        let type_table = test_type_table();
        let ctx = test_ctx(&type_table);

        // maths.add expects both args to have same shape (SameAs constraint)
        let scalar_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let vec3_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::kernel(Shape::Vector { dim: 3 }, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![scalar_arg, vec3_arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&call_expr, &ctx);
        assert!(!errors.is_empty());
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidKernelShape)
        );
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("must have same shape as argument 0"))
        );
    }

    #[test]
    fn test_validate_kernel_unit_exact_mismatch() {
        let type_table = test_type_table();
        let ctx = test_ctx(&type_table);

        // sin expects a dimensionless (or angle) argument, but we give it meters
        let meters_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::meters()),
            },
            Type::kernel(Shape::Scalar, Unit::meters(), None),
            test_span(),
        );

        let call_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "sin"),
                args: vec![meters_arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&call_expr, &ctx);
        assert!(!errors.is_empty());
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidKernelUnit)
        );
    }

    #[test]
    fn test_validate_kernel_unit_sameas_mismatch() {
        let type_table = test_type_table();
        let ctx = test_ctx(&type_table);

        // maths.add expects both args to have same unit (SameAs constraint)
        let meters_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::meters()),
            },
            Type::kernel(Shape::Scalar, Unit::meters(), None),
            test_span(),
        );

        let seconds_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: Some(Unit::seconds()),
            },
            Type::kernel(Shape::Scalar, Unit::seconds(), None),
            test_span(),
        );

        let call_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![meters_arg, seconds_arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&call_expr, &ctx);
        assert!(!errors.is_empty());
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::InvalidKernelUnit)
        );
        assert!(
            errors
                .iter()
                .any(|e| e.message.contains("must have same unit as argument 0"))
        );
    }

    #[test]
    fn test_validate_kernel_valid_call() {
        let type_table = test_type_table();
        let ctx = test_ctx(&type_table);

        // Valid maths.abs call
        let arg = TypedExpr::new(
            ExprKind::Literal {
                value: -5.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "abs"),
                args: vec![arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_expr(&call_expr, &ctx);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_kernel_valid_vector_call() {
        let type_table = test_type_table();
        let ctx = test_ctx(&type_table);

        // Valid vector.dot call with Vector<3>
        let vec3_arg1 = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::kernel(Shape::Vector { dim: 3 }, Unit::meters(), None),
            test_span(),
        );

        let vec3_arg2 = TypedExpr::new(
            ExprKind::Literal {
                value: 2.0,
                unit: None,
            },
            Type::kernel(Shape::Vector { dim: 3 }, Unit::meters(), None),
            test_span(),
        );

        let call_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("vector", "dot"),
                args: vec![vec3_arg1, vec3_arg2],
            },
            Type::kernel(Shape::Scalar, Unit::meters(), None),
            test_span(),
        );

        let errors = validate_expr(&call_expr, &ctx);
        assert!(errors.is_empty());
    }
}
