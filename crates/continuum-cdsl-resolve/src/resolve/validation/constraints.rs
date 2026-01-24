//! Shape and unit constraint validation for kernel calls.
//!
//! Validates that kernel call arguments satisfy shape and unit constraints
//! specified in kernel signatures.

use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::Span;
use continuum_cdsl_ast::foundation::Type;
use continuum_cdsl_ast::{KernelId, TypedExpr};

pub(super) fn validate_shape_constraint(
    arg_type: &Type,
    constraint: &continuum_cdsl_ast::ShapeConstraint,
    arg_index: usize,
    all_args: &[TypedExpr],
    kernel: &KernelId,
    span: Span,
) -> Vec<CompileError> {
    use continuum_cdsl_ast::foundation::Shape;
    use continuum_cdsl_ast::ShapeConstraint;

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
            if *param_index >= all_args.len() {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelShape,
                    span,
                    format!(
                        "kernel {} shape constraint SameAs({}) references out-of-bounds argument (only {} args)",
                        kernel.qualified_name(),
                        param_index,
                        all_args.len()
                    ),
                ));
                return errors;
            }

            let Some(expected_shape) = all_args[*param_index].ty.as_kernel().map(|k| &k.shape)
            else {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelShape,
                    span,
                    format!(
                        "kernel {} shape constraint SameAs({}) references non-kernel argument type {:?}",
                        kernel.qualified_name(),
                        param_index,
                        all_args[*param_index].ty
                    ),
                ));
                return errors;
            };

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

        ShapeConstraint::VectorDim(dim_constraint) => {
            if let Shape::Vector { dim } = arg_shape {
                match matches_dim_constraint(*dim, dim_constraint) {
                    Ok(true) => {
                        // Constraint satisfied
                    }
                    Ok(false) => {
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
                    Err(msg) => {
                        errors.push(CompileError::new(
                            ErrorKind::InvalidKernelShape,
                            span,
                            format!(
                                "kernel {} argument {}: {}",
                                kernel.qualified_name(),
                                arg_index,
                                msg
                            ),
                        ));
                    }
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

        ShapeConstraint::BroadcastWith(param_index) => {
            errors.push(CompileError::new(
                ErrorKind::InvalidKernelShape,
                span,
                format!(
                    "kernel {} uses unsupported BroadcastWith({}) constraint (not yet implemented)",
                    kernel.qualified_name(),
                    param_index
                ),
            ));
        }

        ShapeConstraint::MatrixDims { .. } => {
            errors.push(CompileError::new(
                ErrorKind::InvalidKernelShape,
                span,
                format!(
                    "kernel {} uses unsupported MatrixDims constraint (not yet implemented)",
                    kernel.qualified_name()
                ),
            ));
        }
    }

    errors
}

/// Check if a dimension satisfies a dimension constraint.
///
/// Returns:
/// - `Ok(true)` if dimension satisfies constraint
/// - `Ok(false)` if dimension does not satisfy constraint
/// - `Err(message)` if constraint is unsupported
fn matches_dim_constraint(
    dim: u8,
    constraint: &continuum_cdsl_ast::DimConstraint,
) -> Result<bool, String> {
    use continuum_cdsl_ast::DimConstraint;

    match constraint {
        DimConstraint::Exact(expected) => Ok(dim == *expected),
        DimConstraint::Any => Ok(true),
        DimConstraint::Var(var_id) => Err(format!(
            "dimension variable constraints (Var({})) not yet supported - requires tracking dimension variables across parameters",
            var_id
        )),
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
pub(super) fn validate_unit_constraint(
    arg_type: &Type,
    constraint: &continuum_cdsl_ast::UnitConstraint,
    arg_index: usize,
    all_args: &[TypedExpr],
    kernel: &KernelId,
    span: Span,
) -> Vec<CompileError> {
    use continuum_cdsl_ast::UnitConstraint;

    let mut errors = Vec::new();

    // Extract unit from type (only kernel types have units)
    let Some(arg_unit) = arg_type.as_kernel().map(|k| &k.unit) else {
        // Non-kernel types don't have units - check if they're acceptable
        // Bool is conceptually dimensionless, so accept it for Dimensionless or Any constraints
        let acceptable = match (arg_type, constraint) {
            (Type::Bool, UnitConstraint::Any | UnitConstraint::Dimensionless) => true,
            (_, UnitConstraint::Any) => true,
            _ => false,
        };

        if !acceptable {
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
            // Use is_compatible_with() to allow scale mismatch for dimensionless units
            if !arg_unit.is_compatible_with(expected) {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "{} argument {} expected '{}', found '{}'",
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
                        "{} argument {} must be dimensionless, found '{}'",
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
                        "{} argument {} must be an angle unit, found '{}'",
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
            if *param_index >= all_args.len() {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "kernel {} unit constraint SameAs({}) references out-of-bounds argument (only {} args)",
                        kernel.qualified_name(),
                        param_index,
                        all_args.len()
                    ),
                ));
                return errors;
            }

            let Some(expected_unit) = all_args[*param_index].ty.as_kernel().map(|k| &k.unit) else {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "kernel {} unit constraint SameAs({}) references non-kernel argument type {:?}",
                        kernel.qualified_name(),
                        param_index,
                        all_args[*param_index].ty
                    ),
                ));
                return errors;
            };

            // Allow dimensionless to match any unit (implicit unit adoption)
            // This enables config values without units to work with dimensional signals.
            // Either side being dimensionless allows the match because the dimensionless
            // value will adopt the other's unit at runtime.
            let either_dimensionless =
                arg_unit.is_dimensionless() || expected_unit.is_dimensionless();
            if !either_dimensionless && !arg_unit.is_compatible_with(expected_unit) {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "{} argument {} expected '{}', found '{}'",
                        kernel.qualified_name(),
                        arg_index,
                        expected_unit,
                        arg_unit
                    ),
                ));
            }
        }

        UnitConstraint::SameDimsAs(param_index) => {
            if *param_index >= all_args.len() {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "kernel {} unit constraint SameDimsAs({}) references out-of-bounds argument (only {} args)",
                        kernel.qualified_name(),
                        param_index,
                        all_args.len()
                    ),
                ));
                return errors;
            }

            let Some(expected_unit) = all_args[*param_index].ty.as_kernel().map(|k| &k.unit) else {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "kernel {} unit constraint SameDimsAs({}) references non-kernel argument type {:?}",
                        kernel.qualified_name(),
                        param_index,
                        all_args[*param_index].ty
                    ),
                ));
                return errors;
            };

            // Allow dimensionless to match any unit (implicit unit adoption)
            // This enables config values without units to work with dimensional signals.
            // Either side being dimensionless allows the match because the dimensionless
            // value will adopt the other's unit at runtime.
            let either_dimensionless =
                arg_unit.is_dimensionless() || expected_unit.is_dimensionless();
            if !either_dimensionless
                && arg_unit.dimensional_type() != expected_unit.dimensional_type()
            {
                errors.push(CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "{} argument {} must have same dimensional type as argument {}, expected '{}', found '{}'",
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

    errors
}
