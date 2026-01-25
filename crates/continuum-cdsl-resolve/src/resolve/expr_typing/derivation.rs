//! Type derivation for kernel calls.
//!
//! This module implements the logic for deriving the return type of kernel
//! calls based on their signature and the types of their arguments. This
//! includes shape derivation (e.g., Scalar, Vector) and unit derivation
//! (e.g., multiplication, division of units).

use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::{KernelType, Shape, Type, Unit};
use continuum_cdsl_ast::TypedExpr;
use continuum_kernel_types::rational::Rational;
use continuum_kernel_types::ValueType;

/// Extracts the [`KernelType`] from a typed argument at the specified index.
///
/// # Parameters
/// - `args`: The list of typed argument expressions.
/// - `idx`: The index of the argument to extract.
/// - `span`: Source location for error reporting.
/// - `derivation_kind`: Name of the derivation type (for error messages).
///
/// # Returns
/// A reference to the [`KernelType`] if the argument exists and is a kernel type.
///
/// # Errors
/// Returns [`ErrorKind::Internal`] if the index is out of bounds or the argument
/// is not a kernel type.
pub fn get_kernel_arg<'a>(
    args: &'a [TypedExpr],
    idx: usize,
    span: continuum_cdsl_ast::foundation::Span,
    derivation_kind: &str,
) -> Result<&'a KernelType, Vec<CompileError>> {
    let arg = args.get(idx).ok_or_else(|| {
        vec![CompileError::new(
            ErrorKind::Internal,
            span,
            format!(
                "invalid parameter index {} in {} derivation",
                idx, derivation_kind
            ),
        )]
    })?;
    arg.ty.as_kernel().ok_or_else(|| {
        vec![CompileError::new(
            ErrorKind::Internal,
            span,
            format!("parameter {} is not a kernel type", idx),
        )]
    })
}

/// Derives the result type of a kernel call based on its signature and arguments.
///
/// # Parameters
/// - `sig`: The signature of the kernel being called.
/// - `args`: The resolved types of the arguments.
/// - `span`: Source location of the call.
///
/// # Returns
/// The derived [`Type`] of the kernel's return value.
///
/// # Errors
/// Returns an error if:
/// - A shape or unit derivation variant is not yet implemented.
/// - A unit operation (multiplication/division) is invalid.
/// - An argument index in the signature's derivation rules is out of bounds.
pub fn derive_return_type(
    sig: &continuum_cdsl_ast::KernelSignature,
    args: &[TypedExpr],
    span: continuum_cdsl_ast::foundation::Span,
) -> Result<Type, Vec<CompileError>> {
    use continuum_cdsl_ast::{ShapeDerivation, UnitDerivation};

    if sig.returns.value_type == ValueType::Bool {
        return Ok(Type::Bool);
    }

    if !sig.returns.value_type.is_numeric() {
        return Err(vec![CompileError::new(
            ErrorKind::Internal,
            span,
            format!(
                "non-boolean kernel return should be numeric, got {:?}",
                sig.returns.value_type
            ),
        )]);
    }

    let (shape, bounds_from_shape) = match &sig.returns.shape {
        ShapeDerivation::Exact(s) => (s.clone(), None),
        ShapeDerivation::Scalar => (Shape::Scalar, None),
        ShapeDerivation::SameAs(idx) => {
            let kt = get_kernel_arg(args, *idx, span, "shape")?;
            (kt.shape.clone(), kt.bounds.clone())
        }
        ShapeDerivation::VectorDim(dim_constraint) => {
            use continuum_cdsl_ast::DimConstraint;
            let dim = match dim_constraint {
                DimConstraint::Exact(d) => *d,
                DimConstraint::Any => {
                    return Err(vec![CompileError::new(
                        ErrorKind::UnsupportedDSLFeature,
                        span,
                        "VectorDim(Any) requires exact dimension for return type".to_string(),
                    )]);
                }
                DimConstraint::Var(_) => {
                    return Err(vec![CompileError::new(
                        ErrorKind::UnsupportedDSLFeature,
                        span,
                        "VectorDim(Var) not yet supported for return types".to_string(),
                    )]);
                }
            };
            if dim < 2 || dim > 4 {
                return Err(vec![CompileError::new(
                    ErrorKind::TypeMismatch,
                    span,
                    format!(
                        "vector dimension {} not supported (must be 2, 3, or 4)",
                        dim
                    ),
                )]);
            }
            (Shape::Vector { dim }, None)
        }
        ShapeDerivation::MatrixDims { rows, cols } => {
            use continuum_cdsl_ast::DimConstraint;
            let (r, c) = match (rows, cols) {
                (DimConstraint::Exact(r), DimConstraint::Exact(c)) => (*r, *c),
                _ => {
                    return Err(vec![CompileError::new(
                        ErrorKind::UnsupportedDSLFeature,
                        span,
                        format!(
                            "MatrixDims requires exact dimensions, got rows={:?} cols={:?}",
                            rows, cols
                        ),
                    )]);
                }
            };
            if r < 2 || r > 4 || c < 2 || c > 4 {
                return Err(vec![CompileError::new(
                    ErrorKind::TypeMismatch,
                    span,
                    format!(
                        "matrix dimensions {}x{} not supported (must be 2-4 x 2-4)",
                        r, c
                    ),
                )]);
            }
            (Shape::Matrix { rows: r, cols: c }, None)
        }
        _ => {
            return Err(vec![CompileError::new(
                ErrorKind::UnsupportedDSLFeature,
                span,
                format!(
                    "shape derivation variant not yet implemented: {:?}",
                    sig.returns.shape
                ),
            )]);
        }
    };

    let unit = match &sig.returns.unit {
        UnitDerivation::Exact(u) => *u,
        UnitDerivation::Dimensionless => Unit::DIMENSIONLESS,
        UnitDerivation::SameAs(idx) => {
            let kt = get_kernel_arg(args, *idx, span, "unit")?;
            kt.unit
        }
        UnitDerivation::Multiply(indices) => {
            let mut result = Unit::DIMENSIONLESS;
            for &idx in indices {
                let kt = get_kernel_arg(args, idx, span, "unit multiply")?;
                result = result.multiply(&kt.unit).ok_or_else(|| {
                    vec![CompileError::new(
                        ErrorKind::Internal,
                        span,
                        format!(
                            "cannot multiply non-multiplicative units (parameter {})",
                            idx
                        ),
                    )]
                })?;
            }
            result
        }
        UnitDerivation::Divide(a, b) => {
            let kt_a = get_kernel_arg(args, *a, span, "unit divide numerator")?;
            let kt_b = get_kernel_arg(args, *b, span, "unit divide denominator")?;
            kt_a.unit.divide(&kt_b.unit).ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::Internal,
                    span,
                    format!(
                        "cannot divide non-multiplicative units (parameters {} / {})",
                        a, b
                    ),
                )]
            })?
        }
        UnitDerivation::Sqrt(idx) => {
            let kt = get_kernel_arg(args, *idx, span, "unit sqrt")?;
            kt.unit.sqrt().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "cannot take square root of unit {} (all dimension exponents must be even)",
                        kt.unit
                    ),
                )]
            })?
        }
        UnitDerivation::Inverse(idx) => {
            let kt = get_kernel_arg(args, *idx, span, "unit inverse")?;
            kt.unit.inverse().ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::Internal,
                    span,
                    format!("cannot invert non-multiplicative unit (parameter {})", idx),
                )]
            })?
        }
        UnitDerivation::Power(idx, exponent) => {
            let kt = get_kernel_arg(args, *idx, span, "unit power base")?;
            kt.unit.pow(*exponent).ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "cannot raise unit {} to power {} (non-multiplicative units cannot be exponentiated)",
                        kt.unit, exponent
                    ),
                )]
            })?
        }
        UnitDerivation::PowerRational(idx, num, denom) => {
            let kt = get_kernel_arg(args, *idx, span, "unit power base")?;
            let exponent = Rational::new(*num, *denom);
            kt.unit.pow_rational(exponent).ok_or_else(|| {
                vec![CompileError::new(
                    ErrorKind::InvalidKernelUnit,
                    span,
                    format!(
                        "cannot raise unit {} to power {}/{} (non-multiplicative units cannot be exponentiated)",
                        kt.unit, num, denom
                    ),
                )]
            })?
        }
    };

    Ok(Type::Kernel(KernelType {
        shape,
        unit,
        bounds: bounds_from_shape,
    }))
}
