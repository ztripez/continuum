//! Unit resolution and dimensional arithmetic for the Continuum DSL.
//!
//! This module implements the physical unit system, including:
//! - **Base unit resolution** - Mapping symbols (m, kg, s) to dimensions.
//! - **Compound unit arithmetic** - Multiplication, division, and powers.
//! - **Dimensional analysis** - Ensuring physical consistency during compilation.
//! - **Overflow protection** - Checked arithmetic for dimensional exponents.

use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::UnitExpr;
use continuum_cdsl_ast::foundation::{Span, Unit, UnitDimensions, UnitKind};

/// Resolves a parsed [`UnitExpr`] into a semantic [`Unit`].
///
/// Converts unit syntax to dimensional unit values with dimensional exponents.
/// Handles unit arithmetic (multiply, divide, power) and validates kind compatibility.
///
/// # Parameters
///
/// - `unit_expr`: Optional unit syntax; `None` yields dimensionless units.
/// - `span`: Source location for error reporting.
///
/// # Returns
///
/// A resolved [`Unit`] with dimensional exponents computed from the expression.
pub fn resolve_unit_expr(unit_expr: Option<&UnitExpr>, span: Span) -> Result<Unit, CompileError> {
    match unit_expr {
        None => Ok(Unit::DIMENSIONLESS),
        Some(UnitExpr::Dimensionless) => Ok(Unit::DIMENSIONLESS),
        Some(UnitExpr::Base(name)) => resolve_base_unit(name, span),
        Some(UnitExpr::Multiply(lhs, rhs)) => {
            let lhs_unit = resolve_unit_expr(Some(lhs), span)?;
            let rhs_unit = resolve_unit_expr(Some(rhs), span)?;
            multiply_units(&lhs_unit, &rhs_unit, span)
        }
        Some(UnitExpr::Divide(numerator, denominator)) => {
            let num_unit = resolve_unit_expr(Some(numerator), span)?;
            let den_unit = resolve_unit_expr(Some(denominator), span)?;
            divide_units(&num_unit, &den_unit, span)
        }
        Some(UnitExpr::Power(base, exponent)) => {
            let base_unit = resolve_unit_expr(Some(base), span)?;
            power_unit(&base_unit, *exponent, span)
        }
    }
}

/// Resolve a base unit name to a Unit.
pub fn resolve_base_unit(name: &str, span: Span) -> Result<Unit, CompileError> {
    let unit = match name {
        // SI base units
        "m" => Unit::meters(),
        "kg" => Unit::kilograms(),
        "s" => Unit::seconds(),
        "K" => Unit::kelvin(),
        "A" => Unit::amperes(),
        "mol" => Unit::moles(),
        "cd" => Unit::candelas(),
        "rad" => Unit::radians(),

        _ => {
            return Err(CompileError::new(
                ErrorKind::InvalidUnit,
                span,
                format!("Unknown base unit: {}", name),
            ));
        }
    };
    Ok(unit)
}

/// Multiply two units.
pub fn multiply_units(lhs: &Unit, rhs: &Unit, span: Span) -> Result<Unit, CompileError> {
    if !lhs.is_multiplicative() || !rhs.is_multiplicative() {
        return Err(CompileError::new(
            ErrorKind::InvalidUnit,
            span,
            "Cannot multiply non-multiplicative units (affine/logarithmic)".to_string(),
        ));
    }

    let dims = add_dimensions(lhs.dims(), rhs.dims(), span)?;
    Ok(Unit::new(UnitKind::Multiplicative, dims))
}

/// Divide two units.
pub fn divide_units(
    numerator: &Unit,
    denominator: &Unit,
    span: Span,
) -> Result<Unit, CompileError> {
    if !numerator.is_multiplicative() || !denominator.is_multiplicative() {
        return Err(CompileError::new(
            ErrorKind::InvalidUnit,
            span,
            "Cannot divide non-multiplicative units (affine/logarithmic)".to_string(),
        ));
    }

    let dims = subtract_dimensions(numerator.dims(), denominator.dims(), span)?;
    Ok(Unit::new(UnitKind::Multiplicative, dims))
}

/// Raise a unit to a power.
pub fn power_unit(base: &Unit, exponent: i8, span: Span) -> Result<Unit, CompileError> {
    if !base.is_multiplicative() {
        return Err(CompileError::new(
            ErrorKind::InvalidUnit,
            span,
            "Cannot raise non-multiplicative units to powers (affine/logarithmic)".to_string(),
        ));
    }

    let dims = scale_dimensions(base.dims(), exponent, span)?;
    Ok(Unit::new(UnitKind::Multiplicative, dims))
}

/// Add dimensional exponents (for multiplication).
fn add_dimensions(
    lhs: &UnitDimensions,
    rhs: &UnitDimensions,
    span: Span,
) -> Result<UnitDimensions, CompileError> {
    Ok(UnitDimensions {
        length: checked_i8_op(lhs.length, rhs.length, "length", span, i8::checked_add)?,
        mass: checked_i8_op(lhs.mass, rhs.mass, "mass", span, i8::checked_add)?,
        time: checked_i8_op(lhs.time, rhs.time, "time", span, i8::checked_add)?,
        temperature: checked_i8_op(
            lhs.temperature,
            rhs.temperature,
            "temperature",
            span,
            i8::checked_add,
        )?,
        current: checked_i8_op(lhs.current, rhs.current, "current", span, i8::checked_add)?,
        amount: checked_i8_op(lhs.amount, rhs.amount, "amount", span, i8::checked_add)?,
        luminosity: checked_i8_op(
            lhs.luminosity,
            rhs.luminosity,
            "luminosity",
            span,
            i8::checked_add,
        )?,
        angle: checked_i8_op(lhs.angle, rhs.angle, "angle", span, i8::checked_add)?,
    })
}

/// Subtract dimensional exponents (for division).
fn subtract_dimensions(
    lhs: &UnitDimensions,
    rhs: &UnitDimensions,
    span: Span,
) -> Result<UnitDimensions, CompileError> {
    Ok(UnitDimensions {
        length: checked_i8_op(lhs.length, rhs.length, "length", span, i8::checked_sub)?,
        mass: checked_i8_op(lhs.mass, rhs.mass, "mass", span, i8::checked_sub)?,
        time: checked_i8_op(lhs.time, rhs.time, "time", span, i8::checked_sub)?,
        temperature: checked_i8_op(
            lhs.temperature,
            rhs.temperature,
            "temperature",
            span,
            i8::checked_sub,
        )?,
        current: checked_i8_op(lhs.current, rhs.current, "current", span, i8::checked_sub)?,
        amount: checked_i8_op(lhs.amount, rhs.amount, "amount", span, i8::checked_sub)?,
        luminosity: checked_i8_op(
            lhs.luminosity,
            rhs.luminosity,
            "luminosity",
            span,
            i8::checked_sub,
        )?,
        angle: checked_i8_op(lhs.angle, rhs.angle, "angle", span, i8::checked_sub)?,
    })
}

/// Scale dimensional exponents (for power).
fn scale_dimensions(
    dims: &UnitDimensions,
    scale: i8,
    span: Span,
) -> Result<UnitDimensions, CompileError> {
    Ok(UnitDimensions {
        length: checked_i8_op(dims.length, scale, "length", span, i8::checked_mul)?,
        mass: checked_i8_op(dims.mass, scale, "mass", span, i8::checked_mul)?,
        time: checked_i8_op(dims.time, scale, "time", span, i8::checked_mul)?,
        temperature: checked_i8_op(
            dims.temperature,
            scale,
            "temperature",
            span,
            i8::checked_mul,
        )?,
        current: checked_i8_op(dims.current, scale, "current", span, i8::checked_mul)?,
        amount: checked_i8_op(dims.amount, scale, "amount", span, i8::checked_mul)?,
        luminosity: checked_i8_op(dims.luminosity, scale, "luminosity", span, i8::checked_mul)?,
        angle: checked_i8_op(dims.angle, scale, "angle", span, i8::checked_mul)?,
    })
}

/// Checked i8 arithmetic with dimension name in error.
fn checked_i8_op<F>(a: i8, b: i8, dim_name: &str, span: Span, op: F) -> Result<i8, CompileError>
where
    F: FnOnce(i8, i8) -> Option<i8>,
{
    op(a, b).ok_or_else(|| {
        CompileError::new(
            ErrorKind::InvalidUnit,
            span,
            format!("Dimension exponent overflow for {}", dim_name),
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl_ast::foundation::Span;

    fn test_span() -> Span {
        Span::new(0, 10, 20, 1)
    }

    #[test]
    fn test_resolve_base_units() {
        let span = test_span();
        assert!(resolve_base_unit("m", span).is_ok());
        assert!(resolve_base_unit("kg", span).is_ok());
        assert!(resolve_base_unit("s", span).is_ok());
        assert!(resolve_base_unit("K", span).is_ok());
        assert!(resolve_base_unit("A", span).is_ok());
        assert!(resolve_base_unit("mol", span).is_ok());
        assert!(resolve_base_unit("cd", span).is_ok());
        assert!(resolve_base_unit("rad", span).is_ok());

        // Unknown unit should fail
        let err = resolve_base_unit("xyz", span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
    }

    #[test]
    fn test_unit_multiplication() {
        let m = Unit::meters();
        let s = Unit::seconds();
        let result = multiply_units(&m, &s, test_span()).unwrap();
        assert_eq!(result.dims().length, 1);
        assert_eq!(result.dims().time, 1);
    }

    #[test]
    fn test_unit_division() {
        let m = Unit::meters();
        let s = Unit::seconds();
        let result = divide_units(&m, &s, test_span()).unwrap();
        assert_eq!(result.dims().length, 1);
        assert_eq!(result.dims().time, -1);
    }

    #[test]
    fn test_unit_power() {
        let m = Unit::meters();
        let result = power_unit(&m, 2, test_span()).unwrap();
        assert_eq!(result.dims().length, 2);
    }

    #[test]
    fn test_dimension_overflow_fails() {
        let span = test_span();
        let left = Unit::new(UnitKind::Multiplicative, UnitDimensions::METER);
        let right = power_unit(&left, 120, span).unwrap();
        let err = multiply_units(&right, &right, span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("overflow"));
    }
}
