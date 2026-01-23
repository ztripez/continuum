//! Unit resolution and dimensional arithmetic for the Continuum DSL.
//!
//! This module implements the physical unit system, including:
//! - **Base unit resolution** - Mapping symbols (m, kg, s) to dimensions.
//! - **SI prefix support** - Metric prefixes (k, M, G, m, μ, n, etc.)
//! - **Compound unit arithmetic** - Multiplication, division, and powers.
//! - **Dimensional analysis** - Ensuring physical consistency during compilation.
//! - **Overflow protection** - Checked arithmetic for dimensional exponents.

use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::{Span, Unit, UnitDimensions, UnitKind};
use continuum_cdsl_ast::UnitExpr;

/// SI metric prefixes with their scale factors (powers of 10).
///
/// Ordered by scale for deterministic iteration.
/// Note: ASCII 'u' is accepted as fallback for 'μ' (micro).
const SI_PREFIXES: &[(&str, i32)] = &[
    ("Y", 24),  // yotta
    ("Z", 21),  // zetta
    ("E", 18),  // exa
    ("P", 15),  // peta
    ("T", 12),  // tera
    ("G", 9),   // giga
    ("M", 6),   // mega
    ("k", 3),   // kilo
    ("h", 2),   // hecto
    ("da", 1),  // deca
    ("d", -1),  // deci
    ("c", -2),  // centi
    ("m", -3),  // milli
    ("μ", -6),  // micro
    ("u", -6),  // micro (ASCII fallback)
    ("n", -9),  // nano
    ("p", -12), // pico
    ("f", -15), // femto
    ("a", -18), // atto
    ("z", -21), // zepto
    ("y", -24), // yocto
];

/// Units that must NOT be decomposed with prefix parsing.
///
/// These are checked before attempting prefix decomposition to avoid
/// ambiguity (e.g., 'm' = meter, not milli-something).
const RESERVED_UNITS: &[&str] = &[
    // SI base units
    "m",   // meter (not milli)
    "kg",  // kilogram (already has prefix)
    "s",   // second
    "K",   // kelvin
    "A",   // ampere (but 'a' is atto prefix)
    "mol", // mole (not milli-ol)
    "cd",  // candela (not centi-d)
    "rad", // radian
    // SI derived units
    "N",  // newton
    "J",  // joule
    "W",  // watt
    "Pa", // pascal (not peta-a)
    // Time units (non-SI but recognized)
    "yr",  // year
    "day", // day
    // Mass convenience
    "g", // gram
];

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

/// Try to parse a unit name with SI prefix.
///
/// Returns `Some((prefix_scale, base_unit_name))` if a valid prefix is found,
/// otherwise `None`.
///
/// # Strategy
///
/// Try each prefix from longest to shortest to avoid ambiguity
/// (e.g., "da" before "d").
fn try_parse_prefix(name: &str) -> Option<(f64, &str)> {
    // Try two-character prefixes first (da)
    if name.len() > 2 {
        if let Some(&(_, exp)) = SI_PREFIXES
            .iter()
            .find(|(p, _)| p.len() == 2 && name.starts_with(p))
        {
            let base = &name[2..];
            return Some((10.0_f64.powi(exp), base));
        }
    }

    // Try one-character prefixes
    if name.len() > 1 {
        if let Some(&(_, exp)) = SI_PREFIXES
            .iter()
            .find(|(p, _)| p.len() == 1 && name.starts_with(p))
        {
            let base = &name[1..];
            return Some((10.0_f64.powi(exp), base));
        }
    }

    None
}

/// Try to resolve a unit name without prefix parsing (exact match only).
///
/// Returns `Some(Unit)` if the name is a recognized base or derived unit,
/// otherwise `None`.
fn try_exact_base_unit(name: &str) -> Option<Unit> {
    match name {
        // SI base units
        "m" => Some(Unit::meters()),
        "kg" => Some(Unit::kilograms()),
        "s" => Some(Unit::seconds()),
        "K" => Some(Unit::kelvin()),
        "A" => Some(Unit::amperes()),
        "mol" => Some(Unit::moles()),
        "cd" => Some(Unit::candelas()),
        "rad" => Some(Unit::radians()),

        // SI derived units
        "N" => Some(Unit::newtons()),
        "J" => Some(Unit::joules()),
        "W" => Some(Unit::watts()),
        "Pa" => Some(Unit::pascals()),

        // Non-SI but recognized units
        "g" => Some(Unit::grams()),  // mass (for prefix convenience)
        "yr" => Some(Unit::years()), // time (Julian year)
        "day" => Some(Unit::days()), // time

        _ => None,
    }
}

/// Resolve a base unit name to a Unit with SI prefix support.
///
/// # Resolution Strategy
///
/// 1. Try exact match first (handles reserved units: m, mol, Pa, cd, etc.)
/// 2. If exact match fails, try SI prefix parsing
/// 3. If neither works, return error
///
/// # Examples
///
/// ```ignore
/// resolve_base_unit("m", span)    // -> meters (exact match, not milli)
/// resolve_base_unit("km", span)   // -> meters with scale=1000.0
/// resolve_base_unit("Myr", span)  // -> years with scale=1e6
/// resolve_base_unit("mol", span)  // -> moles (exact match, not milli-ol)
/// ```
pub fn resolve_base_unit(name: &str, span: Span) -> Result<Unit, CompileError> {
    // 1. Try exact match first (reserved units)
    if let Some(unit) = try_exact_base_unit(name) {
        return Ok(unit);
    }

    // 2. Try prefix parsing (only if not reserved)
    if !RESERVED_UNITS.contains(&name) {
        if let Some((prefix_scale, base_name)) = try_parse_prefix(name) {
            if let Some(base_unit) = try_exact_base_unit(base_name) {
                // Apply prefix scale to base unit
                return Ok(Unit::new(
                    *base_unit.kind(),
                    *base_unit.dims(),
                    base_unit.scale() * prefix_scale,
                ));
            }
        }
    }

    // 3. Unknown unit
    Err(CompileError::new(
        ErrorKind::InvalidUnit,
        span,
        format!("Unknown base unit: {}", name),
    ))
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
    let scale = lhs.scale() * rhs.scale();
    Ok(Unit::new(UnitKind::Multiplicative, dims, scale))
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
    let scale = numerator.scale() / denominator.scale();
    Ok(Unit::new(UnitKind::Multiplicative, dims, scale))
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
    let scale = base.scale().powi(exponent as i32);
    Ok(Unit::new(UnitKind::Multiplicative, dims, scale))
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

    #[test]
    fn test_unit_power_zero() {
        // Zero power should produce dimensionless unit
        let m = Unit::meters();
        let result = power_unit(&m, 0, test_span()).unwrap();
        assert_eq!(result, Unit::DIMENSIONLESS);
        assert_eq!(result.dims().length, 0);
        assert_eq!(result.dims().mass, 0);
    }

    #[test]
    fn test_unit_power_negative() {
        // Negative power should invert dimensions
        let m = Unit::meters();
        let result = power_unit(&m, -1, test_span()).unwrap();
        assert_eq!(result.dims().length, -1);

        // m^-2 should give length = -2
        let result2 = power_unit(&m, -2, test_span()).unwrap();
        assert_eq!(result2.dims().length, -2);
    }

    #[test]
    fn test_multiply_non_multiplicative_unit_fails() {
        // Attempting to multiply affine units should fail
        let span = test_span();
        let affine = Unit::new(
            UnitKind::Affine { offset: 273.15 },
            UnitDimensions::DIMENSIONLESS,
        );
        let mult = Unit::meters();

        let err = multiply_units(&affine, &mult, span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("non-multiplicative") || err.message.contains("affine"));
    }

    #[test]
    fn test_divide_non_multiplicative_unit_fails() {
        // Attempting to divide affine units should fail
        let span = test_span();
        let affine = Unit::new(
            UnitKind::Affine { offset: 273.15 },
            UnitDimensions::DIMENSIONLESS,
        );
        let mult = Unit::meters();

        let err = divide_units(&affine, &mult, span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("non-multiplicative") || err.message.contains("affine"));

        // Also test numerator being affine
        let err2 = divide_units(&mult, &affine, span).unwrap_err();
        assert_eq!(err2.kind, ErrorKind::InvalidUnit);
    }

    #[test]
    fn test_power_non_multiplicative_unit_fails() {
        // Attempting to raise affine unit to power should fail
        let span = test_span();
        let affine = Unit::new(UnitKind::Affine { offset: 273.15 }, UnitDimensions::METER);

        let err = power_unit(&affine, 2, span).unwrap_err();
        assert_eq!(err.kind, ErrorKind::InvalidUnit);
        assert!(err.message.contains("non-multiplicative") || err.message.contains("affine"));
    }
}
