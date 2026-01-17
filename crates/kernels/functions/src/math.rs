//! Pure Math Functions
//!
//! Standard mathematical functions that don't depend on dt.

use continuum_kernel_macros::kernel_fn;

// === Basic math ===

/// Absolute value: `abs(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn abs(x: f64) -> f64 {
    x.abs()
}

/// Square root: `sqrt(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = Sqrt(0)
)]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Power: `pow(base, exp)`
///
/// Raises base to the power of exp. Note: Unit derivation for pow is complex
/// (base^exp) and requires exp to be a compile-time constant for proper typing.
/// This function uses a simple type constraint.
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0)],
    unit_in = [UnitAny, UnitDimensionless],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Clamp: `clamp(value, min, max)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0), SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0), UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.clamp(min, max)
}

/// Linear interpolation: `lerp(a, b, t)` → `a + t * (b - a)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0), SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0), UnitDimensionless],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Linear interpolation (alias): `mix(a, b, t)` → `a + t * (b - a)`
///
/// Same as `lerp`, provided for GLSL/shader compatibility.
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0), SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0), UnitDimensionless],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn mix(a: f64, b: f64, t: f64) -> f64 {
    a + t * (b - a)
}

/// Step function: `step(edge, x)` → 0.0 if x < edge, else 1.0
///
/// Sharp transition at edge. Common in shader programming.
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn step(edge: f64, x: f64) -> f64 {
    if x < edge { 0.0 } else { 1.0 }
}

/// Smooth Hermite interpolation: `smoothstep(e0, e1, x)`
///
/// Returns 0 if x ≤ e0, 1 if x ≥ e1, smooth interpolation between.
/// Uses cubic Hermite polynomial: 3t² - 2t³
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0), SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0), UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Saturate: `saturate(x)` → clamp to [0, 1]
///
/// Equivalent to `clamp(x, 0.0, 1.0)`. Common in shader programming.
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn saturate(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

// === Trigonometry ===

/// Sine: `sin(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [Angle],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine: `cos(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [Angle],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Tangent: `tan(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [Angle],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

/// Arctangent: `atan(x)`
///
/// Returns angle in radians. Output unit is Angle, which is a dimensionless
/// quantity in the unit system but marked separately for type safety.
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// Arctangent with two parameters: `atan2(y, x)`
///
/// Returns angle in radians. Output unit is dimensionless (angle is treated
/// as dimensionless in the unit system but with type safety).
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0)],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// Arcsine: `asin(x)` → angle in radians whose sine is x
///
/// # Panics
///
/// Returns NaN if x is outside [-1, 1]
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn asin(x: f64) -> f64 {
    x.asin()
}

/// Arccosine: `acos(x)` → angle in radians whose cosine is x
///
/// # Panics
///
/// Returns NaN if x is outside [-1, 1]
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn acos(x: f64) -> f64 {
    x.acos()
}

// === Hyperbolic ===

/// Hyperbolic sine: `sinh(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn sinh(x: f64) -> f64 {
    x.sinh()
}

/// Hyperbolic cosine: `cosh(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn cosh(x: f64) -> f64 {
    x.cosh()
}

/// Hyperbolic tangent: `tanh(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn tanh(x: f64) -> f64 {
    x.tanh()
}

/// Inverse hyperbolic sine: `asinh(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn asinh(x: f64) -> f64 {
    x.asinh()
}

/// Inverse hyperbolic cosine: `acosh(x)` → non-negative value whose hyperbolic cosine is x
///
/// # Panics
///
/// Returns NaN if x < 1
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn acosh(x: f64) -> f64 {
    x.acosh()
}

/// Inverse hyperbolic tangent: `atanh(x)`
///
/// # Panics
///
/// Returns NaN if x is outside (-1, 1)
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitDimensionless],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn atanh(x: f64) -> f64 {
    x.atanh()
}

// === Exponential / Logarithmic ===

/// Exponential: `exp(x)` → `e^x`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// Natural log: `ln(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn ln(x: f64) -> f64 {
    x.ln()
}

/// Log base 10: `log10(x)`
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn log10(x: f64) -> f64 {
    x.log10()
}

/// Logarithm with arbitrary base: `log(x, base)`
///
/// Computes log_base(x) = ln(x) / ln(base)
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0)],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn log(x: f64, base: f64) -> f64 {
    x.ln() / base.ln()
}

/// Log base 2: `log2(x)` → log₂(x)
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitDimensionless],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn log2(x: f64) -> f64 {
    x.log2()
}

/// Cube root: `cbrt(x)` → ∛x
///
/// Takes the cube root. Unit derivation for cbrt (x^(1/3)) is not directly
/// expressible in the type system, so we use UnitDerivSameAs(0) as a conservative constraint.
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// Fractional part: `fract(x)` → x - floor(x)
///
/// Returns the fractional component of a number.
/// For example: fract(3.7) = 0.7
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn fract(x: f64) -> f64 {
    x.fract()
}

// === Rounding ===

/// Floor: `floor(x)` → largest integer ≤ x
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn floor(x: f64) -> f64 {
    x.floor()
}

/// Ceiling: `ceil(x)` → smallest integer ≥ x
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn ceil(x: f64) -> f64 {
    x.ceil()
}

/// Round: `round(x)` → nearest integer (half-way rounds away from zero)
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn round(x: f64) -> f64 {
    x.round()
}

/// Truncate: `trunc(x)` → integer part (round toward zero)
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn trunc(x: f64) -> f64 {
    x.trunc()
}

/// Sign: `sign(x)` → -1.0 if x < 0, 0.0 if x == 0, 1.0 if x > 0
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = Dimensionless
)]
pub fn sign(x: f64) -> f64 {
    if x == 0.0 { 0.0 } else { x.signum() }
}

/// Modulo: `mod(a, b)` → `a % b` (always positive)
#[kernel_fn(
    name = "mod",
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn modulo(a: f64, b: f64) -> f64 {
    ((a % b) + b) % b
}

/// Wrap value to range: `wrap(value, min, max)` → value wrapped to [min, max)
///
/// Useful for cyclic values like angles (0 to 2π) or phases.
///
/// # Panics
///
/// Panics if `min` or `max` are not finite, or if `max <= min`.
#[kernel_fn(
    namespace = "maths",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0), SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0), UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn wrap(value: f64, min: f64, max: f64) -> f64 {
    assert!(min.is_finite(), "wrap: min must be finite, got {}", min);
    assert!(max.is_finite(), "wrap: max must be finite, got {}", max);
    assert!(
        max > min,
        "wrap: max must be greater than min, got range [{}, {})",
        min,
        max
    );

    let range = max - min;
    let offset = value - min;
    min + ((offset % range) + range) % range
}

// === Variadic ===

/// Minimum: `min(a, b, ...)`
#[kernel_fn(
    namespace = "maths",
    variadic,
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn min(args: &[f64]) -> f64 {
    args.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// Maximum: `max(a, b, ...)`
#[kernel_fn(
    namespace = "maths",
    variadic,
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn max(args: &[f64]) -> f64 {
    args.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Sum: `sum(a, b, ...)`
#[kernel_fn(
    namespace = "maths",
    variadic,
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn sum(args: &[f64]) -> f64 {
    args.iter().sum()
}

#[cfg(test)]
mod tests {
    use continuum_kernel_registry::{
        Arity, Value, eval_in_namespace, get_in_namespace, is_known_in,
    };

    #[test]
    fn test_pure_functions_registered() {
        assert!(is_known_in("maths", "abs"));
        assert!(is_known_in("maths", "sqrt"));
        assert!(is_known_in("maths", "sin"));
        assert!(is_known_in("maths", "cos"));
        assert!(is_known_in("maths", "tan"));
        assert!(is_known_in("maths", "atan"));
        assert!(is_known_in("maths", "atan2"));
        assert!(is_known_in("maths", "asin"));
        assert!(is_known_in("maths", "acos"));
        assert!(is_known_in("maths", "sinh"));
        assert!(is_known_in("maths", "cosh"));
        assert!(is_known_in("maths", "tanh"));
        assert!(is_known_in("maths", "asinh"));
        assert!(is_known_in("maths", "acosh"));
        assert!(is_known_in("maths", "atanh"));
        assert!(is_known_in("maths", "exp"));
        assert!(is_known_in("maths", "ln"));
        assert!(is_known_in("maths", "log10"));
        assert!(is_known_in("maths", "log"));
        assert!(is_known_in("maths", "log2"));
        assert!(is_known_in("maths", "cbrt"));
        assert!(is_known_in("maths", "fract"));
        assert!(is_known_in("maths", "pow"));
        assert!(is_known_in("maths", "clamp"));
        assert!(is_known_in("maths", "lerp"));
        assert!(is_known_in("maths", "mix"));
        assert!(is_known_in("maths", "step"));
        assert!(is_known_in("maths", "smoothstep"));
        assert!(is_known_in("maths", "saturate"));
        assert!(is_known_in("maths", "floor"));
        assert!(is_known_in("maths", "ceil"));
        assert!(is_known_in("maths", "round"));
        assert!(is_known_in("maths", "trunc"));
        assert!(is_known_in("maths", "sign"));
    }

    #[test]
    fn test_variadic_functions_registered() {
        assert!(is_known_in("maths", "min"));
        assert!(is_known_in("maths", "max"));
        assert!(is_known_in("maths", "sum"));

        assert_eq!(
            get_in_namespace("maths", "min").unwrap().arity,
            Arity::Variadic
        );
        assert_eq!(
            get_in_namespace("maths", "max").unwrap().arity,
            Arity::Variadic
        );
        assert_eq!(
            get_in_namespace("maths", "sum").unwrap().arity,
            Arity::Variadic
        );
    }

    #[test]
    fn test_pure_dont_require_dt() {
        assert!(!get_in_namespace("maths", "abs").unwrap().requires_dt());
        assert!(!get_in_namespace("maths", "sqrt").unwrap().requires_dt());
        assert!(!get_in_namespace("maths", "sin").unwrap().requires_dt());
    }

    #[test]
    fn test_eval_abs() {
        let args = [Value::Scalar(-5.0)];
        assert_eq!(
            eval_in_namespace("maths", "abs", &args, 1.0),
            Some(Value::Scalar(5.0))
        );
    }

    #[test]
    fn test_eval_sqrt() {
        let args = [Value::Scalar(16.0)];
        assert_eq!(
            eval_in_namespace("maths", "sqrt", &args, 1.0),
            Some(Value::Scalar(4.0))
        );
    }

    #[test]
    fn test_eval_sum() {
        let args = [
            Value::Scalar(1.0),
            Value::Scalar(2.0),
            Value::Scalar(3.0),
            Value::Scalar(4.0),
        ];
        assert_eq!(
            eval_in_namespace("maths", "sum", &args, 1.0),
            Some(Value::Scalar(10.0))
        );
    }

    #[test]
    fn test_eval_min_max() {
        let args = [Value::Scalar(3.0), Value::Scalar(1.0), Value::Scalar(2.0)];
        assert_eq!(
            eval_in_namespace("maths", "min", &args, 1.0),
            Some(Value::Scalar(1.0))
        );
        assert_eq!(
            eval_in_namespace("maths", "max", &args, 1.0),
            Some(Value::Scalar(3.0))
        );
    }

    #[test]
    fn test_eval_clamp() {
        let args1 = [Value::Scalar(5.0), Value::Scalar(0.0), Value::Scalar(10.0)];
        assert_eq!(
            eval_in_namespace("maths", "clamp", &args1, 1.0),
            Some(Value::Scalar(5.0))
        );
        let args2 = [Value::Scalar(-5.0), Value::Scalar(0.0), Value::Scalar(10.0)];
        assert_eq!(
            eval_in_namespace("maths", "clamp", &args2, 1.0),
            Some(Value::Scalar(0.0))
        );
        let args3 = [Value::Scalar(15.0), Value::Scalar(0.0), Value::Scalar(10.0)];
        assert_eq!(
            eval_in_namespace("maths", "clamp", &args3, 1.0),
            Some(Value::Scalar(10.0))
        );
    }

    #[test]
    fn test_eval_wrap() {
        use std::f64::consts::PI;

        // Value within range stays the same
        let args1 = [Value::Scalar(1.0), Value::Scalar(0.0), Value::Scalar(10.0)];
        assert_eq!(
            eval_in_namespace("maths", "wrap", &args1, 1.0),
            Some(Value::Scalar(1.0))
        );

        // Value above max wraps around
        let args2 = [Value::Scalar(12.0), Value::Scalar(0.0), Value::Scalar(10.0)];
        assert_eq!(
            eval_in_namespace("maths", "wrap", &args2, 1.0),
            Some(Value::Scalar(2.0))
        );

        // Value below min wraps around
        let args3 = [Value::Scalar(-2.0), Value::Scalar(0.0), Value::Scalar(10.0)];
        assert_eq!(
            eval_in_namespace("maths", "wrap", &args3, 1.0),
            Some(Value::Scalar(8.0))
        );

        // Angle wrapping (0 to 2π)
        let tau = 2.0 * PI;
        let args4 = [
            Value::Scalar(tau + 1.0),
            Value::Scalar(0.0),
            Value::Scalar(tau),
        ];
        let result = eval_in_namespace("maths", "wrap", &args4, 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // Negative angle
        let args5 = [Value::Scalar(-PI), Value::Scalar(0.0), Value::Scalar(tau)];
        let result = eval_in_namespace("maths", "wrap", &args5, 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - PI).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "wrap: max must be greater than min")]
    fn test_eval_wrap_invalid_range() {
        let args = [Value::Scalar(1.0), Value::Scalar(10.0), Value::Scalar(0.0)];
        eval_in_namespace("maths", "wrap", &args, 1.0);
    }

    #[test]
    #[should_panic(expected = "wrap: max must be greater than min")]
    fn test_eval_wrap_zero_range() {
        let args = [Value::Scalar(1.0), Value::Scalar(10.0), Value::Scalar(10.0)];
        eval_in_namespace("maths", "wrap", &args, 1.0);
    }

    #[test]
    fn test_eval_floor() {
        assert_eq!(
            eval_in_namespace("maths", "floor", &[Value::Scalar(3.7)], 1.0),
            Some(Value::Scalar(3.0))
        );
        assert_eq!(
            eval_in_namespace("maths", "floor", &[Value::Scalar(-3.7)], 1.0),
            Some(Value::Scalar(-4.0))
        );
    }

    #[test]
    fn test_eval_ceil() {
        assert_eq!(
            eval_in_namespace("maths", "ceil", &[Value::Scalar(3.2)], 1.0),
            Some(Value::Scalar(4.0))
        );
        assert_eq!(
            eval_in_namespace("maths", "ceil", &[Value::Scalar(-3.2)], 1.0),
            Some(Value::Scalar(-3.0))
        );
    }

    #[test]
    fn test_eval_round() {
        assert_eq!(
            eval_in_namespace("maths", "round", &[Value::Scalar(3.5)], 1.0),
            Some(Value::Scalar(4.0))
        );
        assert_eq!(
            eval_in_namespace("maths", "round", &[Value::Scalar(3.4)], 1.0),
            Some(Value::Scalar(3.0))
        );
        assert_eq!(
            eval_in_namespace("maths", "round", &[Value::Scalar(-3.5)], 1.0),
            Some(Value::Scalar(-4.0))
        );
    }

    #[test]
    fn test_eval_trunc() {
        assert_eq!(
            eval_in_namespace("maths", "trunc", &[Value::Scalar(3.7)], 1.0),
            Some(Value::Scalar(3.0))
        );
        assert_eq!(
            eval_in_namespace("maths", "trunc", &[Value::Scalar(-3.7)], 1.0),
            Some(Value::Scalar(-3.0))
        );
    }

    #[test]
    fn test_eval_sign() {
        assert_eq!(
            eval_in_namespace("maths", "sign", &[Value::Scalar(5.0)], 1.0),
            Some(Value::Scalar(1.0))
        );
        assert_eq!(
            eval_in_namespace("maths", "sign", &[Value::Scalar(-5.0)], 1.0),
            Some(Value::Scalar(-1.0))
        );
        assert_eq!(
            eval_in_namespace("maths", "sign", &[Value::Scalar(0.0)], 1.0),
            Some(Value::Scalar(0.0))
        );
    }

    #[test]
    fn test_eval_asin() {
        use std::f64::consts::PI;

        // asin(0) = 0
        let result = eval_in_namespace("maths", "asin", &[Value::Scalar(0.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // asin(1) = π/2
        let result = eval_in_namespace("maths", "asin", &[Value::Scalar(1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - PI / 2.0).abs() < 1e-10);

        // asin(-1) = -π/2
        let result = eval_in_namespace("maths", "asin", &[Value::Scalar(-1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result + PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_acos() {
        use std::f64::consts::PI;

        // acos(1) = 0
        let result = eval_in_namespace("maths", "acos", &[Value::Scalar(1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // acos(0) = π/2
        let result = eval_in_namespace("maths", "acos", &[Value::Scalar(0.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - PI / 2.0).abs() < 1e-10);

        // acos(-1) = π
        let result = eval_in_namespace("maths", "acos", &[Value::Scalar(-1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - PI).abs() < 1e-10);
    }

    #[test]
    fn test_eval_log() {
        // log(8, 2) = 3  (log base 2 of 8)
        let result = eval_in_namespace(
            "maths",
            "log",
            &[Value::Scalar(8.0), Value::Scalar(2.0)],
            1.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();
        assert!((result - 3.0).abs() < 1e-10);

        // log(1000, 10) = 3  (log base 10 of 1000)
        let result = eval_in_namespace(
            "maths",
            "log",
            &[Value::Scalar(1000.0), Value::Scalar(10.0)],
            1.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();
        assert!((result - 3.0).abs() < 1e-10);

        // log(27, 3) = 3  (log base 3 of 27)
        let result = eval_in_namespace(
            "maths",
            "log",
            &[Value::Scalar(27.0), Value::Scalar(3.0)],
            1.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();
        assert!((result - 3.0).abs() < 1e-10);

        // log(1, x) = 0 for any base
        let result = eval_in_namespace(
            "maths",
            "log",
            &[Value::Scalar(1.0), Value::Scalar(5.0)],
            1.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_sinh() {
        // sinh(0) = 0
        let result = eval_in_namespace("maths", "sinh", &[Value::Scalar(0.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // sinh(1) ≈ 1.175201
        let result = eval_in_namespace("maths", "sinh", &[Value::Scalar(1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 1.175201193643801).abs() < 1e-10);

        // sinh(-1) = -sinh(1)
        let result = eval_in_namespace("maths", "sinh", &[Value::Scalar(-1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result + 1.175201193643801).abs() < 1e-10);
    }

    #[test]
    fn test_eval_cosh() {
        // cosh(0) = 1
        let result = eval_in_namespace("maths", "cosh", &[Value::Scalar(0.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // cosh(1) ≈ 1.543081
        let result = eval_in_namespace("maths", "cosh", &[Value::Scalar(1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 1.5430806348152437).abs() < 1e-10);

        // cosh(-1) = cosh(1) (even function)
        let result = eval_in_namespace("maths", "cosh", &[Value::Scalar(-1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 1.5430806348152437).abs() < 1e-10);
    }

    #[test]
    fn test_eval_tanh() {
        // tanh(0) = 0
        let result = eval_in_namespace("maths", "tanh", &[Value::Scalar(0.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // tanh(1) ≈ 0.761594
        let result = eval_in_namespace("maths", "tanh", &[Value::Scalar(1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.7615941559557649).abs() < 1e-10);

        // tanh(-1) = -tanh(1) (odd function)
        let result = eval_in_namespace("maths", "tanh", &[Value::Scalar(-1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result + 0.7615941559557649).abs() < 1e-10);
    }

    #[test]
    fn test_eval_asinh() {
        // asinh(0) = 0
        let result = eval_in_namespace("maths", "asinh", &[Value::Scalar(0.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // asinh(sinh(1)) = 1
        let sinh_1 = 1.175201193643801f64;
        let result = eval_in_namespace("maths", "asinh", &[Value::Scalar(sinh_1)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // asinh(-x) = -asinh(x) (odd function)
        let result = eval_in_namespace("maths", "asinh", &[Value::Scalar(-sinh_1)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_acosh() {
        // acosh(1) = 0
        let result = eval_in_namespace("maths", "acosh", &[Value::Scalar(1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // acosh(cosh(1)) = 1
        let cosh_1 = 1.5430806348152437f64;
        let result = eval_in_namespace("maths", "acosh", &[Value::Scalar(cosh_1)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // acosh(2) ≈ 1.316958
        let result = eval_in_namespace("maths", "acosh", &[Value::Scalar(2.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 1.3169578969248166).abs() < 1e-10);
    }

    #[test]
    fn test_eval_atanh() {
        // atanh(0) = 0
        let result = eval_in_namespace("maths", "atanh", &[Value::Scalar(0.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // atanh(tanh(0.5)) = 0.5
        let tanh_half = 0.46211715726000974f64;
        let result = eval_in_namespace("maths", "atanh", &[Value::Scalar(tanh_half)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.5).abs() < 1e-10);

        // atanh(-x) = -atanh(x) (odd function)
        let result = eval_in_namespace("maths", "atanh", &[Value::Scalar(-tanh_half)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result + 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eval_step() {
        // x < edge → 0.0
        assert_eq!(
            eval_in_namespace(
                "maths",
                "step",
                &[Value::Scalar(5.0), Value::Scalar(3.0)],
                1.0
            ),
            Some(Value::Scalar(0.0))
        );
        // x >= edge → 1.0
        assert_eq!(
            eval_in_namespace(
                "maths",
                "step",
                &[Value::Scalar(5.0), Value::Scalar(5.0)],
                1.0
            ),
            Some(Value::Scalar(1.0))
        );
        assert_eq!(
            eval_in_namespace(
                "maths",
                "step",
                &[Value::Scalar(5.0), Value::Scalar(7.0)],
                1.0
            ),
            Some(Value::Scalar(1.0))
        );
    }

    #[test]
    fn test_eval_smoothstep() {
        // x <= edge0 → 0.0
        assert_eq!(
            eval_in_namespace(
                "maths",
                "smoothstep",
                &[Value::Scalar(0.0), Value::Scalar(1.0), Value::Scalar(-1.0)],
                1.0
            ),
            Some(Value::Scalar(0.0))
        );
        // x >= edge1 → 1.0
        assert_eq!(
            eval_in_namespace(
                "maths",
                "smoothstep",
                &[Value::Scalar(0.0), Value::Scalar(1.0), Value::Scalar(2.0)],
                1.0
            ),
            Some(Value::Scalar(1.0))
        );
        // midpoint (x=0.5) → 0.5
        let result = eval_in_namespace(
            "maths",
            "smoothstep",
            &[Value::Scalar(0.0), Value::Scalar(1.0), Value::Scalar(0.5)],
            1.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();
        assert!((result - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_eval_saturate() {
        // x < 0 → 0
        assert_eq!(
            eval_in_namespace("maths", "saturate", &[Value::Scalar(-5.0)], 1.0),
            Some(Value::Scalar(0.0))
        );
        // 0 <= x <= 1 → x
        assert_eq!(
            eval_in_namespace("maths", "saturate", &[Value::Scalar(0.5)], 1.0),
            Some(Value::Scalar(0.5))
        );
        // x > 1 → 1
        assert_eq!(
            eval_in_namespace("maths", "saturate", &[Value::Scalar(5.0)], 1.0),
            Some(Value::Scalar(1.0))
        );
    }

    #[test]
    fn test_eval_fract() {
        // fract(3.7) = 0.7
        let result = eval_in_namespace("maths", "fract", &[Value::Scalar(3.7)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.7).abs() < 1e-10);

        // fract(5.0) = 0.0
        let result = eval_in_namespace("maths", "fract", &[Value::Scalar(5.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.0).abs() < 1e-10);

        // fract(-2.3) = -0.3 (Rust's fract returns negative for negative inputs)
        let result = eval_in_namespace("maths", "fract", &[Value::Scalar(-2.3)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result + 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_eval_cbrt() {
        // cbrt(8) = 2
        let result = eval_in_namespace("maths", "cbrt", &[Value::Scalar(8.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 2.0).abs() < 1e-10);

        // cbrt(27) = 3
        let result = eval_in_namespace("maths", "cbrt", &[Value::Scalar(27.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 3.0).abs() < 1e-10);

        // cbrt(-8) = -2 (cube root of negative number is negative)
        let result = eval_in_namespace("maths", "cbrt", &[Value::Scalar(-8.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result + 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_log2() {
        // log2(8) = 3
        let result = eval_in_namespace("maths", "log2", &[Value::Scalar(8.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 3.0).abs() < 1e-10);

        // log2(1024) = 10
        let result = eval_in_namespace("maths", "log2", &[Value::Scalar(1024.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 10.0).abs() < 1e-10);

        // log2(1) = 0
        let result = eval_in_namespace("maths", "log2", &[Value::Scalar(1.0)], 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_eval_lerp_and_mix() {
        // lerp(0, 10, 0.5) = 5
        let result = eval_in_namespace(
            "maths",
            "lerp",
            &[Value::Scalar(0.0), Value::Scalar(10.0), Value::Scalar(0.5)],
            1.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();
        assert!((result - 5.0).abs() < 1e-10);

        // mix should produce the same result as lerp
        let result_mix = eval_in_namespace(
            "maths",
            "mix",
            &[Value::Scalar(0.0), Value::Scalar(10.0), Value::Scalar(0.5)],
            1.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();
        assert!((result_mix - 5.0).abs() < 1e-10);

        // lerp at t=0 returns a
        let result = eval_in_namespace(
            "maths",
            "lerp",
            &[Value::Scalar(3.0), Value::Scalar(7.0), Value::Scalar(0.0)],
            1.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();
        assert!((result - 3.0).abs() < 1e-10);

        // lerp at t=1 returns b
        let result = eval_in_namespace(
            "maths",
            "lerp",
            &[Value::Scalar(3.0), Value::Scalar(7.0), Value::Scalar(1.0)],
            1.0,
        )
        .unwrap()
        .as_scalar()
        .unwrap();
        assert!((result - 7.0).abs() < 1e-10);
    }
}
