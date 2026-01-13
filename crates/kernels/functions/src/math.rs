//! Pure Math Functions
//!
//! Standard mathematical functions that don't depend on dt.

use continuum_kernel_macros::kernel_fn;

// === Basic math ===

/// Absolute value: `abs(x)`
#[kernel_fn(namespace = "maths")]
pub fn abs(x: f64) -> f64 {
    x.abs()
}

/// Square root: `sqrt(x)`
#[kernel_fn(namespace = "maths")]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Power: `pow(base, exp)`
#[kernel_fn(namespace = "maths")]
pub fn pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Clamp: `clamp(value, min, max)`
#[kernel_fn(namespace = "maths")]
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.clamp(min, max)
}

// === Trigonometry ===

/// Sine: `sin(x)`
#[kernel_fn(namespace = "maths")]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine: `cos(x)`
#[kernel_fn(namespace = "maths")]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// Tangent: `tan(x)`
#[kernel_fn(namespace = "maths")]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

/// Arctangent: `atan(x)`
#[kernel_fn(namespace = "maths")]
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// Arctangent with two parameters: `atan2(y, x)`
#[kernel_fn(namespace = "maths")]
pub fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

// === Exponential / Logarithmic ===

/// Exponential: `exp(x)` → `e^x`
#[kernel_fn(namespace = "maths")]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// Natural log: `ln(x)`
#[kernel_fn(namespace = "maths")]
pub fn ln(x: f64) -> f64 {
    x.ln()
}

/// Log base 10: `log10(x)`
#[kernel_fn(namespace = "maths")]
pub fn log10(x: f64) -> f64 {
    x.log10()
}

/// Modulo: `mod(a, b)` → `a % b` (always positive)
#[kernel_fn(name = "mod", namespace = "maths")]
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
#[kernel_fn(namespace = "maths")]
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
#[kernel_fn(namespace = "maths", variadic)]
pub fn min(args: &[f64]) -> f64 {
    args.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// Maximum: `max(a, b, ...)`
#[kernel_fn(namespace = "maths", variadic)]
pub fn max(args: &[f64]) -> f64 {
    args.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Sum: `sum(a, b, ...)`
#[kernel_fn(namespace = "maths", variadic)]
pub fn sum(args: &[f64]) -> f64 {
    args.iter().sum()
}

#[cfg(test)]
mod tests {
    use continuum_kernel_registry::{Arity, eval_in_namespace, get_in_namespace, is_known_in};

    #[test]
    fn test_pure_functions_registered() {
        assert!(is_known_in("maths", "abs"));
        assert!(is_known_in("maths", "sqrt"));
        assert!(is_known_in("maths", "sin"));
        assert!(is_known_in("maths", "cos"));
        assert!(is_known_in("maths", "tan"));
        assert!(is_known_in("maths", "atan"));
        assert!(is_known_in("maths", "atan2"));
        assert!(is_known_in("maths", "exp"));
        assert!(is_known_in("maths", "ln"));
        assert!(is_known_in("maths", "log10"));
        assert!(is_known_in("maths", "pow"));
        assert!(is_known_in("maths", "clamp"));
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
        assert_eq!(eval_in_namespace("maths", "abs", &[-5.0], 1.0), Some(5.0));
    }

    #[test]
    fn test_eval_sqrt() {
        assert_eq!(eval_in_namespace("maths", "sqrt", &[16.0], 1.0), Some(4.0));
    }

    #[test]
    fn test_eval_sum() {
        assert_eq!(
            eval_in_namespace("maths", "sum", &[1.0, 2.0, 3.0, 4.0], 1.0),
            Some(10.0)
        );
    }

    #[test]
    fn test_eval_min_max() {
        assert_eq!(
            eval_in_namespace("maths", "min", &[3.0, 1.0, 2.0], 1.0),
            Some(1.0)
        );
        assert_eq!(
            eval_in_namespace("maths", "max", &[3.0, 1.0, 2.0], 1.0),
            Some(3.0)
        );
    }

    #[test]
    fn test_eval_clamp() {
        assert_eq!(
            eval_in_namespace("maths", "clamp", &[5.0, 0.0, 10.0], 1.0),
            Some(5.0)
        );
        assert_eq!(
            eval_in_namespace("maths", "clamp", &[-5.0, 0.0, 10.0], 1.0),
            Some(0.0)
        );
        assert_eq!(
            eval_in_namespace("maths", "clamp", &[15.0, 0.0, 10.0], 1.0),
            Some(10.0)
        );
    }

    #[test]
    fn test_eval_wrap() {
        use std::f64::consts::PI;

        // Value within range stays the same
        assert_eq!(
            eval_in_namespace("maths", "wrap", &[1.0, 0.0, 10.0], 1.0),
            Some(1.0)
        );

        // Value above max wraps around
        assert_eq!(
            eval_in_namespace("maths", "wrap", &[12.0, 0.0, 10.0], 1.0),
            Some(2.0)
        );

        // Value below min wraps around
        assert_eq!(
            eval_in_namespace("maths", "wrap", &[-2.0, 0.0, 10.0], 1.0),
            Some(8.0)
        );

        // Angle wrapping (0 to 2π)
        let tau = 2.0 * PI;
        let result = eval_in_namespace("maths", "wrap", &[tau + 1.0, 0.0, tau], 1.0).unwrap();
        assert!((result - 1.0).abs() < 1e-10);

        // Negative angle
        let result = eval_in_namespace("maths", "wrap", &[-PI, 0.0, tau], 1.0).unwrap();
        assert!((result - PI).abs() < 1e-10);
    }

    #[test]
    #[should_panic(expected = "wrap: max must be greater than min")]
    fn test_eval_wrap_invalid_range() {
        eval_in_namespace("maths", "wrap", &[1.0, 10.0, 0.0], 1.0);
    }

    #[test]
    #[should_panic(expected = "wrap: max must be greater than min")]
    fn test_eval_wrap_zero_range() {
        eval_in_namespace("maths", "wrap", &[1.0, 10.0, 10.0], 1.0);
    }
}
