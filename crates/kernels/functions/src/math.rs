//! Pure Math Functions
//!
//! Standard mathematical functions that don't depend on dt.

use continuum_kernel_macros::kernel_fn;

// === Basic math ===

/// Absolute value: `abs(x)`
#[kernel_fn(name = "abs")]
pub fn abs(x: f64) -> f64 {
    x.abs()
}

/// Square root: `sqrt(x)`
#[kernel_fn(name = "sqrt")]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// Power: `pow(base, exp)`
#[kernel_fn(name = "pow")]
pub fn pow(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Clamp: `clamp(value, min, max)`
#[kernel_fn(name = "clamp")]
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.clamp(min, max)
}

// === Trigonometry ===

/// Sine: `sin(x)`
#[kernel_fn(name = "sin")]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// Cosine: `cos(x)`
#[kernel_fn(name = "cos")]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

// === Exponential / Logarithmic ===

/// Exponential: `exp(x)` → `e^x`
#[kernel_fn(name = "exp")]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// Natural log: `ln(x)`
#[kernel_fn(name = "ln")]
pub fn ln(x: f64) -> f64 {
    x.ln()
}

/// Log base 10: `log10(x)`
#[kernel_fn(name = "log10")]
pub fn log10(x: f64) -> f64 {
    x.log10()
}

/// Modulo: `mod(a, b)` → `a % b` (always positive)
#[kernel_fn(name = "mod")]
pub fn modulo(a: f64, b: f64) -> f64 {
    ((a % b) + b) % b
}

// === Variadic ===

/// Minimum: `min(a, b, ...)`
#[kernel_fn(name = "min", variadic)]
pub fn min(args: &[f64]) -> f64 {
    args.iter().cloned().fold(f64::INFINITY, f64::min)
}

/// Maximum: `max(a, b, ...)`
#[kernel_fn(name = "max", variadic)]
pub fn max(args: &[f64]) -> f64 {
    args.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Sum: `sum(a, b, ...)`
#[kernel_fn(name = "sum", variadic)]
pub fn sum(args: &[f64]) -> f64 {
    args.iter().sum()
}

#[cfg(test)]
mod tests {
    use continuum_kernel_registry::{Arity, eval, get, is_known};

    #[test]
    fn test_pure_functions_registered() {
        assert!(is_known("abs"));
        assert!(is_known("sqrt"));
        assert!(is_known("sin"));
        assert!(is_known("cos"));
        assert!(is_known("exp"));
        assert!(is_known("ln"));
        assert!(is_known("log10"));
        assert!(is_known("pow"));
        assert!(is_known("clamp"));
    }

    #[test]
    fn test_variadic_functions_registered() {
        assert!(is_known("min"));
        assert!(is_known("max"));
        assert!(is_known("sum"));

        assert_eq!(get("min").unwrap().arity, Arity::Variadic);
        assert_eq!(get("max").unwrap().arity, Arity::Variadic);
        assert_eq!(get("sum").unwrap().arity, Arity::Variadic);
    }

    #[test]
    fn test_pure_dont_require_dt() {
        assert!(!get("abs").unwrap().requires_dt());
        assert!(!get("sqrt").unwrap().requires_dt());
        assert!(!get("sin").unwrap().requires_dt());
    }

    #[test]
    fn test_eval_abs() {
        assert_eq!(eval("abs", &[-5.0], 1.0), Some(5.0));
    }

    #[test]
    fn test_eval_sqrt() {
        assert_eq!(eval("sqrt", &[16.0], 1.0), Some(4.0));
    }

    #[test]
    fn test_eval_sum() {
        assert_eq!(eval("sum", &[1.0, 2.0, 3.0, 4.0], 1.0), Some(10.0));
    }

    #[test]
    fn test_eval_min_max() {
        assert_eq!(eval("min", &[3.0, 1.0, 2.0], 1.0), Some(1.0));
        assert_eq!(eval("max", &[3.0, 1.0, 2.0], 1.0), Some(3.0));
    }

    #[test]
    fn test_eval_clamp() {
        assert_eq!(eval("clamp", &[5.0, 0.0, 10.0], 1.0), Some(5.0));
        assert_eq!(eval("clamp", &[-5.0, 0.0, 10.0], 1.0), Some(0.0));
        assert_eq!(eval("clamp", &[15.0, 0.0, 10.0], 1.0), Some(10.0));
    }
}
