//! Mathematical constants registry for the DSL.
//!
//! Provides a centralized registry of mathematical constants that can be used
//! in expressions and type range specifications. Constants are registered with
//! both ASCII and Unicode variants for user convenience.

use std::collections::HashMap;
use std::sync::LazyLock;

/// A mathematical constant entry with its value and metadata.
#[derive(Debug, Clone, Copy)]
pub struct MathConstDef {
    /// The numeric value of the constant.
    pub value: f64,
    /// Short description of the constant.
    pub description: &'static str,
}

/// Registry of all mathematical constants.
///
/// Maps constant names (both ASCII and Unicode variants) to their definitions.
/// This is the single source of truth for all built-in math constants.
pub static MATH_CONSTS: LazyLock<HashMap<&'static str, MathConstDef>> = LazyLock::new(|| {
    let mut m = HashMap::new();

    // Pi - ratio of circumference to diameter
    let pi = MathConstDef {
        value: std::f64::consts::PI,
        description: "Pi (3.14159...): ratio of circumference to diameter",
    };
    m.insert("PI", pi);
    m.insert("π", pi);

    // Tau - ratio of circumference to radius (2*pi)
    let tau = MathConstDef {
        value: std::f64::consts::TAU,
        description: "Tau (6.28318...): ratio of circumference to radius (2*pi)",
    };
    m.insert("TAU", tau);
    m.insert("τ", tau);

    // E - Euler's number
    let e = MathConstDef {
        value: std::f64::consts::E,
        description: "Euler's number (2.71828...): base of natural logarithm",
    };
    m.insert("E", e);
    m.insert("ℯ", e);

    // Phi - Golden ratio
    let phi = MathConstDef {
        value: 1.618_033_988_749_895,
        description: "Golden ratio (1.61803...): (1 + sqrt(5)) / 2",
    };
    m.insert("PHI", phi);
    m.insert("φ", phi);

    // Common physics/engineering constants that might be useful in ranges
    // These use only ASCII names to keep it simple

    // Sqrt(2)
    m.insert(
        "SQRT2",
        MathConstDef {
            value: std::f64::consts::SQRT_2,
            description: "Square root of 2 (1.41421...)",
        },
    );

    // Sqrt(3)
    m.insert(
        "SQRT3",
        MathConstDef {
            value: 1.732_050_807_568_877,
            description: "Square root of 3 (1.73205...)",
        },
    );

    // 1/Pi
    m.insert(
        "FRAC_1_PI",
        MathConstDef {
            value: std::f64::consts::FRAC_1_PI,
            description: "1/π (0.31831...)",
        },
    );

    // 2/Pi
    m.insert(
        "FRAC_2_PI",
        MathConstDef {
            value: std::f64::consts::FRAC_2_PI,
            description: "2/π (0.63662...)",
        },
    );

    // Pi/2
    m.insert(
        "FRAC_PI_2",
        MathConstDef {
            value: std::f64::consts::FRAC_PI_2,
            description: "π/2 (1.57080...)",
        },
    );

    // Pi/3
    m.insert(
        "FRAC_PI_3",
        MathConstDef {
            value: std::f64::consts::FRAC_PI_3,
            description: "π/3 (1.04720...)",
        },
    );

    // Pi/4
    m.insert(
        "FRAC_PI_4",
        MathConstDef {
            value: std::f64::consts::FRAC_PI_4,
            description: "π/4 (0.78540...)",
        },
    );

    // Pi/6
    m.insert(
        "FRAC_PI_6",
        MathConstDef {
            value: std::f64::consts::FRAC_PI_6,
            description: "π/6 (0.52360...)",
        },
    );

    // Ln(2)
    m.insert(
        "LN_2",
        MathConstDef {
            value: std::f64::consts::LN_2,
            description: "Natural logarithm of 2 (0.69315...)",
        },
    );

    // Ln(10)
    m.insert(
        "LN_10",
        MathConstDef {
            value: std::f64::consts::LN_10,
            description: "Natural logarithm of 10 (2.30259...)",
        },
    );

    m
});

/// Look up a math constant by name.
///
/// Returns the constant's value if found, or None if the name is not recognized.
pub fn lookup(name: &str) -> Option<f64> {
    MATH_CONSTS.get(name).map(|def| def.value)
}

/// Get all registered constant names (for parser generation).
pub fn all_names() -> impl Iterator<Item = &'static str> {
    MATH_CONSTS.keys().copied()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lookup_pi() {
        assert!((lookup("PI").unwrap() - std::f64::consts::PI).abs() < 1e-15);
        assert!((lookup("π").unwrap() - std::f64::consts::PI).abs() < 1e-15);
    }

    #[test]
    fn test_lookup_tau() {
        assert!((lookup("TAU").unwrap() - std::f64::consts::TAU).abs() < 1e-15);
        assert!((lookup("τ").unwrap() - std::f64::consts::TAU).abs() < 1e-15);
    }

    #[test]
    fn test_lookup_unknown() {
        assert!(lookup("UNKNOWN").is_none());
    }
}
