//! Unit Dimensional Analysis
//!
//! This module provides dimensional analysis for physics safety by representing
//! units in terms of SI base dimensions and enforcing dimensional consistency
//! during compilation.
//!
//! # SI Base Dimensions
//!
//! All physical units can be expressed as combinations of 7 SI base dimensions:
//! - Length (L) - meters (m)
//! - Mass (M) - kilograms (kg)
//! - Time (T) - seconds (s)
//! - Electric current (I) - amperes (A)
//! - Temperature (Θ) - kelvin (K)
//! - Amount of substance (N) - moles (mol)
//! - Luminous intensity (J) - candelas (cd)
//!
//! # Dimensional Algebra
//!
//! Units follow algebraic rules:
//! - Multiplication: dimensions add (e.g., m × m = m²)
//! - Division: dimensions subtract (e.g., m / s = m·s⁻¹)
//! - Powers: dimensions scale (e.g., m² = m·m)
//! - Addition/Subtraction: requires identical dimensions
//!
//! # Example
//!
//! ```ignore
//! let velocity = Unit::parse("m/s")?;      // L¹·T⁻¹
//! let time = Unit::parse("s")?;            // T¹
//! let distance = velocity.multiply(&time); // L¹ (m/s × s = m)
//! ```

use std::fmt;

use serde::{Deserialize, Serialize};

/// Represents a physical unit as a combination of SI base dimensions.
///
/// Each dimension is stored as an integer exponent. Dimensionless quantities
/// have all exponents equal to zero.
///
/// # Examples
///
/// - Velocity (m/s): length=1, time=-1, others=0
/// - Force (N = kg·m/s²): mass=1, length=1, time=-2
/// - Energy (J = kg·m²/s²): mass=1, length=2, time=-2
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Unit {
    /// Length dimension exponent (L) - base unit: meter (m)
    pub length: i8,
    /// Mass dimension exponent (M) - base unit: kilogram (kg)
    pub mass: i8,
    /// Time dimension exponent (T) - base unit: second (s)
    pub time: i8,
    /// Electric current dimension exponent (I) - base unit: ampere (A)
    pub current: i8,
    /// Temperature dimension exponent (Θ) - base unit: kelvin (K)
    pub temperature: i8,
    /// Amount of substance dimension exponent (N) - base unit: mole (mol)
    pub amount: i8,
    /// Luminous intensity dimension exponent (J) - base unit: candela (cd)
    pub luminosity: i8,
    /// Angle dimension (for radians) - treated as dimensionless but tracked
    pub angle: i8,
}

impl Unit {
    /// Dimensionless unit constant (all exponents zero).
    pub const DIMENSIONLESS: Unit = Unit {
        length: 0,
        mass: 0,
        time: 0,
        current: 0,
        temperature: 0,
        amount: 0,
        luminosity: 0,
        angle: 0,
    };

    /// Creates a dimensionless unit (all exponents zero).
    ///
    /// Dimensionless quantities include:
    /// - Pure numbers
    /// - Ratios of like quantities
    /// - Arguments to transcendental functions
    #[inline]
    pub const fn dimensionless() -> Self {
        Self {
            length: 0,
            mass: 0,
            time: 0,
            current: 0,
            temperature: 0,
            amount: 0,
            luminosity: 0,
            angle: 0,
        }
    }

    /// Returns true if this unit is dimensionless.
    #[inline]
    pub fn is_dimensionless(&self) -> bool {
        self.length == 0
            && self.mass == 0
            && self.time == 0
            && self.current == 0
            && self.temperature == 0
            && self.amount == 0
            && self.luminosity == 0
            && self.angle == 0
    }

    /// Returns true if this unit represents an angle (radians).
    #[inline]
    pub fn is_angle(&self) -> bool {
        self.angle != 0
            && self.length == 0
            && self.mass == 0
            && self.time == 0
            && self.current == 0
            && self.temperature == 0
            && self.amount == 0
            && self.luminosity == 0
    }

    /// Multiplies two units by adding their dimension exponents.
    ///
    /// # Examples
    ///
    /// - m × m = m² (length: 1 + 1 = 2)
    /// - m × s⁻¹ = m/s (length: 1, time: -1)
    #[inline]
    pub fn multiply(&self, other: &Unit) -> Unit {
        Unit {
            length: self.length + other.length,
            mass: self.mass + other.mass,
            time: self.time + other.time,
            current: self.current + other.current,
            temperature: self.temperature + other.temperature,
            amount: self.amount + other.amount,
            luminosity: self.luminosity + other.luminosity,
            angle: self.angle + other.angle,
        }
    }

    /// Divides two units by subtracting their dimension exponents.
    ///
    /// # Examples
    ///
    /// - m / s = m·s⁻¹ (velocity)
    /// - J / s = W (power)
    #[inline]
    pub fn divide(&self, other: &Unit) -> Unit {
        Unit {
            length: self.length - other.length,
            mass: self.mass - other.mass,
            time: self.time - other.time,
            current: self.current - other.current,
            temperature: self.temperature - other.temperature,
            amount: self.amount - other.amount,
            luminosity: self.luminosity - other.luminosity,
            angle: self.angle - other.angle,
        }
    }

    /// Raises a unit to an integer power by scaling its exponents.
    ///
    /// # Examples
    ///
    /// - m² = m.power(2) (length: 2)
    /// - s⁻² = s.power(-2) (time: -2)
    #[inline]
    pub fn power(&self, exp: i8) -> Unit {
        Unit {
            length: self.length * exp,
            mass: self.mass * exp,
            time: self.time * exp,
            current: self.current * exp,
            temperature: self.temperature * exp,
            amount: self.amount * exp,
            luminosity: self.luminosity * exp,
            angle: self.angle * exp,
        }
    }

    /// Returns the inverse (reciprocal) of this unit.
    ///
    /// # Examples
    ///
    /// - (m/s)⁻¹ = s/m
    /// - s⁻¹ → s
    #[inline]
    pub fn inverse(&self) -> Unit {
        self.power(-1)
    }

    /// Takes the square root of a unit (halves all exponents).
    ///
    /// Returns `None` if any exponent is odd (cannot take exact root).
    ///
    /// # Examples
    ///
    /// - sqrt(m²) = m
    /// - sqrt(m) = None (cannot represent m^0.5)
    pub fn sqrt(&self) -> Option<Unit> {
        if self.length % 2 != 0
            || self.mass % 2 != 0
            || self.time % 2 != 0
            || self.current % 2 != 0
            || self.temperature % 2 != 0
            || self.amount % 2 != 0
            || self.luminosity % 2 != 0
            || self.angle % 2 != 0
        {
            return None;
        }

        Some(Unit {
            length: self.length / 2,
            mass: self.mass / 2,
            time: self.time / 2,
            current: self.current / 2,
            temperature: self.temperature / 2,
            amount: self.amount / 2,
            luminosity: self.luminosity / 2,
            angle: self.angle / 2,
        })
    }

    /// Parses a unit string into dimensional representation.
    pub fn parse(unit_str: &str) -> Option<Self> {
        let unit_str = unit_str.trim();

        if unit_str.is_empty() || unit_str == "1" {
            return Some(Unit::dimensionless());
        }

        // Extremely simple recursive descent parser for units with parentheses
        Self::parse_recursive(unit_str)
    }

    fn parse_recursive(s: &str) -> Option<Unit> {
        let s = s.trim();
        if s.is_empty() {
            return Some(Unit::dimensionless());
        }

        // Handle parentheses
        if s.starts_with('(') && s.ends_with(')') {
            return Self::parse_recursive(&s[1..s.len() - 1]);
        }

        // Handle division (right-associative for multiple slashes: a/b/c = a/(b*c))
        if let Some(pos) = s.find('/') {
            let num = &s[..pos];
            let den = &s[pos + 1..];
            let u_num = Self::parse_recursive(num)?;
            let u_den = Self::parse_recursive(den)?;
            return Some(u_num.divide(&u_den));
        }

        // Handle products
        if s.contains('·') || s.contains('*') {
            let parts: Vec<&str> = s.split(&['·', '*'][..]).collect();
            let mut result = Unit::dimensionless();
            for part in parts {
                let unit = Self::parse_recursive(part)?;
                result = result.multiply(&unit);
            }
            return Some(result);
        }

        // Handle space-separated products (e.g., "kg m")
        if s.contains(' ') {
            let parts: Vec<&str> = s.split_whitespace().collect();
            let mut result = Unit::dimensionless();
            for part in parts {
                let unit = Self::parse_recursive(part)?;
                result = result.multiply(&unit);
            }
            return Some(result);
        }

        // Fallback to single unit
        Self::parse_single(s)
    }

    /// Parses a single unit with optional exponent like "m", "m²", "m^2", "s⁻¹"
    fn parse_single(unit_str: &str) -> Option<Unit> {
        let unit_str = unit_str.trim();
        if unit_str.is_empty() || unit_str == "1" {
            return Some(Unit::dimensionless());
        }

        // Extract base unit and exponent
        let (base, exp) = Self::extract_exponent(unit_str)?;

        // Look up base unit
        let base_unit = Self::lookup_base_unit(base)?;

        Some(base_unit.power(exp))
    }

    /// Extracts base unit and exponent from strings like "m²", "m^2", "m", "s⁻¹"
    fn extract_exponent(unit_str: &str) -> Option<(&str, i8)> {
        // Check for superscript exponents: ², ³, ⁻¹, etc.
        if let Some((base, exp)) = Self::parse_superscript_exponent(unit_str) {
            return Some((base, exp));
        }

        // Check for caret notation: m^2, s^-1
        if let Some(pos) = unit_str.find('^') {
            let base = &unit_str[..pos];
            let exp_str = &unit_str[pos + 1..];
            let exp: i8 = exp_str.parse().ok()?;
            return Some((base, exp));
        }

        // No exponent, default to 1
        Some((unit_str, 1))
    }

    /// Parses superscript exponents like ², ³, ⁻¹, ⁻²
    fn parse_superscript_exponent(unit_str: &str) -> Option<(&str, i8)> {
        // Find where superscripts start
        let superscripts = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹', '⁻'];

        let super_start = unit_str.find(|c| superscripts.contains(&c))?;
        let base = &unit_str[..super_start];
        let exp_str = &unit_str[super_start..];

        let exp = Self::parse_superscript_number(exp_str)?;

        Some((base, exp))
    }

    /// Converts superscript digits to a number
    fn parse_superscript_number(s: &str) -> Option<i8> {
        let mut negative = false;
        let mut num: i8 = 0;

        for c in s.chars() {
            match c {
                '⁻' => negative = true,
                '⁰' => num = num * 10,
                '¹' => num = num * 10 + 1,
                '²' => num = num * 10 + 2,
                '³' => num = num * 10 + 3,
                '⁴' => num = num * 10 + 4,
                '⁵' => num = num * 10 + 5,
                '⁶' => num = num * 10 + 6,
                '⁷' => num = num * 10 + 7,
                '⁸' => num = num * 10 + 8,
                '⁹' => num = num * 10 + 9,
                _ => return None,
            }
        }

        // If no digits after ⁻, assume -1
        if negative && num == 0 {
            num = 1;
        }

        Some(if negative { -num } else { num })
    }

    /// Static table of unit definitions.
    /// Each entry: (aliases, unit)
    const UNIT_TABLE: &'static [(&'static [&'static str], Unit)] = &[
        // SI Base Units - Length
        (
            &["m", "meter", "meters"],
            Unit {
                length: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // SI Base Units - Mass
        (
            &["kg", "kilogram", "kilograms"],
            Unit {
                mass: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        (
            &["g", "gram", "grams"],
            Unit {
                mass: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // SI Base Units - Time
        (
            &["s", "sec", "second", "seconds"],
            Unit {
                time: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // SI Base Units - Electric current
        (
            &["A", "amp", "ampere", "amperes"],
            Unit {
                current: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // SI Base Units - Temperature
        (
            &["K", "kelvin"],
            Unit {
                temperature: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // SI Base Units - Amount of substance
        (
            &["mol", "mole", "moles"],
            Unit {
                amount: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // SI Base Units - Luminous intensity
        (
            &["cd", "candela", "candelas"],
            Unit {
                luminosity: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Angle
        (
            &["rad", "radian", "radians"],
            Unit {
                angle: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        (
            &["sr", "steradian", "steradians"],
            Unit {
                angle: 2,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Frequency (Hz = s⁻¹)
        (
            &["Hz", "hertz"],
            Unit {
                time: -1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Force (N = kg·m/s²)
        (
            &["N", "newton", "newtons"],
            Unit {
                mass: 1,
                length: 1,
                time: -2,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Pressure (Pa = N/m²)
        (
            &["Pa", "pascal", "pascals"],
            Unit {
                mass: 1,
                length: -1,
                time: -2,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Energy (J = N·m)
        (
            &["J", "joule", "joules"],
            Unit {
                mass: 1,
                length: 2,
                time: -2,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Power (W = J/s)
        (
            &["W", "watt", "watts"],
            Unit {
                mass: 1,
                length: 2,
                time: -3,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Electric charge (C = A·s)
        (
            &["C", "coulomb", "coulombs"],
            Unit {
                current: 1,
                time: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Electric potential (V = W/A)
        (
            &["V", "volt", "volts"],
            Unit {
                mass: 1,
                length: 2,
                time: -3,
                current: -1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Capacitance (F = C/V)
        (
            &["F", "farad", "farads"],
            Unit {
                mass: -1,
                length: -2,
                time: 4,
                current: 2,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Resistance (Ω = V/A)
        (
            &["Ω", "ohm", "ohms"],
            Unit {
                mass: 1,
                length: 2,
                time: -3,
                current: -2,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Magnetic flux (Wb = V·s)
        (
            &["Wb", "weber", "webers"],
            Unit {
                mass: 1,
                length: 2,
                time: -2,
                current: -1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Magnetic flux density (T = Wb/m²)
        (
            &["T", "tesla", "teslas"],
            Unit {
                mass: 1,
                time: -2,
                current: -1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Inductance (H = Wb/A)
        (
            &["H", "henry", "henries"],
            Unit {
                mass: 1,
                length: 2,
                time: -2,
                current: -2,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Luminous flux (lm = cd·sr)
        (
            &["lm", "lumen", "lumens"],
            Unit {
                luminosity: 1,
                angle: 2,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Illuminance (lx = lm/m²)
        (
            &["lx", "lux"],
            Unit {
                luminosity: 1,
                length: -2,
                angle: 2,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Angle (degrees)
        (
            &["deg", "degree", "degrees", "°"],
            Unit {
                angle: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Time units
        (
            &["yr", "year", "years", "a"],
            Unit {
                time: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        (
            &["d", "day", "days"],
            Unit {
                time: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        (
            &["h", "hr", "hour", "hours"],
            Unit {
                time: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        (
            &["min", "minute", "minutes"],
            Unit {
                time: 1,
                ..Unit::DIMENSIONLESS
            },
        ),
        // Dimensionless
        (&["%", "percent"], Unit::DIMENSIONLESS),
    ];

    /// Looks up a base unit by name and returns its dimensional representation.
    fn lookup_base_unit(name: &str) -> Option<Unit> {
        // Search the static table for matching alias
        for (aliases, unit) in Self::UNIT_TABLE {
            if aliases.contains(&name) {
                return Some(*unit);
            }
        }
        None
    }

    /// Converts the unit to its canonical string representation.
    pub fn to_string_canonical(&self) -> String {
        if self.is_dimensionless() {
            return "1".to_string();
        }

        let mut parts = Vec::new();
        let mut neg_parts = Vec::new();

        // Positive exponents
        Self::append_dimension(&mut parts, "m", self.length);
        Self::append_dimension(&mut parts, "kg", self.mass);
        Self::append_dimension(&mut parts, "s", self.time);
        Self::append_dimension(&mut parts, "A", self.current);
        Self::append_dimension(&mut parts, "K", self.temperature);
        Self::append_dimension(&mut parts, "mol", self.amount);
        Self::append_dimension(&mut parts, "cd", self.luminosity);
        Self::append_dimension(&mut parts, "rad", self.angle);

        // Negative exponents (for division notation)
        Self::append_neg_dimension(&mut neg_parts, "m", self.length);
        Self::append_neg_dimension(&mut neg_parts, "kg", self.mass);
        Self::append_neg_dimension(&mut neg_parts, "s", self.time);
        Self::append_neg_dimension(&mut neg_parts, "A", self.current);
        Self::append_neg_dimension(&mut neg_parts, "K", self.temperature);
        Self::append_neg_dimension(&mut neg_parts, "mol", self.amount);
        Self::append_neg_dimension(&mut neg_parts, "cd", self.luminosity);
        Self::append_neg_dimension(&mut neg_parts, "rad", self.angle);

        let num = if parts.is_empty() {
            "1".to_string()
        } else {
            parts.join("·")
        };

        if neg_parts.is_empty() {
            num
        } else {
            format!("{}/{}", num, neg_parts.join("·"))
        }
    }

    fn append_dimension(parts: &mut Vec<String>, name: &str, exp: i8) {
        if exp > 0 {
            if exp == 1 {
                parts.push(name.to_string());
            } else {
                parts.push(format!("{}{}", name, Self::superscript(exp)));
            }
        }
    }

    fn append_neg_dimension(parts: &mut Vec<String>, name: &str, exp: i8) {
        if exp < 0 {
            let abs_exp = exp.abs();
            if abs_exp == 1 {
                parts.push(name.to_string());
            } else {
                parts.push(format!("{}{}", name, Self::superscript(abs_exp)));
            }
        }
    }

    fn superscript(n: i8) -> String {
        let chars: Vec<char> = n
            .to_string()
            .chars()
            .map(|c| match c {
                '0' => '⁰',
                '1' => '¹',
                '2' => '²',
                '3' => '³',
                '4' => '⁴',
                '5' => '⁵',
                '6' => '⁶',
                '7' => '⁷',
                '8' => '⁸',
                '9' => '⁹',
                '-' => '⁻',
                _ => c,
            })
            .collect();
        chars.into_iter().collect()
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_canonical())
    }
}

/// Errors that can occur during dimensional analysis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DimensionError {
    /// Addition or subtraction of incompatible units.
    IncompatibleUnits {
        /// The expected unit.
        expected: Unit,
        /// The actual unit found.
        found: Unit,
        /// Description of the operation.
        operation: String,
    },
    /// Function requires dimensionless argument.
    RequiresDimensionless {
        /// The function name.
        function: String,
        /// The actual unit of the argument.
        found: Unit,
    },
    /// Function requires angle argument.
    RequiresAngle {
        /// The function name.
        function: String,
        /// The actual unit of the argument.
        found: Unit,
    },
    /// Cannot take square root of unit with odd exponents.
    InvalidSqrt {
        /// The unit that cannot be rooted.
        unit: Unit,
    },
    /// Unknown unit string.
    UnknownUnit {
        /// The unit string that could not be parsed.
        unit_str: String,
    },
}

impl fmt::Display for DimensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimensionError::IncompatibleUnits {
                expected,
                found,
                operation,
            } => {
                write!(
                    f,
                    "incompatible units in {}: expected {}, found {}",
                    operation, expected, found
                )
            }
            DimensionError::RequiresDimensionless { function, found } => {
                write!(
                    f,
                    "function '{}' requires dimensionless argument, got {}",
                    function, found
                )
            }
            DimensionError::RequiresAngle { function, found } => {
                write!(
                    f,
                    "function '{}' requires angle argument, got {}",
                    function, found
                )
            }
            DimensionError::InvalidSqrt { unit } => {
                write!(
                    f,
                    "cannot take square root of {}: would result in fractional exponents",
                    unit
                )
            }
            DimensionError::UnknownUnit { unit_str } => {
                write!(f, "unknown unit: '{}'", unit_str)
            }
        }
    }
}

impl std::error::Error for DimensionError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensionless() {
        let d = Unit::dimensionless();
        assert!(d.is_dimensionless());
        assert_eq!(d.to_string(), "1");
    }

    #[test]
    fn test_parse_base_units() {
        // Length
        let m = Unit::parse("m").unwrap();
        assert_eq!(m.length, 1);
        assert!(m.mass == 0 && m.time == 0);

        // Mass
        let kg = Unit::parse("kg").unwrap();
        assert_eq!(kg.mass, 1);

        // Time
        let s = Unit::parse("s").unwrap();
        assert_eq!(s.time, 1);

        // Temperature
        let k = Unit::parse("K").unwrap();
        assert_eq!(k.temperature, 1);
    }

    #[test]
    fn test_parse_derived_units() {
        // Newton: kg·m/s²
        let n = Unit::parse("N").unwrap();
        assert_eq!(n.mass, 1);
        assert_eq!(n.length, 1);
        assert_eq!(n.time, -2);

        // Pascal: kg/(m·s²)
        let pa = Unit::parse("Pa").unwrap();
        assert_eq!(pa.mass, 1);
        assert_eq!(pa.length, -1);
        assert_eq!(pa.time, -2);

        // Watt: kg·m²/s³
        let w = Unit::parse("W").unwrap();
        assert_eq!(w.mass, 1);
        assert_eq!(w.length, 2);
        assert_eq!(w.time, -3);
    }

    #[test]
    fn test_parse_compound_units() {
        // Velocity: m/s
        let v = Unit::parse("m/s").unwrap();
        assert_eq!(v.length, 1);
        assert_eq!(v.time, -1);

        // Acceleration: m/s²
        let a = Unit::parse("m/s²").unwrap();
        assert_eq!(a.length, 1);
        assert_eq!(a.time, -2);

        // Power per area: W/m²
        let flux = Unit::parse("W/m²").unwrap();
        assert_eq!(flux.mass, 1);
        assert_eq!(flux.length, 0); // 2 - 2 = 0
        assert_eq!(flux.time, -3);
    }

    #[test]
    fn test_parse_exponents() {
        // m²
        let m2 = Unit::parse("m²").unwrap();
        assert_eq!(m2.length, 2);

        // m^2
        let m2_caret = Unit::parse("m^2").unwrap();
        assert_eq!(m2_caret.length, 2);

        // s⁻¹
        let hz = Unit::parse("s⁻¹").unwrap();
        assert_eq!(hz.time, -1);

        // s^-2
        let s_neg2 = Unit::parse("s^-2").unwrap();
        assert_eq!(s_neg2.time, -2);
    }

    #[test]
    fn test_multiply() {
        let m = Unit::parse("m").unwrap();
        let s = Unit::parse("s").unwrap();

        // m × s = m·s
        let ms = m.multiply(&s);
        assert_eq!(ms.length, 1);
        assert_eq!(ms.time, 1);

        // m × m = m²
        let m2 = m.multiply(&m);
        assert_eq!(m2.length, 2);
    }

    #[test]
    fn test_divide() {
        let m = Unit::parse("m").unwrap();
        let s = Unit::parse("s").unwrap();

        // m / s = m·s⁻¹ (velocity)
        let v = m.divide(&s);
        assert_eq!(v.length, 1);
        assert_eq!(v.time, -1);
    }

    #[test]
    fn test_power() {
        let m = Unit::parse("m").unwrap();

        // m² = m.power(2)
        let m2 = m.power(2);
        assert_eq!(m2.length, 2);

        // m⁻¹ = m.power(-1)
        let m_inv = m.power(-1);
        assert_eq!(m_inv.length, -1);
    }

    #[test]
    fn test_sqrt() {
        let m2 = Unit::parse("m²").unwrap();
        let m = m2.sqrt().unwrap();
        assert_eq!(m.length, 1);

        // Cannot sqrt m (odd exponent)
        let m1 = Unit::parse("m").unwrap();
        assert!(m1.sqrt().is_none());
    }

    #[test]
    fn test_dimensional_consistency() {
        // Force = mass × acceleration
        let kg = Unit::parse("kg").unwrap();
        let m_s2 = Unit::parse("m/s²").unwrap();
        let force = kg.multiply(&m_s2);

        let newton = Unit::parse("N").unwrap();
        assert_eq!(force, newton);

        // Energy = force × distance
        let m = Unit::parse("m").unwrap();
        let energy = newton.multiply(&m);

        let joule = Unit::parse("J").unwrap();
        assert_eq!(energy, joule);
    }

    #[test]
    fn test_angle_detection() {
        let rad = Unit::parse("rad").unwrap();
        assert!(rad.is_angle());
        assert!(!rad.is_dimensionless());

        let m = Unit::parse("m").unwrap();
        assert!(!m.is_angle());
    }

    #[test]
    fn test_parse_complex_units() {
        // Pascal-seconds
        let pas = Unit::parse("Pa*s").unwrap();
        assert_eq!(pas.mass, 1);
        assert_eq!(pas.length, -1);
        assert_eq!(pas.time, -1);

        // Gravitational constant: m³/(kg*s²)
        let g = Unit::parse("m^3/(kg*s^2)").unwrap();
        assert_eq!(g.length, 3);
        assert_eq!(g.mass, -1);
        assert_eq!(g.time, -2);

        // Unicode superscripts
        let g_uni = Unit::parse("m³/(kg*s²)").unwrap();
        assert_eq!(g_uni, g);
    }
}
