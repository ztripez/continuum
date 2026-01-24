//! Unit dimensional analysis with affine and logarithmic support
//!
//! This module provides compile-time dimensional analysis for physics safety.
//! Units are represented in terms of SI base dimensions with three kinds:
//!
//! - **Multiplicative** — standard SI units (m, kg, s, etc.)
//! - **Affine** — temperature scales with offsets (°C, °F)
//! - **Logarithmic** — logarithmic scales (dB, pH, etc.)
//!
//! # Unit Algebra by Kind
//!
//! | Operation | Multiplicative | Affine | Logarithmic |
//! |-----------|---------------|--------|-------------|
//! | `a + b` | ✓ same dims | ✗ forbidden | ✗ forbidden |
//! | `a - b` | ✓ same dims | ✓ → Multiplicative (delta) | ✗ forbidden |
//! | `a * k` | ✓ | ✗ forbidden | ✗ forbidden |
//! | `a / k` | ✓ | ✗ forbidden | ✗ forbidden |
//! | `a == b` | ✓ | ✓ | ✓ |
//! | `a < b` | ✓ | ✓ | ✓ |
//!
//! # Examples
//!
//! ```rust
//! # use continuum_kernel_types::unit::*;
//! // Multiplicative units
//! let velocity = Unit::meters().divide(&Unit::seconds()).unwrap();
//! assert_eq!(velocity.dims().length, 1);
//! assert_eq!(velocity.dims().time, -1);
//!
//! // Affine units (temperature)
//! let celsius = Unit::celsius();
//! assert!(matches!(celsius.kind(), UnitKind::Affine { .. }));
//!
//! // Logarithmic units
//! let decibels = Unit::decibels();
//! assert!(matches!(decibels.kind(), UnitKind::Logarithmic { .. }));
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;

/// A physical unit with dimensional analysis and kind classification.
///
/// Units combine dimensional exponents (SI base dimensions) with a kind
/// that determines algebraic rules and a scale factor relative to the
/// SI coherent unit.
///
/// # Scale Factor
///
/// The scale represents the multiplicative factor relative to the SI coherent
/// unit for this dimension. For example:
/// - meter (m): scale = 1.0
/// - kilometer (km): scale = 1000.0
/// - millimeter (mm): scale = 0.001
/// - year (yr): scale = 31_557_600.0 (Julian year in seconds)
///
/// Units with different scales are dimensionally equivalent but not
/// type-compatible for arithmetic operations (policy-dependent).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Unit {
    kind: UnitKind,
    dims: UnitDimensions,
    /// Scale factor relative to SI coherent unit (1.0 = base SI unit)
    scale: f64,
}

/// Unit kind — determines algebraic rules.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum UnitKind {
    /// Standard SI units (m, kg, s, etc.)
    /// Supports full arithmetic: +, -, *, /, ^
    Multiplicative,

    /// Temperature scales with offsets (°C = K - 273.15, °F, etc.)
    /// Addition forbidden, subtraction yields Multiplicative delta
    Affine {
        /// Offset from base unit (e.g., 273.15 for Celsius)
        offset: f64,
    },

    /// Logarithmic scales (dB, pH, etc.)
    /// Arithmetic forbidden except comparison
    Logarithmic {
        /// Logarithm base (e.g., 10.0 for dB and pH)
        base: f64,
    },
}

/// SI base dimensional exponents.
///
/// Each dimension represents a power of the corresponding SI base unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UnitDimensions {
    /// Length dimension exponent (L) - base unit: meter (m)
    pub length: i8,
    /// Mass dimension exponent (M) - base unit: kilogram (kg)
    pub mass: i8,
    /// Time dimension exponent (T) - base unit: second (s)
    pub time: i8,
    /// Temperature dimension exponent (Θ) - base unit: kelvin (K)
    pub temperature: i8,
    /// Electric current dimension exponent (I) - base unit: ampere (A)
    pub current: i8,
    /// Amount of substance dimension exponent (N) - base unit: mole (mol)
    pub amount: i8,
    /// Luminous intensity dimension exponent (J) - base unit: candela (cd)
    pub luminosity: i8,
    /// Angle dimension - base unit: radian (rad)
    /// Treated as dimensionless in SI but tracked for type safety
    pub angle: i8,
}

/// Dimensional type - the type-level properties of a unit.
///
/// This represents the **type-compatible** aspects of a unit:
/// - What kind of algebra is allowed (kind)
/// - What physical dimension it represents (dims)
///
/// The scale factor is **not** part of the type - it's value-level metadata.
/// This means two units with the same DimensionalType but different scales
/// (e.g., meters vs kilometers) can be compatible for certain operations.
///
/// # Dimensional Compatibility
///
/// Two units are **dimensionally compatible** if they have the same DimensionalType.
/// This is weaker than full equality, which also checks scale.
///
/// # Examples
///
/// ```rust
/// # use continuum_kernel_types::unit::*;
/// // Same DimensionalType, different scales
/// let m = Unit::meters();     // scale = 1.0
/// let km = Unit::new(UnitKind::Multiplicative, UnitDimensions::METER, 1000.0);
///
/// // They have the same dimensional type (Multiplicative + length dimension)
/// assert_eq!(m.dimensional_type(), km.dimensional_type());
///
/// // But they're not exactly equal (different scales)
/// assert_ne!(m, km);
///
/// // For dimensionless units, scale differences are ignored for compatibility
/// let ppmv = Unit::new(UnitKind::Multiplicative, UnitDimensions::DIMENSIONLESS, 1e-6);
/// let pure = Unit::DIMENSIONLESS;  // scale = 1.0
/// assert!(ppmv.is_compatible_with(&pure));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DimensionalType {
    /// Unit kind (Multiplicative, Affine, Logarithmic)
    pub kind: UnitKind,
    /// Dimensional exponents
    pub dims: UnitDimensions,
}

impl Unit {
    /// Dimensionless unit constant (all exponents zero).
    pub const DIMENSIONLESS: Unit = Unit {
        kind: UnitKind::Multiplicative,
        dims: UnitDimensions::DIMENSIONLESS,
        scale: 1.0,
    };

    /// Create a new unit with specified kind, dimensions, and scale.
    ///
    /// # Parameters
    /// - `kind`: Unit kind (Multiplicative, Affine, Logarithmic)
    /// - `dims`: Dimensional exponents
    /// - `scale`: Scale factor relative to SI coherent unit (1.0 = base)
    pub const fn new(kind: UnitKind, dims: UnitDimensions, scale: f64) -> Self {
        Self { kind, dims, scale }
    }

    /// Create a dimensionless unit.
    pub const fn dimensionless() -> Self {
        Self::DIMENSIONLESS
    }

    /// Get the unit kind.
    pub const fn kind(&self) -> &UnitKind {
        &self.kind
    }

    /// Get the dimensional exponents.
    pub const fn dims(&self) -> &UnitDimensions {
        &self.dims
    }

    /// Get the scale factor relative to SI coherent unit.
    pub const fn scale(&self) -> f64 {
        self.scale
    }

    /// Get the dimensional type (kind + dimensions, without scale).
    ///
    /// The dimensional type represents the type-level properties of this unit.
    /// Units with the same dimensional type but different scales are
    /// dimensionally compatible.
    pub const fn dimensional_type(&self) -> DimensionalType {
        DimensionalType {
            kind: self.kind,
            dims: self.dims,
        }
    }

    /// Check if this unit is dimensionless.
    pub fn is_dimensionless(&self) -> bool {
        self.dims.is_dimensionless()
    }

    /// Check if this unit is multiplicative.
    pub fn is_multiplicative(&self) -> bool {
        matches!(self.kind, UnitKind::Multiplicative)
    }

    /// Check if this unit is affine.
    pub fn is_affine(&self) -> bool {
        matches!(self.kind, UnitKind::Affine { .. })
    }

    /// Check if this unit is logarithmic.
    pub fn is_logarithmic(&self) -> bool {
        matches!(self.kind, UnitKind::Logarithmic { .. })
    }

    // ============================================================================
    // Base SI units (Multiplicative)
    // ============================================================================

    /// Meter (m) - length
    pub const fn meters() -> Self {
        Self::new(UnitKind::Multiplicative, UnitDimensions::METER, 1.0)
    }

    /// Kilogram (kg) - mass
    pub const fn kilograms() -> Self {
        Self::new(UnitKind::Multiplicative, UnitDimensions::KILOGRAM, 1.0)
    }

    /// Second (s) - time
    pub const fn seconds() -> Self {
        Self::new(UnitKind::Multiplicative, UnitDimensions::SECOND, 1.0)
    }

    /// Kelvin (K) - temperature (Multiplicative, not Affine)
    pub const fn kelvin() -> Self {
        Self::new(UnitKind::Multiplicative, UnitDimensions::KELVIN, 1.0)
    }

    /// Ampere (A) - electric current
    pub const fn amperes() -> Self {
        Self::new(UnitKind::Multiplicative, UnitDimensions::AMPERE, 1.0)
    }

    /// Mole (mol) - amount of substance
    pub const fn moles() -> Self {
        Self::new(UnitKind::Multiplicative, UnitDimensions::MOLE, 1.0)
    }

    /// Candela (cd) - luminous intensity
    pub const fn candelas() -> Self {
        Self::new(UnitKind::Multiplicative, UnitDimensions::CANDELA, 1.0)
    }

    /// Radian (rad) - angle
    pub const fn radians() -> Self {
        Self::new(UnitKind::Multiplicative, UnitDimensions::RADIAN, 1.0)
    }

    // ============================================================================
    // Non-SI but recognized base units
    // ============================================================================

    /// Gram (g) - mass
    /// Note: SI base is kilogram, but gram is recognized for prefix convenience
    /// (allows mg, g, kg, Mg naturally)
    pub const fn grams() -> Self {
        Self::new(UnitKind::Multiplicative, UnitDimensions::KILOGRAM, 0.001)
    }

    /// Year (yr) - time
    /// Julian year: 365.25 days = 31,557,600 seconds
    /// Used in astronomical/geological contexts (Ma, Ga)
    pub const fn years() -> Self {
        Self::new(
            UnitKind::Multiplicative,
            UnitDimensions::SECOND,
            31_557_600.0,
        )
    }

    /// Day (day) - time
    /// 86,400 seconds
    pub const fn days() -> Self {
        Self::new(UnitKind::Multiplicative, UnitDimensions::SECOND, 86_400.0)
    }

    // ============================================================================
    // Derived SI units (Multiplicative)
    // ============================================================================

    /// Newton (N) - force: kg⋅m/s²
    pub const fn newtons() -> Self {
        Self::new(
            UnitKind::Multiplicative,
            UnitDimensions {
                length: 1,
                mass: 1,
                time: -2,
                temperature: 0,
                current: 0,
                amount: 0,
                luminosity: 0,
                angle: 0,
            },
            1.0,
        )
    }

    /// Joule (J) - energy/work: kg⋅m²/s²
    pub const fn joules() -> Self {
        Self::new(
            UnitKind::Multiplicative,
            UnitDimensions {
                length: 2,
                mass: 1,
                time: -2,
                temperature: 0,
                current: 0,
                amount: 0,
                luminosity: 0,
                angle: 0,
            },
            1.0,
        )
    }

    /// Watt (W) - power: kg⋅m²/s³ = J/s
    pub const fn watts() -> Self {
        Self::new(
            UnitKind::Multiplicative,
            UnitDimensions {
                length: 2,
                mass: 1,
                time: -3,
                temperature: 0,
                current: 0,
                amount: 0,
                luminosity: 0,
                angle: 0,
            },
            1.0,
        )
    }

    /// Pascal (Pa) - pressure: kg/(m⋅s²) = N/m²
    pub const fn pascals() -> Self {
        Self::new(
            UnitKind::Multiplicative,
            UnitDimensions {
                length: -1,
                mass: 1,
                time: -2,
                temperature: 0,
                current: 0,
                amount: 0,
                luminosity: 0,
                angle: 0,
            },
            1.0,
        )
    }

    // ============================================================================
    // Affine units (temperature scales with offsets)
    // ============================================================================

    /// Celsius (°C) - affine temperature scale
    /// °C = K - 273.15
    pub const fn celsius() -> Self {
        Self::new(
            UnitKind::Affine { offset: 273.15 },
            UnitDimensions::KELVIN,
            1.0,
        )
    }

    /// Fahrenheit (°F) - affine temperature scale
    /// °F = (9/5)K - 459.67
    pub const fn fahrenheit() -> Self {
        Self::new(
            UnitKind::Affine { offset: 459.67 },
            UnitDimensions::KELVIN,
            1.0,
        )
    }

    // ============================================================================
    // Logarithmic units
    // ============================================================================

    /// Decibel (dB) - logarithmic scale (base 10)
    pub const fn decibels() -> Self {
        Self::new(
            UnitKind::Logarithmic { base: 10.0 },
            UnitDimensions::DIMENSIONLESS,
            1.0,
        )
    }

    /// pH - logarithmic scale (base 10)
    pub const fn ph() -> Self {
        Self::new(
            UnitKind::Logarithmic { base: 10.0 },
            UnitDimensions::DIMENSIONLESS,
            1.0,
        )
    }

    // ============================================================================
    // Unit algebra (Multiplicative only)
    // ============================================================================

    /// Multiply two units (dimensions add, scales multiply).
    ///
    /// Only valid for Multiplicative units.
    /// For Affine/Logarithmic, returns None.
    pub fn multiply(&self, other: &Unit) -> Option<Unit> {
        if !self.is_multiplicative() || !other.is_multiplicative() {
            return None;
        }

        Some(Unit::new(
            UnitKind::Multiplicative,
            self.dims.multiply(&other.dims),
            self.scale * other.scale,
        ))
    }

    /// Divide two units (dimensions subtract, scales divide).
    ///
    /// Only valid for Multiplicative units.
    /// For Affine/Logarithmic, returns None.
    pub fn divide(&self, other: &Unit) -> Option<Unit> {
        if !self.is_multiplicative() || !other.is_multiplicative() {
            return None;
        }

        Some(Unit::new(
            UnitKind::Multiplicative,
            self.dims.divide(&other.dims),
            self.scale / other.scale,
        ))
    }

    /// Raise unit to a power (dimensions scale, scale raised to power).
    ///
    /// Only valid for Multiplicative units.
    /// For Affine/Logarithmic, returns None.
    pub fn pow(&self, exponent: i8) -> Option<Unit> {
        if !self.is_multiplicative() {
            return None;
        }

        Some(Unit::new(
            UnitKind::Multiplicative,
            self.dims.pow(exponent),
            self.scale.powi(exponent as i32),
        ))
    }

    /// Square root of a unit.
    ///
    /// Only valid for Multiplicative units where all dimension exponents are even.
    /// Returns None if the unit is not multiplicative or has odd exponents.
    ///
    /// # Examples
    /// - sqrt(m²) = m
    /// - sqrt(m²/s²) = m/s
    /// - sqrt(m) = None (odd exponent)
    pub fn sqrt(&self) -> Option<Unit> {
        if !self.is_multiplicative() {
            return None;
        }

        // Check if all exponents are even
        if self.dims.length % 2 != 0
            || self.dims.mass % 2 != 0
            || self.dims.time % 2 != 0
            || self.dims.temperature % 2 != 0
            || self.dims.current % 2 != 0
            || self.dims.amount % 2 != 0
            || self.dims.luminosity % 2 != 0
            || self.dims.angle % 2 != 0
        {
            return None;
        }

        Some(Unit::new(
            UnitKind::Multiplicative,
            UnitDimensions {
                length: self.dims.length / 2,
                mass: self.dims.mass / 2,
                time: self.dims.time / 2,
                temperature: self.dims.temperature / 2,
                current: self.dims.current / 2,
                amount: self.dims.amount / 2,
                luminosity: self.dims.luminosity / 2,
                angle: self.dims.angle / 2,
            },
            self.scale.sqrt(),
        ))
    }

    /// Multiplicative inverse of a unit (1/unit).
    ///
    /// Only valid for Multiplicative units.
    /// Equivalent to pow(-1) but more explicit.
    ///
    /// # Examples
    /// - inverse(m/s) = s/m
    /// - inverse(kg) = 1/kg
    pub fn inverse(&self) -> Option<Unit> {
        self.pow(-1)
    }

    /// Add two units (must have same dimensions).
    ///
    /// Only valid for Multiplicative units with matching dimensions.
    /// Affine and Logarithmic forbid addition.
    pub fn add(&self, other: &Unit) -> Option<Unit> {
        if !self.is_multiplicative() || !other.is_multiplicative() {
            return None;
        }

        if self.dims != other.dims {
            return None;
        }

        Some(*self)
    }

    /// Subtract two units.
    ///
    /// - Multiplicative: must have same dimensions, result is Multiplicative
    /// - Affine: must have same dimensions, result is Multiplicative (delta)
    /// - Logarithmic: forbidden
    pub fn subtract(&self, other: &Unit) -> Option<Unit> {
        if self.is_logarithmic() || other.is_logarithmic() {
            return None;
        }

        if self.dims != other.dims {
            return None;
        }

        // Affine - Affine = Multiplicative (temperature difference)
        // Scale preserved for affine subtraction (delta T has same scale)
        Some(Unit::new(UnitKind::Multiplicative, self.dims, self.scale))
    }

    /// Check if two units are compatible for comparison.
    ///
    /// Units are comparable if they have the same dimensions,
    /// regardless of kind.
    pub fn is_comparable(&self, other: &Unit) -> bool {
        self.dims == other.dims
    }

    /// Check if two units are compatible for type checking.
    ///
    /// Units are compatible if:
    /// 1. They have the same kind (Multiplicative, Affine, Logarithmic)
    /// 2. They have the same dimensional exponents
    /// 3. Either:
    ///    - They have the same scale, OR
    ///    - They are both dimensionless (scale difference allowed)
    ///
    /// This is used for kernel type validation where dimensionless units
    /// with different scales (e.g., ppmv vs pure ratio) should be accepted.
    ///
    /// # Rationale
    ///
    /// Scale is a value-level property, not a type-level property.
    /// For dimensionless quantities, comparing `ppmv` (scale 1e-6) with
    /// a bare `0.0` (scale 1.0) is physically meaningful - both are ratios.
    ///
    /// For dimensional quantities, scale mismatches indicate unit conversion
    /// issues (meters vs kilometers), so we remain strict.
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use continuum_kernel_types::unit::*;
    /// // Dimensionless units with different scales are compatible
    /// let ppmv = Unit::new(UnitKind::Multiplicative, UnitDimensions::DIMENSIONLESS, 1e-6);
    /// let pure = Unit::DIMENSIONLESS;  // scale = 1.0
    /// assert!(ppmv.is_compatible_with(&pure));
    ///
    /// // Dimensional units with different scales are NOT compatible
    /// let m = Unit::meters();   // scale = 1.0
    /// let km = Unit::new(UnitKind::Multiplicative, UnitDimensions::METER, 1000.0);
    /// assert!(!m.is_compatible_with(&km));
    ///
    /// // Different dimensions are never compatible
    /// let s = Unit::seconds();
    /// assert!(!m.is_compatible_with(&s));
    /// ```
    pub fn is_compatible_with(&self, other: &Unit) -> bool {
        // Kind and dimensions must always match
        if self.kind != other.kind || self.dims != other.dims {
            return false;
        }

        // For dimensionless units, allow scale mismatch
        if self.is_dimensionless() {
            return true;
        }

        // For dimensional units, scale must match
        // Using approximate equality for floating point
        (self.scale - other.scale).abs() < 1e-10
    }
}

impl UnitDimensions {
    /// Dimensionless constant (all exponents zero).
    pub const DIMENSIONLESS: UnitDimensions = UnitDimensions {
        length: 0,
        mass: 0,
        time: 0,
        temperature: 0,
        current: 0,
        amount: 0,
        luminosity: 0,
        angle: 0,
    };

    /// Meter dimension (length = 1)
    pub const METER: UnitDimensions = UnitDimensions {
        length: 1,
        ..Self::DIMENSIONLESS
    };

    /// Kilogram dimension (mass = 1)
    pub const KILOGRAM: UnitDimensions = UnitDimensions {
        mass: 1,
        ..Self::DIMENSIONLESS
    };

    /// Second dimension (time = 1)
    pub const SECOND: UnitDimensions = UnitDimensions {
        time: 1,
        ..Self::DIMENSIONLESS
    };

    /// Kelvin dimension (temperature = 1)
    pub const KELVIN: UnitDimensions = UnitDimensions {
        temperature: 1,
        ..Self::DIMENSIONLESS
    };

    /// Ampere dimension (current = 1)
    pub const AMPERE: UnitDimensions = UnitDimensions {
        current: 1,
        ..Self::DIMENSIONLESS
    };

    /// Mole dimension (amount = 1)
    pub const MOLE: UnitDimensions = UnitDimensions {
        amount: 1,
        ..Self::DIMENSIONLESS
    };

    /// Candela dimension (luminosity = 1)
    pub const CANDELA: UnitDimensions = UnitDimensions {
        luminosity: 1,
        ..Self::DIMENSIONLESS
    };

    /// Radian dimension (angle = 1)
    pub const RADIAN: UnitDimensions = UnitDimensions {
        angle: 1,
        ..Self::DIMENSIONLESS
    };

    /// Check if all dimensions are zero.
    pub fn is_dimensionless(&self) -> bool {
        self.length == 0
            && self.mass == 0
            && self.time == 0
            && self.temperature == 0
            && self.current == 0
            && self.amount == 0
            && self.luminosity == 0
            && self.angle == 0
    }

    /// Multiply dimensions (add exponents).
    pub fn multiply(&self, other: &UnitDimensions) -> UnitDimensions {
        UnitDimensions {
            length: self.length + other.length,
            mass: self.mass + other.mass,
            time: self.time + other.time,
            temperature: self.temperature + other.temperature,
            current: self.current + other.current,
            amount: self.amount + other.amount,
            luminosity: self.luminosity + other.luminosity,
            angle: self.angle + other.angle,
        }
    }

    /// Divide dimensions (subtract exponents).
    pub fn divide(&self, other: &UnitDimensions) -> UnitDimensions {
        UnitDimensions {
            length: self.length - other.length,
            mass: self.mass - other.mass,
            time: self.time - other.time,
            temperature: self.temperature - other.temperature,
            current: self.current - other.current,
            amount: self.amount - other.amount,
            luminosity: self.luminosity - other.luminosity,
            angle: self.angle - other.angle,
        }
    }

    /// Raise dimensions to a power (scale exponents).
    pub fn pow(&self, exponent: i8) -> UnitDimensions {
        UnitDimensions {
            length: self.length * exponent,
            mass: self.mass * exponent,
            time: self.time * exponent,
            temperature: self.temperature * exponent,
            current: self.current * exponent,
            amount: self.amount * exponent,
            luminosity: self.luminosity * exponent,
            angle: self.angle * exponent,
        }
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            UnitKind::Multiplicative => write!(f, "{}", self.dims),
            UnitKind::Affine { offset } => {
                write!(f, "{}+{}", self.dims, offset)
            }
            UnitKind::Logarithmic { base } => {
                write!(f, "log{}({})", base, self.dims)
            }
        }
    }
}

impl fmt::Display for UnitDimensions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "1");
        }

        let mut parts = Vec::new();

        if self.length != 0 {
            parts.push(format_dim("m", self.length));
        }
        if self.mass != 0 {
            parts.push(format_dim("kg", self.mass));
        }
        if self.time != 0 {
            parts.push(format_dim("s", self.time));
        }
        if self.temperature != 0 {
            parts.push(format_dim("K", self.temperature));
        }
        if self.current != 0 {
            parts.push(format_dim("A", self.current));
        }
        if self.amount != 0 {
            parts.push(format_dim("mol", self.amount));
        }
        if self.luminosity != 0 {
            parts.push(format_dim("cd", self.luminosity));
        }
        if self.angle != 0 {
            parts.push(format_dim("rad", self.angle));
        }

        write!(f, "{}", parts.join("·"))
    }
}

fn format_dim(symbol: &str, exponent: i8) -> String {
    match exponent {
        1 => symbol.to_string(),
        exp => format!("{}^{}", symbol, exp),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensionless() {
        let unit = Unit::dimensionless();
        assert!(unit.is_dimensionless());
        assert!(unit.is_multiplicative());
    }

    #[test]
    fn test_base_units() {
        let m = Unit::meters();
        assert_eq!(m.dims().length, 1);
        assert!(m.is_multiplicative());

        let kg = Unit::kilograms();
        assert_eq!(kg.dims().mass, 1);

        let s = Unit::seconds();
        assert_eq!(s.dims().time, 1);
    }

    #[test]
    fn test_multiply() {
        let m = Unit::meters();
        let s = Unit::seconds();

        // Test multiply: m * s = m·s
        let m_s = m.multiply(&s).unwrap();
        assert_eq!(m_s.dims().length, 1);
        assert_eq!(m_s.dims().time, 1);

        // Test multiply to create area: m * m = m²
        let area = m.multiply(&m).unwrap();
        assert_eq!(area.dims().length, 2);
    }

    #[test]
    fn test_divide() {
        let m = Unit::meters();
        let s = Unit::seconds();

        let m_per_s = m.divide(&s).unwrap();
        assert_eq!(m_per_s.dims().length, 1);
        assert_eq!(m_per_s.dims().time, -1);
    }

    #[test]
    fn test_pow() {
        let m = Unit::meters();

        let area = m.pow(2).unwrap();
        assert_eq!(area.dims().length, 2);

        let inv_s = Unit::seconds().pow(-1).unwrap();
        assert_eq!(inv_s.dims().time, -1);
    }

    #[test]
    fn test_affine_addition_forbidden() {
        let c1 = Unit::celsius();
        let c2 = Unit::celsius();

        assert!(c1.add(&c2).is_none());
    }

    #[test]
    fn test_affine_subtraction_yields_multiplicative() {
        let c1 = Unit::celsius();
        let c2 = Unit::celsius();

        let delta = c1.subtract(&c2).unwrap();
        assert!(delta.is_multiplicative());
        assert_eq!(delta.dims().temperature, 1);
    }

    #[test]
    fn test_logarithmic_arithmetic_forbidden() {
        let db = Unit::decibels();

        assert!(db.add(&db).is_none());
        assert!(db.subtract(&db).is_none());
        assert!(db.multiply(&db).is_none());
    }

    #[test]
    fn test_affine_multiply_divide_pow_forbidden() {
        let c = Unit::celsius();
        let k = Unit::kelvin();

        // Affine units cannot be multiplied
        assert!(c.multiply(&k).is_none());
        assert!(c.multiply(&c).is_none());

        // Affine units cannot be divided
        assert!(c.divide(&k).is_none());
        assert!(c.divide(&c).is_none());

        // Affine units cannot be raised to powers
        assert!(c.pow(2).is_none());
        assert!(c.pow(-1).is_none());
    }

    #[test]
    fn test_multiplicative_add_success_and_mismatch() {
        let m1 = Unit::meters();
        let m2 = Unit::meters();

        // Same dimensions: addition allowed
        assert!(m1.add(&m2).is_some());

        // Different dimensions: addition forbidden
        let s = Unit::seconds();
        assert!(m1.add(&s).is_none());
    }

    #[test]
    fn test_multiplicative_subtract() {
        let m1 = Unit::meters();
        let m2 = Unit::meters();

        // Same dimensions: subtraction allowed
        let result = m1.subtract(&m2).unwrap();
        assert!(result.is_multiplicative());
        assert_eq!(result.dims().length, 1);

        // Different dimensions: subtraction forbidden
        let s = Unit::seconds();
        assert!(m1.subtract(&s).is_none());
    }

    #[test]
    fn test_comparison_compatibility() {
        let c = Unit::celsius();
        let k = Unit::kelvin();

        // Both have temperature dimension, so comparable
        assert!(c.is_comparable(&k));

        let m = Unit::meters();
        assert!(!c.is_comparable(&m));
    }

    #[test]
    fn test_display() {
        let m_per_s = Unit::meters().divide(&Unit::seconds()).unwrap();
        let display = format!("{}", m_per_s);
        assert!(display.contains("m"));
        assert!(display.contains("s"));
    }
}
