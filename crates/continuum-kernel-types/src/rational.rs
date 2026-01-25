//! Rational number type for fractional dimensional exponents
//!
//! This module provides exact rational arithmetic for dimensional analysis.
//! Fractional exponents are represented as `num/denom` pairs and automatically
//! normalized to lowest terms using GCD.
//!
//! # Examples
//!
//! ```rust
//! use continuum_kernel_types::rational::Rational;
//!
//! let half = Rational::new(1, 2);
//! let third = Rational::new(1, 3);
//!
//! // Arithmetic operations preserve exactness
//! assert_eq!(half + third, Rational::new(5, 6));
//! assert_eq!(half * third, Rational::new(1, 6));
//!
//! // Automatic normalization
//! assert_eq!(Rational::new(2, 4), Rational::new(1, 2));
//! assert_eq!(Rational::new(6, 9), Rational::new(2, 3));
//! ```

use serde::{Deserialize, Serialize};
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// A rational number represented as `numerator / denominator`.
///
/// Automatically normalized to lowest terms on construction.
/// Denominator is always positive (sign is in numerator).
///
/// # Invariants
///
/// - `denom > 0` (enforced via debug assertion)
/// - `gcd(num.abs(), denom) == 1` (normalized on construction)
/// - `denom == 0` is forbidden (will panic in debug mode)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rational {
    /// Numerator (can be negative)
    pub num: i8,
    /// Denominator (always positive, never zero)
    pub denom: u8,
}

impl Rational {
    /// Create a new rational number, automatically normalized to lowest terms.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `denom == 0`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_kernel_types::rational::Rational;
    ///
    /// let half = Rational::new(1, 2);
    /// assert_eq!(half.num, 1);
    /// assert_eq!(half.denom, 2);
    ///
    /// // Automatic normalization
    /// let also_half = Rational::new(2, 4);
    /// assert_eq!(also_half, half);
    ///
    /// // Negative numerators
    /// let neg_half = Rational::new(-1, 2);
    /// assert_eq!(neg_half.num, -1);
    /// assert_eq!(neg_half.denom, 2);
    /// ```
    pub fn new(num: i8, denom: u8) -> Self {
        debug_assert!(denom > 0, "Denominator must be positive (got 0)");

        // Handle zero numerator (0/n = 0/1)
        if num == 0 {
            return Rational { num: 0, denom: 1 };
        }

        // Compute GCD and normalize
        let gcd = gcd(num.unsigned_abs(), denom);
        let num = num / gcd as i8;
        let denom = denom / gcd;

        Rational { num, denom }
    }

    /// Create an integer rational (n/1).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_kernel_types::rational::Rational;
    ///
    /// let three = Rational::integer(3);
    /// assert_eq!(three.num, 3);
    /// assert_eq!(three.denom, 1);
    /// ```
    pub const fn integer(n: i8) -> Self {
        Rational { num: n, denom: 1 }
    }

    /// Zero (0/1).
    pub const ZERO: Self = Rational { num: 0, denom: 1 };

    /// One (1/1).
    pub const ONE: Self = Rational { num: 1, denom: 1 };

    /// Check if this rational is zero.
    pub const fn is_zero(self) -> bool {
        self.num == 0
    }

    /// Check if this rational is an integer (denom == 1).
    pub const fn is_integer(self) -> bool {
        self.denom == 1
    }

    /// Convert to f64 (lossy).
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_kernel_types::rational::Rational;
    ///
    /// let half = Rational::new(1, 2);
    /// assert_eq!(half.to_f64(), 0.5);
    ///
    /// let third = Rational::new(1, 3);
    /// assert!((third.to_f64() - 0.33333333).abs() < 1e-7);
    /// ```
    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.denom as f64
    }
}

/// Compute greatest common divisor using Euclidean algorithm.
///
/// # Examples
///
/// ```rust
/// use continuum_kernel_types::rational::gcd;
///
/// assert_eq!(gcd(12, 8), 4);
/// assert_eq!(gcd(7, 13), 1);
/// assert_eq!(gcd(0, 5), 5);
/// ```
pub fn gcd(mut a: u8, mut b: u8) -> u8 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

// === Arithmetic Operations ===

impl Add for Rational {
    type Output = Rational;

    /// Add two rationals: `a/b + c/d = (a*d + b*c) / (b*d)`.
    ///
    /// Result is automatically normalized.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_kernel_types::rational::Rational;
    ///
    /// let half = Rational::new(1, 2);
    /// let third = Rational::new(1, 3);
    /// assert_eq!(half + third, Rational::new(5, 6));
    /// ```
    fn add(self, other: Rational) -> Rational {
        let num = self.num as i16 * other.denom as i16 + other.num as i16 * self.denom as i16;
        let denom = self.denom as u16 * other.denom as u16;

        // Clamp to i8/u8 range (dimensional exponents should never overflow)
        let num = num.clamp(i8::MIN as i16, i8::MAX as i16) as i8;
        let denom = denom.min(u8::MAX as u16) as u8;

        Rational::new(num, denom)
    }
}

impl Sub for Rational {
    type Output = Rational;

    /// Subtract two rationals: `a/b - c/d = (a*d - b*c) / (b*d)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_kernel_types::rational::Rational;
    ///
    /// let half = Rational::new(1, 2);
    /// let third = Rational::new(1, 3);
    /// assert_eq!(half - third, Rational::new(1, 6));
    /// ```
    fn sub(self, other: Rational) -> Rational {
        let num = self.num as i16 * other.denom as i16 - other.num as i16 * self.denom as i16;
        let denom = self.denom as u16 * other.denom as u16;

        let num = num.clamp(i8::MIN as i16, i8::MAX as i16) as i8;
        let denom = denom.min(u8::MAX as u16) as u8;

        Rational::new(num, denom)
    }
}

impl Mul for Rational {
    type Output = Rational;

    /// Multiply two rationals: `(a/b) * (c/d) = (a*c) / (b*d)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_kernel_types::rational::Rational;
    ///
    /// let half = Rational::new(1, 2);
    /// let third = Rational::new(1, 3);
    /// assert_eq!(half * third, Rational::new(1, 6));
    /// ```
    fn mul(self, other: Rational) -> Rational {
        let num = self.num as i16 * other.num as i16;
        let denom = self.denom as u16 * other.denom as u16;

        let num = num.clamp(i8::MIN as i16, i8::MAX as i16) as i8;
        let denom = denom.min(u8::MAX as u16) as u8;

        Rational::new(num, denom)
    }
}

impl Neg for Rational {
    type Output = Rational;

    /// Negate a rational: `-(a/b) = (-a)/b`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_kernel_types::rational::Rational;
    ///
    /// let half = Rational::new(1, 2);
    /// assert_eq!(-half, Rational::new(-1, 2));
    /// ```
    fn neg(self) -> Rational {
        Rational {
            num: -self.num,
            denom: self.denom,
        }
    }
}

impl fmt::Display for Rational {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denom == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.denom)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization() {
        assert_eq!(Rational::new(2, 4), Rational::new(1, 2));
        assert_eq!(Rational::new(6, 9), Rational::new(2, 3));
        assert_eq!(Rational::new(12, 8), Rational::new(3, 2));
    }

    #[test]
    fn test_zero() {
        assert_eq!(Rational::new(0, 1), Rational::ZERO);
        assert_eq!(Rational::new(0, 5), Rational::ZERO);
        assert!(Rational::ZERO.is_zero());
    }

    #[test]
    fn test_integer() {
        let three = Rational::integer(3);
        assert_eq!(three.num, 3);
        assert_eq!(three.denom, 1);
        assert!(three.is_integer());
    }

    #[test]
    fn test_addition() {
        let half = Rational::new(1, 2);
        let third = Rational::new(1, 3);
        assert_eq!(half + third, Rational::new(5, 6));
    }

    #[test]
    fn test_subtraction() {
        let half = Rational::new(1, 2);
        let third = Rational::new(1, 3);
        assert_eq!(half - third, Rational::new(1, 6));
    }

    #[test]
    fn test_multiplication() {
        let half = Rational::new(1, 2);
        let third = Rational::new(1, 3);
        assert_eq!(half * third, Rational::new(1, 6));

        let two_thirds = Rational::new(2, 3);
        let three_halves = Rational::new(3, 2);
        assert_eq!(two_thirds * three_halves, Rational::ONE);
    }

    #[test]
    fn test_negation() {
        let half = Rational::new(1, 2);
        assert_eq!(-half, Rational::new(-1, 2));
        assert_eq!(-(-half), half);
    }

    #[test]
    fn test_to_f64() {
        let half = Rational::new(1, 2);
        assert_eq!(half.to_f64(), 0.5);

        let third = Rational::new(1, 3);
        assert!((third.to_f64() - 0.33333333).abs() < 1e-7);
    }

    #[test]
    fn test_gcd() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(7, 13), 1);
        assert_eq!(gcd(0, 5), 5);
        assert_eq!(gcd(6, 9), 3);
    }
}
