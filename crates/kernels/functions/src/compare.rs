//! Comparison Operations
//!
//! Numeric comparison operations for conditional logic in DSL.

use continuum_kernel_macros::kernel_fn;

/// Helper macro for comparison kernels with standard type constraints
///
/// All comparison operations have identical signatures:
/// - Two parameters of matching shape and unit
/// - Return dimensionless boolean with same shape as inputs
macro_rules! comparison_kernel {
    (
        $(#[$meta:meta])*
        $vis:vis fn $name:ident($a:ident: $a_ty:ty, $b:ident: $b_ty:ty) -> $ret:ty $body:block
    ) => {
        $(#[$meta])*
        #[kernel_fn(
            namespace = "compare",
            purity = Pure,
            shape_in = [Any, SameAs(0)],
            unit_in = [UnitAny, UnitSameAs(0)],
            shape_out = ShapeSameAs(0),
            unit_out = Dimensionless
        )]
        $vis fn $name($a: $a_ty, $b: $b_ty) -> $ret $body
    };
}

comparison_kernel! {
    /// Equal: `compare.eq(a, b)`
    ///
    /// Returns true if `a` equals `b` within floating-point epsilon tolerance.
    ///
    /// Uses `f64::EPSILON` (~2.22e-16) as the comparison threshold to handle
    /// floating-point rounding errors. NaN values are never equal (returns false).
    ///
    /// # Parameters
    ///
    /// - `a`: First numeric value (any unit)
    /// - `b`: Second numeric value (must match `a` unit)
    ///
    /// # Returns
    ///
    /// Boolean result (dimensionless): true if `|a - b| < ε`, false otherwise
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// compare.eq(1.0, 1.0)              // → true
    /// compare.eq(1.0, 1.0000000000001)  // → true (within epsilon)
    /// compare.eq(1.0, 2.0)              // → false
    /// compare.eq(NaN, NaN)              // → false (NaN never equals NaN)
    /// ```
    pub fn eq(a: f64, b: f64) -> bool {
        (a - b).abs() < f64::EPSILON
    }
}

comparison_kernel! {
    /// Not equal: `compare.ne(a, b)`
    ///
    /// Returns true if `a` does not equal `b` (beyond floating-point epsilon tolerance).
    ///
    /// Uses `f64::EPSILON` (~2.22e-16) as the comparison threshold. NaN values are
    /// always considered not equal to any value including themselves (returns true).
    ///
    /// # Parameters
    ///
    /// - `a`: First numeric value (any unit)
    /// - `b`: Second numeric value (must match `a` unit)
    ///
    /// # Returns
    ///
    /// Boolean result (dimensionless): true if `|a - b| >= ε`, false otherwise
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// compare.ne(1.0, 2.0)              // → true
    /// compare.ne(1.0, 1.0)              // → false
    /// compare.ne(1.0, 1.0000000000001)  // → false (within epsilon)
    /// compare.ne(NaN, NaN)              // → true (NaN never equals anything)
    /// ```
    pub fn ne(a: f64, b: f64) -> bool {
        (a - b).abs() >= f64::EPSILON
    }
}

comparison_kernel! {
    /// Less than: `compare.lt(a, b)`
    ///
    /// Returns true if `a` is strictly less than `b`.
    ///
    /// # Parameters
    ///
    /// - `a`: First numeric value (any unit)
    /// - `b`: Second numeric value (must match `a` unit)
    ///
    /// # Returns
    ///
    /// Boolean result (dimensionless): true if `a < b`, false otherwise
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// compare.lt(1.0, 2.0)   // → true
    /// compare.lt(2.0, 1.0)   // → false
    /// compare.lt(1.0, 1.0)   // → false
    /// ```
    pub fn lt(a: f64, b: f64) -> bool {
        a < b
    }
}

comparison_kernel! {
    /// Less than or equal: `compare.le(a, b)`
    ///
    /// Returns true if `a` is less than or equal to `b`.
    ///
    /// # Parameters
    ///
    /// - `a`: First numeric value (any unit)
    /// - `b`: Second numeric value (must match `a` unit)
    ///
    /// # Returns
    ///
    /// Boolean result (dimensionless): true if `a <= b`, false otherwise
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// compare.le(1.0, 2.0)   // → true
    /// compare.le(1.0, 1.0)   // → true
    /// compare.le(2.0, 1.0)   // → false
    /// ```
    pub fn le(a: f64, b: f64) -> bool {
        a <= b
    }
}

comparison_kernel! {
    /// Greater than: `compare.gt(a, b)`
    ///
    /// Returns true if `a` is strictly greater than `b`.
    ///
    /// # Parameters
    ///
    /// - `a`: First numeric value (any unit)
    /// - `b`: Second numeric value (must match `a` unit)
    ///
    /// # Returns
    ///
    /// Boolean result (dimensionless): true if `a > b`, false otherwise
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// compare.gt(2.0, 1.0)   // → true
    /// compare.gt(1.0, 2.0)   // → false
    /// compare.gt(1.0, 1.0)   // → false
    /// ```
    pub fn gt(a: f64, b: f64) -> bool {
        a > b
    }
}

comparison_kernel! {
    /// Greater than or equal: `compare.ge(a, b)`
    ///
    /// Returns true if `a` is greater than or equal to `b`.
    ///
    /// # Parameters
    ///
    /// - `a`: First numeric value (any unit)
    /// - `b`: Second numeric value (must match `a` unit)
    ///
    /// # Returns
    ///
    /// Boolean result (dimensionless): true if `a >= b`, false otherwise
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// compare.ge(2.0, 1.0)   // → true
    /// compare.ge(1.0, 1.0)   // → true
    /// compare.ge(1.0, 2.0)   // → false
    /// ```
    pub fn ge(a: f64, b: f64) -> bool {
        a >= b
    }
}
