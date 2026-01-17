//! Logic Operations
//!
//! Boolean logic operations for conditional evaluation in DSL.

use continuum_kernel_macros::kernel_fn;

/// Logical AND: `logic.and(a, b)`
///
/// Returns true if both operands are true, false otherwise.
///
/// # Parameters
///
/// - `a`: First boolean operand (dimensionless)
/// - `b`: Second boolean operand (dimensionless)
///
/// # Returns
///
/// Boolean result (dimensionless): `a AND b`
///
/// # Examples
///
/// ```cdsl
/// logic.and(true, true)   // → true
/// logic.and(true, false)  // → false
/// logic.and(false, false) // → false
/// ```
#[kernel_fn(
    namespace = "logic",
    purity = Pure,
    shape_in = [Any, Any],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = ShapeSameAs(0),
    unit_out = Dimensionless
)]
pub fn and(a: bool, b: bool) -> bool {
    a && b
}

/// Logical OR: `logic.or(a, b)`
///
/// Returns true if at least one operand is true, false if both are false.
///
/// # Parameters
///
/// - `a`: First boolean operand (dimensionless)
/// - `b`: Second boolean operand (dimensionless)
///
/// # Returns
///
/// Boolean result (dimensionless): `a OR b`
///
/// # Examples
///
/// ```cdsl
/// logic.or(true, false)   // → true
/// logic.or(false, true)   // → true
/// logic.or(false, false)  // → false
/// ```
#[kernel_fn(
    namespace = "logic",
    purity = Pure,
    shape_in = [Any, Any],
    unit_in = [UnitDimensionless, UnitDimensionless],
    shape_out = ShapeSameAs(0),
    unit_out = Dimensionless
)]
pub fn or(a: bool, b: bool) -> bool {
    a || b
}

/// Logical NOT: `logic.not(a)`
///
/// Returns the logical negation of the operand.
///
/// # Parameters
///
/// - `a`: Boolean operand to negate (dimensionless)
///
/// # Returns
///
/// Boolean result (dimensionless): `NOT a`
///
/// # Examples
///
/// ```cdsl
/// logic.not(true)   // → false
/// logic.not(false)  // → true
/// ```
#[kernel_fn(
    namespace = "logic",
    purity = Pure,
    shape_in = [Any],
    unit_in = [UnitDimensionless],
    shape_out = ShapeSameAs(0),
    unit_out = Dimensionless
)]
pub fn not(a: bool) -> bool {
    !a
}

/// Select: `logic.select(condition, if_true, if_false)`
///
/// Returns `if_true` if `condition` is true, otherwise returns `if_false`.
///
/// **Important:** This is pure, eager evaluation - both `if_true` and `if_false`
/// are evaluated before selection. Use for simple value selection without side effects.
///
/// # Parameters
///
/// - `condition`: Boolean condition to evaluate (dimensionless)
/// - `if_true`: Value returned when condition is true (any unit)
/// - `if_false`: Value returned when condition is false (must match `if_true` unit)
///
/// # Returns
///
/// Numeric value with same unit as `if_true` and `if_false`
///
/// # Examples
///
/// ```cdsl
/// logic.select(true, 10.0, 20.0)   // → 10.0
/// logic.select(false, 10.0, 20.0)  // → 20.0
///
/// // Works with any unit (e.g., temperature)
/// let temp_k = logic.select(is_hot, 373.15<K>, 273.15<K>)
/// ```
#[kernel_fn(
    namespace = "logic",
    purity = Pure,
    shape_in = [Any, Any, SameAs(1)],
    unit_in = [UnitDimensionless, UnitAny, UnitSameAs(1)],
    shape_out = ShapeSameAs(1),
    unit_out = UnitDerivSameAs(1)
)]
pub fn select(condition: bool, if_true: f64, if_false: f64) -> f64 {
    if condition { if_true } else { if_false }
}
