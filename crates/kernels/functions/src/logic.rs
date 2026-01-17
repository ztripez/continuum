//! Logic Operations
//!
//! Boolean logic operations for conditional evaluation in DSL.

use continuum_kernel_macros::kernel_fn;
use continuum_kernel_types::prelude::*;

/// Logical AND: `logic.and(a, b)`
///
/// Returns true if both operands are true.
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
/// Returns true if either operand is true.
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
/// Returns `if_true` if `condition` is true, otherwise `if_false`.
///
/// This is a pure, eager evaluation (both branches are evaluated).
/// Use for simple value selection without side effects.
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
