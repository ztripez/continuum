//! Comparison Operations
//!
//! Numeric comparison operations for conditional logic in DSL.

use continuum_kernel_macros::kernel_fn;
use continuum_kernel_types::prelude::*;

/// Equal: `compare.eq(a, b)`
///
/// Returns true if a equals b (within floating-point epsilon).
#[kernel_fn(
    namespace = "compare",
    purity = Pure,
    shape_in = [Any, SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = ShapeSameAs(0),
    unit_out = Dimensionless
)]
pub fn eq(a: f64, b: f64) -> bool {
    (a - b).abs() < f64::EPSILON
}

/// Not equal: `compare.ne(a, b)`
///
/// Returns true if a does not equal b.
#[kernel_fn(
    namespace = "compare",
    purity = Pure,
    shape_in = [Any, SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = ShapeSameAs(0),
    unit_out = Dimensionless
)]
pub fn ne(a: f64, b: f64) -> bool {
    (a - b).abs() >= f64::EPSILON
}

/// Less than: `compare.lt(a, b)`
///
/// Returns true if a is less than b.
#[kernel_fn(
    namespace = "compare",
    purity = Pure,
    shape_in = [Any, SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = ShapeSameAs(0),
    unit_out = Dimensionless
)]
pub fn lt(a: f64, b: f64) -> bool {
    a < b
}

/// Less than or equal: `compare.le(a, b)`
///
/// Returns true if a is less than or equal to b.
#[kernel_fn(
    namespace = "compare",
    purity = Pure,
    shape_in = [Any, SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = ShapeSameAs(0),
    unit_out = Dimensionless
)]
pub fn le(a: f64, b: f64) -> bool {
    a <= b
}

/// Greater than: `compare.gt(a, b)`
///
/// Returns true if a is greater than b.
#[kernel_fn(
    namespace = "compare",
    purity = Pure,
    shape_in = [Any, SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = ShapeSameAs(0),
    unit_out = Dimensionless
)]
pub fn gt(a: f64, b: f64) -> bool {
    a > b
}

/// Greater than or equal: `compare.ge(a, b)`
///
/// Returns true if a is greater than or equal to b.
#[kernel_fn(
    namespace = "compare",
    purity = Pure,
    shape_in = [Any, SameAs(0)],
    unit_in = [UnitAny, UnitSameAs(0)],
    shape_out = ShapeSameAs(0),
    unit_out = Dimensionless
)]
pub fn ge(a: f64, b: f64) -> bool {
    a >= b
}
