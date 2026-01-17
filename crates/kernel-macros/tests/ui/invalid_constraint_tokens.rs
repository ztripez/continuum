// Test: kernel_fn with invalid tokens in constraint expressions should fail

use continuum_kernel_macros::kernel_fn;

#[kernel_fn(
    namespace = "test",
    purity = Pure,
    shape_in = [ThisIsNotAValidConstraint],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn test_fn(x: f64) -> f64 {
    x
}

fn main() {}
