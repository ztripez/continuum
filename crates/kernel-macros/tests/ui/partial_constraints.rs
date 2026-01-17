// Test: kernel_fn with partial constraints (shape_in without unit_in) should fail

use continuum_kernel_macros::kernel_fn;

#[kernel_fn(
    namespace = "test",
    purity = Pure,
    shape_in = [AnyScalar],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn test_fn(x: f64) -> f64 {
    x
}

fn main() {}
