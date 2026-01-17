// Test: kernel_fn with unit_inference + type constraints should fail

use continuum_kernel_macros::kernel_fn;

#[kernel_fn(
    namespace = "test",
    unit_inference = "preserve_first",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn test_fn(x: f64) -> f64 {
    x
}

fn main() {}
