// Test: kernel_fn without namespace attribute should fail

use continuum_kernel_macros::kernel_fn;

#[kernel_fn(
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
