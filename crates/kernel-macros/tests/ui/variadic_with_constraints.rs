// Test: kernel_fn with variadic + type constraints should fail

use continuum_kernel_macros::kernel_fn;

#[kernel_fn(
    namespace = "test",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    variadic
)]
pub fn test_fn(args: &[f64]) -> f64 {
    args[0]
}

fn main() {}
