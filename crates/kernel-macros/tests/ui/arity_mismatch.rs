// Test: kernel_fn with arity mismatch (shape_in.len() != params.len()) should fail

use continuum_kernel_macros::kernel_fn;

#[kernel_fn(
    namespace = "test",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],  // 2 constraints
    unit_in = [UnitAny, UnitAny],       // 2 constraints
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn test_fn(x: f64) -> f64 {
    // 1 parameter - mismatch!
    x
}

fn main() {}
