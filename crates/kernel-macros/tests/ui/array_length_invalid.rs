use continuum_kernel_macros::kernel_fn;

#[kernel_fn(
    namespace = "test",
    purity = Pure,
    shape_in = [AnyScalar],
    unit_in = [UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn bad_return(x: f64) -> [f64; 5] {
    [x, x, x, x, x]
}

fn main() {}
