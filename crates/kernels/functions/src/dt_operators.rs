//! dt-Robust Operators
//!
//! These operators provide numerically stable alternatives to raw dt usage.
//! They are designed to produce consistent results regardless of timestep size.

use continuum_foundation::Dt;
use continuum_kernel_macros::kernel_fn;

/// Integration: `integrate(prev, rate)` → `prev + rate * dt`
#[kernel_fn(name = "integrate")]
pub fn integrate(prev: f64, rate: f64, dt: Dt) -> f64 {
    prev + rate * dt
}

/// Exponential decay: `decay(value, halflife)` → `value * 0.5^(dt/halflife)`
#[kernel_fn(name = "decay")]
pub fn decay(value: f64, halflife: f64, dt: Dt) -> f64 {
    value * 0.5_f64.powf(dt / halflife)
}

/// Exponential relaxation: `relax(current, target, tau)` → approaches target
#[kernel_fn(name = "relax")]
pub fn relax(current: f64, target: f64, tau: f64, dt: Dt) -> f64 {
    let alpha = std::f64::consts::E.powf(-dt / tau);
    target + (current - target) * alpha
}

/// Exponential relaxation: `relax_to(current, target, tau)` → approaches target
/// Alias for `relax`
#[kernel_fn(name = "relax_to")]
pub fn relax_to(current: f64, target: f64, tau: f64, dt: Dt) -> f64 {
    relax(current, target, tau, dt)
}

#[cfg(test)]
mod tests {
    use continuum_kernel_registry::{Arity, eval, get, is_known};

    #[test]
    fn test_integrate_registered() {
        assert!(is_known("integrate"));
        let desc = get("integrate").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
        assert!(desc.requires_dt());
    }

    #[test]
    fn test_integrate_eval() {
        // integrate(10, 5, dt=0.1) = 10 + 5*0.1 = 10.5
        let result = eval("integrate", &[10.0, 5.0], 0.1).unwrap();
        assert!((result - 10.5).abs() < 1e-10);
    }

    #[test]
    fn test_decay_registered() {
        assert!(is_known("decay"));
        let desc = get("decay").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
        assert!(desc.requires_dt());
    }

    #[test]
    fn test_decay_eval() {
        // decay(100, 10, dt=10) = 100 * 0.5^1 = 50
        let result = eval("decay", &[100.0, 10.0], 10.0).unwrap();
        assert!((result - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_relax_registered() {
        assert!(is_known("relax"));
        let desc = get("relax").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(3));
        assert!(desc.requires_dt());
    }

    #[test]
    fn test_relax_eval() {
        // relax(0, 100, tau=1, dt=1) ≈ 63.2
        let result = eval("relax", &[0.0, 100.0, 1.0], 1.0).unwrap();
        let expected = 100.0 * (1.0 - std::f64::consts::E.powf(-1.0));
        assert!((result - expected).abs() < 0.01);
    }
}
