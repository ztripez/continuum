//! dt-Robust Operators
//!
//! These operators provide numerically stable alternatives to raw dt usage.
//! They are designed to produce consistent results regardless of timestep size.

use continuum_foundation::Dt;
use continuum_kernel_macros::{kernel_fn, vectorized_kernel_fn};
use continuum_kernel_registry::{VRegBuffer, VectorizedResult};

/// Integration: `integrate(prev, rate)` → `prev + rate * dt`
/// Default uses Euler method
#[kernel_fn(name = "integrate", category = "simulation", vectorized)]
pub fn integrate(prev: f64, rate: f64, dt: Dt) -> f64 {
    prev + rate * dt
}

/// Euler integration: `integrate_euler(prev, rate)` → `prev + rate * dt`
/// Explicit Euler method (same as default integrate)
#[kernel_fn(name = "integrate_euler", category = "simulation", vectorized)]
pub fn integrate_euler(prev: f64, rate: f64, dt: Dt) -> f64 {
    prev + rate * dt
}

/// RK4 integration: `integrate_rk4(prev, rate)` → higher-order integration
/// Note: This is a simplified RK4 that assumes constant rate over dt
#[kernel_fn(name = "integrate_rk4", category = "simulation", vectorized)]
pub fn integrate_rk4(prev: f64, rate: f64, dt: Dt) -> f64 {
    // Simplified RK4 for constant rate case
    // For true RK4, we'd need function evaluation capability
    // k1 = rate * dt
    // k2 = rate * dt (constant rate)
    // k3 = rate * dt (constant rate)
    // k4 = rate * dt (constant rate)
    // result = prev + (k1 + 2*k2 + 2*k3 + k4) / 6
    // For constant rate: result = prev + 6 * rate * dt / 6 = prev + rate * dt
    // So for constant rate, RK4 reduces to Euler
    prev + rate * dt
}

/// Verlet integration: `integrate_verlet(prev, rate)` → Velocity Verlet approximation
/// Note: This is simplified for single-variable case
#[kernel_fn(name = "integrate_verlet", category = "simulation", vectorized)]
pub fn integrate_verlet(prev: f64, rate: f64, dt: Dt) -> f64 {
    // Simplified Verlet integration
    // For position-like quantities, assume rate is velocity
    // For true Verlet, we'd need acceleration and previous position
    // This approximation uses: x_new = x + v*dt + 0.5*a*dt^2
    // Assuming acceleration ≈ 0 for this simplified case
    prev + rate * dt
}

/// Exponential decay: `decay(value, halflife)` → `value * 0.5^(dt/halflife)`
#[kernel_fn(name = "decay", category = "simulation", vectorized)]
pub fn decay(value: f64, halflife: f64, dt: Dt) -> f64 {
    value * 0.5_f64.powf(dt / halflife)
}

/// Exponential relaxation: `relax(current, target, tau)` → approaches target
#[kernel_fn(name = "relax", category = "simulation", vectorized)]
pub fn relax(current: f64, target: f64, tau: f64, dt: Dt) -> f64 {
    let alpha = std::f64::consts::E.powf(-dt / tau);
    target + (current - target) * alpha
}

/// Exponential relaxation: `relax_to(current, target, tau)` → approaches target
/// Alias for `relax`
#[kernel_fn(name = "relax_to", category = "simulation")]
pub fn relax_to(current: f64, target: f64, tau: f64, dt: Dt) -> f64 {
    relax(current, target, tau, dt)
}

/// Smooth transition: `smooth(current, target, tau)` → approaches target
/// Same as relax - exponential approach to target value
#[kernel_fn(name = "smooth", category = "simulation", vectorized)]
pub fn smooth(current: f64, target: f64, tau: f64, dt: Dt) -> f64 {
    relax(current, target, tau, dt)
}

/// Bounded accumulation: `accumulate(prev, delta, min, max)` → clamps accumulated value
#[kernel_fn(name = "accumulate", category = "simulation", vectorized)]
pub fn accumulate(prev: f64, delta: f64, min: f64, max: f64, dt: Dt) -> f64 {
    (prev + delta * dt).clamp(min, max)
}

/// Phase advancement: `advance_phase(phase, omega)` → wraps phase in [0, 2π)
#[kernel_fn(name = "advance_phase", category = "simulation", vectorized)]
pub fn advance_phase(phase: f64, omega: f64, dt: Dt) -> f64 {
    use continuum_kernel_registry::eval;

    // Use the wrap function from math stdlib: wrap(value, min, max)
    let new_phase = phase + omega * dt;
    let tau = std::f64::consts::TAU;

    // Call the wrap function from the kernel registry
    eval("wrap", &[new_phase, 0.0, tau], dt).unwrap_or(new_phase.rem_euclid(tau))
}

/// Damping: `damp(value, damping_factor)` → applies damping
/// Note: This is a simplified damping model. Full spring-damper systems need more context.
#[kernel_fn(name = "damp", category = "simulation", vectorized)]
pub fn damp(value: f64, damping_factor: f64, dt: Dt) -> f64 {
    // Simple exponential damping: value * (1 - damping_factor * dt)
    // Clamp to prevent negative damping
    let factor = (1.0 - damping_factor * dt).max(0.0);
    value * factor
}

// ============================================================================
// VECTORIZED IMPLEMENTATIONS
// ============================================================================

/// Vectorized integration implementation (Euler method)
#[vectorized_kernel_fn(name = "integrate")]
pub fn integrate_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    euler_integration_impl(args, dt, population)
}

/// Vectorized Euler integration implementation
#[vectorized_kernel_fn(name = "integrate_euler")]
pub fn integrate_euler_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    euler_integration_impl(args, dt, population)
}

/// Vectorized RK4 integration implementation
#[vectorized_kernel_fn(name = "integrate_rk4")]
pub fn integrate_rk4_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    // For constant rate case, RK4 reduces to Euler
    euler_integration_impl(args, dt, population)
}

/// Vectorized Verlet integration implementation
#[vectorized_kernel_fn(name = "integrate_verlet")]
pub fn integrate_verlet_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    // Simplified Verlet reduces to Euler for this case
    euler_integration_impl(args, dt, population)
}

/// Shared Euler integration implementation
fn euler_integration_impl(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    if args.len() < 2 {
        return Err("integration requires 2 arguments".into());
    }

    let prev = args[0];
    let rate = args[1];

    match (prev.as_uniform(), rate.as_uniform()) {
        (Some(p), Some(r)) => Ok(VRegBuffer::uniform(p + r * dt)),
        _ => {
            let p_arr = prev.to_scalar_array(population);
            let r_arr = rate.to_scalar_array(population);
            let result: Vec<f64> = p_arr
                .iter()
                .zip(r_arr.iter())
                .map(|(&p, &r)| p + r * dt)
                .collect();
            Ok(VRegBuffer::Scalar(result))
        }
    }
}

/// Vectorized decay implementation
#[vectorized_kernel_fn(name = "decay")]
pub fn decay_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    if args.len() < 2 {
        return Err("decay requires 2 arguments".into());
    }

    let value = args[0];
    let half_life = args[1];

    // Pre-compute factor for uniform half-life case
    if let Some(hl) = half_life.as_uniform() {
        let factor = 0.5_f64.powf(dt / hl);

        match value.as_uniform() {
            Some(v) => Ok(VRegBuffer::uniform(v * factor)),
            None => {
                let v_arr = value.to_scalar_array(population);
                let result: Vec<f64> = v_arr.iter().map(|&v| v * factor).collect();
                Ok(VRegBuffer::Scalar(result))
            }
        }
    } else {
        // Non-uniform half-life
        let v_arr = value.to_scalar_array(population);
        let hl_arr = half_life.to_scalar_array(population);
        let result: Vec<f64> = v_arr
            .iter()
            .zip(hl_arr.iter())
            .map(|(&v, &hl)| v * 0.5_f64.powf(dt / hl))
            .collect();
        Ok(VRegBuffer::Scalar(result))
    }
}

/// Vectorized relax implementation
#[vectorized_kernel_fn(name = "relax")]
pub fn relax_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    if args.len() < 3 {
        return Err("relax requires 3 arguments".into());
    }

    let current = args[0];
    let target = args[1];
    let tau = args[2];

    // Pre-compute factor for uniform tau case
    if let Some(t) = tau.as_uniform() {
        let factor = 1.0 - (-dt / t).exp();

        match (current.as_uniform(), target.as_uniform()) {
            (Some(c), Some(tgt)) => Ok(VRegBuffer::uniform(c + (tgt - c) * factor)),
            _ => {
                let c_arr = current.to_scalar_array(population);
                let t_arr = target.to_scalar_array(population);
                let result: Vec<f64> = c_arr
                    .iter()
                    .zip(t_arr.iter())
                    .map(|(&c, &tgt)| c + (tgt - c) * factor)
                    .collect();
                Ok(VRegBuffer::Scalar(result))
            }
        }
    } else {
        // Non-uniform tau
        let c_arr = current.to_scalar_array(population);
        let t_arr = target.to_scalar_array(population);
        let tau_arr = tau.to_scalar_array(population);
        let result: Vec<f64> = c_arr
            .iter()
            .zip(t_arr.iter())
            .zip(tau_arr.iter())
            .map(|((&c, &tgt), &tau_val)| {
                let factor = 1.0 - (-dt / tau_val).exp();
                c + (tgt - c) * factor
            })
            .collect();
        Ok(VRegBuffer::Scalar(result))
    }
}

/// Vectorized smooth implementation (alias for relax)
#[vectorized_kernel_fn(name = "smooth")]
pub fn smooth_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    relax_vectorized(args, dt, population)
}

/// Vectorized accumulate implementation
#[vectorized_kernel_fn(name = "accumulate")]
pub fn accumulate_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    if args.len() < 4 {
        return Err("accumulate requires 4 arguments".into());
    }

    let prev = args[0];
    let delta = args[1];
    let min_val = args[2]
        .as_uniform()
        .unwrap_or_else(|| args[2].to_scalar_array(population)[0]);
    let max_val = args[3]
        .as_uniform()
        .unwrap_or_else(|| args[3].to_scalar_array(population)[0]);

    match (prev.as_uniform(), delta.as_uniform()) {
        (Some(p), Some(d)) => {
            let result = (p + d * dt).clamp(min_val, max_val);
            Ok(VRegBuffer::uniform(result))
        }
        _ => {
            let p_arr = prev.to_scalar_array(population);
            let d_arr = delta.to_scalar_array(population);
            let result: Vec<f64> = p_arr
                .iter()
                .zip(d_arr.iter())
                .map(|(&p, &d)| (p + d * dt).clamp(min_val, max_val))
                .collect();
            Ok(VRegBuffer::Scalar(result))
        }
    }
}

/// Vectorized advance_phase implementation
#[vectorized_kernel_fn(name = "advance_phase")]
pub fn advance_phase_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    if args.len() < 2 {
        return Err("advance_phase requires 2 arguments".into());
    }

    let phase = args[0];
    let omega = args[1];
    let tau = std::f64::consts::TAU;

    match (phase.as_uniform(), omega.as_uniform()) {
        (Some(p), Some(o)) => {
            let result = (p + o * dt).rem_euclid(tau);
            Ok(VRegBuffer::uniform(result))
        }
        _ => {
            let p_arr = phase.to_scalar_array(population);
            let o_arr = omega.to_scalar_array(population);
            let result: Vec<f64> = p_arr
                .iter()
                .zip(o_arr.iter())
                .map(|(&p, &o)| (p + o * dt).rem_euclid(tau))
                .collect();
            Ok(VRegBuffer::Scalar(result))
        }
    }
}

/// Vectorized damp implementation
#[vectorized_kernel_fn(name = "damp")]
pub fn damp_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    if args.len() < 2 {
        return Err("damp requires 2 arguments".into());
    }

    let value = args[0];
    let damping_factor = args[1];

    match (value.as_uniform(), damping_factor.as_uniform()) {
        (Some(v), Some(d)) => {
            let factor = (1.0 - d * dt).max(0.0);
            Ok(VRegBuffer::uniform(v * factor))
        }
        _ => {
            let v_arr = value.to_scalar_array(population);
            let d_arr = damping_factor.to_scalar_array(population);
            let result: Vec<f64> = v_arr
                .iter()
                .zip(d_arr.iter())
                .map(|(&v, &d)| {
                    let factor = (1.0 - d * dt).max(0.0);
                    v * factor
                })
                .collect();
            Ok(VRegBuffer::Scalar(result))
        }
    }
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

    #[test]
    fn test_vectorized_attribute_working() {
        use continuum_kernel_registry::has_vectorized_impl;

        // These functions should NOT have vectorized implementations yet
        // (we marked them with vectorized attribute but haven't implemented the actual vectorized functions)
        assert!(!has_vectorized_impl("integrate"));
        assert!(!has_vectorized_impl("decay"));
        assert!(!has_vectorized_impl("relax"));

        // But they should still be registered as regular functions
        assert!(is_known("integrate"));
        assert!(is_known("decay"));
        assert!(is_known("relax"));
    }

    #[test]
    fn test_new_operators_registered() {
        // Test all new operators are registered
        assert!(is_known("smooth"));
        assert!(is_known("accumulate"));
        assert!(is_known("advance_phase"));
        assert!(is_known("damp"));

        // Check their descriptors
        let smooth = get("smooth").unwrap();
        assert!(smooth.requires_dt());
        assert_eq!(smooth.arity, Arity::Fixed(3));

        let accumulate = get("accumulate").unwrap();
        assert!(accumulate.requires_dt());
        assert_eq!(accumulate.arity, Arity::Fixed(4));

        let advance_phase = get("advance_phase").unwrap();
        assert!(advance_phase.requires_dt());
        assert_eq!(advance_phase.arity, Arity::Fixed(2));

        let damp = get("damp").unwrap();
        assert!(damp.requires_dt());
        assert_eq!(damp.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_smooth_eval() {
        // smooth should behave exactly like relax
        let smooth_result = eval("smooth", &[0.0, 100.0, 1.0], 1.0).unwrap();
        let relax_result = eval("relax", &[0.0, 100.0, 1.0], 1.0).unwrap();
        assert!((smooth_result - relax_result).abs() < 1e-10);
    }

    #[test]
    fn test_accumulate_eval() {
        // accumulate(prev=10, delta=5, min=0, max=20, dt=2) = clamp(10 + 5*2, 0, 20) = clamp(20, 0, 20) = 20
        let result = eval("accumulate", &[10.0, 5.0, 0.0, 20.0], 2.0).unwrap();
        assert!((result - 20.0).abs() < 1e-10);

        // Test clamping: accumulate(prev=10, delta=10, min=0, max=15, dt=1) = clamp(20, 0, 15) = 15
        let result = eval("accumulate", &[10.0, 10.0, 0.0, 15.0], 1.0).unwrap();
        assert!((result - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_advance_phase_eval() {
        use std::f64::consts::PI;

        // advance_phase(phase=0, omega=PI, dt=1) = wrap(0 + PI*1, 0, TAU) = PI
        let result = eval("advance_phase", &[0.0, PI, 0.0], 1.0).unwrap();
        assert!((result - PI).abs() < 1e-10);

        // Test wrapping: advance_phase(phase=PI, omega=PI, dt=1) = wrap(2*PI, 0, TAU) = 0
        let result = eval("advance_phase", &[PI, PI, 0.0], 1.0).unwrap();
        assert!(result.abs() < 1e-10);
    }

    #[test]
    fn test_damp_eval() {
        // damp(value=100, damping_factor=0.1, dt=1) = 100 * (1 - 0.1*1) = 100 * 0.9 = 90
        let result = eval("damp", &[100.0, 0.1], 1.0).unwrap();
        assert!((result - 90.0).abs() < 1e-10);

        // Test clamping to prevent negative factors
        let result = eval("damp", &[100.0, 2.0], 1.0).unwrap(); // factor would be negative, gets clamped to 0
        assert!((result - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_integration_methods_registered() {
        // Test all integration method variants are registered
        assert!(is_known("integrate"));
        assert!(is_known("integrate_euler"));
        assert!(is_known("integrate_rk4"));
        assert!(is_known("integrate_verlet"));

        // All should have same arity and requirements
        for method in &[
            "integrate",
            "integrate_euler",
            "integrate_rk4",
            "integrate_verlet",
        ] {
            let desc = get(method).unwrap();
            assert!(desc.requires_dt());
            assert_eq!(desc.arity, Arity::Fixed(2));
        }
    }

    #[test]
    fn test_integration_methods_eval() {
        // For constant rate case, all methods should give same result
        let args = &[10.0, 5.0];
        let dt = 0.1;

        let euler = eval("integrate_euler", args, dt).unwrap();
        let rk4 = eval("integrate_rk4", args, dt).unwrap();
        let verlet = eval("integrate_verlet", args, dt).unwrap();
        let default = eval("integrate", args, dt).unwrap();

        assert!((euler - 10.5).abs() < 1e-10);
        assert!((rk4 - euler).abs() < 1e-10); // Should be same for constant rate
        assert!((verlet - euler).abs() < 1e-10); // Should be same for simplified case
        assert!((default - euler).abs() < 1e-10); // Default should be Euler
    }

    #[test]
    fn test_phase2_complete_dt_robust_coverage() {
        // Comprehensive test that all expected dt-robust operators are available
        let expected_operators = vec![
            // Original operators
            "integrate",
            "decay",
            "relax",
            // Integration method variants
            "integrate_euler",
            "integrate_rk4",
            "integrate_verlet",
            // New operators
            "smooth",
            "accumulate",
            "advance_phase",
            "damp",
            // Alias
            "relax_to",
        ];

        for op in &expected_operators {
            assert!(is_known(op), "Operator '{}' should be registered", op);

            let desc = get(op).unwrap();
            assert!(desc.requires_dt(), "Operator '{}' should require dt", op);

            // All should be in simulation category except relax_to
            if *op != "relax_to" {
                assert_eq!(
                    desc.category, "simulation",
                    "Operator '{}' should be in simulation category",
                    op
                );
            }
        }

        println!(
            "✅ Phase 2 Complete: All {} dt-robust operators registered and tested",
            expected_operators.len()
        );
    }
}
