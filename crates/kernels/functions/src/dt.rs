//! dt-Robust Operators
//!
//! These operators provide numerically stable alternatives to raw dt usage.
//! They are designed to produce consistent results regardless of timestep size.

use continuum_foundation::{Dt, Value};
use continuum_kernel_macros::{kernel_fn, vectorized_kernel_fn};
use continuum_kernel_registry::{VRegBuffer, VectorizedResult, eval_in_namespace};

/// Integration: `integrate(prev, rate)` → `prev + rate * dt`
/// Default uses Euler method
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    vectorized
)]
pub fn integrate(prev: f64, rate: f64, dt: Dt) -> f64 {
    prev + rate * dt
}

/// Euler integration: `integrate_euler(prev, rate)` → `prev + rate * dt`
/// Explicit Euler method (same as default integrate)
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    vectorized
)]
pub fn integrate_euler(prev: f64, rate: f64, dt: Dt) -> f64 {
    prev + rate * dt
}

/// RK4 integration: `integrate_rk4(prev, rate)` → higher-order integration
/// Note: This is a simplified RK4 that assumes constant rate over dt
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    vectorized
)]
pub fn integrate_rk4(prev: f64, rate: f64, dt: Dt) -> f64 {
    // Simplified RK4 for constant rate case
    // For constant rate: result = prev + 6 * rate * dt / 6 = prev + rate * dt
    prev + rate * dt
}

/// Verlet integration: `integrate_verlet(prev, rate)` → Velocity Verlet approximation
/// Note: This is simplified for single-variable case
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    vectorized
)]
pub fn integrate_verlet(prev: f64, rate: f64, dt: Dt) -> f64 {
    // Simplified Verlet integration
    prev + rate * dt
}

/// Exponential decay: `decay(value, halflife)` → `value * 0.5^(dt/halflife)`
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    vectorized
)]
pub fn decay(value: f64, halflife: f64, dt: Dt) -> f64 {
    value * 0.5_f64.powf(dt / halflife)
}

/// Exponential relaxation: `relax(current, target, tau)` → approaches target
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0), AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0), UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    vectorized
)]
pub fn relax(current: f64, target: f64, tau: f64, dt: Dt) -> f64 {
    let alpha = std::f64::consts::E.powf(-dt / tau);
    target + (current - target) * alpha
}

/// Exponential relaxation: `relax_to(current, target, tau)` → approaches target
/// Alias for `relax`
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0), AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0), UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0)
)]
pub fn relax_to(current: f64, target: f64, tau: f64, dt: Dt) -> f64 {
    relax(current, target, tau, dt)
}

/// Smooth transition: `smooth(current, target, tau)` → approaches target
/// Same as relax - exponential approach to target value
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, SameAs(0), AnyScalar],
    unit_in = [UnitAny, UnitSameAs(0), UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    vectorized
)]
pub fn smooth(current: f64, target: f64, tau: f64, dt: Dt) -> f64 {
    relax(current, target, tau, dt)
}

/// Bounded accumulation: `accumulate(prev, delta, min, max)` → clamps accumulated value
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar, SameAs(0), SameAs(0)],
    unit_in = [UnitAny, UnitAny, UnitSameAs(0), UnitSameAs(0)],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    vectorized
)]
pub fn accumulate(prev: f64, delta: f64, min: f64, max: f64, dt: Dt) -> f64 {
    (prev + delta * dt).clamp(min, max)
}

/// Phase advancement: `advance_phase(phase, omega)` → wraps phase in [0, 2π)
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    vectorized
)]
pub fn advance_phase(phase: f64, omega: f64, dt: Dt) -> f64 {
    // Use the wrap function from math stdlib: wrap(value, min, max)
    let new_phase = phase + omega * dt;
    let tau = std::f64::consts::TAU;

    // Call the wrap function from the kernel registry
    eval_in_namespace(
        "maths",
        "wrap",
        &[
            Value::Scalar(new_phase),
            Value::Scalar(0.0),
            Value::Scalar(tau),
        ],
        dt,
    )
    .and_then(|v| v.as_scalar())
    .unwrap_or(new_phase.rem_euclid(tau))
}

/// Damping: `damp(value, damping_factor)` → applies damping
/// Note: This is a simplified damping model. Full spring-damper systems need more context.
#[kernel_fn(
    namespace = "dt",
    purity = Pure,
    shape_in = [AnyScalar, AnyScalar],
    unit_in = [UnitAny, UnitAny],
    shape_out = Scalar,
    unit_out = UnitDerivSameAs(0),
    vectorized
)]
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
#[vectorized_kernel_fn(name = "integrate", namespace = "dt")]
pub fn integrate_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    euler_integration_impl(args, dt, population)
}

/// Vectorized Euler integration implementation
#[vectorized_kernel_fn(name = "integrate_euler", namespace = "dt")]
pub fn integrate_euler_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    euler_integration_impl(args, dt, population)
}

/// Vectorized RK4 integration implementation
#[vectorized_kernel_fn(name = "integrate_rk4", namespace = "dt")]
pub fn integrate_rk4_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    euler_integration_impl(args, dt, population)
}

/// Vectorized Verlet integration implementation
#[vectorized_kernel_fn(name = "integrate_verlet", namespace = "dt")]
pub fn integrate_verlet_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
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

    match (prev.get_scalar(0), rate.get_scalar(0)) {
        // Optimization for uniform values
        (Some(p), Some(r)) if prev.as_uniform().is_some() && rate.as_uniform().is_some() => {
            Ok(VRegBuffer::uniform_scalar(p + r * dt))
        }
        _ => {
            let p_arr = prev
                .to_scalar_array(population)
                .ok_or("Type mismatch: expected scalar array")?;
            let r_arr = rate
                .to_scalar_array(population)
                .ok_or("Type mismatch: expected scalar array")?;
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
#[vectorized_kernel_fn(name = "decay", namespace = "dt")]
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

    if let Some(hl) = half_life.as_uniform().and_then(|v| v.as_scalar()) {
        let factor = 0.5_f64.powf(dt / hl);

        if let Some(v) = value.as_uniform().and_then(|v| v.as_scalar()) {
            Ok(VRegBuffer::uniform_scalar(v * factor))
        } else {
            let v_arr = value
                .to_scalar_array(population)
                .ok_or("Expected scalar array")?;
            let result: Vec<f64> = v_arr.iter().map(|&v| v * factor).collect();
            Ok(VRegBuffer::Scalar(result))
        }
    } else {
        let v_arr = value
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
        let hl_arr = half_life
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
        let result: Vec<f64> = v_arr
            .iter()
            .zip(hl_arr.iter())
            .map(|(&v, &hl)| v * 0.5_f64.powf(dt / hl))
            .collect();
        Ok(VRegBuffer::Scalar(result))
    }
}

/// Vectorized relax implementation
#[vectorized_kernel_fn(name = "relax", namespace = "dt")]
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

    if let Some(t) = tau.as_uniform().and_then(|v| v.as_scalar()) {
        let factor = 1.0 - (-dt / t).exp();

        if current.as_uniform().is_some() && target.as_uniform().is_some() {
            let c = current.get_scalar(0).unwrap();
            let tgt = target.get_scalar(0).unwrap();
            Ok(VRegBuffer::uniform_scalar(c + (tgt - c) * factor))
        } else {
            let c_arr = current
                .to_scalar_array(population)
                .ok_or("Expected scalar array")?;
            let t_arr = target
                .to_scalar_array(population)
                .ok_or("Expected scalar array")?;
            let result: Vec<f64> = c_arr
                .iter()
                .zip(t_arr.iter())
                .map(|(&c, &tgt)| c + (tgt - c) * factor)
                .collect();
            Ok(VRegBuffer::Scalar(result))
        }
    } else {
        let c_arr = current
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
        let t_arr = target
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
        let tau_arr = tau
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
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
#[vectorized_kernel_fn(name = "smooth", namespace = "dt")]
pub fn smooth_vectorized(
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> VectorizedResult<VRegBuffer> {
    relax_vectorized(args, dt, population)
}

/// Vectorized accumulate implementation
#[vectorized_kernel_fn(name = "accumulate", namespace = "dt")]
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

    // Helper to get value or first element of array (clunky but matches previous logic)
    let get_val = |buf: &VRegBuffer, idx: usize| -> f64 { buf.get_scalar(idx).unwrap_or(0.0) };

    // If everything is uniform
    if prev.as_uniform().is_some()
        && delta.as_uniform().is_some()
        && args[2].as_uniform().is_some()
        && args[3].as_uniform().is_some()
    {
        let p = get_val(prev, 0);
        let d = get_val(delta, 0);
        let min_val = get_val(args[2], 0);
        let max_val = get_val(args[3], 0);

        let result = (p + d * dt).clamp(min_val, max_val);
        Ok(VRegBuffer::uniform_scalar(result))
    } else {
        let p_arr = prev
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
        let d_arr = delta
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;

        // For min/max, they might be uniform or arrays
        let min_arr = args[2]
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
        let max_arr = args[3]
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;

        let result: Vec<f64> = p_arr
            .iter()
            .zip(d_arr.iter())
            .zip(min_arr.iter())
            .zip(max_arr.iter())
            .map(|(((&p, &d), &min_v), &max_v)| (p + d * dt).clamp(min_v, max_v))
            .collect();
        Ok(VRegBuffer::Scalar(result))
    }
}

/// Vectorized advance_phase implementation
#[vectorized_kernel_fn(name = "advance_phase", namespace = "dt")]
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

    if phase.as_uniform().is_some() && omega.as_uniform().is_some() {
        let p = phase.get_scalar(0).unwrap();
        let o = omega.get_scalar(0).unwrap();
        let result = (p + o * dt).rem_euclid(tau);
        Ok(VRegBuffer::uniform_scalar(result))
    } else {
        let p_arr = phase
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
        let o_arr = omega
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
        let result: Vec<f64> = p_arr
            .iter()
            .zip(o_arr.iter())
            .map(|(&p, &o)| (p + o * dt).rem_euclid(tau))
            .collect();
        Ok(VRegBuffer::Scalar(result))
    }
}

/// Vectorized damp implementation
#[vectorized_kernel_fn(name = "damp", namespace = "dt")]
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

    if value.as_uniform().is_some() && damping_factor.as_uniform().is_some() {
        let v = value.get_scalar(0).unwrap();
        let d = damping_factor.get_scalar(0).unwrap();
        let factor = (1.0 - d * dt).max(0.0);
        Ok(VRegBuffer::uniform_scalar(v * factor))
    } else {
        let v_arr = value
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
        let d_arr = damping_factor
            .to_scalar_array(population)
            .ok_or("Expected scalar array")?;
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

#[cfg(test)]
mod tests {
    use continuum_kernel_registry::{
        Arity, Value, eval_in_namespace, get_in_namespace, is_known_in,
    };

    #[test]
    fn test_integrate_registered() {
        assert!(is_known_in("dt", "integrate"));
        let desc = get_in_namespace("dt", "integrate").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
        assert!(desc.requires_dt());
    }

    #[test]
    fn test_integrate_eval() {
        // integrate(10, 5, dt=0.1) = 10 + 5*0.1 = 10.5
        let args = [Value::Scalar(10.0), Value::Scalar(5.0)];
        let result = eval_in_namespace("dt", "integrate", &args, 0.1).unwrap();
        assert!((result.as_scalar().unwrap() - 10.5).abs() < 1e-10);
    }

    #[test]
    fn test_decay_registered() {
        assert!(is_known_in("dt", "decay"));
        let desc = get_in_namespace("dt", "decay").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(2));
        assert!(desc.requires_dt());
    }

    #[test]
    fn test_decay_eval() {
        // decay(100, 10, dt=10) = 100 * 0.5^1 = 50
        let args = [Value::Scalar(100.0), Value::Scalar(10.0)];
        let result = eval_in_namespace("dt", "decay", &args, 10.0).unwrap();
        assert!((result.as_scalar().unwrap() - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_relax_registered() {
        assert!(is_known_in("dt", "relax"));
        let desc = get_in_namespace("dt", "relax").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(3));
        assert!(desc.requires_dt());
    }

    #[test]
    fn test_relax_eval() {
        // relax(0, 100, tau=1, dt=1) ≈ 63.2
        let args = [Value::Scalar(0.0), Value::Scalar(100.0), Value::Scalar(1.0)];
        let result = eval_in_namespace("dt", "relax", &args, 1.0).unwrap();
        let expected = 100.0 * (1.0 - std::f64::consts::E.powf(-1.0));
        assert!((result.as_scalar().unwrap() - expected).abs() < 0.01);
    }

    #[test]
    fn test_vectorized_attribute_working() {
        use continuum_kernel_registry::has_vectorized_impl;

        // Vectorized implementations should be registered
        assert!(has_vectorized_impl("dt", "integrate"));
        assert!(has_vectorized_impl("dt", "integrate_euler"));
        assert!(has_vectorized_impl("dt", "integrate_rk4"));
        assert!(has_vectorized_impl("dt", "integrate_verlet"));
        assert!(has_vectorized_impl("dt", "decay"));
        assert!(has_vectorized_impl("dt", "relax"));
        assert!(has_vectorized_impl("dt", "smooth"));
        assert!(has_vectorized_impl("dt", "accumulate"));
        assert!(has_vectorized_impl("dt", "advance_phase"));
        assert!(has_vectorized_impl("dt", "damp"));

        // All should still be registered as regular functions
        assert!(is_known_in("dt", "integrate"));
        assert!(is_known_in("dt", "decay"));
        assert!(is_known_in("dt", "relax"));
    }

    #[test]
    fn test_new_operators_registered() {
        // Test all new operators are registered
        assert!(is_known_in("dt", "smooth"));
        assert!(is_known_in("dt", "accumulate"));
        assert!(is_known_in("dt", "advance_phase"));
        assert!(is_known_in("dt", "damp"));

        // Check their descriptors
        let smooth = get_in_namespace("dt", "smooth").unwrap();
        assert!(smooth.requires_dt());
        assert_eq!(smooth.arity, Arity::Fixed(3));

        let accumulate = get_in_namespace("dt", "accumulate").unwrap();
        assert!(accumulate.requires_dt());
        assert_eq!(accumulate.arity, Arity::Fixed(4));

        let advance_phase = get_in_namespace("dt", "advance_phase").unwrap();
        assert!(advance_phase.requires_dt());
        assert_eq!(advance_phase.arity, Arity::Fixed(2));

        let damp = get_in_namespace("dt", "damp").unwrap();
        assert!(damp.requires_dt());
        assert_eq!(damp.arity, Arity::Fixed(2));
    }

    #[test]
    fn test_smooth_eval() {
        // smooth should behave exactly like relax
        let args = [Value::Scalar(0.0), Value::Scalar(100.0), Value::Scalar(1.0)];
        let smooth_result = eval_in_namespace("dt", "smooth", &args, 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        let relax_result = eval_in_namespace("dt", "relax", &args, 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((smooth_result - relax_result).abs() < 1e-10);
    }

    #[test]
    fn test_accumulate_eval() {
        // accumulate(prev=10, delta=5, min=0, max=20, dt=2) = clamp(10 + 5*2, 0, 20) = clamp(20, 0, 20) = 20
        let args = [
            Value::Scalar(10.0),
            Value::Scalar(5.0),
            Value::Scalar(0.0),
            Value::Scalar(20.0),
        ];
        let result = eval_in_namespace("dt", "accumulate", &args, 2.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 20.0).abs() < 1e-10);

        // Test clamping: accumulate(prev=10, delta=10, min=0, max=15, dt=1) = clamp(20, 0, 15) = 15
        let args2 = [
            Value::Scalar(10.0),
            Value::Scalar(10.0),
            Value::Scalar(0.0),
            Value::Scalar(15.0),
        ];
        let result2 = eval_in_namespace("dt", "accumulate", &args2, 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result2 - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_advance_phase_eval() {
        use std::f64::consts::PI;

        // advance_phase(phase=0, omega=PI, dt=1) = wrap(0 + PI*1, 0, TAU) = PI
        let args = [Value::Scalar(0.0), Value::Scalar(PI)];
        let result = eval_in_namespace("dt", "advance_phase", &args, 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - PI).abs() < 1e-10);

        // Test wrapping: advance_phase(phase=PI, omega=PI, dt=1) = wrap(2*PI, 0, TAU) = 0
        let args2 = [Value::Scalar(PI), Value::Scalar(PI)];
        let result2 = eval_in_namespace("dt", "advance_phase", &args2, 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!(result2.abs() < 1e-10);
    }

    #[test]
    fn test_damp_eval() {
        // damp(value=100, damping_factor=0.1, dt=1) = 100 * (1 - 0.1*1) = 100 * 0.9 = 90
        let args = [Value::Scalar(100.0), Value::Scalar(0.1)];
        let result = eval_in_namespace("dt", "damp", &args, 1.0)
            .unwrap()
            .as_scalar()
            .unwrap();
        assert!((result - 90.0).abs() < 1e-10);

        // Test clamping to prevent negative factors
        let args2 = [Value::Scalar(100.0), Value::Scalar(2.0)];
        let result2 = eval_in_namespace("dt", "damp", &args2, 1.0)
            .unwrap()
            .as_scalar()
            .unwrap(); // factor would be negative, gets clamped to 0
        assert!((result2 - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_integration_methods_registered() {
        // Test all integration method variants are registered
        assert!(is_known_in("dt", "integrate"));
        assert!(is_known_in("dt", "integrate_euler"));
        assert!(is_known_in("dt", "integrate_rk4"));
        assert!(is_known_in("dt", "integrate_verlet"));

        // All should have same arity and requirements
        for method in &[
            "integrate",
            "integrate_euler",
            "integrate_rk4",
            "integrate_verlet",
        ] {
            let desc = get_in_namespace("dt", method).unwrap();
            assert!(desc.requires_dt());
            assert_eq!(desc.arity, Arity::Fixed(2));
        }
    }

    #[test]
    fn test_integration_methods_eval() {
        // For constant rate case, all methods should give same result
        let args = [Value::Scalar(10.0), Value::Scalar(5.0)];
        let dt = 0.1;

        let euler = eval_in_namespace("dt", "integrate_euler", &args, dt)
            .unwrap()
            .as_scalar()
            .unwrap();
        let rk4 = eval_in_namespace("dt", "integrate_rk4", &args, dt)
            .unwrap()
            .as_scalar()
            .unwrap();
        let verlet = eval_in_namespace("dt", "integrate_verlet", &args, dt)
            .unwrap()
            .as_scalar()
            .unwrap();
        let default = eval_in_namespace("dt", "integrate", &args, dt)
            .unwrap()
            .as_scalar()
            .unwrap();

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
            assert!(
                is_known_in("dt", op),
                "Operator '{}' should be registered",
                op
            );

            let desc = get_in_namespace("dt", op).unwrap();
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
    }
}
