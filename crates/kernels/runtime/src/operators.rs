//! dt-Robust Operators
//!
//! These operators provide numerically stable alternatives to raw dt usage.
//! They are designed to produce consistent results regardless of timestep size.
//!
//! # Design Principles
//!
//! All operators:
//! - Are deterministic (same inputs → same outputs)
//! - Are stable (bounded output for bounded input at any reasonable dt)
//! - Are convergent (approach correct continuous solution as dt → 0)
//! - Are symmetric (dt=0.1 twice ≈ dt=0.2 once)

use std::f64::consts::{E, TAU};

/// Integration methods for numerical integration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegrationMethod {
    /// Simple Euler integration (first-order)
    #[default]
    Euler,
    /// Midpoint method (second-order)
    Midpoint,
    /// Classic Runge-Kutta (fourth-order)
    Rk4,
}

/// Integrate a rate over time
///
/// Accumulates a rate value over the timestep.
///
/// # Arguments
/// * `prev` - Previous value
/// * `rate` - Rate of change (units per second)
/// * `dt` - Timestep in seconds
///
/// # Example
/// ```
/// use continuum_runtime::operators::integrate;
///
/// let position = 10.0;
/// let velocity = 5.0;
/// let dt = 0.1;
/// let new_position = integrate(position, velocity, dt);
/// assert!((new_position - 10.5).abs() < 1e-10);
/// ```
#[inline]
pub fn integrate(prev: f64, rate: f64, dt: f64) -> f64 {
    prev + rate * dt
}

/// Integrate with a specific method (for multi-step methods, rate_fn provides rate at any point)
///
/// For simple cases, use `integrate()`. This is for when you need higher-order methods.
#[inline]
pub fn integrate_with_method<F>(prev: f64, rate_fn: F, dt: f64, method: IntegrationMethod) -> f64
where
    F: Fn(f64) -> f64,
{
    match method {
        IntegrationMethod::Euler => prev + rate_fn(prev) * dt,
        IntegrationMethod::Midpoint => {
            let k1 = rate_fn(prev);
            let mid = prev + k1 * dt * 0.5;
            prev + rate_fn(mid) * dt
        }
        IntegrationMethod::Rk4 => {
            let k1 = rate_fn(prev);
            let k2 = rate_fn(prev + k1 * dt * 0.5);
            let k3 = rate_fn(prev + k2 * dt * 0.5);
            let k4 = rate_fn(prev + k3 * dt);
            prev + (k1 + 2.0 * k2 + 2.0 * k3 + k4) * dt / 6.0
        }
    }
}

/// Exponential decay toward zero
///
/// Uses the exact exponential solution, stable at any dt.
///
/// # Arguments
/// * `value` - Current value
/// * `halflife` - Time for value to decay to half (in seconds)
/// * `dt` - Timestep in seconds
///
/// # Formula
/// `value * 0.5^(dt/halflife)`
///
/// # Example
/// ```
/// use continuum_runtime::operators::decay;
///
/// let value = 100.0;
/// let halflife = 10.0;
/// let dt = 10.0;
/// let decayed = decay(value, halflife, dt);
/// assert!((decayed - 50.0).abs() < 1e-10);
/// ```
#[inline]
pub fn decay(value: f64, halflife: f64, dt: f64) -> f64 {
    value * 0.5_f64.powf(dt / halflife)
}

/// Exponential decay with rate constant
///
/// # Arguments
/// * `value` - Current value
/// * `rate` - Decay rate constant (1/seconds)
/// * `dt` - Timestep in seconds
///
/// # Formula
/// `value * e^(-rate * dt)`
#[inline]
pub fn decay_rate(value: f64, rate: f64, dt: f64) -> f64 {
    value * E.powf(-rate * dt)
}

/// Exponential decay with time constant (tau)
///
/// # Arguments
/// * `value` - Current value
/// * `tau` - Time constant in seconds (time to decay to ~37%)
/// * `dt` - Timestep in seconds
///
/// # Formula
/// `value * e^(-dt/tau)`
#[inline]
pub fn decay_tau(value: f64, tau: f64, dt: f64) -> f64 {
    value * E.powf(-dt / tau)
}

/// Exponential relaxation toward a target value
///
/// Uses the exact exponential solution, stable at any dt.
///
/// # Arguments
/// * `current` - Current value
/// * `target` - Target value to relax toward
/// * `tau` - Time constant (63% of the way after tau seconds)
/// * `dt` - Timestep in seconds
///
/// # Formula
/// `target + (current - target) * e^(-dt/tau)`
///
/// # Example
/// ```
/// use continuum_runtime::operators::relax;
///
/// let current = 0.0;
/// let target = 100.0;
/// let tau = 1.0;
/// let dt = 1.0;
/// let relaxed = relax(current, target, tau, dt);
/// // After one time constant, should be ~63% of the way
/// assert!((relaxed - 63.212).abs() < 0.01);
/// ```
#[inline]
pub fn relax(current: f64, target: f64, tau: f64, dt: f64) -> f64 {
    let alpha = E.powf(-dt / tau);
    target + (current - target) * alpha
}

/// Exponential relaxation with halflife
///
/// # Arguments
/// * `current` - Current value
/// * `target` - Target value to relax toward
/// * `halflife` - Time to reach halfway to target
/// * `dt` - Timestep in seconds
#[inline]
pub fn relax_halflife(current: f64, target: f64, halflife: f64, dt: f64) -> f64 {
    let alpha = 0.5_f64.powf(dt / halflife);
    target + (current - target) * alpha
}

/// Bounded accumulation (integrate with clamping)
///
/// Integrates a delta over time, clamping to bounds.
///
/// # Arguments
/// * `prev` - Previous value
/// * `delta` - Rate of change (units per second)
/// * `dt` - Timestep in seconds
/// * `min` - Minimum bound
/// * `max` - Maximum bound
///
/// # Example
/// ```
/// use continuum_runtime::operators::accumulate;
///
/// let prev = 90.0;
/// let delta = 50.0;
/// let dt = 1.0;
/// let result = accumulate(prev, delta, dt, 0.0, 100.0);
/// assert_eq!(result, 100.0); // Clamped to max
/// ```
#[inline]
pub fn accumulate(prev: f64, delta: f64, dt: f64, min: f64, max: f64) -> f64 {
    (prev + delta * dt).clamp(min, max)
}

/// Advance a cyclic phase value
///
/// Advances a phase by omega * dt and wraps at the period boundary.
///
/// # Arguments
/// * `phase` - Current phase value
/// * `omega` - Angular velocity (radians per second for default period)
/// * `dt` - Timestep in seconds
///
/// # Example
/// ```
/// use continuum_runtime::operators::advance_phase;
/// use std::f64::consts::TAU;
///
/// let phase = 0.0;
/// let omega = TAU; // One full rotation per second
/// let dt = 0.5;
/// let new_phase = advance_phase(phase, omega, dt);
/// assert!((new_phase - std::f64::consts::PI).abs() < 1e-10);
/// ```
#[inline]
pub fn advance_phase(phase: f64, omega: f64, dt: f64) -> f64 {
    advance_phase_bounded(phase, omega, dt, 0.0, TAU)
}

/// Advance a cyclic phase with custom bounds
///
/// # Arguments
/// * `phase` - Current phase value
/// * `omega` - Rate of change per second
/// * `dt` - Timestep in seconds
/// * `min` - Minimum bound (wraps to this)
/// * `max` - Maximum bound (wraps at this)
#[inline]
pub fn advance_phase_bounded(phase: f64, omega: f64, dt: f64, min: f64, max: f64) -> f64 {
    let period = max - min;
    let new_phase = phase + omega * dt;
    min + (new_phase - min).rem_euclid(period)
}

/// Exponential moving average (smoothing)
///
/// # Arguments
/// * `prev` - Previous smoothed value
/// * `input` - New input value
/// * `tau` - Time constant in seconds
/// * `dt` - Timestep in seconds
///
/// # Example
/// ```
/// use continuum_runtime::operators::smooth;
///
/// let prev = 0.0;
/// let input = 100.0;
/// let tau = 1.0;
/// let dt = 0.1;
/// let smoothed = smooth(prev, input, tau, dt);
/// assert!(smoothed > 0.0 && smoothed < 100.0);
/// ```
#[inline]
pub fn smooth(prev: f64, input: f64, tau: f64, dt: f64) -> f64 {
    relax(prev, input, tau, dt)
}

/// Smooth with equivalent N-sample EMA behavior
///
/// Provides smoothing equivalent to an N-sample exponential moving average.
///
/// # Arguments
/// * `prev` - Previous smoothed value
/// * `input` - New input value
/// * `samples` - Equivalent number of samples
/// * `dt` - Timestep in seconds (used for consistency, behavior is sample-based)
#[inline]
pub fn smooth_samples(prev: f64, input: f64, samples: f64, _dt: f64) -> f64 {
    let alpha = 2.0 / (samples + 1.0);
    prev + alpha * (input - prev)
}

/// Second-order spring-damper system state
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DampedState {
    /// Position
    pub position: f64,
    /// Velocity
    pub velocity: f64,
}

/// Second-order spring-damper system (critically damped or underdamped)
///
/// Simulates a mass-spring-damper system relaxing toward a target.
///
/// # Arguments
/// * `state` - Current position and velocity
/// * `target` - Target position
/// * `omega` - Natural frequency (radians per second)
/// * `zeta` - Damping ratio (1.0 = critically damped, <1.0 = underdamped)
/// * `dt` - Timestep in seconds
///
/// # Returns
/// New state (position, velocity)
///
/// # Example
/// ```
/// use continuum_runtime::operators::{damp, DampedState};
///
/// let state = DampedState { position: 0.0, velocity: 0.0 };
/// let target = 1.0;
/// let omega = 2.0 * std::f64::consts::PI; // 1 Hz natural frequency
/// let zeta = 0.7; // Slightly underdamped
/// let dt = 0.01;
///
/// let new_state = damp(state, target, omega, zeta, dt);
/// assert!(new_state.position > 0.0);
/// ```
pub fn damp(state: DampedState, target: f64, omega: f64, zeta: f64, dt: f64) -> DampedState {
    // Use semi-implicit Euler for stability
    // x'' = -omega^2 * (x - target) - 2 * zeta * omega * v

    let displacement = state.position - target;
    let acceleration = -omega * omega * displacement - 2.0 * zeta * omega * state.velocity;

    // Semi-implicit: update velocity first, then position
    let new_velocity = state.velocity + acceleration * dt;
    let new_position = state.position + new_velocity * dt;

    DampedState {
        position: new_position,
        velocity: new_velocity,
    }
}

/// Spring-damper with stiffness and damping coefficients
///
/// # Arguments
/// * `state` - Current position and velocity
/// * `target` - Target position
/// * `stiffness` - Spring stiffness (N/m equivalent)
/// * `damping` - Damping coefficient (Ns/m equivalent)
/// * `mass` - Effective mass (default 1.0)
/// * `dt` - Timestep in seconds
#[inline]
pub fn damp_spring(
    state: DampedState,
    target: f64,
    stiffness: f64,
    damping: f64,
    mass: f64,
    dt: f64,
) -> DampedState {
    let omega = (stiffness / mass).sqrt();
    let zeta = damping / (2.0 * (stiffness * mass).sqrt());
    damp(state, target, omega, zeta, dt)
}

/// Vec3 variant of relaxation
///
/// Relaxes each component independently toward target.
#[inline]
pub fn relax_vec3(
    current: [f64; 3],
    target: [f64; 3],
    tau: f64,
    dt: f64,
) -> [f64; 3] {
    [
        relax(current[0], target[0], tau, dt),
        relax(current[1], target[1], tau, dt),
        relax(current[2], target[2], tau, dt),
    ]
}

/// Vec3 variant of decay
#[inline]
pub fn decay_vec3(value: [f64; 3], halflife: f64, dt: f64) -> [f64; 3] {
    let factor = 0.5_f64.powf(dt / halflife);
    [
        value[0] * factor,
        value[1] * factor,
        value[2] * factor,
    ]
}

/// Vec3 variant of integration
#[inline]
pub fn integrate_vec3(prev: [f64; 3], rate: [f64; 3], dt: f64) -> [f64; 3] {
    [
        prev[0] + rate[0] * dt,
        prev[1] + rate[1] * dt,
        prev[2] + rate[2] * dt,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_integrate() {
        let pos = 10.0;
        let vel = 5.0;
        let dt = 0.1;
        assert!((integrate(pos, vel, dt) - 10.5).abs() < 1e-10);
    }

    #[test]
    fn test_decay_halflife() {
        let value = 100.0;
        let halflife = 10.0;

        // After one halflife, should be 50%
        let result = decay(value, halflife, halflife);
        assert!((result - 50.0).abs() < 1e-10);

        // After two halflives, should be 25%
        let result = decay(value, halflife, 2.0 * halflife);
        assert!((result - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_decay_symmetry() {
        let value = 100.0;
        let halflife = 10.0;

        // Two steps of dt=5 should equal one step of dt=10
        let two_steps = decay(decay(value, halflife, 5.0), halflife, 5.0);
        let one_step = decay(value, halflife, 10.0);
        assert!((two_steps - one_step).abs() < 1e-10);
    }

    #[test]
    fn test_relax_tau() {
        let current = 0.0;
        let target = 100.0;
        let tau = 1.0;

        // After one time constant, should be ~63.2% of the way
        let result = relax(current, target, tau, tau);
        let expected = target * (1.0 - E.powf(-1.0));
        assert!((result - expected).abs() < 0.01);
    }

    #[test]
    fn test_relax_symmetry() {
        let current = 0.0;
        let target = 100.0;
        let tau = 1.0;

        // Two steps should equal one double step
        let two_steps = relax(relax(current, target, tau, 0.5), target, tau, 0.5);
        let one_step = relax(current, target, tau, 1.0);
        assert!((two_steps - one_step).abs() < 1e-10);
    }

    #[test]
    fn test_accumulate_clamping() {
        // Should clamp to max
        let result = accumulate(90.0, 50.0, 1.0, 0.0, 100.0);
        assert_eq!(result, 100.0);

        // Should clamp to min
        let result = accumulate(10.0, -50.0, 1.0, 0.0, 100.0);
        assert_eq!(result, 0.0);

        // Normal accumulation within bounds
        let result = accumulate(50.0, 10.0, 1.0, 0.0, 100.0);
        assert_eq!(result, 60.0);
    }

    #[test]
    fn test_advance_phase() {
        // Half rotation
        let phase = advance_phase(0.0, TAU, 0.5);
        assert!((phase - PI).abs() < 1e-10);

        // Full rotation wraps to 0
        let phase = advance_phase(0.0, TAU, 1.0);
        assert!(phase.abs() < 1e-10);

        // Beyond full rotation
        let phase = advance_phase(0.0, TAU, 1.25);
        assert!((phase - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_advance_phase_negative() {
        // Negative omega (retrograde)
        let phase = advance_phase(PI, -TAU, 0.25);
        assert!((phase - PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_damp_converges() {
        let mut state = DampedState {
            position: 0.0,
            velocity: 0.0,
        };
        let target = 1.0;
        let omega = TAU; // 1 Hz
        let zeta = 1.0; // Critically damped
        let dt = 0.01;

        // Run for many steps
        for _ in 0..1000 {
            state = damp(state, target, omega, zeta, dt);
        }

        // Should have converged close to target
        assert!((state.position - target).abs() < 0.01);
        assert!(state.velocity.abs() < 0.01);
    }

    #[test]
    fn test_damp_underdamped_oscillates() {
        let mut state = DampedState {
            position: 0.0,
            velocity: 0.0,
        };
        let target = 1.0;
        let omega = TAU; // 1 Hz
        let zeta = 0.1; // Underdamped
        let dt = 0.01;

        let mut max_position: f64 = 0.0;

        // Run for half a period
        for _ in 0..50 {
            state = damp(state, target, omega, zeta, dt);
            max_position = max_position.max(state.position);
        }

        // Underdamped should overshoot
        assert!(max_position > target);
    }

    #[test]
    fn test_smooth_equals_relax() {
        let prev = 10.0;
        let input = 100.0;
        let tau = 2.0;
        let dt = 0.1;

        let smoothed = smooth(prev, input, tau, dt);
        let relaxed = relax(prev, input, tau, dt);

        assert!((smoothed - relaxed).abs() < 1e-10);
    }

    #[test]
    fn test_vec3_relax() {
        let current = [0.0, 0.0, 0.0];
        let target = [100.0, 200.0, 300.0];
        let tau = 1.0;
        let dt = 1.0;

        let result = relax_vec3(current, target, tau, dt);

        // Each component should relax independently
        let expected_factor = 1.0 - E.powf(-1.0);
        assert!((result[0] - 100.0 * expected_factor).abs() < 0.01);
        assert!((result[1] - 200.0 * expected_factor).abs() < 0.01);
        assert!((result[2] - 300.0 * expected_factor).abs() < 0.01);
    }

    #[test]
    fn test_rk4_more_accurate() {
        // For a simple system, RK4 should be more accurate than Euler
        // dy/dt = y (exponential growth)
        let y0 = 1.0;
        let dt = 0.1;
        let rate_fn = |y: f64| y;

        let euler = integrate_with_method(y0, rate_fn, dt, IntegrationMethod::Euler);
        let rk4 = integrate_with_method(y0, rate_fn, dt, IntegrationMethod::Rk4);
        let exact = E.powf(dt);

        // RK4 should be closer to exact
        assert!((rk4 - exact).abs() < (euler - exact).abs());
    }
}
