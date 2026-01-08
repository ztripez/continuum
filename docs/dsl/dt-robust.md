# Continuum DSL — dt-Robust Operators

This document defines **dt-robust operators** and the rules for timestep access in the DSL.

---

## 1. The Problem with Raw dt

Naive timestep usage creates fragile simulations:

```
// FRAGILE — behavior changes with dt
resolve {
  prev + velocity * dt  // Euler integration, unstable at large dt
}
```

Problems:
- Different eras use different `dt` values
- Numerical stability depends on step size
- Results are not invariant under temporal resolution changes

If changing `dt` changes qualitative behavior, the model is wrong.

---

## 2. The Solution: dt-Robust Operators

Instead of raw `dt`, authors use **semantic operators** that express intent:

| Intent | Raw (fragile) | dt-Robust |
|--------|---------------|-----------|
| Accumulate rate | `prev + rate * dt` | `integrate(prev, rate)` |
| Decay toward zero | `prev * (1 - k * dt)` | `decay(prev, halflife)` |
| Relax toward target | `prev + (target - prev) * k * dt` | `relax(prev, target, tau)` |
| Bounded accumulation | `clamp(prev + delta * dt, 0, max)` | `accumulate(prev, delta, 0..max)` |
| Phase advancement | `wrap(prev + omega * dt, 0, TAU)` | `advance_phase(prev, omega)` |

The engine implements these with proper numerical methods.

---

## 3. Raw dt Access: Explicit Opt-In

To access raw `dt`, a signal must declare `: dt_raw`:

```
signal.terra.rotation.phase {
  : Scalar<rad, 0..TAU>
  : strata(terra.rotation)
  : dt_raw  // explicit opt-in

  resolve {
    wrap(prev + signal.terra.rotation.omega * dt, 0, TAU)
  }
}
```

Without `: dt_raw`, referencing `dt` is a **compile error**.

### When to Use dt_raw

Legitimate uses:
- Energy = Power × dt (physics definition)
- Custom numerical methods with known stability
- Phase/angle advancement
- Discrete event timing

If uncertain, prefer dt-robust operators.

---

## 4. dt-Robust Operator Reference

### Integration

```
integrate(prev, rate)
integrate(prev, rate, method: euler)
integrate(prev, rate, method: rk4)
integrate(prev, rate, method: verlet)
```

Accumulates a rate over time. Default method chosen for stability.

### Decay

```
decay(value, halflife)        // value * 0.5^(dt/halflife)
decay(value, rate: k)         // value * e^(-k*dt)
decay(value, tau)             // value * e^(-dt/tau)
```

Exponential decay toward zero. Exact solution, stable at any dt.

### Relaxation

```
relax(current, target, tau)
relax(current, target, tau, method: exp)
relax(current, target, halflife)
```

Exponential relaxation toward a target value:
- `tau` — time constant (63% of the way after tau)
- `halflife` — time to reach halfway

Exact exponential solution, stable at any dt.

### Bounded Accumulation

```
accumulate(prev, delta, bounds)
accumulate(prev, delta, 0..max)
accumulate(prev, delta, min..max)
```

Integrates with clamping. Prevents overflow/underflow.

### Phase Advancement

```
advance_phase(phase, omega)
advance_phase(phase, omega, period)
advance_phase(phase, omega, 0..TAU)
```

Advances a cyclic quantity, wrapping at bounds. Default period is TAU (2π).

### Smoothing

```
smooth(prev, input, tau)
smooth(prev, input, samples: N)
```

Exponential moving average. `samples: N` gives equivalent N-sample EMA behavior.

### Damping (Spring-Damper)

```
damp(position, velocity, target, stiffness, damping)
damp(state, target, omega, zeta)  // canonical form
```

Second-order spring-damper system. Stable even at large dt.

---

## 5. Operator Behavior Guarantees

All dt-robust operators:

1. **Deterministic** — same inputs → same outputs
2. **Stable** — bounded output for bounded input at any reasonable dt
3. **Convergent** — approach correct continuous solution as dt → 0
4. **Symmetric** — dt=0.1 twice equals dt=0.2 once (within numerical precision)

Operators do NOT:
- Hide runaway behavior (assertions still apply)
- Implicitly clamp values
- Change qualitative dynamics

---

## 6. Assertions Still Apply

Using dt-robust operators does not bypass bounds checking:

```
signal.terra.temperature {
  : Scalar<K, 0..1e6>  // bounds enforced
  : strata(terra.thermal)

  resolve {
    // relax() is stable, but if target is extreme, assertion fires
    relax(prev, signal.terra.heat_input * 1e10, 1000 <s>)
  }
}
```

If a value diverges despite using dt-robust operators, the simulation fails loudly.

---

## 7. Custom Numerical Methods

For advanced use, raw dt with explicit method declaration:

```
signal.terra.orbit.position {
  : Vec3<m>
  : strata(terra.orbital)
  : dt_raw
  : integrator(symplectic_euler)  // declare method for analysis

  resolve {
    // Symplectic Euler for energy conservation
    let new_velocity = prev.velocity + signal.terra.gravity * dt
    let new_position = prev.position + new_velocity * dt
    { position: new_position, velocity: new_velocity }
  }
}
```

The `: integrator(...)` annotation documents intent and enables tooling to verify stability.

---

## 8. Examples

### Temperature Relaxation

```
signal.terra.surface.temperature {
  : Scalar<K, 50..500>
  : strata(terra.atmosphere)

  resolve {
    let equilibrium = fn.radiative_equilibrium(
      signal.stellar.flux,
      signal.terra.albedo
    )
    relax(prev, equilibrium, config.terra.thermal_tau)
  }
}
```

### Radioactive Decay

```
signal.terra.core.radiogenic_heat {
  : Scalar<W, 0..1e14>
  : strata(terra.thermal)

  resolve {
    decay(prev, config.terra.radiogenic_halflife)
  }
}
```

### Orbital Phase (Raw dt)

```
signal.terra.orbit.true_anomaly {
  : Scalar<rad, 0..TAU>
  : strata(terra.orbital)
  : dt_raw

  resolve {
    let n = fn.mean_motion(signal.stellar.mass, signal.terra.orbit.semi_major)
    advance_phase(prev, n)
  }
}
```

### Damped Oscillator

```
signal.terra.chandler_wobble {
  : Vec2<rad>
  : strata(terra.rotation)

  resolve {
    damp(prev, Vec2(0, 0), omega: config.chandler_frequency, zeta: config.chandler_damping)
  }
}
```

---

## Summary

- Raw `dt` requires explicit `: dt_raw` declaration
- Prefer dt-robust operators for stability
- Operators express intent, not mechanism
- Assertions still validate bounds
- Use raw dt only when physics demands it

If a simulation behaves differently at different dt values,
the model is wrong unless dt_raw is explicitly used and justified.
