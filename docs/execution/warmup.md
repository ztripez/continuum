# Warmup Phase

This document defines the **warmup phase** in Continuum execution.

Warmup is a bounded, deterministic, pre-causal phase that prepares
simulation state before time begins to advance.

---

## 1. Purpose

Warmup exists to prepare values that must be **stable before causal history begins**.

Common use cases:

- **Fixed-point convergence** — signals that must reach equilibrium
- **Derived table construction** — lookup tables computed from initial conditions
- **Reconstruction cache priming** — observer infrastructure preparation
- **Invariant validation** — pre-execution consistency checks

Warmup allows the simulation to "settle" before tick 0.

---

## 2. Warmup Is Not Time

During warmup:

- no ticks advance
- `dt` is not defined
- no causal history is produced
- no fields are emitted
- no observers are invoked
- no chronicles execute

Warmup is **outside time**.

---

## 3. Warmup Phases

Warmup executes a subset of phases:

| Phase | During Warmup |
|-------|---------------|
| Configure | Yes |
| Collect | No |
| Resolve | Yes (warmup-tagged only) |
| Fracture | No |
| Measure | No |

Only signals and operators explicitly tagged for warmup execution participate.

---

## 4. DSL Syntax

### Warmup Blocks in Signals

Signals that require warmup declare a `warmup` block:

```
signal terra.thermal.equilibrium {
  : Scalar<K>
  : strata(terra.thermal)

  warmup {
    : iterations(100)
    : convergence(1e-6)

    iterate {
      let flux_in = physics.radiogenic_heat(config.terra.core.heat_budget)
      let flux_out = physics.surface_radiation(prev)
      prev + (flux_in - flux_out) * 0.1
    }
  }

  resolve {
    // Normal tick resolution
    prev + collected * dt
  }
}
```

### Warmup Block Attributes

| Attribute | Description |
|-----------|-------------|
| `: iterations(N)` | Maximum warmup iterations (required) |
| `: convergence(epsilon)` | Convergence threshold (optional) |

### Convergence

If `convergence` is specified:
- Warmup terminates early when `|current - previous| < epsilon`
- All warmup signals must converge for early termination
- If any signal fails to converge within iterations, execution fails

If `convergence` is omitted:
- Warmup runs exactly `iterations` times

---

## 5. Warmup Operators

Operators may also participate in warmup:

```
operator terra.thermal.warmup_budget {
  : strata(terra.thermal)
  : phase(warmup)

  warmup {
    let base_heat = physics.compute_initial_heat(config.terra.core.mass)
    signal.terra.thermal.budget <- base_heat
  }
}
```

Warmup operators:
- Execute once per warmup iteration
- May write to signal input accumulators
- Must be deterministic

---

## 6. Warmup Dependencies

Warmup signals and operators form their own DAG:

- Dependencies are inferred from reads
- Topological ordering is computed
- Parallel execution is allowed within levels

Warmup DAG is separate from tick execution DAG.

---

## 7. Warmup Execution Order

1. Apply scenario initial conditions
2. Execute Configure phase
3. Execute warmup DAG:
   - For each iteration until max or convergence:
     - Clear warmup input accumulators
     - Execute warmup operators (topological order)
     - Resolve warmup signals (topological order)
     - Check convergence criteria
4. If convergence fails: emit fault, halt
5. Copy warmup results to tick 0 initial state
6. Begin causal execution

---

## 8. Warmup and Strata

Warmup respects stratum membership but ignores stride:

- Signals execute based on their stratum's warmup participation
- Era gating does not apply during warmup
- All warmup-tagged items in active strata execute

Warmup is stratum-aware but not stratum-gated.

---

## 9. Determinism

Warmup must be fully deterministic:

- Same inputs produce same outputs
- Iteration order is fixed
- Convergence checks are reproducible
- No dependency on wall time or external state

Warmup is part of replay identity.

---

## 10. Faults

Warmup may produce faults:

| Fault | Cause |
|-------|-------|
| `WarmupDivergence` | Signal failed to converge within iterations |
| `WarmupNaN` | Warmup produced NaN value |
| `WarmupInfinite` | Warmup produced infinite value |
| `WarmupAssertion` | Warmup assertion failed |

Warmup faults are fatal by default.

---

## 11. What Warmup Must Not Do

Warmup must not:

- Access `dt` (time has not started)
- Emit fields (no observers during warmup)
- Trigger fractures (no causality during warmup)
- Produce causal history (warmup is pre-causal)
- Depend on tick number (always 0)

If warmup requires time, the model is wrong.

---

## 12. Complete Example

```
const {
  physics.stefan_boltzmann: 5.67e-8 <W/m^2/K^4>
}

config {
  terra.thermal.initial_mass: 6e24 <kg>
  terra.thermal.convergence_rate: 0.05
}

signal terra.thermal.equilibrium_temp {
  : Scalar<K, 200..6000>
  : strata(terra.thermal)

  warmup {
    : iterations(500)
    : convergence(0.01 <K>)

    iterate {
      let flux_in = physics.internal_heat(config.terra.thermal.initial_mass)
      let flux_out = const.physics.stefan_boltzmann * (prev ^ 4)
      let delta = (flux_in - flux_out) * config.terra.thermal.convergence_rate
      prev + delta
    }
  }

  resolve {
    let heating = collected
    let radiation = const.physics.stefan_boltzmann * (prev ^ 4)
    prev + (heating - radiation) * dt
  }
}

signal terra.thermal.crust_temp {
  : Scalar<K, 200..2000>
  : strata(terra.thermal)

  warmup {
    : iterations(500)
    : convergence(0.01 <K>)

    iterate {
      // Depends on equilibrium_temp during warmup
      signal.terra.thermal.equilibrium_temp * 0.1
    }
  }

  resolve {
    signal.terra.thermal.equilibrium_temp * 0.1 + collected * dt
  }
}
```

---

## Summary

| Aspect | Warmup Behavior |
|--------|-----------------|
| Time | Not advancing |
| Ticks | None |
| dt | Undefined |
| Fields | Not emitted |
| Observers | Not invoked |
| Fractures | Not evaluated |
| DAG | Separate warmup DAG |
| Determinism | Required |
| Convergence | Optional, bounded |

Warmup is the bridge between scenario instantiation and causal history.

---

## See Also

- @execution/lifecycle.md — Execution stages overview
- @scenario.md — Initial conditions and parameters
- @dsl/syntax.md — DSL syntax reference
