# Signals

This document defines **Signals** in Continuum.

Signals are the **authoritative state** of the simulation.
All causality flows through signals.

---

## 1. What a Signal Is

A **Signal** represents a single, authoritative quantity in the simulation.

A signal:
- has a stable identity
- has a type
- has exactly one resolved value per tick
- is resolved deterministically

Signals define **what is true** in the simulation at a given tick.

---

## 2. Signals and Causality

Signals are the foundation of causality.

- If something influences the simulation, it must be a signal
- If something is not a signal, it must not affect execution
- All causal dependencies are expressed through signal reads

There is no hidden state.

---

## 3. Signal Resolution

Signals are resolved during the **Resolve phase**.

Resolution:
- consumes inputs accumulated during Collect
- may read resolved signals from the previous tick
- produces exactly one resolved value

Resolution must be:
- deterministic
- order-independent
- free of side effects

---

## 4. Signal Inputs

Signals do not mutate directly.

Instead, they receive **inputs** during Collect and Fracture.

Inputs:
- are accumulated
- must be commutative unless explicitly constrained
- are cleared after resolution

This allows:
- parallel accumulation
- deterministic resolution
- explicit causality

---

## 5. Signal Reads

Signal reads are:
- inferred, not declared
- read-only
- resolved-value only

A signal may read:
- its own previous value
- other signals’ resolved values

A signal must never read:
- fields
- observer state
- unresolved inputs

---

## 6. Assertions and Validation

Signals may declare **assertions**.

Assertions:
- validate resolved values
- do not modify values
- produce faults on failure

Assertions exist to detect:
- runaway values
- impossible physics
- numerical instability

Silent correction is forbidden by default.

---

## 7. Signal Initialization

Signals that use `prev` (read their previous value) require initialization.

### Explicit Initialization

Use the `:initial()` attribute to declare initial values:

```cdsl
signal atmosphere.co2_ppmv {
    : Scalar<ppmv, 100..10000>
    : stratum(atmosphere)
    : initial(280.0)  # Earth preindustrial level
    
    resolve {
        # 'prev' starts at 280.0 on first tick
        dt.relax(prev, equilibrium, tau)
    }
}
```

### Initialization Rules

- **Explicit is required:** Signals using `prev` must have `:initial(value)`
- **Literals only:** Value must be a numeric literal, not an expression
- **Fail-hard:** Missing initialization causes runtime panic before first tick
- **Processed at compile time:** Initial values are part of the resolved IR

### Literal Resolve Initialization

Signals with literal resolve blocks are auto-initialized:

```cdsl
signal demo.constant {
    resolve { 42.0 }  # Auto-initialized to 42.0
}
```

### Why Explicit Initialization

1. **No magic:** Initial state is visible in CDSL
2. **One truth:** Declared once, used everywhere
3. **Fail-hard:** Missing values detected at startup
4. **Documentation:** Initial conditions are self-documenting

---

## 8. Signals and Time

Signals are resolved once per tick per stratum.

- A signal does not advance time
- A signal does not own `dt`
- Time enters signal resolution explicitly via context

Signals must be robust to changes in `dt`.

---

## 9. Signals and Determinism

Signals are fully deterministic.

Given:
- the same World
- the same Scenario
- the same seed
- the same tick

a signal must resolve to the same value.

Any deviation is a bug.

---

## 9. What Signals Are Not

Signals are **not**:
- mutable variables
- ECS components
- observer data
- cached fields
- scripting globals

They are **authoritative causal state**.

---

## Summary

- Signals define authoritative truth
- All causality flows through signals
- Signals resolve deterministically
- Inputs are accumulated, not mutated
- Assertions detect errors, not hide them

If it affects causality and isn’t a signal,
the model is wrong.
