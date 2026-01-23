# Fractures

This document defines **Fractures** in Continuum.

Fractures are the mechanism by which **emergent tension** is detected and acted upon.

---

## 1. What a Fracture Is

A **Fracture** represents a detected condition of instability, conflict, or transition
within the simulation.

Fractures:
- are not scripted events
- are not authored outcomes
- arise from resolved state

They model **emergence**, not control flow.

---

## 2. Fractures and the Fracture Phase

Fractures execute during the **Fracture phase**.

During this phase:
- resolved signals are inspected
- tension conditions are evaluated
- signal inputs are accumulated for the next tick's Collect phase

Fractures do not:
- read fields
- mutate signals directly
- observe future state
- access the current tick's inputs (only write to next tick's inputs)

---

## 3. Fractures vs Events

Fractures are not events.

| Aspect | Fracture | Event |
|------|----------|-------|
| Trigger | Emergent condition | Authored trigger |
| Purpose | Model instability | Drive narrative |
| Writes | Signal inputs | Arbitrary effects |
| Causality | Continuous | Discrete |

If something is authored “to happen”, it is not a fracture.

---

## 4. Fracture Declaration

Fractures are declared in the DSL.

A fracture defines:
- the condition to detect (signal-only, in `when` block)
- the response (signal input accumulation, in `collect` block)
- scope and phase constraints (via attributes)

Fractures must be:
- deterministic
- side-effect free except for declared outputs
- idempotent per tick

---

## 5. Fractures and Eras

Fractures may:
- influence era transitions indirectly (via signals)
- be active or inactive depending on era

Fractures must not:
- directly switch eras
- mutate execution policy

Execution policy responds to signals, not fractures.

---

## 6. Fractures and Determinism

Fracture behavior must be deterministic.

Given the same resolved signals at a tick:
- the same fractures must trigger
- the same outputs must be produced

If fracture outcomes differ, determinism is broken.

---

## 7. Fracture Restrictions

Fractures have specific restrictions to maintain causality and determinism:

### 7.1 No dt.raw in Collect Blocks

Fractures **cannot use `dt.raw`** in their `collect` blocks.

**Why?** Fractures detect emergent conditions and accumulate signal inputs for the next tick. The magnitude of these inputs should be condition-dependent, not time-step dependent. Using `dt.raw` would make fracture responses vary with simulation timestep, violating the principle that fractures respond to **state**, not **time**.

**Wrong:**
```cdsl
fracture thermal.coupling {
    when { signal.temp > 1000 <K> }
    collect {
        let energy = signal.power * dt.raw in  # ❌ NOT ALLOWED
        signal.heat <- energy
    }
}
```

**Right approach #1 - Let signal handle dt:**
```cdsl
fracture thermal.coupling {
    when { signal.temp > 1000 <K> }
    collect {
        signal.heat_power <- signal.power  # Accumulate power, signal integrates
    }
}

signal heat {
    resolve {
        dt.integrate(prev, inputs)  # Signal handles dt.raw
    }
}
```

**Right approach #2 - Use regular cross-domain coupling:**

If this isn't an emergent tension condition but regular coupling, don't use a fracture at all. Use a signal that reads from another signal, or reorganize the signal dependencies.

### 7.2 Rates vs Instantaneous Deltas

Fractures can emit two types of contributions:

1. **Rates** — continuous processes (e.g., volcanic outgassing in kg/s)
2. **Instantaneous deltas** — discrete events (e.g., meteor impact adds X mass)

**For continuous processes, accumulate rates:**
```cdsl
fracture atmosphere.volcanic_co2 {
    collect {
        # Accumulate ppmv/s rate - signal integrates
        let emission_rate_ppmv_s = activity * base_rate * 1e-15 in
        signal.atmosphere.co2_ppmv <- emission_rate_ppmv_s
    }
}

signal atmosphere.co2_ppmv {
    resolve {
        # inputs contains rates - integrate over dt
        dt.integrate(prev, inputs)
    }
}
```

**For discrete events, accumulate the actual delta:**
```cdsl
fracture impact.mass_delivery {
    when { signal.impact.detected }
    collect {
        # Accumulate actual mass delivered (not a rate)
        signal.atmosphere.dust_mass <- signal.impact.ejecta_kg
    }
}

signal atmosphere.dust_mass {
    resolve {
        # inputs contains actual deltas - just add
        prev + inputs
    }
}
```

**The key distinction:**
- If the process is **continuous** and the emission scales with time → accumulate a **rate**, signal uses `dt.integrate()`
- If the process is **discrete** and happens instantaneously → accumulate the **delta**, signal uses `prev + inputs`

Never use `dt.s()` or `dt.raw` in fracture collect blocks. The signal's resolve block is responsible for dt-correct integration.

### 7.3 Dangerous Functions

Fractures that use dangerous functions (like `maths.clamp`) must declare them explicitly using `: uses()` declarations.

```cdsl
fracture thermal.safety_limit {
    : strata(thermal)
    : uses(maths.clamping)  // Required for maths.clamp
    
    when { signal.temp > 1000 <K> }
    collect {
        // Clamp is legitimate here - physical safety limit
        signal.temp_adjustment <- maths.clamp(signal.delta, -10 <K>, 10 <K>)
    }
}
```

The `: uses()` declaration system is now fully implemented for all executable primitives: signals, members, fractures, operators, and impulses.

---

## 8. What Fractures Are Not

Fractures are **not**:
- conditional logic
- branching control flow
- scripted transitions
- observer hooks

They are **emergent causal responses**.

---

## Summary

- Fractures detect emergent tension
- They run in the Fracture phase (after Resolve, before Measure)
- They accumulate inputs for the next tick's Collect phase
- They use `when { }` blocks for conditions and `collect { }` blocks for inputs
- They are not authored outcomes

If fractures feel like scripting,
the abstraction has been violated.
