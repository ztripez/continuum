# Execution Phases

This document defines the **execution phases** that occur during
causal simulation ticks.

Phases define **what kind of work is allowed when** and enforce
the boundary between causality and observation.

---

## Phase Model

Each simulation tick executes a fixed sequence of phases.

Phases are:
- ordered
- deterministic
- barriered

No phase may overlap another.
No phase may be skipped.

---

## Phase Order

The canonical phase order is:

1. Configure
2. Collect
3. Resolve
4. Fracture
5. Measure

This order is invariant.

---

## 1. Configure

**Purpose:** finalize per-tick execution context and initialize stateful signals.

Configure is primarily **engine-internal**. Most simulation logic should use other phases.

### What Configure Does

Configure is responsible for:
- creating the tick context (`TickContext` with tick number, sim_time, dt, era)
- executing signal `:initial(...)` blocks for stateful signals (signals using `prev`)
- committing member signal initial values (copy current → previous buffers)
- validating that all signals in the Resolve DAG are initialized

### What Configure Does NOT Do

Configure does **not**:
- advance time (sim_time reflects the *start* of the tick)
- resolve signals from inputs (that's Resolve phase)
- emit fields or observer data (that's Measure phase)
- modify config/const values (frozen at scenario application)

### DSL Usage

**There is currently no user-facing DSL syntax for Configure blocks.**

Signal initialization uses the `:initial(...)` directive, which is executed during Configure:

```cdsl
signal atmosphere.temperature {
    : Scalar<K>
    : initial(288.0)  # Executed in Configure phase, tick 0
    
    resolve {
        prev + heating_rate * dt  # prev starts at 288.0
    }
}
```

### When Would You Use Configure?

In most cases, **you don't**. Signal initialization (`:initial(...)`) is the main user-facing feature.

Future DSL extensions might allow user-defined Configure operators for:
- per-tick context setup that must happen before input collection
- derived execution parameters that change per-era
- dynamic stratum gating logic

But currently, Configure is engine internals only.

### Implementation Notes

Config and const values are frozen execution parameters loaded during Scenario
Application (see `execution/lifecycle.md` § 4 and `dsl/language.md` § 5). The
Configure phase does not modify these values.

Configure prepares the tick; it does not execute simulation logic.

---

## 2. Collect

**Purpose:** gather inputs to signals.

Collect is responsible for:
- collecting signal inputs
- applying impulses
- accumulating commutative contributions

During Collect:
- signals are not resolved
- only inputs are accumulated

Collect defines *what will influence resolution*, not results themselves.

---

## 3. Resolve

**Purpose:** compute authoritative state.

Resolve is responsible for:
- resolving signals from their inputs
- producing the authoritative state for the tick
- validating invariants via assertions

Resolve:
- reads resolved signals from the previous tick
- writes resolved signals for the current tick
- must be deterministic and order-independent

Resolve is the core of causality.

---

## 4. Fracture

**Purpose:** detect tension and accumulate inputs for the next tick.

Fracture is responsible for:
- detecting instability, conflict, or threshold conditions
- accumulating signal inputs for the **next tick's Collect phase**

Fracture:
- reads resolved signals from the **current tick**
- writes inputs for the **next tick**
- may influence future resolution (one tick ahead)
- must not read fields or observers
- must not access current tick's inputs

**Key insight:** Fractures bridge ticks. They detect emergent conditions in resolved state and schedule inputs for the next execution cycle.

Fracture exists to model **emergent change**, not scripted events.

---

## 5. Measure

**Purpose:** produce observations.

Measure is responsible for:
- emitting fields
- producing lens artifacts
- triggering observer-only logic

Measure:
- may read resolved signals
- may write fields
- must not write signals

Measure is strictly non-causal.

---

## Phase Boundaries and Safety

Phase boundaries are strict.

- Data written in a phase is not visible to earlier phases
- Fields are visible only after Measure
- Observers may attach only after Measure

Violating a phase boundary is a system error.

---

## Parallelism Within Phases

Within a phase:
- work may be executed in parallel
- ordering is derived from the execution graph
- parallelism must not change outcomes

Between phases:
- a full barrier is enforced

Parallelism is an optimization, never a semantic feature.

---

## What Phases Are Not

Phases are **not**:
- optional
- domain-specific
- a scripting convenience
- observer-controlled

They are the **structural backbone** of execution.

---

## Practical Example: Phase Usage

Here's how a signal flows through phases during a tick:

```cdsl
signal atmosphere.temperature {
    : Scalar<K>
    : initial(288.0)         # Configure phase (tick 0 only)
    
    collect {
        inputs                # Collect phase: accumulate all inputs
    }
    
    resolve {
        # Resolve phase: compute authoritative state
        prev + heating_rate * dt + inputs
    }
    
    assert {
        # Resolve phase: validate invariants
        self > 0 <K>
        self < 1000 <K>
    }
}

# Fracture phase: detect emergent conditions
fracture atmosphere.runaway_greenhouse {
    trigger { atmosphere.temperature > 373 <K> }
    
    effect {
        # Schedule input for NEXT tick's Collect phase
        emit atmosphere.co2_mass: 1e12 <kg>
    }
}

# Measure phase: emit observations
field atmosphere.temperature_celsius {
    : Grid2D<celsius>
    
    measure {
        (atmosphere.temperature - 273.15) <celsius>
    }
}
```

**Execution order for one tick:**

1. **Configure**: Initialize temperature to 288K (tick 0 only), set up tick context
2. **Collect**: Accumulate inputs to temperature signal
3. **Resolve**: Compute new temperature value, validate assertions
4. **Fracture**: Check if temperature > 373K, schedule CO₂ emission for next tick
5. **Measure**: Emit temperature field in celsius for observation

---

## Which Phase Should I Use?

**Quick reference for placing DSL logic:**

| What you're doing | Phase to use |
|-------------------|--------------|
| Initialize stateful signal | `:initial(...)` directive (Configure) |
| Accumulate inputs to a signal | `collect { }` block (Collect) |
| Apply an impulse | Impulse handler (Collect) |
| Resolve authoritative state | `resolve { }` block (Resolve) |
| Validate invariants | `assert { }` block (Resolve) |
| Detect emergent conditions | `fracture { }` block (Fracture) |
| Emit observable data | `field { }` or `measure { }` block (Measure) |
| Log/analyze simulation state | `chronicle { }` block (Measure) |

**Rule of thumb:**
- If it affects causality → Resolve (or Collect for inputs)
- If it's a response to resolved state → Fracture
- If it's observation only → Measure
- If it's initialization → `:initial(...)`

---

## Summary

- Phases define *what happens during a tick*
- Order is fixed and invariant
- Configure is engine-internal (initialization only)
- Resolve defines causality
- Measure defines observation
- Boundaries are strictly enforced

If logic appears in the wrong phase, the model is wrong.
