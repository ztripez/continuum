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

**Purpose:** finalize per-tick execution context.

Configure is responsible for:
- aligning clocks
- applying era-specific execution policy
- activating or gating strata
- preparing per-tick state

Configure does **not**:
- advance time
- mutate signals
- emit fields
- modify config/const values (already frozen at world initialization)

### Config and Const Values

Config and const values are **frozen execution parameters** that are loaded once
during world initialization (lifecycle stage 4: Scenario Application) and remain
immutable throughout all execution phases.

- `const.*` — Global simulation constants loaded from `const {}` blocks
  - Immutable, not scenario-overridable
  - Example: `const { physics.boltzmann: 1.380649e-23 }`

- `config.*` — Configuration parameters loaded from `config {}` blocks
  - World provides defaults via `default` field
  - Scenarios may override config values (not const values)
  - Example: `config { thermal.decay_halflife: 1.42e17 }`

Both config and const values:
- Are available to all phases via `load_config()` and `load_const()`
- Behave like frozen parameters (no prev/current distinction like signals)
- Are NOT recomputed per-tick (frozen at initialization)
- Must be compile-time literals (Scalar or Vec3 with literal components)

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

## Summary

- Phases define *what happens during a tick*
- Order is fixed and invariant
- Resolve defines causality
- Measure defines observation
- Boundaries are strictly enforced

If logic appears in the wrong phase, the model is wrong.
