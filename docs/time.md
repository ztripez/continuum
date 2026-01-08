# Time

This document defines how **time** is modeled in Continuum.

Time is a **first-class part of causality**.
It is not a rendering concern, a UI concern, or a convenience abstraction.

Execution policy over time (eras, strata gating, cadence) is defined elsewhere.

---

## 1. Time as a Model Dimension

Continuum simulations are **time-driven**.

Time advancement is:
- explicit
- deterministic
- part of the causal execution model

There is no implicit “frame time”, “update loop”, or coupling to wall-clock time.

All simulation progress occurs through **ticks** with a defined **dt**.

---

## 2. Ticks and dt

A **tick** is the atomic unit of simulation advancement.

Each tick advances simulated time by a duration `dt` expressed in seconds.

- `dt` is always explicit
- `dt` is selected by the active era
- `dt` must never be inferred from real time or execution speed

Changing `dt` changes **temporal resolution**, not **meaning**.

If changing `dt` alters qualitative behavior,
the model is incorrect.

---

## 3. Time and Multi-Rate Execution

Not all simulation logic runs every tick.

Continuum supports **multi-rate execution** through **strata**.

From the perspective of time:
- ticks advance global simulated time
- strata determine *whether* logic executes on a given tick
- skipped execution does not advance additional time

Time advances uniformly.
Execution frequency varies.

(See `strata.md` for full semantics.)

---

## 4. Time and Eras

Time itself is continuous across the simulation.

**Eras do not create separate timelines.**

From the perspective of time:
- exactly one era is active at any tick
- the active era selects the current `dt`
- era changes do not reset or fork time

Eras define **execution policy**, not time itself.

(See `eras.md` for full semantics.)

---

## 5. Derived Time Units

Human-readable time units (years, kyr, Myr, etc.) are **derived**, not fundamental.

- A base unit is derived before execution begins
- Unit labels and thresholds are fixed for the duration of execution
- Units are used for:
  - configuration
  - interpretation
  - DSL literals and formatting

Derived units must never influence causality.

They exist to make time **interpretable**, not authoritative.

---

## 6. Time and Determinism

Time behavior must be fully replayable.

A replay requires:
- world specification
- scenario specification
- initial seed
- era sequence
- `dt` sequence

Wall-clock time, frame rate, and execution speed must not affect results.

---

## 7. What Time Is Not

Time is **not**:
- a rendering frame
- a real-time clock
- an observer concern
- a scheduling heuristic
- an execution policy

Time is part of the **causal model**.

---

## Summary

- Time advancement is explicit and deterministic
- Ticks advance simulated time
- `dt` controls resolution, not meaning
- Strata and eras affect execution, not time itself
- Derived units are interpretive only
- Observation must never affect time progression

If time behavior feels implicit, convenient, or adjustable during execution,
the model is wrong.
