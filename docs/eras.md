# Eras

This document defines **Eras** in Continuum.

Eras are the mechanism by which **execution policy changes over simulated time**.
They control *how* the simulation runs, not *what* the simulation is.

---

## 1. What an Era Is

An **Era** defines a named execution regime.

An era specifies:
- the base timestep (`dt`)
- which strata are active or gated
- cadence modifiers for active strata
- the conditions under which the era transitions

An era does **not** define simulation logic.
It defines **how frequently logic executes**.

---

## 2. Why Eras Exist

Many simulations require different execution regimes over their lifetime.

Common reasons include:
- coarse timesteps during early evolution
- finer timesteps as systems become sensitive
- delayed activation of subsystems
- selective suspension of expensive processes

Eras allow the simulation to **change resolution and scope**
without changing its causal structure.

---

## 3. Eras and Time

Eras operate on top of the global simulation timeline.

- Time advances via ticks and `dt`
- An era defines the `dt` used while it is active
- Era changes do not reset or fork time

Eras change **temporal resolution**, not meaning.

If changing eras alters qualitative behavior,
the model is incorrect.

---

## 4. Era Membership and Activation

At any tick, exactly one era is active.

- The active era determines execution policy
- All other eras are inactive
- Era identity is explicit and stable

There is no implicit blending or interpolation between eras.

---

## 5. Strata Control Within Eras

Eras control strata execution.

Within an era:
- strata may be active
- strata may be gated (paused)
- strata may execute at reduced cadence

Strata identity does not change across eras.
Only their execution eligibility does.

---

## 6. Era Transitions

Era transitions are **explicit and signal-driven**.

A transition specifies:
- a source era
- a target era
- a condition expressed over resolved signals

Transitions:
- are evaluated at tick boundaries
- must be deterministic
- must not depend on fields or observers

An era transition changes **execution policy**, not simulation state.

---

## 7. Initial and Terminal Eras

A World defines:
- exactly one initial era
- zero or more terminal eras

A terminal era:
- has no outgoing transitions
- may run indefinitely
- may represent steady-state or real-time execution

Reaching a terminal era is not a failure.

---

## 8. Eras and Determinism

Era behavior must be deterministic.

Given:
- the same World
- the same Scenario
- the same seed
- the same causal history

era transitions must occur at the same ticks
and select the same target eras.

Non-deterministic era switching is forbidden.

---

## 9. Eras and Scenarios

Scenarios must not alter era structure.

A Scenario may:
- select the initial era (if allowed by the World)
- configure parameters used by era transition conditions

A Scenario must not:
- add eras
- remove eras
- alter transition logic

Era structure belongs to the World.

---

## 10. What Eras Are Not

Eras are **not**:
- phases
- domains
- timelines
- world variants
- observer constructs

They are **execution policy regimes**.

---

## Summary

- Eras define execution regimes
- They control `dt`, strata gating, and cadence
- Eras do not define logic or state
- Transitions are signal-driven and deterministic
- Eras change resolution, not meaning

If an era change alters causality,
the abstraction has been violated.
