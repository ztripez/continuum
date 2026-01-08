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
- additional signal inputs may be emitted

Fractures do not:
- read fields
- mutate signals directly
- observe future state

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
- the condition to detect (signal-only)
- the response (signal input emission)
- scope and phase constraints

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

## 7. What Fractures Are Not

Fractures are **not**:
- conditional logic
- branching control flow
- scripted transitions
- observer hooks

They are **emergent causal responses**.

---

## Summary

- Fractures detect emergent tension
- They run after resolution
- They emit causal inputs
- They are not authored outcomes

If fractures feel like scripting,
the abstraction has been violated.
