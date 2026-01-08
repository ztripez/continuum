# Impulses

This document defines **Impulses** in Continuum.

Impulses are the **only mechanism by which external influence enters a World**.

---

## 1. What an Impulse Is

An **Impulse** represents an explicit, modeled external influence applied to a World.

An impulse:
- has a stable identity
- has a typed payload
- is applied at a specific tick
- participates in causality

Impulses are part of the simulation model.
They are not shortcuts or overrides.

---

## 2. Why Impulses Exist

Worlds are causally closed, but not isolated.

Impulses exist to model:
- interventions
- boundary conditions
- user actions
- experimental perturbations

If an external effect influences the simulation,
it must do so via an impulse.

---

## 3. Impulses and Causality

Impulses:
- contribute inputs to signals
- are applied during the **Collect phase**
- influence future resolution deterministically

Impulses are not special-cased.
Once applied, they are indistinguishable from other causal inputs.

---

## 4. Impulse Declaration

Impulses are declared in the DSL.

A declaration defines:
- payload schema
- valid targets
- application constraints
- validation rules

An impulse must not:
- write signals directly
- bypass assertions
- mutate state outside the Collect phase

---

## 5. Impulses and Scenarios

Scenarios may schedule impulses.

Impulse scheduling:
- is part of scenario configuration
- is fixed before execution begins
- is part of deterministic input

Changing impulse schedules defines a new Scenario.

---

## 6. Impulses and Determinism

Impulses are fully deterministic.

Given:
- the same World
- the same Scenario (including impulse schedule)
- the same seed

impulse effects must be identical.

If impulse timing or ordering differs, execution is incorrect.

---

## 7. What Impulses Are Not

Impulses are **not**:
- ad-hoc overrides
- scripting hooks
- observer callbacks
- debugging tools

They are **first-class causal inputs**.

---

## Summary

- Impulses are the only external causal entry point
- They apply during Collect
- They influence signals via normal resolution
- They are part of replay identity

If something affects the simulation from “outside” and isn’t an impulse,
the model is wrong.
