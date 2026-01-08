Let’s lock the responsibilities:

### **World**

* Defines *what exists*
* Defines time, eras, strata, execution rules
* Defines signals, operators, fields, impulses
* Immutable causal structure

### **Scenario**

* Defines *how this world is instantiated*
* Sets:

  * initial signal values
  * signal parameters / constants
  * enabled/disabled subsystems (within declared bounds)
* No logic
* No new signals
* No new operators

### **Run**

* Executes:

  * a World
  * under a Scenario
  * with a seed
* Produces causal history
* Attaches observers

This avoids **configuration drift** and keeps causality pure.

---

## New document: `scenario.md` (or `signal-configuration.md`)

Below is a **clean, root-level doc** that fits your existing structure and terminology.

I recommend calling the file:

```
./scenario.md
```

and linking it from `world.md` and `tools/run.md`.

---

## `scenario.md`

```md
# Scenarios

This document defines **Scenarios** in Continuum.

A Scenario configures **how a World is instantiated** for execution.
It does not define causal structure or execution semantics.

---

## 1. What a Scenario Is

A **Scenario** defines a specific configuration of a World.

A Scenario:
- selects initial conditions
- sets signal parameters
- fixes configuration choices within declared bounds

A Scenario is:
- deterministic
- declarative
- reusable across runs

A Scenario is **not** a World.
It cannot introduce new logic.

---

## 2. World → Scenario → Run

Continuum execution is structured as:

```

World → Scenario → Run

```

- **World** defines *what exists*
- **Scenario** defines *how it starts*
- **Run** defines *when and how it executes*

This separation is fundamental.

---

## 3. What a Scenario May Configure

A Scenario may configure:

- initial values of signals
- constant parameters used by signals or operators
- optional feature flags explicitly declared by the World
- impulse schedules (if allowed by the World)

All configuration must target **declared hooks**.

If a value cannot be configured via a declared hook,
it cannot be configured at all.

---

## 4. What a Scenario Must Not Do

A Scenario must not:

- define new signals
- define new operators
- alter dependencies
- change execution phases
- change time semantics
- bypass validation or assertions

A Scenario must never change **causal structure**.

---

## 5. Signal Configuration

Signals may declare **configuration parameters**.

These parameters:
- have names and types
- may have bounds or validation rules
- may have defaults

A Scenario may override these parameters.

Parameter values are:
- fixed for the duration of a run
- visible to signal resolution
- part of the causal model

Changing a parameter produces a **different scenario**.

---

## 6. Initial Conditions

A Scenario may specify **initial signal values**.

Initial conditions:
- apply before the first tick
- must satisfy all signal assertions
- are part of the causal history root

If initial conditions violate invariants, execution must fail.

---

## 7. Scenario Identity

A Scenario has a stable identity defined by:
- the World specification
- the Scenario configuration data

Two runs with:
- the same World
- the same Scenario
- the same seed

must produce identical causal history.

---

## 8. Scenarios and Reuse

Scenarios are designed to be reused.

Common use cases include:
- parameter sweeps
- ensemble runs
- sensitivity analysis
- comparative studies

A single World may have many Scenarios.

---

## 9. Scenarios and Determinism

Scenario configuration is part of the deterministic input set.

Replay requires:
- world specification
- scenario specification
- initial seed
- era sequence

If changing a Scenario changes results, this is expected.
If the same Scenario produces different results, it is a bug.

---

## 10. What a Scenario Is Not

A Scenario is **not**:
- a scripting layer
- a patch to the World
- a dynamic configuration system
- an observer

It is a **static instantiation description**.

---

## Summary

- Worlds define causal structure
- Scenarios define initial conditions and parameters
- Runs execute Worlds under Scenarios
- Scenarios must not alter logic
- Scenario configuration is part of replay identity

If a configuration choice changes causality,
it belongs in the World, not the Scenario.
