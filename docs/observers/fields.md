# Fields

This document defines **Fields** in Continuum.

Fields are **observations** of the simulation.
They never participate in causality.

---

## 1. What a Field Is

A **Field** represents measured or derived data emitted by the simulation.

A field:
- has a stable identity
- has a payload type
- may be spatial or non-spatial
- is produced during execution but never consumed by it

Fields exist to make simulation state **observable**.

---

## 2. Fields vs Signals

Signals and fields are intentionally separated.

| Aspect | Signal | Field |
|------|--------|-------|
| Role | Authority | Observation |
| Phase | Resolve | Measure |
| Can influence execution | Yes | No |
| Deterministic | Yes | Yes |
| Required for causality | Yes | No |

Fields must never affect signals.

---

## 3. Field Emission

Fields are emitted during the **Measure phase**.

Emission:
- reads resolved signals
- produces samples or aggregates
- does not modify simulation state

Field emission must be:
- deterministic
- side-effect free
- non-causal

---

## 4. Field Structure

A field consists of:
- a descriptor (identity, topology, units)
- one or more samples
- optional metadata

The exact storage and reconstruction is observer-defined.

---

## 5. Fields and Observers

Fields are consumed by observers.

Observers may:
- reconstruct fields
- sample fields
- visualize fields
- store fields

Observers must not:
- modify signals
- influence execution
- affect timing or ordering

---

## 6. Fields and Time

Fields are associated with:
- a tick
- a stratum
- an era

Fields do not advance time.
They only observe it.

---

## 7. Fields and Determinism

Field emission must be deterministic.

Given the same causal history:
- the same fields must be emitted
- with the same values
- in the same order

Observer presentation may differ.

---

## 8. What Fields Are Not

Fields are **not**:
- authoritative state
- inputs to signals
- configuration
- caches
- mutable data stores

They are **read-only observations**.

---

## Summary

- Fields are observer-only
- Fields never influence causality
- Fields are emitted during Measure
- Signals and fields must never mix

If a field influences execution,
the observer boundary is broken.
