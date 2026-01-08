Here is a **clean, authoritative `observers/lens.md`** that matches the rest of your system: strict observer boundary, no causal leakage, no legacy assumptions.

This document defines **Lens as the observation interface**, not a tool, not a renderer, not a cache.

---

```md
# Lens

This document defines the **Lens** in Continuum.

The Lens is the **primary observation interface** for simulation output.
It reconstructs and exposes **Fields** for inspection, analysis, and visualization.

The Lens is strictly **non-causal**.

---

## 1. What the Lens Is

The **Lens** is an observer-layer system that:

- ingests emitted fields
- reconstructs field representations
- exposes query and sampling interfaces
- provides data to visualization and analysis tools

The Lens does **not** influence execution.
It observes causal history after the fact.

---

## 2. Lens and the Observer Boundary

The observer boundary is absolute.

The Lens:
- may read fields
- may read resolved signals (if explicitly allowed)
- must never write signals
- must never affect execution order
- must never affect timing or determinism

Removing the Lens must not change simulation results.

If removing the Lens changes outcomes, the boundary has been violated.

---

## 3. Lens Inputs

The Lens consumes **Field emissions** produced during the **Measure phase**.

Each field emission includes:
- a field identifier
- a tick and stratum
- payload samples
- optional metadata

The Lens does not consume:
- signal inputs
- unresolved state
- execution internals

---

## 4. Field Reconstruction

Fields may be emitted in minimal or partial form.

The Lens is responsible for:
- reconstructing spatial structure
- interpolating or aggregating samples (observer-only)
- producing queryable representations

Reconstruction:
- must be deterministic
- must not introduce new causal meaning
- must not feed back into execution

Reconstruction policies are observer concerns, not simulation concerns.

---

## 5. Lens Queries

The Lens exposes read-only queries.

Typical queries include:
- field value at a position
- field slices or projections
- temporal sampling
- aggregate statistics

Queries must:
- be pure
- be side-effect free
- never block execution

---

## 6. Lens and Time

Lens data is indexed by:
- tick
- era
- stratum

The Lens does not advance time.
It only observes it.

Temporal interpolation or resampling is allowed,
but must be clearly marked as derived observation.

---

## 7. Lens and Determinism

Lens behavior should be deterministic with respect to its inputs.

Given the same emitted fields:
- the same reconstructed fields should be produced
- the same query results should be returned

Minor floating-point differences in visualization are acceptable,
but the underlying data must be stable.

Lens determinism is important for analysis,
but never for causality.

---

## 8. Lens and Performance

The Lens may:
- cache reconstructed fields
- downsample data
- stream data incrementally

These are performance optimizations only.

Performance decisions must not:
- alter field values
- hide missing data
- change interpretation silently

---

## 9. Lens vs Other Observers

The Lens is the **foundational observer**.

Other observers (e.g. chronicles, visualizers, analytics tools):
- build on top of the Lens
- consume its outputs
- never bypass it to access execution internals

The Lens defines the canonical observer interface.

---

## 10. What the Lens Is Not

The Lens is **not**:
- part of the execution graph
- a simulation cache
- a scripting surface
- a control mechanism
- an authority on truth

Truth belongs to signals.
Meaning belongs to observers.

---

## Summary

- The Lens observes fields, never signals
- It reconstructs and exposes data
- It is strictly non-causal
- It must be removable without effect
- It defines the observer interface boundary

If the Lens influences execution,
the system is broken.
```

---

### Status

With `lens.md` in place:

* Observers are now fully specified:

  * `fields.md` — what is observed
  * `lens.md` — how observation is exposed
  * `chronicles.md` — how meaning is assigned
* The observer boundary is airtight
* Tooling (`run`, `analyze`, visualization) has a clean anchor

If you want to continue, the next logical targets would be:

* a light pass on `observers/fields.md` to ensure alignment, or
* `tools/run.md` to reflect Lens-first observation clearly.

But `lens.md` itself is now complete and correct.
