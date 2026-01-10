# Lens

This document defines the **Lens** in Continuum.

Lens is the **canonical observer boundary**. End programs (tools, visualizers, analyzers)
must consume field data **only through Lens**.

Lens is strictly **non-causal**. Removing Lens must not change simulation outcomes.

---

## 1. What the Lens Is

Lens is an observer-layer system that:
- ingests field emissions from Measure
- structures samples using virtual topology
- reconstructs continuous field functions
- provides deterministic query APIs for tools and visualization

Lens does **not** influence execution or scheduling.

---

## 2. Observer Boundary (Absolute)

The boundary is strict.

Allowed:
- Lens reads field emissions
- Lens reconstructs and serves observations
- End programs query Lens APIs

Forbidden:
- End programs read FieldSnapshot directly
- Domains query Lens
- Lens writes signals or affects execution order

If removing Lens changes outcomes, the boundary is broken.

---

## 3. Inputs and Transport

Lens consumes **field emissions** produced during Measure.

Each emission includes:
- field identifier
- tick and stratum
- payload samples
- optional metadata

`FieldSnapshot` is **internal transport only**. It is not a public API.
End programs must never read `FieldSnapshot` directly.

---

## 4. Fields Are Functions

A field is a function:

```
f : Position -> Value
```

Samples are **constraints**, not final data.
Any observer usage must go through reconstruction.

If code loops over raw samples as final data, it violates the Lens contract.

---

## 5. Reconstruction (Mandatory)

Lens reconstructs a continuous (or piecewise-continuous) field function from samples.

Reconstruction:
- is deterministic
- is observer-only
- never feeds back into execution

Reconstruction methods may vary (nearest, linear, RBF, etc.) but the boundary does not.

---

## 6. Virtual Topology (Structure)

Lens organizes samples using **virtual topology** to provide:
- stable tile/region identifiers
- deterministic spatial partitioning
- coherent LOD/refinement paths

Virtual topology is observer-owned structure only. It contains no values.

---

## 7. Temporal Interpolation

Lens may serve data at fractional time using temporal interpolation.

Rules (observer-only):
- Scalars: linear interpolation
- Vectors: lerp then normalize
- Metadata: categorical, choose prev or next

Playback may lag simulation by a configurable interval to ensure bracketing frames.

---

## 8. Query API (Conceptual)

Lens exposes read-only queries. The exact types may evolve, but the boundary does not.

Examples:
- `latest(field_id)` -> reconstruction
- `at(field_id, tick)` -> reconstruction
- `tile(field_id, tile_id, tick)` -> reconstruction
- `query(field_id, position, time)` -> value
- `query_playback(field_id, position, playback)` -> value
- `query_batch(field_id, positions, time)` -> values

Queries must be pure and deterministic.

---

## 9. Refinement

Lens may request refinement when uncertainty is too high.

Refinement requests:
- are observer-only
- are deduplicated and ordered deterministically
- result in **new sampling of signals** (not resampling the point cloud)

---

## 10. Determinism

Given identical field emissions and playback policy:
- Lens must produce the same reconstructions
- Query results must be stable

Minor visualization rounding differences are acceptable, but the data must be stable.

---

## 11. What Lens Is Not

Lens is not:
- part of the execution DAG
- a simulation cache
- a scripting surface
- a renderer
- a source of truth

Truth belongs to signals. Meaning belongs to observers.

---

## Summary

- Lens is the canonical observer boundary
- FieldSnapshot is internal transport only
- Reconstruction is mandatory
- Virtual topology provides structure for queries and refinement
- Lens is deterministic and removable without causal impact
