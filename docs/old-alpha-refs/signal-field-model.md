# SIGNAL & FIELD MODEL

> For complete execution order, signal dependencies, and fracture flow,
> see @./execution-flow.md

---

# CORE RULE

## Signals are authority. Fields are observation.

### Signals (simulation-internal)

Signals are the **authoritative coupling layer** of the engine.

* Typed
* Deterministic
* Used by simulation systems
* Resolved once per tick/stratum

Signals define **what the simulation knows**.

### Fields (observer-only)

Fields are **measurements** of resolved signals.

* Lossy
* Quantized
* Sampled
* Used only by observers

Fields define **what observers see**.

Simulation systems must never depend on fields.

---

# CHRONICLE AS OBSERVER BOUNDARY (HARD RULE)

Lens is the **only observer interface**.

End programs (apps, tools, visuals, analytics) must **never** consume domain field output directly.

Rules:
* `FieldSnapshot` is **internal transport** only (measurement → lens ingest).
* End programs must **only** query `continuum-lens` APIs.
* No app/tool/visual may read `LatestFieldSnapshots`, subscribe to field events, or reconstruct fields outside Lens.
* If an end program needs a field, Lens must serve it.

Removing Lens must not change simulation outcomes; it only changes what can be observed.

---

# FIELDS ARE FUNCTIONS (HARD RULE)

A field is a mathematical function:

```
f : Position → Value
```

Samples are **constraints**, not the field itself. Density is never the solution.

Forbidden reasoning:
* "just sample more densely"
* "increase the grid resolution"
* "add more points until it looks right"

End programs must use **reconstruction** via Lens to query `f(x)`.

---

# VIRTUAL TOPOLOGY (OBSERVER-OWNED STRUCTURE)

Virtual topology defines **regions, tiles, adjacency, LOD, and sampling patterns**.

It contains **no values** and is **not simulation state**.

Lens owns virtual topology and uses it to organize constraints and refinement.

Primer:
* Topology is a **stable indexing space** over a manifold; it is not a mesh and not data.
* It defines **tile ids, adjacency, and refinement rules** (LOD), so queries are structured.
* It answers **where** to sample and **how** to subdivide; it never answers **what** the value is.
* Reconstruction uses topology to interpret constraints and to determine uncertainty.
* End programs never bypass it; they query Lens with topology-aware requests.

---

# RECONSTRUCTION & REFINEMENT

All end-program usage of fields goes through:

```
FieldSamples + ReconstructionSpec → f(x)
```

Rules:
* Reconstruction is observer-only and deterministic.
* Queries use Lens; raw samples are never rendered directly.
* Refinement resamples **signals**, not the point cloud.
* New constraints are ingested by Lens and may update caches.

---

# SIGNAL MODEL (ENGINE CONTRACT)

For any signal `S`:

1. **collect** — systems add deltas into `SignalInputs<S>`
2. **resolve** — one resolver computes next value from
   `{prev_state, inputs, dt, WorldSeed}`
3. **clear** — inputs are reset
4. **measure (optional)** — observers emit `FieldSnapshot`

### Rules

* Input deltas must be commutative unless explicitly documented.
* Non-commutative resolution must define a canonical stable ordering.
* Prefer resolved signals over raw ECS state when available.
* Signals are strongly typed — no stringly-typed buses.

### Post-Resolve Hooks

Signals may declare an optional `post_resolve` hook for derived resource updates:

```rust
#[signal(
    domain = "terra",
    title = "Plate Accel Derived",
    description = "Triggers PlateAccel rebuild when PlatesCatalog changes",
    reads = "PlatesCatalog",
    resolve_on = "terra.tectonics",
    post_resolve = "crate::geophysics::plates::rebuild_plate_accel_hook"
)]
pub struct PlateAccelDerived(pub ());
```

**Hook signature:** `fn(&mut World, ResolveCtx, &WorldSeed)`

**Use cases:**
* Derived resource updates (acceleration structures, caches)
* Side effects that depend on resolved signal values
* Operations requiring full world access after resolution

**Execution order:**
1. Standard signal resolver runs
2. Post-resolve hook runs with full world access
3. Next signal in dependency order proceeds

The hook participates in compute graph ordering via the signal's `reads` dependencies.

> **Deep dive**: See @./execution-flow.md for complete signal dependency resolution,
> compute graph construction, and level-by-level execution.

---

# FRACTURE (SIMULATION KERNEL)

`continuum-fracture` is **engine kernel**, not a domain and not an addon.

Fracture is responsible for **detecting and resolving tension** in the simulation.

## Responsibilities

* Executes as a fixed kernel phase.
* Runs **registered tension detectors**.
* Owns scheduling and execution order.
* Enforces determinism.

## Detector contract

Detectors:

* **Read:** resolved signals and simulation-authoritative ECS state
* **Write:** signal input deltas and/or typed tension tokens
* **Never read:** fields, lens, visuals, or history

Domains and coupling crates **register detectors**; they do not run fracture logic themselves.

Observers may later **measure fracture outputs into fields**, but fracture itself is signal-native.

> **Deep dive**: See @./execution-flow.md for complete fracture execution order,
> detector contract, and how tension-driven inputs flow to the next tick.

---

# FIELDS-FIRST OBSERVATION

All inspection, visualization, tooling, and history operate on **fields**, never on domain-specific snapshots.
Those fields are **not public**; end programs see them only **through Lens**.

Canonical observer payload:

```
FieldSample { position: Vec3, scalar: f32, vector: VecN, metadata: u32 }
FieldDescriptor { id, name, topology, unit, value_range? }
FieldSnapshot { tick, descriptor, samples }
```

> `VecN` denotes a domain-defined tangent/vector space.
> The engine does not assume dimensionality or geometry.

If you think you need `PlateSnapshot`, `ForceSnapshot`, `SettlementSnapshot`, etc.
→ **emit another field**.

---

# OBSERVABILITY-FIRST RULE

A simulation feature is **not complete** unless it is observable.

When adding a simulation feature, you must define:

1. The authoritative **signal(s)**
2. A **field emission plan**
3. An **observer interpretation plan** (visual, analytical, or derived)

Visualization is one observer.
Headless analysis is another.
Fields are the contract.

---

# DEBUG_VIEW MANDATE (HARD RULE)

Every observable primitive **must** declare an explicit `debug_view`.

This applies to:
- `#[signal(...)]` — must include `debug_view = "<hint>"`
- `#[field(...)]` — must include `debug_view = "<hint>"`
- `#[impulse(...)]` — must include `debug_view = "<hint>"`
- `#[fracture_detector(...)]` — must include `debug_view = "<hint>"`

Valid hints:
- `"heatmap"` — scalar field visualization
- `"vector_field"` — directional/vector visualization
- `"timeseries"` — temporal evolution
- `"histogram"` — distribution analysis
- `"events"` — discrete event visualization
- `"tensor"` — matrix/tensor visualization

## Enforcement

The derive macros will **fail compilation** if `debug_view` is omitted.

The debug instrumentation is gated behind the `debug_view` feature flag:
- When enabled: debug fields are auto-generated and registered
- When disabled: zero runtime cost, but the attribute is still required

## Rationale

If something cannot be observed in debug mode, it cannot be diagnosed.
Forcing explicit debug view selection ensures:
1. Every simulation primitive is inspectable
2. Authors think about observation semantics upfront
3. Debug tooling has consistent metadata to render any primitive

---

# OBSERVER TEMPORAL POLICY (PLAYBACK LAG)

Observers are allowed to introduce **temporal presentation policy** as long as it is:
- **Field-only** (never influences simulation outcomes; Lens remains the public interface)
- **Deterministic** given `{FieldSnapshot stream, dt_seconds, playback_policy}`
- **Removable** (changing/removing observers must not change simulation results)

## Default: One-Interval Playback (Interpolation-First)

To avoid "ticking" visuals (e.g. clouds), the observer kernel runs a **playback clock** that is
intentionally **one stratum interval behind** the simulation for the strata it wants to smooth.

Conceptually:

```
t_play = t_sim - Δt_lag
```

Where `Δt_lag` is chosen to ensure the observer has **two bracketing keyframes** for interpolation.

## Field interpolation contract (observer-only)

For each `FieldId`, observers maintain:
- `prev: FieldSnapshot`
- `next: FieldSnapshot`

Rendering/analysis at `t_play` uses a blend factor:

```
alpha = clamp((t_play - t_prev) / (t_next - t_prev), 0..1)
```

Interpolation rules:
- Scalars: `lerp(prev.scalar, next.scalar, alpha)`
- Vectors: `normalize(lerp(prev.vector, next.vector, alpha))` (or magnitude-aware variant when needed)
- Metadata: **do not** interpolate; treat as categorical (choose `prev` or `next`)

This produces smooth observer output without inventing simulation authority and without requiring
simulation to run at render cadence.
