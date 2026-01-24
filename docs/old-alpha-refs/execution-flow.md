# KERNEL EXECUTION FLOW

> **The definitive reference for execution order, signal mutation, and fracture resolution.**

This document describes the complete execution flow of a Continuum simulation tick:
how strata fire, how signals resolve, how signals depend on and mutate other signals,
and how fracture detectors inject tension-driven inputs.

---

## 1. Tick Anatomy (Overview)

A single simulation tick executes the following phases in order:

```
┌─────────────────────────────────────────────────────────────────────┐
│                           ONE TICK                                   │
├─────────────────────────────────────────────────────────────────────┤
│ 1. Strata Advancement     → Determine which strata fire this tick   │
│ 2. Impulse Bookkeeping    → Clear per-tick counters                 │
│ 3. Couplers (per stratum) → Read prev signals → Write SignalInputs  │
│ 4. Signal Resolution      → Level-by-level DAG execution            │
│ 5. Fracture               → Tension detectors → Write SignalInputs  │
│ 6. Field Capture          → Observer-only measurement               │
│ 7. Tick Advance           → SimTime.tick++                          │
└─────────────────────────────────────────────────────────────────────┘
```

**Key insight**: Fracture runs *after* signal resolution. Tension detectors read
the just-resolved signals and write inputs that will be consumed in the *next* tick.

---

## 2. Multi-Rate Strata

Not all systems need to run every tick. Geological processes operate on megayear
timescales; atmospheric processes on hours. Strata provide multi-rate scheduling.

### 2.1 Stratum Schedules

```rust
pub enum StratumSchedule {
    EveryBaseTick,       // Fires every tick
    StrideTicks(u64),    // Fires every N base ticks (at tick % N == 0)
    Manual,              // Never auto-fires; must be triggered explicitly
}
```

### 2.2 How Strata Fire

Each tick, `TimeStrata::advance()` populates `FiredStrata` with which strata fire:

```rust
// Pseudocode
for (id, runtime) in strata {
    let fires = match runtime.schedule {
        EveryBaseTick => true,
        StrideTicks(n) => n != 0 && (tick % n == 0),
        Manual => false,
    };
    if fires {
        fired.ids.insert(id);
    }
}
```

### 2.3 Effective dt

When a stratum fires, its `dt_seconds` accounts for the stride:

```
dt_effective = dt_base * stride
```

For example, if `dt_base = 1000 years` and `stride = 100`, then `dt_effective = 100,000 years`.

### 2.4 Per-Stratum Execution

Within a tick, each fired stratum is processed independently:
1. Run couplers for this stratum
2. Run signal resolution for this stratum (level by level)
3. Run fracture detectors for this stratum

Strata are processed in `StratumId` order (deterministic).

---

## 3. Signal Model

Signals are the **authoritative coupling layer** of the simulation. They define
what the simulation knows.

### 3.1 The Signal Contract

For any signal `S`, each tick/stratum executes:

```
collect → resolve → clear → (optional) measure
```

1. **Collect**: Systems accumulate deltas into `SignalInputs<S>`
2. **Resolve**: One resolver computes next value from `{prev, inputs, dt, seed}`
3. **Clear**: Inputs are reset for next tick
4. **Measure**: (Optional) Observers emit `FieldSnapshot`

### 3.2 Key Types

| Type | Purpose |
|------|---------|
| `SignalSpec` | Static metadata: `PATH`, `ID`, `DESCRIPTOR` |
| `SignalRuntimeSpec` | Runtime binding: `Value`, `Input`, `Resolver`, `READS` |
| `SignalInputs<S, I>` | Accumulated deltas for this tick (cleared after resolve) |
| `SignalResolved<S, V>` | Latest authoritative resolved value |
| `ResolveCtx` | Deterministic context: `tick`, `stratum`, `dt_seconds` |

### 3.3 SignalResolver Trait

```rust
pub trait SignalResolver<S: SignalSpec, V, I>: Send + Sync + 'static {
    /// Optional: refresh resolver-local caches from world state.
    fn pre_resolve(&mut self, world: &World, ctx: ResolveCtx, seed: &WorldSeed) {}

    /// Core: compute next value from {prev, inputs, ctx, seed}.
    /// MUST be deterministic.
    fn resolve(&mut self, prev: &V, inputs: &[I], ctx: ResolveCtx, seed: &WorldSeed) -> V;
}
```

### 3.4 Post-Resolve Hooks

Signals may declare an optional `post_resolve` hook for derived resource updates:

```rust
#[signal(
    domain = "terra",
    title = "Example",
    resolve_on = "terra.stratum",
    post_resolve = "crate::rebuild_acceleration_structure"
)]
pub struct ExampleSignal(pub f64);
```

**Hook signature**: `fn(&mut World, ResolveCtx, &WorldSeed)`

Execution order:
1. Standard resolver runs
2. Post-resolve hook runs with full world access
3. Next signal in dependency order proceeds

---

## 4. How Signals Depend on Other Signals

Signals form a directed acyclic graph (DAG) of dependencies. A signal may read
other signals to compute its value.

### 4.1 Declaring Dependencies

Use the `reads` attribute in the `#[signal(...)]` macro:

```rust
#[signal(
    domain = "terra",
    title = "Surface Temperature",
    reads = "AtmosphericComposition",  // Type name, verified at compile time
    resolve_on = "terra.atmosphere"
)]
pub struct SurfaceTemperatureK(pub f32);
```

### 4.2 Dependency Timing

```rust
pub enum DependencyTiming {
    SameTick,       // Must resolve before this signal (creates DAG edge)
    PreviousTick,   // Reads previous tick's value (no ordering constraint)
}
```

**Same-tick dependencies** create edges in the compute graph. The dependency must
resolve before the dependent signal.

**Previous-tick dependencies** read the already-resolved value from the prior tick.
No ordering constraint; useful for feedback loops.

### 4.3 Cross-Stratum Dependencies

If signal A (stratum X) depends on signal B (stratum Y), the dependency is
**always previous-tick** (no same-tick edge). Signal B will have resolved in
a prior tick or a prior stratum phase.

---

## 5. Compute Graph (Signal Ordering)

The `SignalComputeGraph` transforms flat signal registrations into a structured DAG.

### 5.1 Level-Parallel Execution

Signals are assigned to levels via Kahn's algorithm (topological sort):

```
Level 0: [SignalA, SignalB, SignalC]  ← No dependencies, can run in any order
    ↓
Level 1: [SignalD, SignalE]           ← Depend on level 0 signals
    ↓
Level 2: [SignalF]                    ← Depends on level 1 signals
```

All signals in a level are independent and could theoretically run in parallel.
Currently they run sequentially, sorted by `SignalId` for determinism.

### 5.2 Graph Construction

1. **Collect nodes**: Each `SignalResolveBinding` becomes a `ComputeNode`
2. **Build edges**: `SameTick` dependencies create edges (dependency → dependent)
3. **Topological sort**: Kahn's algorithm assigns levels
4. **Sort within levels**: Signals sorted by `SignalId.raw()` (u64)

### 5.3 Cycle Detection

Circular dependencies are detected at graph construction time:

```rust
// This would panic at startup:
// A reads B, B reads A  →  CycleDetected error
```

### 5.4 Execution

```rust
for stratum in fired_strata {
    let graph = compute_graph.get_stratum(stratum);
    
    // Run couplers first (sorted by name)
    for coupler in graph.couplers {
        coupler.run(world, ctx, seed);
    }
    
    // Then resolve signals level by level
    for level in graph.levels {
        for signal_id in level {
            let node = graph.nodes.get(signal_id);
            (node.resolve)(world, ctx, seed);
        }
    }
}
```

---

## 6. Couplers (Cross-Domain Glue)

Couplers are cross-domain functions that read previous tick's resolved signals
and write to `SignalInputs` for the current tick.

### 6.1 Purpose

Domains cannot depend on each other. Couplers provide the glue:

```rust
// Example: Atmosphere-Ocean coupling
fn atmosphere_ocean_coupler(world: &mut World, ctx: ResolveCtx, seed: &WorldSeed) {
    // Read previous tick's atmosphere state
    let atm = world.resource::<SignalResolved<AtmosphereState>>();
    
    // Write inputs to ocean signals for this tick
    let mut ocean_inputs = world.resource_mut::<SignalInputs<OceanHeat>>();
    ocean_inputs.push(HeatDelta { ... });
}
```

### 6.2 When Couplers Run

**Before signal resolution**, within each stratum. This ensures coupler outputs
are available as inputs when signals resolve.

### 6.3 CouplerNode Structure

```rust
pub struct CouplerNode {
    pub name: &'static str,        // For debugging and deterministic ordering
    pub stratum_id: StratumId,     // Which stratum this fires on
    pub couple: CouplerFn,         // The coupling function
    pub reads: &'static [SignalId],   // Documentation: what it reads
    pub writes: &'static [SignalId],  // Documentation: what it writes
}
```

Couplers within a stratum are sorted by `name` for determinism.

---

## 7. Fracture (Tension Detection)

Fracture is the **tension engine** of the simulation kernel. It detects unstable
conditions and triggers corrective responses by writing signal inputs.

> **"Structure emerges from tension."**

Rather than authoring events manually (e.g., "spawn a volcano here"), you define
**tension detectors** that watch for unstable conditions.

### 7.1 When Fracture Runs

**After signal resolution, before field measurement**:

```
KernelPhase::Resolve   → Signals resolved from inputs
KernelPhase::Fracture  → Tension detectors run (this phase)
KernelPhase::Measure   → Fields emitted for observers
```

This placement allows detectors to:
- **Read** the latest resolved signals
- **Write** inputs that will be resolved in the **next** tick

### 7.2 Detector Contract

**Detectors MAY:**
| Operation | Target |
|-----------|--------|
| **Read** | `SignalResolved<S>` (resolved signals) |
| **Read** | Authoritative ECS state (entities, components) |
| **Write** | `SignalInputs<S>` (for next resolution) |
| **Write** | `ImpulseBus<I>` (discrete events) |

**Detectors MUST NOT:**
| Operation | Reason |
|-----------|--------|
| Read fields, lens, or observer state | Observer-removable invariant |
| Introduce non-determinism | Use `seed.derive(...)` for RNG |
| Mutate resolved signals directly | Signals resolve from inputs only |

### 7.3 Detector Function Signature

```rust
fn my_detector(world: &mut World, ctx: TensionCtx, seed: &WorldSeed) {
    // Read resolved signals
    let temp = world.resource::<SignalResolved<CoreTemp>>();

    // Detect tension
    if temp.value.0 > THRESHOLD {
        // Write signal input for next resolve
        world.resource_mut::<SignalInputs<Eruption>>()
            .push(EruptionDelta { energy: 1e18 });
    }
}
```

### 7.4 Registration

Detectors are registered at link-time via `submit_tension_detector!`:

```rust
continuum_foundation::submit_tension_detector!(TensionDetectorRegistration {
    domain: "terra",
    detector_path: "terra.detectors.hydraulic_erosion",
    detector_id: TensionDetectorId::from_path_const("terra.detectors.hydraulic_erosion"),
    stratum_path: "terra.hydrology",
    stratum_id: <crate::Hydrology as StratumSpec>::ID,
    run: hydraulic_erosion_detector,
    type_name: "hydraulic_erosion_detector",
});
```

### 7.5 Execution Order

Fracture guarantees deterministic execution:

1. **Strata processed in `StratumId` order** (BTreeMap)
2. **Within stratum, detectors sorted by `detector_id`** (u64 hash of path)
3. **Only strata that fired this tick are processed**

---

## 8. Impulses (Discrete Events)

Impulses are **authoritative discrete events** that flow through the simulation.
Unlike signals (continuous values), impulses represent instantaneous occurrences.

### 8.1 ImpulseSpec and ImpulseBus

```rust
#[impulse(
    domain = "terra",
    title = "Volcanic Eruption",
    description = "A volcanic eruption event",
)]
pub struct VolcanicEruption {
    pub index: u32,
    pub heat_j: f64,
    pub ejecta_m: f32,
}

// ImpulseBus<VolcanicEruption> is auto-generated
```

### 8.2 Emitting Impulses

Tension detectors emit impulses via `ImpulseBus::emit()`:

```rust
fn volcanic_detector(world: &mut World, ctx: TensionCtx, seed: &WorldSeed) {
    // ... detect eruption conditions ...
    
    if let Some(mut bus) = world.get_resource_mut::<ImpulseBus<VolcanicEruption>>() {
        bus.emit(ctx.tick, ctx.stratum, VolcanicEruption {
            index: cell_idx,
            heat_j: 1e18,
            ejecta_m: 500.0,
        });
    }
}
```

### 8.3 Impulse Lifecycle

- **Per-tick counters** cleared at start of each tick
- **Items** accumulated during fracture phase
- Other systems can **query** `ImpulseBus<I>` to react to events
- Impulses are authoritative (part of simulation, not observer)

---

## 9. Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                           TICK N                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. Strata Advancement                                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  TimeStrata::advance(tick) → FiredStrata populated           │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  2. Impulse Bookkeeping                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Clear per-tick counters on all ImpulseBus<I>                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  FOR EACH FIRED STRATUM (in StratumId order):                       │
│  │                                                                   │
│  │  3. Couplers                                                     │
│  │  ┌───────────────────────────────────────────────────────┐      │
│  │  │  Read: SignalResolved<A> (tick N-1)                    │      │
│  │  │  Write: SignalInputs<B> (for this tick's resolve)      │      │
│  │  └───────────────────────────────────────────────────────┘      │
│  │                          ↓                                        │
│  │  4. Signal Resolution (level by level)                           │
│  │  ┌───────────────────────────────────────────────────────┐      │
│  │  │  Level 0: [A, B] → resolve independently                │      │
│  │  │  Level 1: [C]    → depends on A or B                    │      │
│  │  │  Level 2: [D, E] → depends on level 0-1                 │      │
│  │  │                                                         │      │
│  │  │  For each signal:                                       │      │
│  │  │    1. pre_resolve(world, ctx, seed)                     │      │
│  │  │    2. resolve(prev, inputs, ctx, seed) → new value      │      │
│  │  │    3. post_resolve(world, ctx, seed)                    │      │
│  │  │    4. SignalInputs cleared                              │      │
│  │  └───────────────────────────────────────────────────────┘      │
│  │                          ↓                                        │
│  │  5. Fracture (tension detection)                                 │
│  │  ┌───────────────────────────────────────────────────────┐      │
│  │  │  For each detector (sorted by detector_id):             │      │
│  │  │    - Read: SignalResolved<*> (just resolved)            │      │
│  │  │    - Detect: tension conditions                         │      │
│  │  │    - Write: SignalInputs<*> (for tick N+1)              │      │
│  │  │            ImpulseBus<*> (discrete events)              │      │
│  │  └───────────────────────────────────────────────────────┘      │
│  │                                                                   │
│  └───────────────────────────────────────────────────────────────┘  │
│                              ↓                                       │
│  6. Field Capture (observer-only, may be strided)                   │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Emit FieldSnapshot for observers (Lens ingestion)           │   │
│  │  Does NOT affect simulation (removable)                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              ↓                                       │
│  7. Tick Advance                                                    │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  SimTime.tick++                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                               ↓
                           TICK N+1
                    (fracture-written inputs consumed)
```

---

## 10. Worked Example: Thermal-Tectonic Cascade

This example shows how a tension detector triggers a signal cascade across ticks.

### Setup

- `CoreHeat` signal: heat content of planetary core (Joules)
- `MantleVigor` signal: convective vigor of mantle (dimensionless)
- `PlateMotion` signal: velocity field of tectonic plates (m/Myr)

### Tick N: Resolution

1. `CoreHeat` resolves to 2.5e30 J (above threshold)
2. `MantleVigor` resolves based on previous heat

### Tick N: Fracture

3. `thermal_mechanical_coupling` detector runs:
   ```rust
   fn thermal_mechanical_coupling(world: &mut World, ctx: TensionCtx, seed: &WorldSeed) {
       let heat = world.resource::<SignalResolved<CoreHeat>>();
       
       if heat.value > VIGOR_THRESHOLD {
           // Tension detected: core is hot, boost vigor
           let boost = (heat.value - VIGOR_THRESHOLD) * COUPLING_FACTOR;
           world.resource_mut::<SignalInputs<MantleVigor>>()
               .push(VigorDelta { boost });
       }
   }
   ```

4. Detector writes `SignalInputs<MantleVigor>` with vigor boost

### Tick N+1: Resolution

5. `MantleVigor` resolver receives the boost input:
   ```rust
   fn resolve(&mut self, prev: &Vigor, inputs: &[VigorDelta], ctx: ResolveCtx, seed: &WorldSeed) -> Vigor {
       let total_boost: f64 = inputs.iter().map(|d| d.boost).sum();
       Vigor(prev.0 + total_boost * ctx.dt_myr())
   }
   ```

6. `MantleVigor` now has higher value

### Tick N+1: Fracture

7. `derive_plate_motion` detector reads higher vigor:
   ```rust
   fn derive_plate_motion(world: &mut World, ctx: TensionCtx, seed: &WorldSeed) {
       let vigor = world.resource::<SignalResolved<MantleVigor>>();
       
       // Higher vigor → faster plate motion
       let motion_scale = vigor.value.0.powf(0.7);
       world.resource_mut::<SignalInputs<PlateMotion>>()
           .push(MotionDelta { scale: motion_scale });
   }
   ```

### Result

Heat → Vigor → Plate Motion: causality flows through signals and fracture.
No authoring required; the cascade emerges from tension detection.

---

## 11. Determinism Guarantees

Continuum guarantees **perfect reproducibility** given identical inputs.

### 11.1 Ordering Guarantees

| Structure | Guarantee |
|-----------|-----------|
| Strata iteration | `BTreeMap<StratumId, _>` → sorted by `StratumId` |
| Signal compute graph | `BTreeMap<SignalId, _>` → sorted by `SignalId` |
| Within-level signals | Sorted by `SignalId.raw()` (u64) |
| Couplers | Sorted by `name` (string) |
| Tension detectors | Sorted by `detector_id` (u64) |

### 11.2 Prohibited Patterns

- `HashMap` for ordering-sensitive iteration
- `thread_rng()` or `rand::random()`
- `DefaultHasher` (platform-dependent)
- `Instant::now()` or wall-clock time
- Floating-point comparison without epsilon

### 11.3 Replay Contract

Identical `{world_seed, dt_seconds, kernel_version, domain_set}` →
Identical simulation outcomes.

---

## 12. Key Files Reference

| File | Purpose |
|------|---------|
| `crates/kernels/foundation/src/signal.rs` | `SignalSpec`, `SignalId`, dependencies |
| `crates/kernels/foundation/src/signal_runtime.rs` | `SignalInputs`, `SignalResolved`, `SignalResolver` |
| `crates/kernels/foundation/src/phases.rs` | `KernelPhase` enum |
| `crates/kernels/orchestration/src/compute_graph.rs` | `SignalComputeGraph`, DAG construction |
| `crates/kernels/orchestration/src/bevy_host.rs` | `run_kernel_ticks` main loop |
| `crates/kernels/fracture/src/lib.rs` | `FracturePlan`, `run_fracture` |
| `crates/kernels/foundation/src/impulse.rs` | `ImpulseSpec`, `ImpulseBus` |

---

## See Also

- @./signal-field-model.md — Signal/field conceptual model
- @./kernel-architecture.md — Kernel split and crate responsibilities
- @./foundation-primer.md — Foundation primitives reference
