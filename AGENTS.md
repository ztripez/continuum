# CONTINUUM — Worlds, DSL, and Execution Graph

> **Systems evolve. Structure is a consequence.**  
> **Structure emerges from tension.**

Continuum models **causality**, not authored outcomes.

Nothing is simulated by handwritten engine logic.  
Nothing is resolved externally.  
All behavior emerges from **declared signals, operators, constraints, and coupling**
compiled into a **deterministic execution graph**.

---

## What This Repository Is

This repository defines:

- **Continuum Worlds** — a minimal manifest plus DSL describing *what exists* and *how time advances*
- **Continuum DSL** — a declarative simulation language
- A **compiler** that turns DSL into a **typed IR**
- A **graph builder** that derives a **deterministic execution DAG**
- A **runtime** that executes that DAG with safe parallelism

There is **no manually authored simulation code**.
There are **no Rust macros defining simulation behavior**.
All structure comes from the DSL and the graph it produces.

---

## Core Invariants (Authoritative)

These rules must remain true regardless of implementation details.

### Determinism
- All ordering is explicit and stable:
  - file discovery
  - symbol registration
  - IR construction
  - DAG construction
  - execution scheduling
- Randomness is allowed only via explicit seeded derivation.
- A run is replayable from:
```

{ world, scenario, seed, era sequence, dt sequence }

```

### Signals Are Authority
- Authoritative simulation state is expressed as **resolved signals**.
- If something influences causality, it must be a signal.
- Operators may read only resolved signals and authoritative state.

### Fields Are Observation
- Fields are derived measurements, not state.
- Fields exist only for observation (Measure phase).
- Kernel execution must never depend on fields.

### Observer Boundary Is Absolute
- Observers may be removed entirely without changing outcomes.
- Lens, chronicles, and analysis are strictly non-causal.

### Fail Loudly
- No hidden clamps.
- No silent correction.
- Impossible or runaway states are detected via assertions and surfaced as faults.
Got it. Here’s a **hard-cut, agent-friendly version** — no fluff, no philosophy.

---

## ❌ HARD CODING IS FORBIDDEN

**If behavior is encoded in code shape instead of data or abstraction, reject it.**

### Rules

1. **No large `match` / `if-else`**

   * Especially over enums
   * If adding a variant requires editing a `match`, it’s wrong
   * Use traits, tables, or derived order instead

2. **No duplicated structs**

   * If structs differ only by defaults or constants → use one struct + trait/defaults

3. **Enums are not polymorphism**

   * Enums may represent state
   * Behavior must live behind traits, not `match`

4. **No implicit ordering**

   * Never rely on enum order
   * Ordering must be explicit (arrays, metadata, config)

5. **No baked-in knowledge**

   * Code must not “know” domain rules
   * Domain rules must be declared, not hard-coded

6. **If it repeats, generate it**

   * No copy-paste variants
   * Use generics, traits, or codegen

### Auto-reject if:

* Adding a case touches multiple files
* Behavior is expressed via `match`
* Structure encodes policy
* Ordering is implicit

**One-liner for agents:**

> *If logic lives in syntax instead of data or dispatch, it’s wrong.*

If you want it even shorter (like a single paragraph or lint comments), say so.


---

## World → Scenario → Run

Continuum execution is structured as:

```

World → Scenario → Run

```

- **World** defines causal structure and execution policy
- **Scenario** defines initial conditions and parameters
- **Run** executes the world under a scenario with a seed

These roles must never be blurred.

---

## What a World Is

A **World** is a directory containing:

- `world` definition (DSL or `world.yaml`) — execution policy only
- `*.cdsl` — authoritative simulation declarations

### World loading (fixed rule)

- World root = directory containing `world.yaml` OR `world` DSL definition
- Load all `*.yaml` under root (recursive), sorted by path; merge
- Load all `*.cdsl` under root (recursive), sorted by path; compile

This behavior is **not configurable**.

See: `world.md`

---

## Manifest (DSL or `world.yaml`)

The manifest defines **how execution runs**, not simulation logic.

It may define:
- derived time units
- fault and determinism policy
- engine feature flags

It must not define:
- signals
- operators
- fields
- impulses
- strata (defined in DSL)
- eras (defined in DSL)
- observer logic

Logic belongs exclusively in the DSL.

---

## The DSL

The Continuum DSL exists to make simulation logic:

- declarative
- analyzable
- schedulable
- deterministic
- observer-safe

The DSL is **compiled**, not interpreted.

It produces a **typed IR** which is the sole input to execution graph construction.

---

## What the DSL Defines (Authoritative)

All simulation primitives are declared in `*.cdsl`:

- **Signals** — authoritative resolved values
- **Entities** — pure index spaces (identity only)
- **Members** — per-entity authoritative state with own strata
- **Operators** — phase-tagged execution blocks
- **Fields** — observer data + reconstruction hints
- **Impulses** — external causal inputs
- **Fractures** — emergent tension detectors
- **Chronicles** — observer-only interpretation rules
- **Strata** — execution lanes and cadence
- **Eras** — time phases and execution policy

Dependencies are **inferred**, never manually declared.

---

## Execution Model

Execution graphs are built per:

```

(phase × stratum × era)

```

Each graph is a DAG executed as:
- stable topological levels
- full barriers between levels
- parallel execution within a level where safe

### Phases (fixed semantics)

1. **Configure** — execution context, freezing derived data
2. **Collect** — accumulate signal inputs and impulses
3. **Resolve** — resolve authoritative signals
4. **Fracture** — detect and respond to tension
5. **Measure** — emit fields and observer artifacts

Kernel phases must never access fields.

---

## Kernel Compute

The DSL may call namespaced kernel operations (e.g. `maths.*`, `vector.*`, `dt.*`, `physics.*`).

- These are engine-provided primitives
- Implemented via CPU or GPU backends
- Backend choice is automatic and policy-driven
- DSL never specifies shaders, bindings, or dispatch

Only **strict-deterministic** kernels may influence causality.

---

## Assertions and Faults

Signals and operators may declare assertions.

- Assertions validate invariants
- They never mutate values
- Failures emit structured faults
- Policy determines response (fatal, halt, continue)

There are no implicit safety nets.

---

## Documentation Index (Explanatory)

These documents explain the system.  
They do not override the invariants above.

### Philosophy & Principles
- `@docs/manifesto.md`
- `@docs/principles.md`

### World, Scenario, Time
- `@docs/world.md`
- `@docs/scenario.md`
- `@docs/time.md`
- `@docs/strata.md`
- `@docs/eras.md`

### Core Concepts
- `@docs/signals.md`
- `@docs/observers/fields.md`
- `@docs/impulses.md`
- `@docs/fractures.md`
- `@docs/observers/chronicles.md`

### DSL
- `@docs/dsl/language.md`
- `@docs/dsl/syntax.md`
- `@docs/dsl/types-and-units.md`
- `@docs/dsl/dt-robust.md`
- `@docs/dsl/functions.md`
- `@docs/dsl/entities.md`
- `@docs/dsl/assertions.md`

### Execution
- `@docs/execution/lifecycle.md`
- `@docs/execution/phases.md`
- `@docs/execution/ir.md`
- `@docs/execution/dag.md`
- `@docs/execution/dag-construction.md`
- `@docs/execution/tick-execution.md`
- `@docs/execution/kernels.md`
- `@docs/execution/determinism.md`

### Observers
- `@docs/observers/lens.md`

### Tooling
- `tools/run.md`
- `tools/analyze.md`

---

## Contribution Rules

When adding or changing features:

1. Preserve determinism.
2. Keep YAML policy-only.
3. Put structure in DSL, not Rust.
4. Enforce phase and observer boundaries at compile time.
5. Never hide instability — assert and surface it.

When in doubt:

> **Make it explicit, typed, schedulable, and observable.**
