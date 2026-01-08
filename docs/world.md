# Worlds

This document defines what a **World** is in Continuum.

A World defines **causal structure**.
It specifies *what exists* and *how time advances*, but not how the system is instantiated or run.

---

## 1. What a World Is

A **World** is a self-contained specification of a causal system.

It consists of:
- an execution manifest (`world.yaml`)
- a set of DSL declarations (`*.cdsl`)

A World defines:
- signals, operators, and their relationships
- time strata, eras, and execution policy
- the complete causal structure of the system

A World is:
- deterministic
- closed under causality
- independent of instantiation

A World is **not** a scenario, a run, or a content bundle.
It defines *what can happen*, not *how it starts*.

---

## 2. World → Scenario → Run

Continuum execution is structured as:

```

World → Scenario → Run

```

- **World** defines causal structure
- **Scenario** defines initial conditions and parameterization
- **Run** executes the World under a Scenario with a seed

This separation is fundamental and must not be blurred.

---

## 3. World Root and Loading Rules

A World is loaded from a directory.

### Fixed rules (not configurable)

1. **Find root**: The World root is the directory containing `world.yaml`
2. **Load YAML**: Glob all `*.yaml` under root (recursive), sort by path, merge (last wins)
3. **Load DSL**: Glob all `*.cdsl` under root (recursive), sort by path, compile as single unit

There are:
- no include directives
- no glob configuration
- no conditional loading

World structure is defined by the filesystem.

### Example structures

Organized by domain:
```
terra/
  world.yaml
  config/
    physics.yaml
    defaults.yaml
  domains/
    geophysics/
      signals.cdsl
      thermal.cdsl
    atmosphere/
      signals.cdsl
      chemistry.cdsl
  strata.cdsl
  eras.cdsl
```

Or flat:
```
terra/
  world.yaml
  terra.cdsl
```

Both are valid. The engine only cares about sorted glob results.

---

## 4. The `world.yaml` Manifest

### Required structure

```yaml
apiVersion: continuum/v1
kind: World

metadata:
  name: terra
  title: "Earth Planetary Simulation"
```

| Field | Required | Description |
|-------|----------|-------------|
| `apiVersion` | Yes | Engine compatibility version (`continuum/v1`) |
| `kind` | Yes | Always `World` for world manifests |
| `metadata.name` | Yes | Machine identifier (lowercase, no spaces) |
| `metadata.title` | No | Human-readable name |

### What `world.yaml` defines

`world.yaml` defines **execution policy**, not simulation logic.

It may define:
- determinism policy
- fault policy
- engine feature flags

It must not define:
- signals
- operators
- fields
- impulses
- strata (defined in DSL)
- eras (defined in DSL)
- initial conditions
- observer logic

If logic or instantiation appears in `world.yaml`, the model is incorrect.

### Additional YAML files

Other `*.yaml` files under the world root are merged into the manifest.
Use for organizing configuration:

```yaml
# config/physics.yaml
metadata:
  description: "Physics constants override"
```

Files are merged in sorted path order. Later files override earlier ones.

---

## 5. Responsibilities of the DSL

All causal structure is declared in DSL files.

The DSL defines:
- signals (authoritative state)
- operators (phase-tagged execution)
- fields (observer data)
- impulses (external inputs)
- lens and chronicle rules (observer-only)

The DSL is the **single source of truth** for what exists in the World.

---

## 6. World Identity

A World has a stable identity defined by:
- its merged `world.yaml`
- its compiled DSL IR

Together, these define the **world specification**.

World identity is independent of:
- scenarios
- initial conditions
- execution speed
- number of threads
- backend (CPU/GPU)
- observers

Changing a World always produces a new identity.

---

## 7. Worlds and Scenarios

A World may be instantiated by many **Scenarios**.

Scenarios:
- configure initial signal values
- set declared signal parameters
- select optional features explicitly exposed by the World

Scenarios must not:
- alter causal structure
- introduce new logic
- bypass validation or assertions

If a configuration choice changes causality,
it belongs in the World, not the Scenario.

---

## 8. Worlds Are Closed Systems

A World is causally closed.

- All causal influences must be declared
- External interaction occurs only via impulses
- Impulses are part of the causal model

If an effect enters the system without a declared cause,
the World is invalid.

---

## 9. Worlds and Context

A World defines **context**, not meaning.

- Worlds produce causal history
- Observers interpret that history
- Visualization, narrative, and analytics are external

A World does not:
- label outcomes
- decide significance
- encode interpretation

Meaning is assigned outside the World.

---

## 10. Worlds and Change

Worlds are expected to evolve.

When a World changes:
- its specification changes
- its identity changes
- scenario compatibility may break
- replay compatibility may break

Backward compatibility is not guaranteed.
Correctness takes precedence over continuity.

---

## 11. What a World Is Not

A World is **not**:
- an engine plugin
- a scripting environment
- a runtime configuration profile
- a scenario
- a run
- a visualization preset
- a gameplay definition

It is a **causal specification**.

---

## Summary

- A World defines causal structure
- `world.yaml` defines execution policy only
- DSL defines all simulation structure
- Scenarios instantiate Worlds
- Runs execute Worlds under Scenarios
- Worlds produce causal history, not meaning

For execution semantics, see:
- `@docs/execution/lifecycle.md`
- `@docs/execution/phases.md`
- `@docs/execution/dag.md`
