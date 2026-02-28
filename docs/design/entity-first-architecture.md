# Entity-First Architecture

> **Status**: Design  
> **Motivation**: Entities are currently an afterthought — bolted onto a signal-centric model. They should be the building block.

---

## The Problem

The current architecture has two parallel worlds:

```
World
├── globals: IndexMap<Path, Node<()>>        ← signals, fields, operators
├── members: IndexMap<Path, Node<EntityId>>  ← per-entity state
├── entities: IndexMap<Path, Entity>         ← pure index spaces
```

This causes:

1. **Two storage backends** — `SignalStorage` (globals, key-value) vs `MemberSignalBuffer` (members, SoA). Two double-buffer implementations, two advance-tick paths.
2. **Two executor paths** — global signals go through the DAG executor, members through `member_executor.rs` with a separate L1 chunked strategy.
3. **Two AST node types** — `Node<()>` vs `Node<EntityId>`. Same struct, different generic, forces two code paths everywhere.
4. **Flat entity space** — entities can't contain entities. No structural hierarchy.
5. **One-directional aggregation** — globals read entities via `agg.*`, but there's no natural parent-child composition.

The world itself is a special container that holds everything. But conceptually, the world *is* an entity — the root entity with count=1, whose "members" are what we currently call global signals.

---

## The Design

### Core Principle

**Everything is an entity. The world is the root entity.**

- What are currently "global signals" become members of the root entity (count=1)
- Entities can contain child entities (recursive composition)
- There is one storage system, one executor path, one node type
- Entity hierarchy is structural/organizational — it does not restrict signal access

### World Structure

```
World {
    root: Entity,                          // the world itself
    strata: IndexMap<Path, Stratum>,       // execution lanes (world-level)
    eras: IndexMap<Path, Era>,             // time phases (world-level)
}

Entity {
    id: EntityId,
    path: Path,
    count: Option<CountExpr>,              // None for root (implicit 1)
    members: Vec<Node>,                    // signals, fields, operators, etc.
    children: IndexMap<Path, Entity>,      // sub-entities
    topology: Option<TopologyExpr>,
    attributes: Vec<Attribute>,
}
```

Strata and eras remain at world level — they are execution policy, not structural.

### CDSL Syntax

```cdsl
world terra {
    : dt(1.0<s>)

    strata tectonics { : stride(10) }
    strata thermal   { : stride(1) }

    era main {
        : initial
        : dt(1_000_000.0<yr>)
        strata { tectonics: active, thermal: active }
    }

    // These are members of the root entity (world)
    signal atmosphere.temperature {
        : Scalar<K>
        : strata(thermal)
        resolve { prev + heating_rate * dt }
    }

    signal mantle.heat_flux {
        : Scalar<W/m2>
        : strata(thermal)
        resolve { ... }
    }

    // Child entity — count per world instance (so just count total)
    entity plate {
        : count(config.plates.count)

        signal age {
            : Scalar<Myr>
            : strata(tectonics)
            resolve { dt.integrate(prev, 1.0) }
        }

        signal velocity {
            : Vec3<m/s>
            : strata(tectonics)
            resolve { ... }
        }

        // Nested child entity — count per plate
        entity boundary {
            : count(4)

            signal stress {
                : Scalar<Pa>
                : strata(tectonics)
                resolve { ... }
            }
        }
    }

    entity star {
        : count(1)

        signal luminosity {
            : Scalar<W>
            : strata(thermal)
            resolve { prev }
        }
    }
}
```

### No Scope Restrictions

Any signal can read any other signal regardless of entity hierarchy:

```cdsl
// A plate boundary reading star luminosity — perfectly legal
entity plate {
    entity boundary {
        signal thermal_stress {
            : Scalar<Pa>
            : strata(thermal)
            resolve {
                // Read from star (sibling entity tree), atmosphere (root member)
                let flux = entity.star.at(0).luminosity / (4.0 * PI * self.distance^2) in
                let ambient = signal.atmosphere.temperature in
                fn.thermal_stress(flux, ambient, self.conductivity)
            }
        }
    }
}
```

Entity hierarchy defines **identity and storage layout**, not visibility.

### Flat Instance Indexing

Nested entities use per-parent counts, but storage is flat for cache efficiency:

```
plate (count=12)
  boundary (count=4 per plate → 48 total)
```

Flat index for `plate[p].boundary[b]` = `p * 4 + b`

The runtime computes offsets at build time:

```rust
struct EntityLayout {
    /// Path to this entity
    path: Path,
    /// Total instance count (own count × all ancestor counts)
    total_count: usize,
    /// Instances per parent (the declared count)
    count_per_parent: usize,
    /// Parent entity (None for root)
    parent: Option<EntityId>,
    /// Offset into flat storage
    offset: usize,
}
```

Root entity: total_count=1, offset=0
plate: total_count=12, count_per_parent=12, offset depends on layout
plate.boundary: total_count=48, count_per_parent=4

### Aggregation with Hierarchy

Aggregation respects parent-child relationships:

```cdsl
// Sum over ALL boundaries globally
agg.sum(entity.plate.boundary, self.stress)

// Sum over boundaries of a SPECIFIC plate (inside plate resolve)
agg.sum(self.boundary, self.stress)  // 'self.boundary' scopes to current plate's children
```

### Storage: Unified SoA

One storage system for everything. "Global" signals are members of root entity (instance 0):

```rust
struct UnifiedStorage {
    /// SoA buffers by value type, double-buffered
    scalars: DoubleBuffer<f64>,
    vec2s: DoubleBuffer<[f64; 2]>,
    vec3s: DoubleBuffer<[f64; 3]>,
    vec4s: DoubleBuffer<[f64; 4]>,
    booleans: DoubleBuffer<bool>,
    integers: DoubleBuffer<i64>,

    /// Registry mapping signal paths to buffer positions
    registry: SignalRegistry,

    /// Entity layout for computing flat indices
    layouts: IndexMap<EntityId, EntityLayout>,
}
```

A "global" signal like `atmosphere.temperature` is stored as:
- Entity: root (count=1)
- Instance: 0
- Buffer position: computed from registry

A member signal like `plate[7].age` is stored as:
- Entity: plate (count=12)
- Instance: 7
- Buffer position: computed from registry + instance offset

### Execution: One Path

No more separate global executor and member executor:

1. DAG built per (phase × stratum × era) — unchanged
2. Each node in the DAG resolves for all instances of its entity
3. Root entity members resolve for 1 instance (effectively scalar)
4. Child entity members resolve for N instances (vectorized)
5. Parallelism: instances within a node are parallelized, nodes at the same DAG level run in parallel where independent

The current `member_executor.rs` L1 strategy generalizes to all nodes. The current `SignalStorage` path is just the degenerate case where instance_count=1.

---

## What Changes

### AST

| Before | After |
|--------|-------|
| `World { globals, members, entities }` | `World { root: Entity, strata, eras }` |
| `Declaration::Node(Node<()>)` | Gone — everything is a member |
| `Declaration::Member(Node<EntityId>)` | `Node` with entity path |
| `Node<I: Index>` generic | `Node` with `entity: EntityPath` field |
| Flat `entities: IndexMap` | Recursive `Entity.children` |

### Parser

| Before | After |
|--------|-------|
| `world` parsed as metadata | `world` parsed as root entity |
| `entity` declarations flat | `entity` declarations recursive |
| `signal` at top level → `Node<()>` | `signal` at top level → member of root |
| Separate `member` keyword | `member` still works, but nesting is preferred |

### Resolution Pipeline

| Before | After |
|--------|-------|
| `flatten_entity_members()` as first step | Recursive entity tree walk |
| Separate global/member resolution | Single unified resolution pass |
| Two dependency extraction paths | One path, entity-aware |

### Runtime

| Before | After |
|--------|-------|
| `SignalStorage` + `MemberSignalBuffer` | `UnifiedStorage` |
| `executor` + `member_executor` | Single executor with instance-aware dispatch |
| Entity counts flat | Entity layouts with parent-child indexing |

### Examples

All `.cdsl` files rewritten. Current top-level signals move inside `world { }` block.

---

## Migration: Big Bang

Not incremental. The current code compiles and runs terra successfully. We:

1. Branch
2. Rewrite AST, parser, resolution, runtime, examples
3. Terra must compile and produce identical output
4. Merge

No compatibility layer. No dual paths. Fail hard.

---

## Wave Plan

### Wave 0: AST Foundation
- New `Entity` struct with recursive children
- `World` wraps root entity + strata + eras
- `Node` loses the `I: Index` generic — gets `entity: EntityPath` field
- `Declaration` enum simplifies
- All AST tests updated

### Wave 1: Parser
- `world { }` parses as root entity containing members
- Recursive `entity { }` parsing
- Top-level signals inside `world { }` are members of root
- `member` keyword continues to work as standalone syntax

### Wave 2: Resolution Pipeline
- Recursive entity tree walk replaces `flatten_entity_members()`
- Unified node resolution (no global/member split)
- Entity layout computation (flat indices, parent-child offsets)
- Dependency extraction works uniformly

### Wave 3: Runtime Storage
- `UnifiedStorage` replaces `SignalStorage` + `MemberSignalBuffer`
- `EntityLayout` for flat indexing with hierarchy
- Single double-buffer advance-tick path
- Build pipeline uses entity tree

### Wave 4: Runtime Executor
- Single executor handles all nodes
- Instance-parallel execution for all entity levels
- Root entity members (count=1) are the trivial case
- Aggregation respects parent-child scoping

### Wave 5: Examples & Validation
- Rewrite terra, entity-test, all examples
- Terra produces same simulation results
- All existing tests pass or are updated
- New tests for nested entities

---

## Non-Goals (for now)

- **Dynamic entity lifecycle** (spawn/destroy at runtime) — separate concern
- **Relations** (entity-to-entity references) — revisit after this lands
- **Entity inheritance** — not needed, composition via nesting
- **Per-entity strata/eras** — strata/eras remain world-level
