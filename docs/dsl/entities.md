# Continuum DSL — Entities and Member Signals

This document defines **entities** as pure index spaces and **member signals**
as per-entity authoritative state.

---

## 1. The Problem

Many simulations need variable-count collections:
- Multiple moons (1-20 depending on scenario)
- Binary or single star systems
- N tectonic plates
- Atmospheric layers

Each element needs its own state, and different aspects of that state may
evolve at different rates. The DSL must support:
- Variable instance counts
- Per-entity state
- Multi-rate scheduling (different strata for different properties)

---

## 2. The Solution: Entities + Member Signals

The solution separates **identity** from **state**:

### Entities as Pure Index Spaces

An `entity` defines *what exists* — a named collection of instances that can
be indexed and iterated. Entities are pure identity providers:

```cdsl
entity stellar.moon {
    : count(config.stellar.moon_count)
    : count(1..20)  // validation bounds
}
```

### Member Signals as Per-Entity State

A `member` signal defines *what state each instance has*. Each member signal
is a top-level primitive with its own resolve expression and stratum:

```cdsl
member stellar.moon.mass {
    : Scalar<kg>
    : strata(stellar.orbital)
    resolve { prev }
}

member stellar.moon.orbit_radius {
    : Scalar<m>
    : strata(stellar.orbital)
    resolve { prev }
}

member stellar.moon.surface_temp {
    : Scalar<K>
    : strata(stellar.thermal)  // Different stratum!
    resolve {
        fn.equilibrium_temperature(
            signal.stellar.flux_at(self.orbit_radius),
            self.albedo
        )
    }
}
```

This separation enables **multi-rate scheduling** — surface temperature can
update on a thermal stratum while orbital mechanics use an orbital stratum.

---

## 3. Entity Declaration

Entities define identity and instance count:

```cdsl
entity stellar.moon {
    : count(config.stellar.moon_count)  // count from config
    : count(1..20)                      // validation bounds
}

entity terra.plate {
    : count(5..50)  // bounded count
}

entity stellar.star {}  // no constraints
```

### Entity Attributes

| Attribute | Description |
|-----------|-------------|
| `: count(config.path)` | Instance count from scenario configuration |
| `: count(min..max)` | Count validation bounds |

Entities do **not** have:
- Strata (member signals have their own strata)
- Schema blocks (state defined via member signals)
- Resolve blocks (no behavior, just identity)
- Config blocks (per-instance config via scenario)

---

## 4. Member Signal Declaration

Member signals define per-entity authoritative state:

```cdsl
member stellar.moon.mass {
    : Scalar<kg, 1e18..1e24>
    : strata(stellar.orbital)

    resolve { prev }
}

member stellar.moon.orbit_phase {
    : Scalar<rad, 0..TAU>
    : strata(stellar.orbital)

    resolve {
        dt.advance_phase(prev, self.orbit_velocity)
    }
}
```

### Member Signal Attributes

| Attribute | Description |
|-----------|-------------|
| `: Type<unit, range>` | Value type with constraints |
| `: strata(path)` | Stratum binding (required) |
| `: title("...")` | Human-readable name |
| `: symbol("...")` | Display symbol |

### Self Reference

Inside a member signal resolve block, `self.X` reads other member signals
of the same entity instance:

```cdsl
member stellar.moon.velocity {
    : Scalar<m/s>
    : strata(stellar.orbital)

    resolve {
        fn.orbital_velocity(signal.terra.mass, self.orbit_radius)
    }
}
```

---

## 5. Snapshot/Next-State Semantics

Member signals use **snapshot semantics** to enable parallel execution:

### The Rule

- All `self.X` **reads** see the **snapshot** (previous tick values)
- All writes go to the **next-state** buffer (current tick)

### Why This Matters

This separation enables **full parallelism** across all member signal resolvers:

```cdsl
member stellar.moon.velocity {
    : Vec3<m/s>
    : strata(stellar.orbital)

    resolve {
        dt.integrate(prev, acceleration)  // prev = previous tick velocity
    }
}

member stellar.moon.position {
    : Vec3<m>
    : strata(stellar.orbital)

    resolve {
        // self.velocity reads PREVIOUS tick velocity, not just-computed!
        dt.integrate(prev, self.velocity)
    }
}
```

Without snapshot semantics, position would use the just-computed velocity,
making results depend on execution order.

### Execution Model

1. **Tick Start**: All member signals snapshot their current values
2. **Resolve Phase**: All resolvers run in parallel, reading snapshots
3. **Tick End**: Next-state becomes current, ready for next tick

---

## 6. Scenario Configuration

Instance data is provided via scenario configuration:

### YAML Format

```yaml
entities:
  stellar.moon:
    - name: luna
      mass: 7.34e22 kg
      radius: 1.737e6 m
      orbit_radius: 3.844e8 m
      orbit_phase: 0 rad
    - name: phobos
      mass: 1.0659e16 kg
      radius: 11.267e3 m
      orbit_radius: 9.376e6 m
      orbit_phase: 1.2 rad
```

The scenario defines:
- How many instances exist
- Initial values for each member signal

---

## 7. Accessing Entities

### Aggregate Operations

No manual iteration — use built-in aggregators:

```cdsl
signal terra.tidal.total_force {
    : Vec3<N>
    : strata(terra.orbital)

    resolve {
        agg.sum(entity.stellar.moon,
            fn.tidal_force(self.mass, self.orbit_radius, self.orbit_phase)
        )
    }
}

signal stellar.moon_count {
    : Scalar<1, 0..20>
    : strata(stellar.orbital)

    resolve {
        agg.count(entity.stellar.moon)
    }
}
```

### Index Access

```cdsl
signal terra.primary_moon.distance {
    : Scalar<m>
    : strata(terra.orbital)

    resolve {
        entity.stellar.moon[0].orbit_radius
    }
}
```

### Named Access

```cdsl
signal terra.luna.phase {
    : Scalar<rad>
    : strata(terra.orbital)

    resolve {
        entity.stellar.moon["luna"].orbit_phase
    }
}
```

---

## 8. Aggregate Operations Reference

### Reduction

| Operation | Description | Example |
|-----------|-------------|---------|
| `agg.sum(entity, expr)` | Sum over all instances | `agg.sum(entity.moon, self.mass)` |
| `agg.product(entity, expr)` | Product over all | `agg.product(entity.layer, self.transmittance)` |
| `agg.max(entity, expr)` | Maximum value | `agg.max(entity.star, self.luminosity)` |
| `agg.min(entity, expr)` | Minimum value | `agg.min(entity.moon, self.orbit_radius)` |
| `agg.mean(entity, expr)` | Average | `agg.mean(entity.plate, self.age)` |
| `agg.count(entity)` | Number of instances | `agg.count(entity.moon)` |

### Predicates

| Operation | Description | Example |
|-----------|-------------|---------|
| `agg.any(entity, pred)` | Any matches | `agg.any(entity.moon, self.mass > 1e22)` |
| `agg.all(entity, pred)` | All match | `agg.all(entity.star, self.luminosity > 0)` |
| `agg.none(entity, pred)` | None match | `agg.none(entity.plate, self.age < 0)` |

### Filtering

| Operation | Description | Example |
|-----------|-------------|---------|
| `filter(entity, pred)` | Subset for nested ops | `agg.sum(filter(entity.moon, self.mass > 1e20), self.mass)` |
| `first(entity, pred)` | First matching | `first(entity.plate, self.type == Continental)` |

### Spatial

| Operation | Description | Example |
|-----------|-------------|---------|
| `nearest(entity, pos)` | Closest to position | `nearest(entity.plate, position).velocity` |
| `within(entity, pos, radius)` | All within radius | `agg.sum(within(entity.moon, pos, 1e9), self.mass)` |

---

## 9. Entity Interactions

### Self-Exclusion with `other`

For N-body interactions, exclude self:

```cdsl
member stellar.moon.perturbation {
    : Vec3<m/s²>
    : strata(stellar.orbital)

    resolve {
        sum(other(entity.stellar.moon),
            fn.gravitational_acceleration(other.mass, distance(self.position, other.position))
        )
    }
}
```

### Pairwise Operations with `pairs`

For symmetric interactions:

```cdsl
operator stellar.orbital.gravity {
    : strata(stellar.orbital)
    : phase(collect)

    collect {
        // pairs() generates all unique (i,j) combinations, i < j
        for (a, b) in pairs(entity.stellar.body) {
            let r = distance(a.position, b.position)
            let force = fn.gravitational_force(a.mass, b.mass, r)
            let direction = normalize(b.position - a.position)

            a.acceleration <- force / a.mass * direction
            b.acceleration <- -force / b.mass * direction
        }
    }
}
```

---

## 10. Complete Example

```cdsl
// Entity as pure index space
entity stellar.moon {
    : count(config.stellar.moon_count)
    : count(0..20)
}

// Member signals define per-entity state
member stellar.moon.mass {
    : Scalar<kg, 1e15..1e24>
    : strata(stellar.orbital)
    resolve { prev }
}

member stellar.moon.position {
    : Vec3<m>
    : strata(stellar.orbital)

    resolve {
        dt.integrate(prev, self.velocity)
    }
}

member stellar.moon.velocity {
    : Vec3<m/s>
    : strata(stellar.orbital)

    resolve {
        // Gravity from planet
        let planet_gravity = fn.gravitational_acceleration(
            signal.terra.mass,
            magnitude(self.position)
        ) * -normalize(self.position) in

        // Perturbations from other moons
        let perturbations = sum(other(entity.stellar.moon),
            fn.gravitational_acceleration(other.mass, distance(self.position, other.position))
            * normalize(other.position - self.position)
        ) in

        dt.integrate(prev, planet_gravity + perturbations)
    }
}

member stellar.moon.surface_temp {
    : Scalar<K>
    : strata(stellar.thermal)  // Different stratum for thermal!

    resolve {
        fn.equilibrium_temperature(
            signal.stellar.flux_at(self.position),
            self.albedo
        )
    }
}

// Derived signal using entity
signal terra.tidal.amplitude {
    : Scalar<m, 0..100>
    : strata(terra.orbital)

    resolve {
        agg.sum(entity.stellar.moon,
            fn.tidal_amplitude(self.mass, magnitude(self.position), signal.terra.radius)
        )
    }
}
```

---

## 11. Why This Design?

### Separation of Concerns

| Concept | Defines | Has Stratum? |
|---------|---------|--------------|
| Entity | What exists (identity) | No |
| Member Signal | What state it has | Yes |

### Multi-Rate Scheduling

Different member signals can have different strata:

```cdsl
member terra.plate.position { : strata(terra.tectonics) }  // slow
member terra.plate.temperature { : strata(terra.thermal) }  // medium
member terra.plate.surface_stress { : strata(terra.seismic) }  // fast
```

### Clean DAG Construction

Each member signal becomes a DAG node in its stratum's resolve phase.
Dependencies are inferred from self/signal reads.

---

## Summary

| Concept | Purpose |
|---------|---------|
| `entity.path { }` | Define identity/collection |
| `member.entity.field { }` | Define per-entity state |
| `: strata(path)` | Member stratum binding |
| `self.X` | Read member of same instance |
| `other(entity)` | All instances except self |
| `pairs(entity)` | All unique pairs |
| Aggregations | `sum/max/min/mean/count` |
| Index/named access | `entity[i]` / `entity["name"]` |

Key principles:
- Entities are pure index spaces (no behavior)
- Member signals are top-level primitives with own strata
- Multi-rate scheduling via different strata
- Snapshot semantics for parallel execution
- Aggregation is explicit, iteration is implicit
