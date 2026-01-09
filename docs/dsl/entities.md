# Continuum DSL — Entities

This document defines **entities** — dynamic collections of structured state.

---

## 1. The Problem

Many simulations need variable-count collections:
- Multiple moons (1-20 depending on scenario)
- Binary or single star systems
- N tectonic plates
- Atmospheric layers

Each element needs its own state, but the DSL shouldn't require manual iteration.

---

## 2. The Solution: Entity Collections

An `entity` is a **named, indexed collection** of structured state:

- Count is determined by scenario configuration
- Each instance has the same schema
- Iteration is implicit — aggregation is explicit
- The engine handles parallelization

---

## 3. Entity Declaration

```
entity.stellar.moon {
  : strata(stellar.orbital)
  : count(config.stellar.moon_count)
  : count(1..20)  // validation bounds

  schema {
    mass: Scalar<kg, 1e18..1e24>
    radius: Scalar<m, 1e5..1e7>
    orbit_radius: Scalar<m, 1e7..1e10>
    orbit_phase: Scalar<rad, 0..TAU>
    orbit_velocity: Scalar<m/s>
    name: String
  }

  config {
    // Default values, can be overridden by scenario
    orbit_phase: 0 <rad>
  }

  resolve {
    // 'self' refers to current instance
    // This block runs once per moon, engine handles iteration
    self.orbit_phase = advance_phase(self.orbit_phase, self.orbit_velocity)
    self.orbit_velocity = fn.orbital.velocity(signal.terra.mass, self.orbit_radius)
  }
}
```

### Entity Attributes

| Attribute | Description |
|-----------|-------------|
| `: strata(path)` | Stratum binding |
| `: count(config.path)` | Count from config |
| `: count(min..max)` | Count validation bounds |

---

## 4. Scenario Configuration

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

  stellar.star:
    - name: sol_a
      mass: 1.989e30 kg
      luminosity: 3.828e26 W
      position: [0, 0, 0] m
    - name: sol_b
      mass: 1.5e30 kg
      luminosity: 2.1e26 W
      position: [1.5e11, 0, 0] m
```

### DSL Format (for fixed scenarios)

```
scenario.binary_earth {
  entity.stellar.star: [
    { name: "sol_a", mass: 1.989e30 <kg>, luminosity: 3.828e26 <W> },
    { name: "sol_b", mass: 1.5e30 <kg>, luminosity: 2.1e26 <W> }
  ]

  entity.stellar.moon: [
    { name: "luna", mass: 7.34e22 <kg>, orbit_radius: 3.844e8 <m> }
  ]
}
```

---

## 5. Accessing Entities

### Aggregate Operations

No manual iteration — use built-in aggregators:

```
signal.terra.tidal.total_force {
  : Vec3<N>
  : strata(terra.orbital)

  resolve {
    sum(entity.stellar.moon, fn.tidal_force(self.mass, self.orbit_radius, self.orbit_phase))
  }
}

signal.stellar.system.total_luminosity {
  : Scalar<W>
  : strata(stellar.core)

  resolve {
    sum(entity.stellar.star, self.luminosity)
  }
}

signal.terra.plates.average_age {
  : Scalar<Myr>
  : strata(terra.tectonics)

  resolve {
    mean(entity.terra.plate, self.age)
  }
}
```

### Index Access

```
signal.terra.primary_moon.distance {
  : Scalar<m>
  : strata(terra.orbital)

  resolve {
    entity.stellar.moon[0].orbit_radius
  }
}
```

### Named Access

```
signal.terra.luna.phase {
  : Scalar<rad>
  : strata(terra.orbital)

  resolve {
    entity.stellar.moon["luna"].orbit_phase
  }
}
```

---

## 6. Aggregate Operations Reference

### Reduction

| Operation | Description | Example |
|-----------|-------------|---------|
| `sum(entity, expr)` | Sum over all instances | `sum(entity.moon, self.mass)` |
| `product(entity, expr)` | Product over all | `product(entity.layer, self.transmittance)` |
| `max(entity, expr)` | Maximum value | `max(entity.star, self.luminosity)` |
| `min(entity, expr)` | Minimum value | `min(entity.moon, self.orbit_radius)` |
| `mean(entity, expr)` | Average | `mean(entity.plate, self.age)` |
| `count(entity)` | Number of instances | `count(entity.moon)` |

### Predicates

| Operation | Description | Example |
|-----------|-------------|---------|
| `any(entity, pred)` | Any matches | `any(entity.moon, self.mass > 1e22)` |
| `all(entity, pred)` | All match | `all(entity.star, self.luminosity > 0)` |
| `none(entity, pred)` | None match | `none(entity.plate, self.age < 0)` |

### Filtering

| Operation | Description | Example |
|-----------|-------------|---------|
| `filter(entity, pred)` | Subset for nested ops | `sum(filter(entity.moon, self.mass > 1e20), self.mass)` |
| `first(entity, pred)` | First matching | `first(entity.plate, self.type == Continental)` |

### Spatial

| Operation | Description | Example |
|-----------|-------------|---------|
| `nearest(entity, pos)` | Closest to position | `nearest(entity.plate, position).velocity` |
| `within(entity, pos, radius)` | All within radius | `sum(within(entity.moon, pos, 1e9), self.mass)` |

---

## 7. Entity Interactions

### Self-Exclusion with `other`

For N-body interactions, exclude self:

```
entity.stellar.moon {
  ...
  resolve {
    // Gravitational perturbation from other moons
    let perturbation = sum(other(entity.stellar.moon),
      fn.gravitational_acceleration(other.mass, distance(self.position, other.position))
    ) in
    self.velocity = integrate(self.velocity, perturbation + signal.terra.gravity_at(self.position))
  }
}
```

### Pairwise Operations with `pairs`

For symmetric interactions:

```
operator.stellar.orbital.gravity {
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

### Cross-Entity Operations

Interactions between different entity types:

```
signal.terra.moon_shadow_fraction {
  : Scalar<1, 0..1>
  : strata(terra.orbital)

  resolve {
    sum(entity.stellar.moon,
      fn.shadow_cone_intersection(
        self.position,
        self.radius,
        entity.stellar.star[0].position,
        signal.terra.position
      )
    )
  }
}
```

---

## 8. Entity Fields

Entities can emit fields for observation:

```
entity.stellar.moon {
  ...

  field.position {
    : Vec3<m>
    : topology(point_cloud)

    measure {
      self.position
    }
  }

  field.surface_temperature {
    : Scalar<K>
    : topology(point_cloud)

    measure {
      fn.equilibrium_temperature(
        signal.stellar.flux_at(self.position),
        self.albedo
      )
    }
  }
}
```

---

## 9. Dynamic Entity Creation (Fractures)

Entities can spawn via fractures:

```
fracture.terra.tectonics.plate_split {
  when {
    let plate = max(entity.terra.plate, self.area) in
    plate.area > config.terra.max_plate_area
    plate.rift_stress > config.terra.rift_threshold
  }

  emit {
    // Request plate split — engine handles actual creation
    entity.terra.plate.split <- {
      source: plate.id,
      axis: plate.max_stress_axis
    }
  }
}
```

Entity creation/destruction is handled by the engine to maintain determinism.

---

## 10. Entity Ordering

Entity iteration order is **deterministic**:

1. By index (creation order)
2. Stable across ticks
3. New entities appended at end

This ensures:
- Reproducible results
- Safe parallel execution within aggregations
- Predictable `entity[i]` access

---

## 11. Complete Example

```
// Schema
entity.stellar.moon {
  : strata(stellar.orbital)
  : count(config.stellar.moon_count)
  : count(0..20)

  schema {
    name: String
    mass: Scalar<kg, 1e15..1e24>
    radius: Scalar<m, 1e3..1e7>
    position: Vec3<m>
    velocity: Vec3<m/s>
    albedo: Scalar<1, 0..1>
  }

  config {
    albedo: 0.12
  }

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

    self.velocity = integrate(self.velocity, planet_gravity + perturbations)
    self.position = integrate(self.position, self.velocity)
  }
}

// Derived signal using entity
signal.terra.tidal.amplitude {
  : Scalar<m, 0..100>
  : strata(terra.orbital)

  resolve {
    sum(entity.stellar.moon,
      fn.tidal_amplitude(self.mass, magnitude(self.position), signal.terra.radius)
    )
  }
}

// Entity count as signal
signal.stellar.moon_count {
  : Scalar<1, 0..20>
  : strata(stellar.orbital)

  resolve {
    count(entity.stellar.moon)
  }
}
```

---

## Summary

| Concept | Purpose |
|---------|---------|
| `entity.path { }` | Define collection schema |
| `schema { }` | Fields each instance has |
| `self` | Current instance in resolve |
| `other(entity)` | All instances except self |
| `pairs(entity)` | All unique pairs |
| `sum/max/min/mean` | Aggregate operations |
| `entity[i]` | Index access |
| `entity["name"]` | Named access |
| Scenario config | Define instance count and values |

Key principles:
- Iteration is implicit
- Aggregation is explicit
- Count is scenario-controlled
- Order is deterministic
