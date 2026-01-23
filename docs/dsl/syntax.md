# Continuum DSL — Syntax Reference

This document defines the **concrete syntax** of the Continuum DSL.

---

## 1. File Structure

DSL source files use the extension `*.cdsl`.

All files under a World root are:
- discovered recursively
- ordered lexicographically by path
- compiled together as a single unit

There are no includes or conditional loading.

---

## 2. Comments

```
// Single line comment
# Alternative single line comment
/* Multi-line
   comment */
```

---

## 3. Namespaced Identifiers

All entities use namespaced paths with the pattern `[kind].[path]`:

```
signal terra.geophysics.core.temp_k
field terra.surface.temperature_map
strata terra.thermal
era hadean
impulse terra.impact.asteroid
fracture terra.climate.runaway_greenhouse
chronicle terra.events.supercontinent
```

Identifiers are:
- lowercase
- dot-separated
- globally unique within a World

---

## 4. Units and Literals

Literals:

```
42              // Integer
3.14            // Float
true            // Boolean
"hello"         // String
[1.0, 2.0, 3.0] // Vector/Array
```

Units are specified in angle brackets:

```
5.67e-8 <W/m²/K⁴>
100 <K>
1 <Myr>
0.5 <atm>
```

Ranges use `..` syntax:

```
100..10000
0..1
-500..2000
```

---

## 5. World-Level Blocks

### Constants

Immutable universal values:

```
const {
  physics.stefan_boltzmann: 5.67e-8 <W/m²/K⁴>
  physics.gravitational: 6.674e-11 <m³/kg/s²>
  physics.speed_of_light: 299792458 <m/s>
}
```

### Configuration

Scenario-tunable parameters:

```
config {
  terra.thermal.reference_heat: 8e30 <J>
  terra.tectonics.rift_threshold: 1e8 <Pa>
}
```

---

## 6. Types

Custom struct types:

```
type PlateState {
  position: Vec3<m>
  velocity: Vec3<m/s>
  strain: Tensor<3,3,Pa>
  age: Scalar<s>
}

type ImpactEvent {
  mass: Scalar<kg>
  velocity: Vec3<m/s>
  location: Vec2<rad>
}
```

### Built-in Types

#### Scalar Types

| Type | Description | Parameters |
|------|-------------|------------|
| `Scalar<unit>` | Single floating-point value | unit (optional) |
| `Scalar<unit, range>` | Bounded single value | unit, range |

```
: Scalar<K>                  # temperature in Kelvin
: Scalar<m/s>                # velocity magnitude
: Scalar<1>                  # dimensionless ratio
: Scalar<K, 100..10000>      # bounded temperature
: Scalar<1, 0..1>            # normalized fraction
```

#### Vector Types

| Type | Description | Components | Parameters |
|------|-------------|------------|------------|
| `Vec2<unit>` | 2D vector | x, y | unit, magnitude |
| `Vec3<unit>` | 3D vector | x, y, z | unit, magnitude |
| `Vec4<unit>` | 4D vector | x, y, z, w | unit, magnitude |

```
: Vec2<m>                           # 2D position
: Vec3<m/s>                         # 3D velocity
: Vec4<1>                           # homogeneous coordinates
: Vec3<m, magnitude: 1e10..1e12>    # bounded orbital radius
```

Component access:
```
signal.position.x    # x component
signal.velocity.z    # z component
```

#### Quaternion Type

| Type | Description | Components | Parameters |
|------|-------------|------------|------------|
| `Quat` | Unit quaternion for rotations | w, x, y, z | magnitude |

```
: Quat                       # rotation quaternion
: Quat<magnitude: 1>         # enforced unit quaternion
```

Quaternions are stored as Vec4 but with w-first component order (w, x, y, z).
They represent rotations and should typically have magnitude 1.

#### Matrix Types

| Type | Description | Components | Parameters |
|------|-------------|------------|------------|
| `Mat2<unit>` | 2x2 matrix | m00, m10, m01, m11 | unit |
| `Mat3<unit>` | 3x3 matrix | m00..m22 (9 elements) | unit |
| `Mat4<unit>` | 4x4 matrix | m00..m33 (16 elements) | unit |

```
: Mat3<1>                    # rotation matrix
: Mat4<m>                    # transformation matrix
```

Matrices use column-major order for GPU compatibility.
Component naming: `mRC` where R=row, C=column.

#### Tensor Type

| Type | Description | Parameters |
|------|-------------|------------|
| `Tensor<rows, cols, unit>` | General NxM tensor | rows, cols, unit |

```
: Tensor<3,3,Pa>             # 3x3 stress tensor
: Tensor<6,6,Pa>             # elasticity tensor
```

Tensors support structural constraints:
```
: Tensor<3,3,Pa>
  : symmetric                # Tij = Tji
  : positive_definite        # all eigenvalues > 0
```

#### Collection Types

| Type | Description | Parameters |
|------|-------------|------------|
| `Seq<T>` | Ordered sequence | element_type |
| `Grid<W,H,T>` | 2D grid | width, height, element_type |

```
: Seq<Scalar<kg>>            # sequence of masses
: Grid<360,180,Scalar<K>>    # temperature grid
```

### Constraints

Type constraints restrict valid values at compile time and generate runtime assertions.

#### Range Constraints

For scalar types, specify min..max bounds:
```
: Scalar<K, 100..10000>              # temperature 100-10000 K
: Scalar<Pa, 0..1e12>                # pressure 0-1 TPa
: Scalar<1, 0..1>                    # fraction 0-1
: Scalar<1, -1..1>                   # normalized value
```

#### Magnitude Constraints

For vector types, constrain the vector length:
```
: Vec3<m, magnitude: 1e10..1e12>     # orbital radius bounds
: Vec3<m/s, magnitude: 0..3e8>       # sub-light velocity
: Vec4<1, magnitude: 1>              # unit quaternion
: Quat<magnitude: 1>                 # enforced unit rotation
```

#### Structural Constraints

For tensors, enforce mathematical properties:
```
: Tensor<3,3,Pa>
  : symmetric                        # Tij = Tji
  : positive_definite                # all eigenvalues > 0
  : trace_zero                       # sum of diagonal = 0
```

#### Collection Constraints

For sequences, constrain elements and aggregates:
```
: Seq<Scalar<kg>>
  : each(1e20..1e28)                 # each element in range
  : sum(1e25..1e30)                  # total mass bounds
  : count(1..100)                    # element count bounds
```

### Unit Annotations

Units use angle brackets and are normalized to SI base dimensions:

```
# Base units
value: 100.0 <m>
mass: 5.97e24 <kg>
duration: 3600 <s>

# Derived units (automatically decomposed)
force: 9.8 <N>              # = kg·m/s²
pressure: 101325 <Pa>       # = kg/(m·s²)
energy: 4.2e9 <J>           # = kg·m²/s²
power: 1e12 <W>             # = kg·m²/s³

# Compound units
velocity: 30000 <m/s>
density: 5500 <kg/m³>
flux: 1361 <W/m²>

# Equivalent forms (same dimensions)
work_a: 1000 <J>            # M¹·L²·T⁻²
work_b: 1000 <N·m>          # M¹·L²·T⁻² (same)
work_c: 1000 <W·s>          # M¹·L²·T⁻² (same)
```

See `types-and-units.md` for dimensional algebra rules.

---

## 7. Strata

Time strata define execution groupings:

```
strata terra.thermal {
  : title("Thermal")
  : symbol("Q")
  : stride(5)
}

strata terra.tectonics {
  : title("Tectonics")
  : symbol("T")
  : stride(10)
}
```

---

## 8. Eras

Execution policy regimes:

```
era hadean {
  : initial
  : title("Hadean")
  : dt(1 <Myr>)

  config {
    terra.tectonics.viscosity_factor: 0.5
  }

  strata {
    terra.genesis: active
    terra.thermal: active
    terra.tectonics: gated
    terra.atmosphere: gated
  }

  transition {
    to: era.archean
    when {
      signal.terra.geophysics.mantle.heat_j < 9e30 <J>
      signal.time.planet_age > 500 <Myr>
    }
  }
}

era phanerozoic {
  : terminal
  : title("Phanerozoic")
  : dt(10 <kyr>)

  strata {
    terra.thermal: active
    terra.tectonics: active
    terra.atmosphere: active(stride: 2)
  }
}
```

### Era Attributes

| Attribute | Description |
|-----------|-------------|
| `: initial` | Starting era |
| `: terminal` | No outgoing transitions |
| `: dt(value)` | Base timestep |
| `: title("...")` | Human-readable name |

### Strata States

| State | Description |
|-------|-------------|
| `active` | Executes every tick |
| `active(stride: N)` | Executes every Nth tick |
| `gated` | Paused, state preserved |

### Transitions

Multiple transitions allowed; first match wins:

```
transition {
  to: era.target
  when {
    signal.path > value
    signal.other < value
  }
}
```

---

## 9. Signals

Authoritative resolved values:

```
signal terra.geophysics.core.temp_k {
  : Scalar<K, 100..10000>
  : strata(terra.thermal)
  : title("Core Temperature")
  : symbol("T_core")

  const {
    radiative_factor: 0.85
  }

  config {
    initial_temp: 5500 <K>
  }

  resolve {
    let radiation = const.physics.stefan_boltzmann * (prev ^ 4) * const.radiative_factor in
    prev - radiation * config.decay_rate * dt
  }
}
```

### Signal Attributes

| Attribute | Description |
|-----------|-------------|
| `: Type<...>` | Value type with constraints |
| `: strata(path)` | Stratum binding |
| `: title("...")` | Human-readable name |
| `: symbol("...")` | Display symbol |

### Resolve Block

- `prev` — previous resolved value
- `prev.field` — field access for struct types
- `signal.path` — read other resolved signals
- `const.path` — read constants
- `config.path` — read configuration
- `dt.raw` — timestep (prefer dt-robust operators, see @dsl/dt-robust.md)
- `inputs` — accumulated inputs from Collect phase
- `namespace.fn(...)` — engine-provided functions (e.g. `maths.*`, `vector.*`, `dt.*`, `physics.*`)
- `let name = expr in body` — local bindings (ML-style)

### Warmup Block

Signals may declare a warmup block for pre-causal equilibration:

```
signal terra.thermal.equilibrium {
  : Scalar<K>
  : strata(terra.thermal)

  warmup {
    : iterations(100)
    : convergence(1e-6)

    iterate {
      let flux_in = physics.radiogenic_heat(config.terra.core.budget) in
      let flux_out = physics.surface_radiation(prev) in
      prev + (flux_in - flux_out) * 0.1
    }
  }

  resolve {
    prev + inputs * dt
  }
}
```

Warmup attributes:

| Attribute | Description |
|-----------|-------------|
| `: iterations(N)` | Maximum warmup iterations (required) |
| `: convergence(epsilon)` | Convergence threshold (optional) |

In warmup blocks:
- `prev` — current warmup value
- `dt` — **not available** (time has not started)
- `signal.path` — read other warmup signals (current iteration if resolved, else previous)

See @execution/warmup.md for full semantics.

---

## 10. Entities

Entities are pure index spaces that define what instances exist:

```
entity stellar.moon {
    : count(config.stellar.moon_count)
    : count(1..20)
}

entity terra.plate {
    : count(5..50)
}

entity stellar.star {}
```

### Entity Attributes

| Attribute | Description |
|-----------|-------------|
| `: count(config.path)` | Instance count from config |
| `: count(min..max)` | Count validation bounds |

Entities do not have strata, schema, or resolve blocks.
Per-entity state is defined via member signals.

---

## 11. Member Signals

Per-entity authoritative state with own strata:

```
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

member stellar.moon.surface_temp {
    : Scalar<K>
    : strata(stellar.thermal)

    resolve {
        fn.equilibrium_temperature(signal.stellar.flux_at(self.position), self.albedo)
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

- `self.X` — read other member signals of same entity instance
- `prev` — previous tick value of this member signal

Different member signals can have different strata for multi-rate scheduling.

(See `entities.md` for full semantics.)

---

## 12. Fields

Observable derived data:

```
field terra.surface.temperature_map {
  : Grid<2048, 1024, Scalar<K>>
  : strata(terra.atmosphere)
  : topology(sphere_surface)
  : title("Surface Temperature")
  : symbol("T_s")

  measure {
    physics.surface_projection(
      signal.terra.atmosphere.temp_profile,
      signal.terra.geophysics.surface.temp
    )
  }
}
```

### Topology Options

| Topology | Description |
|----------|-------------|
| `sphere_surface` | Data on planetary surface |
| `point_cloud` | Sparse spatial samples |
| `volume` | 3D volumetric data |

---

## 13. Operators

Phase-tagged logic blocks:

```
operator terra.thermal.budget {
  : strata(terra.thermal)
  : phase(collect)

  collect {
    let radiogenic = physics.radiogenic_power(config.terra.thermal.base, signal.time.age) in
    let loss = physics.surface_heat_loss(signal.terra.geophysics.mantle.heat_j) in

    signal.terra.geophysics.mantle.heat_j <- radiogenic - loss
  }
}

operator terra.tectonics.boundary_capture {
  : strata(terra.tectonics)
  : phase(measure)

  measure {
    let segments = signal.terra.tectonics.plate_boundaries in

    for seg in segments {
      field.terra.plates.boundary_type <- seg.position, seg.kind
      field.terra.plates.boundary_shear <- seg.position, seg.shear
    }
  }
}
```

### Phases

| Phase | Purpose | Can Write |
|-------|---------|-----------|
| `warmup` | Pre-causal initialization | `signal.x <- value` |
| `collect` | Gather signal inputs | `signal.x <- value` |
| `measure` | Emit field samples | `field.x <- position, value` |

Warmup operators execute during the warmup phase before causal execution begins.
See @execution/warmup.md for warmup semantics.

---

## 14. Impulses

External causal inputs:

```
impulse terra.impact.asteroid {
  : ImpactEvent

  config {
    max_energy: 1e25 <J>
  }

  apply {
    let crater = physics.impact_physics(payload.mass, payload.velocity) in
    signal.terra.geophysics.surface.energy <- crater.thermal_energy
    signal.terra.atmosphere.dust <- crater.ejecta_mass
  }
}
```

- `payload` — the impulse data (typed per declaration)
- Writes to signals via `<-`

---

## 15. Fractures

Emergent tension detectors:

```
fracture terra.climate.runaway_greenhouse {
  when {
    signal.terra.atmosphere.co2 > 1000 <ppm>
    signal.terra.surface.avg_temp > 350 <K>
  }

  emit {
    signal.terra.atmosphere.feedback <- 1.5
  }
}

fracture terra.tectonics.subduction {
  when {
    let age_diff = signal.terra.tectonics.plate_a.age - signal.terra.tectonics.plate_b.age in
    age_diff > 50e6 <s>
    signal.terra.tectonics.boundary.stress > config.subduction_threshold
  }

  emit {
    signal.terra.tectonics.boundary.mode <- SubductionMode.Active
  }
}
```

---

## 16. Chronicles

Observer-only pattern recognition:

```
chronicle terra.events.supercontinent {
  observe {
    when signal.terra.tectonics.continental_fraction > 0.8 {
      emit event.supercontinent_formed {
        age: signal.time.planet_age
        area: signal.terra.tectonics.continental_area
      }
    }
  }
}

chronicle terra.events.mass_extinction {
  observe {
    when signal.terra.biosphere.diversity.delta < -0.5 {
      emit event.mass_extinction {
        severity: -signal.terra.biosphere.diversity.delta
        cause: infer_cause(signal.terra.atmosphere, signal.terra.surface)
      }
    }
  }
}
```

Chronicles:
- Execute in Measure phase
- Cannot influence causality
- Emit structured events for analysis

---

## 17. Expression Syntax

### Arithmetic

```
a + b
a - b
a * b
a / b
a ^ b      // power
-a         // negation
```

### Comparisons

```
a > b
a < b
a >= b
a <= b
a == b
a != b
```

### Logic

```
a and b
a or b
not a
```

### Locals

```
let name = expr in body
```

The `in` keyword separates the value expression from the body where the binding is used.
Multiple let bindings chain naturally: `let a = 1 in let b = 2 in a + b`

### Conditionals

```
if condition { expr } else { expr }
```

### Iteration (deterministic only)

```
for item in sequence { ... }
sum(sequence)
map(sequence, fn)
fold(sequence, init, fn)
```

---

## 18. Vector Field Access

Vector types (`Vec2`, `Vec3`, `Vec4`) support component access using dot notation:

### Supported Patterns

```
prev.x              // Component of previous value
prev.y
prev.z
prev.w

inputs.x         // Component of accumulated inputs
inputs.y

signal path.x       // Component of another signal
signal path.y
signal path.z
```

These patterns work because `prev`, `inputs`, and `signal.path` are statically known
to refer to resolved storage locations.

### Unsupported Patterns

Field access on computed expressions is **not supported**:

```
// NOT SUPPORTED - accessing component of a local binding
let v = vec3(1.0, 2.0, 3.0) in v.x    // Error!

// NOT SUPPORTED - accessing component of a function result
normalize(velocity).x                   // Error!

// NOT SUPPORTED - chained field access
other_struct.position.x                 // Error!
```

### Workarounds

For patterns not directly supported, restructure to access components at the source:

```
// Instead of: let v = vec3(a, b, c) in v.x
// Use: extract the component directly
a

// Instead of: normalize(velocity).x
// Use: compute inline or use separate signals
let vx = velocity.x in
let vy = velocity.y in
let vz = velocity.z in
let mag = sqrt(vx*vx + vy*vy + vz*vz) in
vx / mag

// For complex cases, split into separate scalar signals
signal velocity_x { resolve { ... } }
signal velocity_y { resolve { ... } }
signal velocity_z { resolve { ... } }
```

The compiler expands vector signal resolve blocks into per-component scalar expressions
at compile time, so component access on `prev`, `inputs`, and `signal.path` is
resolved statically without runtime overhead.

---

## 19. Signal Input Operator

The `<-` operator writes to signal input accumulators:

```
signal target <- value
```

Multiple writes accumulate (must be commutative).

For fields with position:

```
field target <- position, value
```

---

## 20. Reference Prefixes

| Prefix | Meaning |
|--------|---------|
| `signal.` | Resolved signal value |
| `field.` | Field (measure phase only) |
| `const.` | Constant value |
| `config.` | Configuration parameter |
| `prev` | Previous signal value (in resolve blocks) |
| `payload` | Impulse data (in apply blocks) |
| `dt.raw` | Raw timestep (requires `: uses(dt.raw)` declaration, prefer dt-robust operators) |
| `namespace.` | Engine-provided function (e.g. `maths.`, `vector.`, `dt.`, `physics.`) |
| `dt.` | dt-robust integration operators |

---

## 21. Mathematical Constants

Built-in mathematical constants. Both ASCII and Unicode forms are supported:

| Constant | Symbol | Value |
|----------|--------|-------|
| `PI` | `π` | 3.14159... (ratio of circumference to diameter) |
| `TAU` | `τ` | 6.28318... (2π, the circle constant) |
| `E` | `ℯ` | 2.71828... (Euler's number) |
| `I` | `ⅈ` | √-1 (imaginary unit) |
| `PHI` | `φ` | 1.61803... (golden ratio) |

Example usage:

```
resolve {
  dt.advance_phase(prev, signal.omega)
}

# Unicode form for extra flair
resolve {
  prev * ℯ ^ (-signal.rate * τ)
}
```

---

## 22. Complete Example

```
const {
  physics.stefan_boltzmann: 5.67e-8 <W/m²/K⁴>
}

config {
  terra.thermal.decay_rate: 1e-10 <1/s>
}

type ThermalState {
  temperature: Scalar<K>
  flux: Scalar<W/m²>
}

strata terra.thermal {
  : title("Thermal")
  : stride(5)
}

era early {
  : initial
  : dt(1 <Myr>)

  strata {
    terra.thermal: active
  }

  transition {
    to: era.stable
    when {
      signal.terra.core.temp < 5000 <K>
    }
  }
}

era stable {
  : terminal
  : dt(100 <kyr>)

  strata {
    terra.thermal: active
  }
}

signal terra.core.temp {
  : Scalar<K, 100..10000>
  : strata(terra.thermal)

  resolve {
    # Use dt-robust decay operator instead of raw dt
    dt.decay(prev, config.terra.thermal.decay_halflife)
  }
}

field terra.core.temp_field {
  : Scalar<K>
  : strata(terra.thermal)
  : topology(point_cloud)

  measure {
    signal.terra.core.temp
  }
}
```

---

## See Also

- @dsl/language.md — Design principles
- @dsl/types-and-units.md — Type system details
- @dsl/dt-robust.md — dt-robust operators
- @dsl/assertions.md — Assertions and faults
