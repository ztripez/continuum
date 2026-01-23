# Continuum DSL — Functions and Templates

This document defines **functions**, **templates**, and **patterns** for code reuse in the DSL.

---

## 1. The Problem

Without abstraction, authors write repetitive code:
- Same physics equations repeated across signals
- Similar signal structures with different parameters
- Boilerplate for common patterns (accumulators, integrators)

The DSL provides three levels of abstraction to address this.

---

## 2. Functions (`fn`)

Pure, inlined expression reuse.

### Declaration

```
fn physics.stefan_boltzmann_loss(temp: Scalar<K>) -> Scalar<W/m²> {
  const.physics.stefan_boltzmann * (temp ^ 4)
}

fn physics.gravitational_acceleration(mass: Scalar<kg>, radius: Scalar<m>) -> Scalar<m/s²> {
  const.physics.gravitational * mass / (radius ^ 2)
}

fn orbital.mean_motion(central_mass: Scalar<kg>, semi_major: Scalar<m>) -> Scalar<rad/s> {
  sqrt(const.physics.gravitational * central_mass / (semi_major ^ 3))
}

fn orbital.orbital_velocity(central_mass: Scalar<kg>, radius: Scalar<m>) -> Scalar<m/s> {
  sqrt(const.physics.gravitational * central_mass / radius)
}
```

### Usage

```
signal terra.surface.radiation {
  : Scalar<W/m²>
  : strata(terra.thermal)

  resolve {
    fn.physics.stefan_boltzmann_loss(signal.terra.surface.temperature)
  }
}

signal terra.surface.gravity {
  : Scalar<m/s²>
  : strata(terra.genesis)

  resolve {
    fn.physics.gravitational_acceleration(signal.terra.mass, signal.terra.radius)
  }
}
```

### Rules

- Functions are **pure** — no side effects
- Functions are **inlined** at compile time
- Functions cannot access `prev`, `dt`, or write to signals
- Functions can call other functions
- Functions can access `const.*` and `config.*`

---

## 3. Templates (`template`)

Generate multiple entities from a pattern.

### Declaration

```
template.thermal_layer(name: String, depth: Scalar<m>, conductivity: Scalar<W/m/K>) {
  signal.terra.geophysics.{name}.temperature {
    : Scalar<K, 100..10000>
    : strata(terra.thermal)

    config {
      initial: 1500 <K>
    }

    resolve {
      let flux_in = signal.terra.geophysics.{name}.flux_in in
      let flux_out = signal.terra.geophysics.{name}.flux_out in
      dt.relax(prev, prev + (flux_in - flux_out) / {conductivity}, config.terra.thermal.tau)
    }
  }

  signal.terra.geophysics.{name}.flux_out {
    : Scalar<W/m²>
    : strata(terra.thermal)

    resolve {
      fn.physics.conductive_flux(
        signal.terra.geophysics.{name}.temperature,
        signal.terra.geophysics.{name}.temperature_above,
        {depth},
        {conductivity}
      )
    }
  }
}
```

### Instantiation

```
template.thermal_layer("crust", 30000 <m>, 2.5 <W/m/K>)
template.thermal_layer("upper_mantle", 400000 <m>, 3.0 <W/m/K>)
template.thermal_layer("lower_mantle", 2200000 <m>, 4.0 <W/m/K>)
template.thermal_layer("outer_core", 2300000 <m>, 40.0 <W/m/K>)
```

### Generated Signals

The above generates:
- `signal.terra.geophysics.crust.temperature`
- `signal.terra.geophysics.crust.flux_out`
- `signal.terra.geophysics.upper_mantle.temperature`
- `signal.terra.geophysics.upper_mantle.flux_out`
- ... etc

### Rules

- Templates are **expanded** at compile time
- Template parameters are substituted textually
- Templates can generate signals, fields, operators, etc.
- Templates cannot be recursive
- Generated identifiers must be unique

---

## 4. Patterns (`pattern`)

Pre-built templates for very common signal structures.

### Available Patterns

#### Accumulator

```
pattern.accumulator(
  path: terra.atmosphere.co2,
  unit: ppm,
  bounds: 0..1000000,
  strata: terra.atmosphere
)
```

Expands to:

```
signal terra.atmosphere.co2 {
  : Scalar<ppm, 0..1000000>
  : strata(terra.atmosphere)

  resolve {
    dt.accumulate(prev, inputs, 0, 1000000)
  }
}
```

#### Integrator

```
pattern.integrator(
  path: terra.orbit.position,
  rate_path: terra.orbit.velocity,
  type: Vec3<m>,
  strata: terra.orbital
)
```

Expands to:

```
signal terra.orbit.position {
  : Vec3<m>
  : strata(terra.orbital)

  resolve {
    dt.integrate(prev, signal.terra.orbit.velocity)
  }
}
```

#### Relaxer

```
pattern.relaxer(
  path: terra.surface.temperature,
  target_expr: fn.equilibrium_temp(signal.stellar.flux, signal.terra.albedo),
  tau: config.terra.thermal_tau,
  type: Scalar<K, 50..500>,
  strata: terra.thermal
)
```

#### Decay

```
pattern.decay(
  path: terra.core.radiogenic_heat,
  halflife: config.terra.radiogenic_halflife,
  type: Scalar<W, 0..1e14>,
  strata: terra.thermal
)
```

#### Phase

```
pattern.phase(
  path: terra.orbit.true_anomaly,
  rate_expr: fn.orbital.mean_motion(signal.stellar.mass, signal.terra.semi_major),
  strata: terra.orbital
)
```

---

## 5. Combining Abstractions

Functions, templates, and patterns compose:

```
// Function for physics
fn thermal.equilibrium(flux: Scalar<W/m²>, albedo: Scalar<1>) -> Scalar<K> {
  pow(flux * (1 - albedo) / const.physics.stefan_boltzmann, 0.25)
}

// Template using function
template.planetary_surface(name: String) {
  pattern.relaxer(
    path: {name}.surface.temperature,
    target_expr: fn.thermal.equilibrium(signal.stellar.flux_at_{name}, signal.{name}.albedo),
    tau: config.{name}.thermal_tau,
    type: Scalar<K, 50..1000>,
    strata: {name}.thermal
  )

  pattern.accumulator(
    path: {name}.atmosphere.co2,
    unit: ppm,
    bounds: 0..1e6,
    strata: {name}.atmosphere
  )
}

// Instantiate for multiple planets
template.planetary_surface("terra")
template.planetary_surface("venus")
template.planetary_surface("mars")
```

---

## 6. Conditional Templates

Templates can include conditional generation:

```
template.orbital_body(name: String, has_atmosphere: Bool) {
  signal.{name}.orbit.position {
    : Vec3<m>
    : strata(stellar.orbital)

    resolve {
      dt.integrate(prev, signal.{name}.orbit.velocity)
    }
  }

  if {has_atmosphere} {
    signal.{name}.atmosphere.pressure {
      : Scalar<Pa, 0..1e8>
      : strata({name}.atmosphere)

      resolve {
        // atmosphere logic
      }
    }
  }
}

template.orbital_body("terra", true)
template.orbital_body("luna", false)  // no atmosphere signal generated
```

---

## 7. Generic Functions

Functions can be generic over types:

```
fn math.lerp<T>(a: T, b: T, t: Scalar<1>) -> T {
  a + (b - a) * t
}

fn math.clamp<T: Ordered>(value: T, min: T, max: T) -> T {
  if value < min { min }
  else if value > max { max }
  else { value }
}

// Usage
let temp = fn.math.lerp(200 <K>, 400 <K>, 0.5) in  // 300 K
let clamped = fn.math.clamp(signal.terra.pressure, 0 <Pa>, 1e5 <Pa>) in
// use temp and clamped in body expression
```

---

## 8. Standard Library

The DSL includes a standard library of common functions:

### Math

```
fn math.lerp(a, b, t)
fn math.clamp(value, min, max)
fn math.smoothstep(edge0, edge1, x)
fn math.remap(value, in_min, in_max, out_min, out_max)
```

### Vector

```
fn vec.dot(a, b)
fn vec.cross(a, b)
fn vec.magnitude(v)
fn vec.normalize(v)
fn vec.distance(a, b)
fn vec.angle_between(a, b)
```

### Physics

```
fn physics.gravitational_force(m1, m2, r)
fn physics.gravitational_acceleration(m, r)
fn physics.orbital_velocity(m, r)
fn physics.escape_velocity(m, r)
fn physics.schwarzschild_radius(m)
fn physics.stefan_boltzmann_flux(t)
fn physics.planck_peak_wavelength(t)
```

### Orbital

```
fn orbital.mean_motion(m, a)
fn orbital.period(m, a)
fn orbital.vis_viva(m, r, a)
fn orbital.eccentric_anomaly(mean_anomaly, eccentricity)
fn orbital.true_anomaly(eccentric_anomaly, eccentricity)
```

---

## Summary

| Construct | Purpose | Scope |
|-----------|---------|-------|
| `fn.name` | Pure expression reuse | Expressions only |
| `template.name` | Generate multiple entities | Signals, fields, operators |
| `pattern.name` | Common signal structures | Pre-built templates |

- Functions are inlined and pure
- Templates expand at compile time
- Patterns provide high-level abstractions
- All three compose together
- Standard library provides common functions
