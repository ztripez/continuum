# Kernels

This document specifies the **kernel compute system** in Continuum.

Kernels are engine-provided mathematical primitives that execute on CPU or GPU.

---

## 1. What Kernels Are

A **kernel** is a named, typed compute operation provided by the engine.

Kernels:
- are called from DSL via `kernel.*` syntax
- have fixed signatures (inputs, outputs, types)
- may execute on CPU or GPU
- must be deterministic within their declared guarantee level

Kernels are **not** user-defined.
They are part of the engine's primitive library.

---

## 2. Why Kernels Exist

The DSL expresses high-level math.
Some operations require:
- vectorized computation
- parallel reduction
- GPU acceleration
- numerical algorithms

Kernels encapsulate this complexity.

The DSL author writes:
```
kernel.gravity_field(masses, positions, target)
```

The engine decides:
- CPU vs GPU execution
- workgroup dispatch
- memory layout
- synchronization

---

## 3. Kernel Categories

### 3.1 Scalar Math

Basic mathematical operations with unit awareness.

```
kernel.sqrt(x: Scalar<T²>) -> Scalar<T>
kernel.pow(base: Scalar<T>, exp: Scalar<1>) -> Scalar<T^exp>
kernel.exp(x: Scalar<1>) -> Scalar<1>
kernel.log(x: Scalar<1>) -> Scalar<1>
kernel.sin(x: Scalar<rad>) -> Scalar<1>
kernel.cos(x: Scalar<rad>) -> Scalar<1>
kernel.atan2(y: Scalar<T>, x: Scalar<T>) -> Scalar<rad>
kernel.clamp(x: Scalar<T>, lo: Scalar<T>, hi: Scalar<T>) -> Scalar<T>
kernel.lerp(a: Scalar<T>, b: Scalar<T>, t: Scalar<1>) -> Scalar<T>
```

### 3.2 Vector Operations

```
kernel.dot(a: Vec3<T>, b: Vec3<T>) -> Scalar<T²>
kernel.cross(a: Vec3<T>, b: Vec3<T>) -> Vec3<T²>
kernel.normalize(v: Vec3<T>) -> Vec3<1>
kernel.magnitude(v: Vec3<T>) -> Scalar<T>
kernel.distance(a: Vec3<T>, b: Vec3<T>) -> Scalar<T>
```

### 3.3 Matrix Operations

```
kernel.mat_mul(a: Mat4<T>, b: Mat4<U>) -> Mat4<T*U>
kernel.mat_vec_mul(m: Mat4<T>, v: Vec4<U>) -> Vec4<T*U>
kernel.transpose(m: Mat4<T>) -> Mat4<T>
kernel.inverse(m: Mat4<1>) -> Mat4<1>
kernel.determinant(m: Mat4<T>) -> Scalar<T^4>
```

### 3.4 Tensor Operations

```
kernel.tensor_contract(a: Tensor<N,M,T>, b: Tensor<M,P,U>) -> Tensor<N,P,T*U>
kernel.tensor_trace(t: Tensor<N,N,T>) -> Scalar<T>
kernel.tensor_symmetric(t: Tensor<N,N,T>) -> Tensor<N,N,T>
```

### 3.5 Reductions

Deterministic parallel reductions over collections.

```
kernel.sum(seq: Seq<Scalar<T>>) -> Scalar<T>
kernel.product(seq: Seq<Scalar<T>>) -> Scalar<T>
kernel.min(seq: Seq<Scalar<T>>) -> Scalar<T>
kernel.max(seq: Seq<Scalar<T>>) -> Scalar<T>
kernel.mean(seq: Seq<Scalar<T>>) -> Scalar<T>
```

### 3.6 Grid Operations

Operations on spatial grids.

```
kernel.grid_sample(grid: Grid<W,H,T>, uv: Vec2<1>) -> T
kernel.grid_gradient(grid: Grid<W,H,Scalar<T>>) -> Grid<W,H,Vec2<T>>
kernel.grid_laplacian(grid: Grid<W,H,Scalar<T>>) -> Grid<W,H,Scalar<T>>
kernel.grid_blur(grid: Grid<W,H,T>, radius: Scalar<1>) -> Grid<W,H,T>
kernel.grid_convolve(grid: Grid<W,H,T>, kernel: Grid<K,K,Scalar<1>>) -> Grid<W,H,T>
```

### 3.7 Physics Primitives

Domain-specific physics computations.

```
kernel.gravity_acceleration(
    mass: Scalar<kg>,
    distance: Scalar<m>
) -> Scalar<m/s²>

kernel.orbital_velocity(
    central_mass: Scalar<kg>,
    radius: Scalar<m>
) -> Scalar<m/s>

kernel.kepler_position(
    semi_major: Scalar<m>,
    eccentricity: Scalar<1>,
    mean_anomaly: Scalar<rad>
) -> Vec3<m>

kernel.blackbody_radiation(
    temperature: Scalar<K>,
    surface_area: Scalar<m²>
) -> Scalar<W>

kernel.stefan_boltzmann_flux(
    temperature: Scalar<K>
) -> Scalar<W/m²>
```

### 3.8 Field Computations

Spatial field calculations (typically GPU-accelerated).

```
kernel.gravity_field(
    masses: Seq<Scalar<kg>>,
    positions: Seq<Vec3<m>>,
    sample_points: Grid<W,H,Vec3<m>>
) -> Grid<W,H,Vec3<m/s²>>

kernel.temperature_diffusion(
    current: Grid<W,H,Scalar<K>>,
    conductivity: Grid<W,H,Scalar<W/m/K>>,
    dt: Scalar<s>
) -> Grid<W,H,Scalar<K>>

kernel.fluid_advection(
    field: Grid<W,H,T>,
    velocity: Grid<W,H,Vec2<m/s>>,
    dt: Scalar<s>
) -> Grid<W,H,T>
```

### 3.9 Geometric Operations

```
kernel.sphere_surface_area(radius: Scalar<m>) -> Scalar<m²>
kernel.sphere_volume(radius: Scalar<m>) -> Scalar<m³>
kernel.great_circle_distance(
    a: Vec2<rad>,
    b: Vec2<rad>,
    radius: Scalar<m>
) -> Scalar<m>

kernel.ray_sphere_intersect(
    ray_origin: Vec3<m>,
    ray_dir: Vec3<1>,
    sphere_center: Vec3<m>,
    sphere_radius: Scalar<m>
) -> Option<Scalar<m>>
```

---

## 4. Kernel Signatures

Each kernel has a fixed signature in the engine's kernel registry.

```
KernelSignature {
    name: String,
    inputs: Vec<(String, TypeRef)>,
    output: TypeRef,
    determinism: DeterminismLevel,
    preferred_backend: BackendHint,
}
```

### 4.1 Type References

Kernel types use the DSL type system:
- `Scalar<unit>`
- `Vec2<unit>`, `Vec3<unit>`, `Vec4<unit>`
- `Mat4<unit>`
- `Tensor<N,M,unit>`
- `Seq<T>`
- `Grid<W,H,T>`
- `Option<T>`

### 4.2 Unit Polymorphism

Some kernels are unit-polymorphic:

```
kernel.dot<T>(a: Vec3<T>, b: Vec3<T>) -> Scalar<T²>
```

The compiler infers `T` from usage.

---

## 5. Determinism Levels

Kernels declare their determinism guarantee.

### 5.1 Strict Deterministic

```
determinism: Strict
```

- Bitwise identical results across runs
- Bitwise identical across CPU/GPU
- Required for causal phases (Collect, Resolve, Fracture)

### 5.2 Relaxed Deterministic

```
determinism: Relaxed
```

- Semantically equivalent results
- May differ in low-order floating-point bits
- Allowed only in Measure phase
- Used for visualization/observation kernels

### 5.3 Enforcement

The compiler enforces:
- Causal phases may only call `Strict` kernels
- Measure phase may call `Strict` or `Relaxed` kernels

```
// In a signal resolve block:
kernel.gravity_field(...)  // OK if Strict

// In a field measure block:
kernel.approximate_render(...)  // OK if Relaxed
```

---

## 6. Backend Selection

### 6.1 Backend Hints

Kernels declare preferred execution:

```
preferred_backend: CPU      // scalar, small data
preferred_backend: GPU      // parallel, large data
preferred_backend: Auto     // engine decides
```

### 6.2 Runtime Selection

The engine selects backend based on:
- Kernel hint
- Data size
- Available hardware
- Current workload

Selection is transparent to DSL.

### 6.3 Fallback

All GPU kernels have CPU fallbacks.
If GPU unavailable, CPU executes.
Results must be identical (for Strict kernels).

---

## 7. Memory and Data Flow

### 7.1 Input Immutability

Kernel inputs are immutable.
Kernels never modify their arguments.

### 7.2 Output Allocation

Kernel outputs are freshly allocated.
No in-place mutation.

### 7.3 Data Transfer

For GPU kernels:
- Inputs are uploaded before dispatch
- Outputs are downloaded after completion
- Transfer is automatic and cached where possible

### 7.4 No Persistent GPU State

Kernels do not maintain GPU-side state between calls.
Each call is independent.

---

## 8. Calling Kernels from DSL

### 8.1 Basic Call

```
signal.terra.surface.flux {
    resolve {
        kernel.stefan_boltzmann_flux(signal.terra.surface.temperature)
    }
}
```

### 8.2 Chained Calls

```
signal.terra.orbit.position {
    resolve {
        let anomaly = kernel.kepler_mean_to_true(
            signal.terra.orbit.mean_anomaly,
            config.terra.orbit.eccentricity
        )
        kernel.kepler_position(
            config.terra.orbit.semi_major,
            config.terra.orbit.eccentricity,
            anomaly
        )
    }
}
```

### 8.3 With Entity Collections

```
signal.stellar.total_luminosity {
    resolve {
        kernel.sum(
            map(entity.stellar.star, kernel.blackbody_radiation(self.temperature, self.surface_area))
        )
    }
}
```

---

## 9. Custom Kernels

The engine may be extended with domain-specific kernels.

### 9.1 Kernel Definition (Engine Side)

```rust
#[kernel(determinism = Strict, backend = Auto)]
fn radiogenic_heat(
    initial: Scalar<W>,
    half_life: Scalar<s>,
    elapsed: Scalar<s>,
) -> Scalar<W> {
    initial * 0.5_f64.powf(elapsed / half_life)
}
```

### 9.2 Registration

Custom kernels are registered at engine startup.
They appear in the `kernel.*` namespace.

### 9.3 Validation

Custom kernels must:
- Have deterministic implementations
- Pass unit-consistency checks
- Provide CPU fallback if GPU implementation exists

---

## 10. Kernel Batching

### 10.1 Automatic Batching

When multiple nodes call the same kernel:

```
// Three entity instances each call:
kernel.gravity_acceleration(self.mass, self.distance)
```

The engine may batch into single dispatch:

```
kernel.gravity_acceleration_batch(masses[], distances[]) -> accelerations[]
```

### 10.2 Batching Requirements

- Same kernel
- Same types
- Independent inputs
- Within same DAG level

### 10.3 DSL Transparency

Batching is invisible to DSL.
Authors write single-instance code.

---

## 11. Error Handling

### 11.1 Numeric Errors

Kernels detect:
- NaN results
- Infinity (unless explicitly allowed)
- Domain errors (sqrt of negative, log of zero)

### 11.2 Error Propagation

Kernel errors become faults:

```
Fault {
    kind: KernelError,
    kernel: "kernel.sqrt",
    message: "negative input",
    location: SignalId("terra.orbit.velocity"),
}
```

### 11.3 No Silent Clamping

Kernels do not silently clamp or fix errors.
Invalid inputs produce faults.

---

## 12. Kernel Index

Standard kernels provided by the engine:

| Category | Kernels |
|----------|---------|
| Scalar Math | `sqrt`, `pow`, `exp`, `log`, `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `atan2`, `abs`, `sign`, `floor`, `ceil`, `round`, `clamp`, `lerp`, `smoothstep` |
| Vector | `dot`, `cross`, `normalize`, `magnitude`, `distance`, `reflect`, `refract` |
| Matrix | `mat_mul`, `mat_vec_mul`, `transpose`, `inverse`, `determinant` |
| Tensor | `tensor_contract`, `tensor_trace`, `tensor_symmetric`, `tensor_antisymmetric` |
| Reduction | `sum`, `product`, `min`, `max`, `mean`, `variance` |
| Grid | `grid_sample`, `grid_gradient`, `grid_laplacian`, `grid_blur`, `grid_convolve`, `grid_resample` |
| Physics | `gravity_acceleration`, `orbital_velocity`, `kepler_position`, `kepler_velocity`, `blackbody_radiation`, `stefan_boltzmann_flux`, `planck_spectrum` |
| Geometry | `sphere_surface_area`, `sphere_volume`, `great_circle_distance`, `ray_sphere_intersect` |
| Interpolation | `lerp`, `slerp`, `cubic_bezier`, `catmull_rom` |

---

## Summary

- Kernels are engine-provided compute primitives
- Called via `kernel.*` syntax in DSL
- Execute on CPU or GPU transparently
- Must declare determinism level (Strict for causal, Relaxed for observation)
- Unit-typed and validated at compile time
- Batched automatically for efficiency
- Never silently fix errors

The DSL author focuses on **what** to compute.
The engine handles **how** to compute it.
