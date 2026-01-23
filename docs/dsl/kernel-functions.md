# Kernel Function Reference

This document provides a comprehensive reference for all kernel functions available in the Continuum DSL.

Kernels are engine-provided mathematical primitives called via namespaced syntax.
For architecture and execution details, see `@docs/execution/kernels.md`.

---

## Namespaces

| Namespace | Purpose |
|-----------|---------|
| `maths.*` | Scalar mathematical operations |
| `vector.*` | Vector construction and operations |
| `quat.*` | Quaternion operations |
| `matrix.*` | Matrix operations |
| `tensor.*` | Tensor operations |
| `dt.*` | Time-step robust operators |
| `rng.*` | Deterministic random number generation |

---

## 1. Scalar Math (`maths.*`)

### 1.1 Basic Operations

```
maths.abs(x: Scalar<T>) -> Scalar<T>
```
Absolute value. Preserves unit.

```
maths.sign(x: Scalar<T>) -> Scalar<1>
```
Returns -1, 0, or 1 based on sign. Unitless result.

```
maths.neg(x: Scalar<T>) -> Scalar<T>
```
Negation. Equivalent to `-x`.

```
maths.recip(x: Scalar<T>) -> Scalar<1/T>
```
Reciprocal. Returns `1/x` with inverted unit.

---

### 1.2 Powers and Roots

```
maths.sqrt(x: Scalar<T²>) -> Scalar<T>
```
Square root. Unit is halved (e.g., `m²` → `m`).

```
maths.cbrt(x: Scalar<T³>) -> Scalar<T>
```
Cube root. Unit is divided by 3.

```
maths.pow(base: Scalar<T>, exp: Scalar<1>) -> Scalar<T^exp>
```
Power. Exponent must be unitless. Result unit scales with exponent.

```
maths.hypot(x: Scalar<T>, y: Scalar<T>) -> Scalar<T>
```
Hypotenuse: `sqrt(x² + y²)`. More numerically stable than direct computation.

---

### 1.3 Exponential and Logarithmic

```
maths.exp(x: Scalar<1>) -> Scalar<1>
```
Natural exponential. Input and output are unitless.

```
maths.exp2(x: Scalar<1>) -> Scalar<1>
```
Base-2 exponential: `2^x`.

```
maths.ln(x: Scalar<1>) -> Scalar<1>
```
Natural logarithm. Input must be positive and unitless.

```
maths.log(x: Scalar<1>) -> Scalar<1>
```
Alias for `ln`.

```
maths.log2(x: Scalar<1>) -> Scalar<1>
```
Base-2 logarithm.

```
maths.log10(x: Scalar<1>) -> Scalar<1>
```
Base-10 logarithm.

---

### 1.4 Trigonometric

All trigonometric functions expect angles in radians.

```
maths.sin(x: Scalar<rad>) -> Scalar<1>
```
Sine. Input in radians, output unitless.

```
maths.cos(x: Scalar<rad>) -> Scalar<1>
```
Cosine.

```
maths.tan(x: Scalar<rad>) -> Scalar<1>
```
Tangent.

```
maths.asin(x: Scalar<1>) -> Scalar<rad>
```
Arcsine. Input must be in [-1, 1].

```
maths.acos(x: Scalar<1>) -> Scalar<rad>
```
Arccosine. Input must be in [-1, 1].

```
maths.atan(x: Scalar<1>) -> Scalar<rad>
```
Arctangent.

```
maths.atan2(y: Scalar<T>, x: Scalar<T>) -> Scalar<rad>
```
Two-argument arctangent. Returns angle in [-π, π]. Both arguments must have same unit.

---

### 1.5 Hyperbolic

```
maths.sinh(x: Scalar<1>) -> Scalar<1>
```
Hyperbolic sine.

```
maths.cosh(x: Scalar<1>) -> Scalar<1>
```
Hyperbolic cosine.

```
maths.tanh(x: Scalar<1>) -> Scalar<1>
```
Hyperbolic tangent. Output in (-1, 1).

```
maths.asinh(x: Scalar<1>) -> Scalar<1>
```
Inverse hyperbolic sine.

```
maths.acosh(x: Scalar<1>) -> Scalar<1>
```
Inverse hyperbolic cosine. Input must be ≥ 1.

```
maths.atanh(x: Scalar<1>) -> Scalar<1>
```
Inverse hyperbolic tangent. Input must be in (-1, 1).

---

### 1.6 Rounding

```
maths.floor(x: Scalar<T>) -> Scalar<T>
```
Round toward negative infinity.

```
maths.ceil(x: Scalar<T>) -> Scalar<T>
```
Round toward positive infinity.

```
maths.round(x: Scalar<T>) -> Scalar<T>
```
Round to nearest integer (half away from zero).

```
maths.trunc(x: Scalar<T>) -> Scalar<T>
```
Round toward zero (truncate fractional part).

```
maths.fract(x: Scalar<T>) -> Scalar<T>
```
Fractional part: `x - floor(x)`.

---

### 1.7 Comparison and Selection

```
maths.min(a: Scalar<T>, b: Scalar<T>) -> Scalar<T>
```
Minimum of two values.

```
maths.max(a: Scalar<T>, b: Scalar<T>) -> Scalar<T>
```
Maximum of two values.

```
maths.clamp(x: Scalar<T>, lo: Scalar<T>, hi: Scalar<T>) -> Scalar<T>
```
⚠️ **Dangerous Function** - Requires `: uses(maths.clamping)`

Clamp value to range [lo, hi]. All arguments must have same unit.

Silently constrains values to bounds, masking out-of-range conditions that may indicate bugs. **Prefer using assertions** to validate bounds instead (see `@docs/dsl/assertions.md`).

```cdsl
signal example {
    : Scalar<K>
    : uses(maths.clamping)  // Required explicit opt-in
    resolve {
        maths.clamp(prev, 0 <K>, 100 <K>)
    }
}
```

```
maths.saturate(x: Scalar<1>) -> Scalar<1>
```
⚠️ **Dangerous Function** - Requires `: uses(maths.clamping)`

Clamp value to range [0, 1]. Equivalent to `clamp(x, 0.0, 1.0)`.

Silently constrains values to bounds, masking out-of-range conditions that may indicate bugs. **Prefer using assertions** to validate bounds instead (see `@docs/dsl/assertions.md`).

```
maths.step(edge: Scalar<T>, x: Scalar<T>) -> Scalar<1>
```
Step function: returns 0 if x < edge, 1 otherwise.

---

### 1.8 Interpolation

```
maths.lerp(a: Scalar<T>, b: Scalar<T>, t: Scalar<1>) -> Scalar<T>
```
Linear interpolation: `a + (b - a) * t`.

```
maths.smoothstep(edge0: Scalar<T>, edge1: Scalar<T>, x: Scalar<T>) -> Scalar<1>
```
Smooth Hermite interpolation. Returns 0 if x ≤ edge0, 1 if x ≥ edge1, smooth curve between.

```
maths.mix(a: Scalar<T>, b: Scalar<T>, t: Scalar<1>) -> Scalar<T>
```
Alias for `lerp`.

---

### 1.9 Variadic Reduction

These functions accept sequences and perform deterministic parallel reduction.

```
maths.sum(seq: Seq<Scalar<T>>) -> Scalar<T>
```
Sum all elements.

```
maths.product(seq: Seq<Scalar<T>>) -> Scalar<T>
```
Product of all elements.

```
maths.min(seq: Seq<Scalar<T>>) -> Scalar<T>
```
Minimum element.

```
maths.max(seq: Seq<Scalar<T>>) -> Scalar<T>
```
Maximum element.

```
maths.mean(seq: Seq<Scalar<T>>) -> Scalar<T>
```
Arithmetic mean.

---

### 1.10 Angle Operations

```
maths.degrees(x: Scalar<rad>) -> Scalar<deg>
```
Convert radians to degrees.

```
maths.radians(x: Scalar<deg>) -> Scalar<rad>
```
Convert degrees to radians.

```
maths.wrap_angle(x: Scalar<rad>) -> Scalar<rad>
```
Wrap angle to [0, 2π).

```
maths.wrap_angle_signed(x: Scalar<rad>) -> Scalar<rad>
```
Wrap angle to [-π, π).

---

## 2. Vector Operations (`vector.*`)

### 2.1 Construction

```
vector.vec2(x: Scalar<T>, y: Scalar<T>) -> Vec2<T>
```
Construct 2D vector from components.

```
vector.vec3(x: Scalar<T>, y: Scalar<T>, z: Scalar<T>) -> Vec3<T>
```
Construct 3D vector from components.

```
vector.vec4(x: Scalar<T>, y: Scalar<T>, z: Scalar<T>, w: Scalar<T>) -> Vec4<T>
```
Construct 4D vector from components.

```
vector.splat2(v: Scalar<T>) -> Vec2<T>
```
Create Vec2 with all components equal to v.

```
vector.splat3(v: Scalar<T>) -> Vec3<T>
```
Create Vec3 with all components equal to v.

```
vector.splat4(v: Scalar<T>) -> Vec4<T>
```
Create Vec4 with all components equal to v.

```
vector.zero2() -> Vec2<1>
```
Zero vector (2D).

```
vector.zero3() -> Vec3<1>
```
Zero vector (3D).

```
vector.zero4() -> Vec4<1>
```
Zero vector (4D).

```
vector.unit_x() -> Vec3<1>
vector.unit_y() -> Vec3<1>
vector.unit_z() -> Vec3<1>
```
Standard basis vectors.

---

### 2.2 Component Access

```
vector.x(v: Vec3<T>) -> Scalar<T>
vector.y(v: Vec3<T>) -> Scalar<T>
vector.z(v: Vec3<T>) -> Scalar<T>
vector.w(v: Vec4<T>) -> Scalar<T>
```
Extract individual components.

```
vector.xy(v: Vec3<T>) -> Vec2<T>
vector.xz(v: Vec3<T>) -> Vec2<T>
vector.yz(v: Vec3<T>) -> Vec2<T>
```
Extract 2D subvectors (swizzle).

```
vector.xyz(v: Vec4<T>) -> Vec3<T>
```
Extract first three components.

---

### 2.3 Arithmetic

```
vector.add(a: Vec3<T>, b: Vec3<T>) -> Vec3<T>
```
Component-wise addition.

```
vector.sub(a: Vec3<T>, b: Vec3<T>) -> Vec3<T>
```
Component-wise subtraction.

```
vector.mul(v: Vec3<T>, s: Scalar<U>) -> Vec3<T*U>
```
Scalar multiplication.

```
vector.div(v: Vec3<T>, s: Scalar<U>) -> Vec3<T/U>
```
Scalar division.

```
vector.neg(v: Vec3<T>) -> Vec3<T>
```
Negate all components.

```
vector.hadamard(a: Vec3<T>, b: Vec3<U>) -> Vec3<T*U>
```
Component-wise (Hadamard) product.

---

### 2.4 Products

```
vector.dot(a: Vec3<T>, b: Vec3<U>) -> Scalar<T*U>
```
Dot product. Result unit is product of input units.

```
vector.cross(a: Vec3<T>, b: Vec3<U>) -> Vec3<T*U>
```
Cross product (3D only). Result unit is product of input units.

---

### 2.5 Length and Distance

```
vector.length(v: Vec3<T>) -> Scalar<T>
```
Euclidean length (magnitude).

```
vector.length_squared(v: Vec3<T>) -> Scalar<T²>
```
Squared length. Faster than `length` when comparing magnitudes.

```
vector.distance(a: Vec3<T>, b: Vec3<T>) -> Scalar<T>
```
Distance between two points.

```
vector.distance_squared(a: Vec3<T>, b: Vec3<T>) -> Scalar<T²>
```
Squared distance.

```
vector.manhattan(a: Vec3<T>, b: Vec3<T>) -> Scalar<T>
```
Manhattan (L1) distance.

---

### 2.6 Normalization

```
vector.normalize(v: Vec3<T>) -> Vec3<1>
```
Unit vector in same direction. Result is unitless.

```
vector.normalize_or_zero(v: Vec3<T>) -> Vec3<1>
```
Normalize, or return zero vector if input is zero.

```
vector.try_normalize(v: Vec3<T>) -> Option<Vec3<1>>
```
Normalize if possible, None if zero vector.

```
vector.is_normalized(v: Vec3<1>) -> Bool
```
Check if vector has unit length (within tolerance).

---

### 2.7 Interpolation

```
vector.lerp(a: Vec3<T>, b: Vec3<T>, t: Scalar<1>) -> Vec3<T>
```
Linear interpolation between vectors.

```
vector.nlerp(a: Vec3<T>, b: Vec3<T>, t: Scalar<1>) -> Vec3<1>
```
Normalized linear interpolation. Result is unit vector.

```
vector.slerp(a: Vec3<1>, b: Vec3<1>, t: Scalar<1>) -> Vec3<1>
```
Spherical linear interpolation. Both inputs must be unit vectors.

---

### 2.8 Projection and Reflection

```
vector.project(v: Vec3<T>, onto: Vec3<U>) -> Vec3<T>
```
Project v onto direction of `onto`.

```
vector.reject(v: Vec3<T>, from: Vec3<U>) -> Vec3<T>
```
Component of v perpendicular to `from`.

```
vector.reflect(v: Vec3<T>, normal: Vec3<1>) -> Vec3<T>
```
Reflect v across plane defined by normal.

```
vector.refract(v: Vec3<1>, normal: Vec3<1>, eta: Scalar<1>) -> Vec3<1>
```
Refract unit vector through surface. `eta` is ratio of indices of refraction.

---

### 2.9 Angles

```
vector.angle_between(a: Vec3<T>, b: Vec3<T>) -> Scalar<rad>
```
Angle between two vectors in radians.

```
vector.cos_angle_between(a: Vec3<T>, b: Vec3<T>) -> Scalar<1>
```
Cosine of angle. Faster than `angle_between` when only comparing angles.

---

### 2.10 Component-wise Operations

```
vector.abs(v: Vec3<T>) -> Vec3<T>
```
Absolute value of each component.

```
vector.floor(v: Vec3<T>) -> Vec3<T>
```
Floor each component.

```
vector.ceil(v: Vec3<T>) -> Vec3<T>
```
Ceil each component.

```
vector.round(v: Vec3<T>) -> Vec3<T>
```
Round each component.

```
vector.min(a: Vec3<T>, b: Vec3<T>) -> Vec3<T>
```
Component-wise minimum.

```
vector.max(a: Vec3<T>, b: Vec3<T>) -> Vec3<T>
```
Component-wise maximum.

```
vector.clamp(v: Vec3<T>, lo: Vec3<T>, hi: Vec3<T>) -> Vec3<T>
```
Clamp each component.

---

## 3. Quaternion Operations (`quat.*`)

Quaternions represent rotations in 3D space.
Component order is (w, x, y, z) where w is the scalar part.

### 3.1 Construction

```
quat.quat(w: Scalar<1>, x: Scalar<1>, y: Scalar<1>, z: Scalar<1>) -> Quat
```
Construct from components. Not normalized automatically.

```
quat.identity() -> Quat
```
Identity quaternion (no rotation): (1, 0, 0, 0).

```
quat.from_axis_angle(axis: Vec3<1>, angle: Scalar<rad>) -> Quat
```
Create rotation around axis by angle. Axis must be unit vector.

```
quat.from_euler(roll: Scalar<rad>, pitch: Scalar<rad>, yaw: Scalar<rad>) -> Quat
```
Create from Euler angles (XYZ order).

```
quat.from_rotation_x(angle: Scalar<rad>) -> Quat
```
Rotation around X axis.

```
quat.from_rotation_y(angle: Scalar<rad>) -> Quat
```
Rotation around Y axis.

```
quat.from_rotation_z(angle: Scalar<rad>) -> Quat
```
Rotation around Z axis.

```
quat.from_rotation_arc(from: Vec3<1>, to: Vec3<1>) -> Quat
```
Shortest rotation from one direction to another. Both must be unit vectors.

---

### 3.2 Component Access

```
quat.w(q: Quat) -> Scalar<1>
quat.x(q: Quat) -> Scalar<1>
quat.y(q: Quat) -> Scalar<1>
quat.z(q: Quat) -> Scalar<1>
```
Extract individual components.

```
quat.xyz(q: Quat) -> Vec3<1>
```
Extract vector part (x, y, z).

---

### 3.3 Operations

```
quat.normalize(q: Quat) -> Quat
```
Normalize to unit quaternion.

```
quat.conjugate(q: Quat) -> Quat
```
Quaternion conjugate: (w, -x, -y, -z).

```
quat.inverse(q: Quat) -> Quat
```
Quaternion inverse. For unit quaternions, same as conjugate.

```
quat.mul(a: Quat, b: Quat) -> Quat
```
Quaternion multiplication. Represents composition of rotations.

```
quat.rotate(q: Quat, v: Vec3<T>) -> Vec3<T>
```
Rotate vector by quaternion.

```
quat.length(q: Quat) -> Scalar<1>
```
Quaternion magnitude.

```
quat.dot(a: Quat, b: Quat) -> Scalar<1>
```
Quaternion dot product.

---

### 3.4 Interpolation

```
quat.lerp(a: Quat, b: Quat, t: Scalar<1>) -> Quat
```
Linear interpolation (not normalized).

```
quat.nlerp(a: Quat, b: Quat, t: Scalar<1>) -> Quat
```
Normalized linear interpolation.

```
quat.slerp(a: Quat, b: Quat, t: Scalar<1>) -> Quat
```
Spherical linear interpolation. Constant angular velocity.

---

### 3.5 Conversion

```
quat.to_euler(q: Quat) -> Vec3<rad>
```
Convert to Euler angles (XYZ order). Returns (roll, pitch, yaw).

```
quat.to_axis_angle(q: Quat) -> (Vec3<1>, Scalar<rad>)
```
Convert to axis-angle representation.

```
quat.to_rotation_matrix(q: Quat) -> Mat3<1>
```
Convert to 3x3 rotation matrix.

---

### 3.6 Queries

```
quat.angle(q: Quat) -> Scalar<rad>
```
Rotation angle (always positive).

```
quat.axis(q: Quat) -> Vec3<1>
```
Rotation axis (undefined for identity).

```
quat.is_normalized(q: Quat) -> Bool
```
Check if unit quaternion.

---

## 4. Matrix Operations (`matrix.*`)

Matrices use column-major layout.

### 4.1 Construction

```
matrix.identity2() -> Mat2<1>
matrix.identity3() -> Mat3<1>
matrix.identity4() -> Mat4<1>
```
Identity matrices.

```
matrix.zero2() -> Mat2<1>
matrix.zero3() -> Mat3<1>
matrix.zero4() -> Mat4<1>
```
Zero matrices.

```
matrix.from_cols2(c0: Vec2<T>, c1: Vec2<T>) -> Mat2<T>
matrix.from_cols3(c0: Vec3<T>, c1: Vec3<T>, c2: Vec3<T>) -> Mat3<T>
matrix.from_cols4(c0: Vec4<T>, c1: Vec4<T>, c2: Vec4<T>, c3: Vec4<T>) -> Mat4<T>
```
Construct from column vectors.

```
matrix.from_diagonal2(d: Vec2<T>) -> Mat2<T>
matrix.from_diagonal3(d: Vec3<T>) -> Mat3<T>
matrix.from_diagonal4(d: Vec4<T>) -> Mat4<T>
```
Diagonal matrix from vector.

```
matrix.from_scale2(s: Vec2<T>) -> Mat2<T>
matrix.from_scale3(s: Vec3<T>) -> Mat3<T>
```
Scaling matrix.

---

### 4.2 Column/Row Access

```
matrix.col(m: Mat3<T>, i: Int) -> Vec3<T>
```
Extract column i (0-indexed).

```
matrix.row(m: Mat3<T>, i: Int) -> Vec3<T>
```
Extract row i (0-indexed).

---

### 4.3 Arithmetic

```
matrix.add(a: Mat3<T>, b: Mat3<T>) -> Mat3<T>
```
Component-wise addition.

```
matrix.sub(a: Mat3<T>, b: Mat3<T>) -> Mat3<T>
```
Component-wise subtraction.

```
matrix.mul_scalar(m: Mat3<T>, s: Scalar<U>) -> Mat3<T*U>
```
Scalar multiplication.

```
matrix.mul(a: Mat3<T>, b: Mat3<U>) -> Mat3<T*U>
```
Matrix multiplication.

```
matrix.mul_vec(m: Mat3<T>, v: Vec3<U>) -> Vec3<T*U>
```
Matrix-vector multiplication.

---

### 4.4 Properties

```
matrix.transpose(m: Mat3<T>) -> Mat3<T>
```
Matrix transpose.

```
matrix.determinant(m: Mat3<T>) -> Scalar<T³>
```
Determinant. Unit is input unit cubed (for 3x3).

```
matrix.trace(m: Mat3<T>) -> Scalar<T>
```
Sum of diagonal elements.

```
matrix.inverse(m: Mat3<1>) -> Mat3<1>
```
Matrix inverse. Input must be unitless and invertible.

```
matrix.try_inverse(m: Mat3<1>) -> Option<Mat3<1>>
```
Inverse if exists, None if singular.

---

### 4.5 Decomposition

```
matrix.eigenvalues(m: Mat3<1>) -> Vec3<1>
```
Eigenvalues of symmetric matrix.

```
matrix.eigenvectors(m: Mat3<1>) -> Mat3<1>
```
Eigenvectors as columns.

```
matrix.svd_u(m: Mat3<T>) -> Mat3<1>
```
U matrix from SVD: M = U * S * V^T.

```
matrix.svd_s(m: Mat3<T>) -> Vec3<T>
```
Singular values from SVD.

```
matrix.svd_v(m: Mat3<T>) -> Mat3<1>
```
V matrix from SVD.

---

### 4.6 Transforms (4x4)

```
matrix.from_translation(t: Vec3<T>) -> Mat4<T>
```
Translation matrix.

```
matrix.from_rotation(q: Quat) -> Mat4<1>
```
Rotation matrix from quaternion.

```
matrix.from_scale(s: Vec3<T>) -> Mat4<T>
```
Scale matrix.

```
matrix.look_at(eye: Vec3<T>, target: Vec3<T>, up: Vec3<1>) -> Mat4<1>
```
Look-at view matrix.

```
matrix.perspective(fov: Scalar<rad>, aspect: Scalar<1>, near: Scalar<T>, far: Scalar<T>) -> Mat4<1>
```
Perspective projection matrix.

```
matrix.orthographic(left: Scalar<T>, right: Scalar<T>, bottom: Scalar<T>, top: Scalar<T>, near: Scalar<T>, far: Scalar<T>) -> Mat4<1>
```
Orthographic projection matrix.

---

## 5. Time-Step Robust Operators (`dt.*`)

These operators handle variable time steps correctly.
They are essential for stable simulation under varying `dt`.

**Important:** `dt.raw` cannot be used in fracture `collect` blocks. Fractures detect emergent conditions and should accumulate state-dependent inputs, not time-dependent ones. If you need dt-based integration, use signals with `dt.integrate()` or reorganize signal dependencies. See `@docs/fractures.md` section 7.1 for details.

### 5.1 Integration

```
dt.integrate(prev: Scalar<T>, rate: Scalar<T/s>) -> Scalar<T>
```
Euler integration: `prev + rate * dt`.

```
dt.integrate(prev: Vec3<T>, rate: Vec3<T/s>) -> Vec3<T>
```
Vector form of Euler integration.

**Example:**
```
signal terra.orbit.position {
  : Vec3<m>
  resolve {
    dt.integrate(prev, signal.terra.orbit.velocity)
  }
}
```

---

### 5.2 Exponential Decay

```
dt.decay(value: Scalar<T>, halflife: Scalar<s>) -> Scalar<T>
```
Exponential decay toward zero: `value * 0.5^(dt/halflife)`.

Stable for any `dt`. When `dt = halflife`, value halves.

**Example:**
```
signal terra.radiogenic.heat {
  : Scalar<W>
  resolve {
    dt.decay(prev, config.terra.radiogenic.halflife)
  }
}
```

---

### 5.3 Exponential Relaxation

```
dt.relax(current: Scalar<T>, target: Scalar<T>, tau: Scalar<s>) -> Scalar<T>
```
Exponential relaxation toward target: `target + (current - target) * exp(-dt/tau)`.

After time `tau`, approximately 63% of the gap is closed.

**Example:**
```
signal terra.surface.temperature {
  : Scalar<K>
  resolve {
    let equilibrium = fn.thermal.equilibrium_temp(signal.stellar.flux) in
    dt.relax(prev, equilibrium, config.terra.thermal.tau)
  }
}
```

---

### 5.4 Smoothing

```
dt.smooth(current: Scalar<T>, target: Scalar<T>, tau: Scalar<s>) -> Scalar<T>
```
Alias for `dt.relax`. Smooth value changes over time.

---

### 5.5 Bounded Accumulation

```
dt.accumulate(prev: Scalar<T>, delta: Scalar<T>, min: Scalar<T>, max: Scalar<T>) -> Scalar<T>
```
Add delta to previous value, clamped to [min, max].

Equivalent to: `clamp(prev + delta, min, max)`.

**Example:**
```
signal terra.atmosphere.co2 {
  : Scalar<ppm>
  resolve {
    dt.accumulate(prev, inputs.delta, 0 <ppm>, 1000000 <ppm>)
  }
}
```

---

### 5.6 Phase Advancement

```
dt.advance_phase(phase: Scalar<rad>, omega: Scalar<rad/s>) -> Scalar<rad>
```
Advance phase by angular velocity, wrapping to [0, 2π).

Equivalent to: `wrap_angle(phase + omega * dt)`.

Essential for oscillators and orbital mechanics.

**Example:**
```
signal terra.orbit.true_anomaly {
  : Scalar<rad>
  resolve {
    dt.advance_phase(prev, signal.terra.orbit.mean_motion)
  }
}
```

---

### 5.7 Damping

```
dt.damp(value: Scalar<T>, damping: Scalar<1/s>) -> Scalar<T>
```
Apply damping: `value * exp(-damping * dt)`.

Alternative to `decay` using damping coefficient instead of half-life.

Relationship: `damping = ln(2) / halflife`.

---

## 6. Tensor Operations (`tensor.*`)

Operations on general tensors.

```
tensor.contract(a: Tensor<N,M,T>, b: Tensor<M,P,U>) -> Tensor<N,P,T*U>
```
Tensor contraction (generalized matrix multiplication).

```
tensor.trace(t: Tensor<N,N,T>) -> Scalar<T>
```
Trace of square tensor.

```
tensor.symmetric(t: Tensor<N,N,T>) -> Tensor<N,N,T>
```
Symmetric part: `(t + t^T) / 2`.

```
tensor.antisymmetric(t: Tensor<N,N,T>) -> Tensor<N,N,T>
```
Antisymmetric part: `(t - t^T) / 2`.

```
tensor.outer(a: Vec<N,T>, b: Vec<M,U>) -> Tensor<N,M,T*U>
```
Outer product.

---

## 7. Random Number Generation (`rng.*`)

Deterministic pseudo-random number generation derived from the world seed.

Every signal, field, and member has an **implicit backing stream** derived from the world seed and the primitive's path. For members, the entity ID is also incorporated. Streams advance with each `rng.*` call and never reset.

All functions have two forms:
- Without stream parameter: uses the implicit backing stream
- With stream parameter: uses the specified stream

### 7.1 Stream Derivation

```
rng.derive(label: String) -> RngStream
```
Create a substream by mixing the label hash into the current backing stream state.

Use this when you need multiple independent random sequences within a single primitive.

**Example:**
```
signal terra.plate.initial_state {
  : PlateState
  resolve {
    let pos_stream = rng.derive("position") in
    let vel_stream = rng.derive("velocity") in
    PlateState {
      position: rng.in_sphere(pos_stream) * 1000 <km>,
      velocity: rng.unit_vec3(vel_stream) * 0.01 <m/s>
    }
  }
}
```

---

### 7.2 Uniform Distribution

```
rng.uniform() -> Scalar<1>
rng.uniform(stream: RngStream) -> Scalar<1>
```
Uniform random value in [0, 1).

```
rng.uniform_range(min: Scalar<T>, max: Scalar<T>) -> Scalar<T>
rng.uniform_range(stream: RngStream, min: Scalar<T>, max: Scalar<T>) -> Scalar<T>
```
Uniform random value in [min, max). Result has same unit as inputs.

**Example:**
```
member terra.plate.thickness {
  : Scalar<km>
  resolve {
    rng.uniform_range(30 <km>, 100 <km>)
  }
}
```

---

### 7.3 Normal Distribution

```
rng.normal() -> Scalar<1>
rng.normal(stream: RngStream) -> Scalar<1>
```
Standard normal distribution N(0, 1).

```
rng.normal(mean: Scalar<T>, stddev: Scalar<T>) -> Scalar<T>
rng.normal(stream: RngStream, mean: Scalar<T>, stddev: Scalar<T>) -> Scalar<T>
```
Normal distribution N(mean, stddev). Result has same unit as inputs.

**Example:**
```
member terra.plate.density {
  : Scalar<kg/m³>
  resolve {
    rng.normal(2800 <kg/m³>, 200 <kg/m³>)
  }
}
```

---

### 7.4 Geometric Sampling

```
rng.unit_vec2() -> Vec2<1>
rng.unit_vec2(stream: RngStream) -> Vec2<1>
```
Uniform random direction on the unit circle.

```
rng.unit_vec3() -> Vec3<1>
rng.unit_vec3(stream: RngStream) -> Vec3<1>
```
Uniform random direction on the unit sphere.

```
rng.unit_quat() -> Quat
rng.unit_quat(stream: RngStream) -> Quat
```
Uniform random rotation (uniformly distributed over SO(3)).

```
rng.in_disk() -> Vec2<1>
rng.in_disk(stream: RngStream) -> Vec2<1>
```
Uniform random point inside the unit disk.

```
rng.in_sphere() -> Vec3<1>
rng.in_sphere(stream: RngStream) -> Vec3<1>
```
Uniform random point inside the unit sphere.

**Example:**
```
signal terra.plate.seed_direction {
  : Vec3<1>
  phase: configure
  resolve {
    rng.unit_vec3()
  }
}

member terra.plate.angular_velocity {
  : Vec3<rad/s>
  resolve {
    rng.unit_vec3() * rng.normal(0.001 <rad/s>, 0.0002 <rad/s>)
  }
}
```

---

### 7.5 Discrete Sampling

```
rng.bool(probability: Scalar<1>) -> Bool
rng.bool(stream: RngStream, probability: Scalar<1>) -> Bool
```
Bernoulli trial. Returns true with given probability.

```
rng.int_range(min: Int, max: Int) -> Int
rng.int_range(stream: RngStream, min: Int, max: Int) -> Int
```
Uniform random integer in [min, max] (inclusive).

```
rng.weighted_choice(weights: Seq<Scalar<1>>) -> Int
rng.weighted_choice(stream: RngStream, weights: Seq<Scalar<1>>) -> Int
```
Random index selected with probability proportional to weights. Weights do not need to sum to 1.

**Example:**
```
member terra.plate.type {
  : PlateType
  resolve {
    let idx = rng.weighted_choice([0.7, 0.3]) in  # 70% oceanic, 30% continental
    if idx == 0 then PlateType.Oceanic else PlateType.Continental
  }
}
```

---

### 7.6 Backing Stream Model

The implicit backing stream for each primitive is derived as:

```
world_seed
  └─> primitive_path ("terra.plate.velocity")
        └─> entity_id (for members)
              └─> advances with each rng.* call, never resets
                    └─> rng.derive("label") creates substream via state mixing
```

**Key properties:**
- Streams never reset (no arbitrary reset boundaries)
- Each `rng.*` call advances the stream automatically
- Substreams are derived by mixing label hash into parent state
- Determinism preserved because call sequence is deterministic
- Members automatically get per-entity streams (same code, different results per entity)

---

## Quick Reference

### Most Common Functions

| Operation | Function |
|-----------|----------|
| Square root | `maths.sqrt(x)` |
| Clamp ⚠️ | `maths.clamp(x, lo, hi)` (requires `: uses(maths.clamping)`) |
| Linear interpolation | `maths.lerp(a, b, t)` |
| Dot product | `vector.dot(a, b)` |
| Cross product | `vector.cross(a, b)` |
| Normalize | `vector.normalize(v)` |
| Length | `vector.length(v)` |
| Euler integration | `dt.integrate(prev, rate)` |
| Exponential decay | `dt.decay(value, halflife)` |
| Relaxation | `dt.relax(current, target, tau)` |
| Phase wrap | `dt.advance_phase(phase, omega)` |
| Random value | `rng.uniform()` |
| Random direction | `rng.unit_vec3()` |
| Normal distribution | `rng.normal(mean, stddev)` |

### dt.* Summary

| Function | Use Case |
|----------|----------|
| `dt.integrate` | Position from velocity, accumulation over time |
| `dt.decay` | Radioactive decay, cooling toward ambient |
| `dt.relax` | Equilibration, smoothing toward target |
| `dt.accumulate` | Bounded accumulators (resources, populations) |
| `dt.advance_phase` | Orbital anomaly, oscillator phase |
| `dt.damp` | Friction, viscous damping |

### rng.* Summary

| Function | Use Case |
|----------|----------|
| `rng.uniform` | Random value in [0, 1) |
| `rng.uniform_range` | Random value in [min, max) with units |
| `rng.normal` | Gaussian-distributed values |
| `rng.unit_vec3` | Random direction on sphere |
| `rng.unit_quat` | Random rotation |
| `rng.in_sphere` | Random point in unit sphere |
| `rng.bool` | Bernoulli trial |
| `rng.weighted_choice` | Select from weighted options |
| `rng.derive` | Create independent substream |

---

## See Also

- `@docs/execution/kernels.md` — Kernel architecture and GPU execution
- `@docs/dsl/functions.md` — User-defined functions and templates
- `@docs/dsl/types-and-units.md` — Type system and dimensional analysis
