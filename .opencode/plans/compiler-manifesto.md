# Compiler Manifesto

> **One structure. Single source. Composed capabilities. Types that prove correctness.**

---

## The Prime Directive

### Explicit Over Implicit

**Ambiguity is a bug.** Every construct must have exactly one interpretation.

- **No silent coercion** — types don't silently become other types
- **No silent fallback** — unresolved paths are errors, not assumptions
- **No silent collision** — if two things could be confused, forbid the collision
- **No inference where declaration is cheap** — write the type name, it's documentation

When there's a choice between implicit convenience and explicit clarity, choose explicit.

```cdsl
// Explicit type on struct literal (even when "obvious")
signal orbit : OrbitalElements {
    resolve {
        OrbitalElements {           // <- type stated, not inferred
            semi_major: 1000.0<m>,
            eccentricity: 0.1,
        }
    }
}

// Collision forbidden at compile time
signal orbit : OrbitalElements     // has field .semi_major
signal orbit.semi_major : Scalar   // ERROR: ambiguous with orbit.semi_major field
```

**The test:** If a human reader could misinterpret it, the compiler should reject it.

---

## The Four Pillars

### 1. Unified Node

Everything is `Node<I>` with a `RoleId`.

```rust
pub struct Node<I: Index = ()> {
    // Identity
    pub path: Path,
    pub span: Span,
    pub file: Option<PathBuf>,
    
    // Documentation
    pub doc: Option<String>,
    pub title: Option<String>,
    pub symbol: Option<String>,
    
    // Role + role-specific data (prevents invalid states)
    pub role: RoleData,
    
    // Common capabilities (used by multiple roles)
    pub scoping: Option<Scoping>,
    pub assertions: Vec<Assertion>,
    pub executions: Vec<Execution>,
    pub stratum: Option<StratumId>,
    pub output: Option<Type>,   // what it produces (Kernel, User, Bool, etc.)
    pub inputs: Option<Type>,   // what it receives (typically UserType)
    
    // Indexing
    pub index: I,   // () for global, EntityId for per-entity
}

/// Role-specific data — makes invalid states unrepresentable
pub enum RoleData {
    Signal,
    Field { reconstruction: Option<ReconstructionHint> },
    Operator,
    Impulse { payload: Option<Type> },
    Fracture,
    Chronicle,
}

impl RoleData {
    pub fn id(&self) -> RoleId {
        match self {
            Self::Signal => RoleId::Signal,
            Self::Field { .. } => RoleId::Field,
            Self::Operator => RoleId::Operator,
            Self::Impulse { .. } => RoleId::Impulse,
            Self::Fracture => RoleId::Fracture,
            Self::Chronicle => RoleId::Chronicle,
        }
    }
}
```

**`RoleData`** — what it is + role-specific data (Signal, Field with reconstruction, etc.)
**`I: Index`** — where it lives (`()` = global, `EntityId` = per-entity)

Member is `Node<EntityId>` with `role: RoleData::Signal`. Not a separate type.

**Why `RoleData` enum instead of `Option<T>` fields:**
- Invalid states are unrepresentable (Signal can't have reconstruction)
- Role-specific data is co-located with the role tag
- `role.id()` still gives `RoleId` for registry lookup

**Output, Inputs, Payload — all just Type:**
- No separate `Input` or `Payload` structs
- `inputs.field` → `FieldAccess { object: Inputs, field }` (already works)
- `payload.field` → `FieldAccess { object: Payload, field }` (already works)
- Reuses `UserType` and `FieldAccess` machinery — KISS/DRY

**Role via Compile-Time Registry (not type parameter):**

```rust
// RoleId — indexing enum, NOT polymorphism
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum RoleId {
    Signal = 0,
    Field = 1,
    Operator = 2,
    Impulse = 3,
    Fracture = 4,
    Chronicle = 5,
}

impl RoleId {
    pub const COUNT: usize = 6;
    
    #[inline]
    pub const fn spec(self) -> &'static RoleSpec {
        &ROLE_REGISTRY[self as usize]
    }
}

// PhaseSet — bitset for allowed phases (u16 for 9 phases)
#[derive(Copy, Clone, Default)]
pub struct PhaseSet(u16);

impl PhaseSet {
    pub const fn empty() -> Self { Self(0) }
    pub const fn with(self, phase: Phase) -> Self { Self(self.0 | (1 << phase as u16)) }
    pub const fn contains(self, phase: Phase) -> bool { (self.0 & (1 << phase as u16)) != 0 }
}

// CapabilitySet — bitset for context capabilities
#[derive(Copy, Clone, Default)]
pub struct CapabilitySet(u16);

impl CapabilitySet {
    pub const fn empty() -> Self { Self(0) }
    pub const fn with(self, cap: Capability) -> Self { Self(self.0 | (1 << cap as u8)) }
    pub const fn contains(self, cap: Capability) -> bool { (self.0 & (1 << cap as u8)) != 0 }
}

// RoleSpec — all const, built at compile time
pub struct RoleSpec {
    pub name: &'static str,
    pub allowed_phases: PhaseSet,
    pub phase_capabilities: [CapabilitySet; Phase::COUNT],
    pub has_reconstruction: bool,
}

// Compile-time registry — zero runtime cost
pub static ROLE_REGISTRY: [RoleSpec; RoleId::COUNT] = [
    // Signal
    RoleSpec {
        name: "signal",
        allowed_phases: PhaseSet::empty()
            .with(Phase::Resolve)
            .with(Phase::Assert),
        phase_capabilities: phase_caps!(
            Resolve => [Scoping, Signals, Prev, Inputs, Dt],
            Assert => [Scoping, Signals, Prev, Current, Dt]
        ),
        has_reconstruction: false,
    },
    // Field
    RoleSpec {
        name: "field",
        allowed_phases: PhaseSet::empty()
            .with(Phase::Measure)
            .with(Phase::Assert),
        phase_capabilities: phase_caps!(
            Measure => [Scoping, Signals, Dt],
            Assert => [Scoping, Signals, Current, Dt]
        ),
        has_reconstruction: true,
    },
    // Operator — causal only, no observer phases
    RoleSpec {
        name: "operator",
        allowed_phases: PhaseSet::empty()
            .with(Phase::Configure)
            .with(Phase::Collect)
            .with(Phase::Resolve)
            .with(Phase::Fracture),
        phase_capabilities: phase_caps!(/* per-phase */),
        has_reconstruction: false,
    },
    // Impulse
    RoleSpec {
        name: "impulse",
        allowed_phases: PhaseSet::empty().with(Phase::Collect),
        phase_capabilities: phase_caps!(Collect => [Scoping, Signals, Dt, Payload, Emit]),
        has_reconstruction: false,
    },
    // Fracture
    RoleSpec {
        name: "fracture",
        allowed_phases: PhaseSet::empty()
            .with(Phase::Fracture)
            .with(Phase::Assert),
        phase_capabilities: phase_caps!(
            Fracture => [Scoping, Signals, Dt],
            Assert => [Scoping, Signals, Dt]
        ),
        has_reconstruction: false,
    },
    // Chronicle — observer-only, no DAG impact
    RoleSpec {
        name: "chronicle",
        allowed_phases: PhaseSet::empty().with(Phase::Measure),
        phase_capabilities: phase_caps!(Measure => [Scoping, Signals, Dt]),
        has_reconstruction: false,
    },
];

// **Chronicle scheduling:** Chronicles execute in a separate sub-phase AFTER
// all Fields complete. They are not interleaved in the Field DAG.
// This ensures removing Chronicles has zero effect on Field execution.
// See "Measure Phase Sub-Phases" in Execution Model section.
```

**Why compile-time registry:**
- Zero runtime cost (static array, inlined access)
- Data-driven rules (no match-as-polymorphism)
- Extensibility: add `RoleSpec` entry, not code changes
- Fits "structure from data, not code"

**Phase:** See "Types Prove Correctness" section for full definition.
- **Init:** CollectConfig, Initialize, WarmUp
- **Tick:** Configure, Collect, Resolve, Fracture, Measure, Assert

---

### 2. Capability Composition

Contexts are built from orthogonal capabilities, not inheritance.

```rust
// Capabilities — compose as needed
trait HasScoping { fn config(&self, path: &Path) -> Value; fn constant(&self, path: &Path) -> Value; }
trait HasSignals { fn signal(&self, path: &Path) -> Value; }
trait HasPrev    { fn prev(&self) -> &Value; }
trait HasCurrent { fn current(&self) -> &Value; }
trait HasInputs  { fn inputs(&self) -> f64; }
trait HasDt      { fn dt(&self) -> f64; }
trait HasPayload { fn payload(&self) -> &Value; }
trait CanEmit    { fn emit(&self, target: &Path, value: Value); }
trait HasIndex   { fn self_field(&self, name: &str) -> Value; }
```

**Phase contexts implement only what they provide:**

| Phase | Scoping | Signals | Prev | Current | Inputs | Dt | Payload | Emit |
|-------|---------|---------|------|---------|--------|-----|---------|------|
| *Initialization (pre-DAG)* |
| CollectConfig | ✓ | - | - | - | - | - | - | - |
| Initialize | ✓ | - | - | - | - | - | - | - |
| WarmUp | *(runs tick phases until stable)* |
| *Tick Execution (DAG)* |
| Configure | ✓ | - | - | - | - | - | - | - |
| Collect | ✓ | ✓ | - | - | - | ✓ | ✓ | ✓ |
| Resolve | ✓ | ✓ | ✓ | - | ✓ | ✓ | - | - |
| Fracture | ✓ | ✓ | - | - | - | ✓ | - | - |
| Measure | ✓ | ✓ | - | ✓ | - | ✓ | - | - |
| Assert | ✓ | ✓ | ✓ | ✓ | - | ✓ | - | - |

**Note:** This table shows *maximum* capabilities per phase. Each Role gets a subset — see `RoleSpec.phase_capabilities`. For example, Field in Measure doesn't get `Current` (it's producing the value), but gets `Current` in Assert (to validate what was emitted).

**Statement blocks:** Phases with `Emit` capability (`Collect`, impulse `Apply`) use statement blocks (`ExecutionBody::Statements`). All other phases use expression blocks (`ExecutionBody::Expr`). See Rule 10.

**Signal values by phase:**
- **Collect:** Signals return **previous tick values** (not yet resolved this tick)
- **Resolve:** Signals being resolved read `prev`; other signals return previous tick values
- **Fracture/Measure/Assert:** Signals return **current tick values** (just resolved)

**Index is orthogonal.** `Indexed<C>` wraps any context when the node has `I = EntityId`, adding `HasIndex`. Not phase-dependent.

**Compile-time enforcement:** Use `prev` in a measure block → type error, not runtime dummy.

---

### 3. Types Prove Correctness

Every expression carries its type. Type errors are compile errors.

```rust
struct TypedExpr {
    expr: ExprKind,
    ty: Type,
    span: Span,  // source location for error messages
}
```

**Type — what a value is:**

```rust
enum Type {
    Kernel(KernelType),      // numeric computation types
    User(UserTypeId),        // user-defined product types
    Bool,                    // true/false (distinct from Scalar)
    Unit,                    // void (for emit, side effects)
    Seq(Box<Type>),          // collection from map (see note below)
}
```

**Seq<T> usage:** `Seq` is produced by `map` and must be consumed by an aggregate:

```cdsl
// Seq as intermediate — sum consumes it
let total = sum(map(plates, |p| p.area))

// Seq cannot be stored in signals (no Seq-valued signals)
// Seq cannot be returned from resolve blocks
// Seq exists only within expressions
```

**Seq<T> constraints:**

1. **Intermediate-only** — `Seq<T>` cannot be:
   - A signal type (`signal masses : Seq<Scalar<kg>>` → error)
   - A `let` binding unless immediately consumed (`let s = map(...); s` → error)
   - Returned from resolve blocks
   - Stored in any form
   
2. **Purity required** — aggregate bodies must be pure (no effects):
   ```cdsl
   // OK — pure body
   sum(bodies, |b| b.mass)
   
   // ERROR — effectful body
   sum(bodies, |b| { emit(x, b.mass); b.mass })
   ```

3. **Capture rules** — lambda bodies can only reference:
   - `self` (current entity, if in entity context)
   - Bound element (`b` in `|b| ...`)
   - `const`/`config` values
   - Resolved signals
   - Outer `let` bindings (immutable)
   
   Cannot capture: mutable state, other iterators, ambiguous lifetime refs.

4. **Determinism** — iteration and reduction order:
   - All iteration uses **lexical `InstanceId` order** (deterministic)
   - Floating-point reductions use **fixed-tree reduction** for bitwise stability
   - `fold` with non-commutative function: order = InstanceId order

**Compiler check:** `Seq<T>` type only valid as direct argument to aggregate functions. Any other use → `SeqNotConsumed` error.

**KernelType — numeric types with physics:**

```rust
struct KernelType {
    shape: Shape,
    unit: Unit,             // always explicit; dimensionless = all zeros
    bounds: Option<Bounds>, // None = unbounded
}
```

**Shape — geometric structure:**

```rust
enum Shape {
    Scalar,                        // rank 0, dims = []
    Vector { dim: u8 },            // rank 1, dims = [dim]
    Matrix { rows: u8, cols: u8 }, // rank 2, dims = [rows, cols]
    Tensor { dims: Vec<u8> },      // rank n, any higher-dimensional tensor
    
    // === Structured types ===
    Complex,                            // real + imaginary (2 components)
    Quaternion,                         // rotation representation (4 components)
    SymmetricMatrix { dim: u8 },        // n(n+1)/2 independent components
    SkewSymmetricMatrix { dim: u8 },    // n(n-1)/2 independent components
}

impl Shape {
    // Uniform access for operations that don't care about named variants
    fn dims(&self) -> Vec<u8> {
        match self {
            Scalar => vec![],
            Vector { dim } => vec![*dim],
            Matrix { rows, cols } => vec![*rows, *cols],
            Tensor { dims } => dims.clone(),
        }
    }
    
    fn rank(&self) -> usize { self.dims().len() }
}
```

Named variants for the 95% case (readable errors, self-documenting).
`Tensor` for ML, elasticity tensors, or anything rank 3+.

**Structured types:**
- `Complex` — supports complex arithmetic kernels (`complex.mul`, `complex.conj`, `complex.abs`, etc.)
- `Quaternion` — rotation operations (`quat.slerp`, `quat.rotate`, `quat.conjugate`, etc.)
- `SymmetricMatrix` — stores only upper/lower triangle, symmetric operations
- `SkewSymmetricMatrix` — antisymmetric tensors (e.g., angular velocity tensor)

**Structured type interpolation:**
- `Quaternion` uses SLERP (spherical linear interpolation), not component-wise
- `Complex` interpolation is component-wise (real and imaginary separately)
- Symmetric/skew matrices interpolate independent components only

**Unit — dimensional analysis (SI base units with exponents):**

```rust
struct Unit {
    kind: UnitKind,
    dims: UnitDimensions,
}

pub enum UnitKind {
    Multiplicative,                 // standard SI (m, kg, s, etc.)
    Affine { offset: f64 },         // temperature scales (°C = K - 273.15)
    Logarithmic { base: f64 },      // dB, pH, etc.
}

pub struct UnitDimensions {
    length: i8,      // m
    mass: i8,        // kg
    time: i8,        // s
    temperature: i8, // K
    current: i8,     // A
    amount: i8,      // mol
    luminosity: i8,  // cd
    angle: i8,       // rad (for convenience)
}

impl Unit {
    pub const DIMENSIONLESS: Unit = Unit { 
        kind: UnitKind::Multiplicative,
        dims: UnitDimensions { 
            length: 0, mass: 0, time: 0, temperature: 0, 
            current: 0, amount: 0, luminosity: 0, angle: 0 
        }
    };
}

// Examples:
// m/s     = { kind: Multiplicative, dims: { length: 1, time: -1, ..0 } }
// K       = { kind: Multiplicative, dims: { temperature: 1, ..0 } }
// °C      = { kind: Affine { offset: 273.15 }, dims: { temperature: 1, ..0 } }
// dB      = { kind: Logarithmic { base: 10.0 }, dims: dimensionless }
```

**Unit algebra special cases:**
- `T1 - T2` where both are Affine → result is Multiplicative (temperature difference)
- `log(x)` requires dimensionless input → result is Logarithmic
- `exp(x)` requires Logarithmic or dimensionless → result is dimensionless
- Mixing Affine/Logarithmic in addition → compile error

**Bounds — value constraints:**

```rust
struct Bounds {
    min: Option<f64>,  // None = unbounded below
    max: Option<f64>,  // None = unbounded above
}
// 0..1     = { min: Some(0.0), max: Some(1.0) }
// 100..    = { min: Some(100.0), max: None }
// ..10000  = { min: None, max: Some(10000.0) }
```

**UserType — product types:**

```rust
struct UserType {
    id: UserTypeId,
    name: Path,
    fields: Vec<(String, Type)>,  // fields can nest user types
}
```

**Identifiers:**

```rust
struct Path(Vec<String>);  // e.g., ["terra", "core", "temperature"]

// IDs are validated paths with type safety
struct SignalId(Path);
struct EntityId(Path);
struct UserTypeId(Path);
struct StratumId(Path);

// KernelId — explicit namespace + name (always two-level)
struct KernelId {
    namespace: &'static str,  // "maths", "vector", "compare", "logic"
    name: &'static str,       // "sin", "dot", "lt", "and"
}
```

**Span — source location (minimal struct, debug via lookup):**

```rust
struct Span {
    file_id: u16,      // index into SourceMap.files
    start: u32,        // byte offset
    end: u32,          // byte offset
    start_line: u16,   // cached line number (no lookup needed in prod)
}

struct SourceMap {
    files: Vec<SourceFile>,
}

struct SourceFile {
    path: PathBuf,
    source: String,        // original text (can drop in prod if needed)
    line_starts: Vec<u32>, // byte offset of each line start
}

impl SourceMap {
    // Debug lookups
    fn file_path(&self, span: &Span) -> &Path { &self.files[span.file_id].path }
    fn line_col(&self, span: &Span) -> (u32, u32) { /* binary search line_starts */ }
    fn snippet(&self, span: &Span) -> &str { /* source[start..end] */ }
}
```

Covers: error messages, LSP (URI + Range), DAP (breakpoints, stack frames).
Prod minimum: file_id + start_line. Debug: full source lookup.

**Scoping — config and const blocks:**

```rust
struct Scoping {
    config: Vec<ConfigEntry>,
    consts: Vec<ConstEntry>,
}

struct ConfigEntry {
    name: String,
    ty: Type,
    default: Option<TypedExpr>,  // can be omitted, must be provided by scenario
    span: Span,
}

struct ConstEntry {
    name: String,
    ty: Type,
    value: TypedExpr,  // required, immutable
    span: Span,
}
```

**Config — layered override:**
- Scenario → Module → Node (later layers override earlier)
- Like config frameworks: base defaults, environment overrides, scenario overrides
- Missing value without default at runtime → error

**Const — no layering:**
- Duplicate path at any level → compile error
- Truly constant (physical constants, mathematical constants)
- No override mechanism

```cdsl
signal core.temp : Scalar<K> {
    config {
        initial_temp: Scalar<K> = 5000.0  // default, overridable
        cooling_rate: Scalar<K/s>         // no default, scenario must provide
    }
    
    const {
        BOLTZMANN: Scalar<J/K> = 1.380649e-23  // immutable, duplicate = error
    }
}
```

**Assertion — runtime validation:**

```rust
/// Shared validation rule structure (used by assertions and analyzer validations)
struct ValidationRule {
    condition: TypedExpr,  // must evaluate to Bool
    severity: Severity,
    message: String,
    span: Span,
}

enum Severity {
    Fatal,  // halt simulation immediately
    Error,  // log, mark failed, continue
    Warn,   // log, continue
}

// Type aliases for context clarity
type Assertion = ValidationRule;
type AnalyzerValidation = ValidationRule;
```

```cdsl
signal core.temp : Scalar<K, 100..10000> {
    resolve { ... }
    
    assert {
        current > 0 : fatal, "temperature must be positive"
        maths.abs(current - prev) / prev < 0.1 : warn, "changing too fast"
    }
}
```

Assertions run in the **Assert phase** (after Measure) which has `current`, `prev`, and `dt` — enabling delta validation like `abs(current - prev) < threshold`.
Bounds on type (`100..10000`) are validated separately, automatically.

**Phase — simulation lifecycle phases:**

```rust
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Phase {
    // Initialization (pre-DAG, run once at start)
    CollectConfig = 0,  // gather config values from scenario
    Initialize = 1,     // set initial signal values from configs
    WarmUp = 2,         // run ticks until convergence (see WarmUpPolicy)
    
    // Tick execution (DAG, every simulation tick)
    Configure = 3,      // finalize per-tick execution context
    Collect = 4,        // gather inputs to signals, apply impulses
    Resolve = 5,        // compute authoritative state
    Fracture = 6,       // detect and respond to tension
    Measure = 7,        // produce observations
    Assert = 8,         // validate invariants (has Prev + Current for deltas)
}

impl Phase {
    pub const COUNT: usize = 9;
    
    pub const fn is_init(self) -> bool {
        matches!(self, Self::CollectConfig | Self::Initialize | Self::WarmUp)
    }
    
    pub const fn is_tick(self) -> bool {
        !self.is_init()
    }
}
```

> **Note:** Foundation docs (`docs/execution/phases.md`) currently define 5 tick phases.
> Will be updated to match this spec after sign-off.

**Capability — context capabilities for phase execution:**

```rust
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
#[repr(u8)]
pub enum Capability {
    Scoping = 0,   // config/const access
    Signals = 1,   // signal read access
    Prev = 2,      // previous tick value
    Current = 3,   // just-resolved value
    Inputs = 4,    // accumulated inputs
    Dt = 5,        // time step
    Payload = 6,   // impulse payload
    Emit = 7,      // emit to signal
    Index = 8,     // entity self-reference
}

impl Capability {
    pub const COUNT: usize = 9;
}
```

**ReconstructionHint — how to reconstruct continuous fields from samples (Field-specific):**

```rust
pub struct ReconstructionHint {
    pub domain: Domain,
    pub method: InterpolationMethod,
    pub boundary: BoundaryCondition,
    pub conservative: bool,  // preserve integrals (for flux fields like mass, energy)
}

/// Domain determines distance metric and coordinate handling
pub enum Domain {
    Cartesian,                      // Euclidean distance in R^n
    Spherical { radius: f64 },      // Geodesic (great-circle) distance on sphere
}

/// Interpolation kernel applied using domain's distance metric
pub enum InterpolationMethod {
    // === Basic ===
    NearestNeighbor,                            // closest sample
    Linear,                                     // linear blend by distance
    Cubic,                                      // cubic spline
    
    // === Scattered Data (weighted) ===
    IDW { power: f64 },                         // inverse distance weighting
    RBF { kernel: RBFKernel },                  // radial basis functions
    NaturalNeighbor,                            // Voronoi-based, C1 continuous
    
    // === Geostatistical ===
    Kriging { variogram: Variogram },           // optimal for spatially correlated data
    
    // === Spectral (global) ===
    SphericalHarmonics { max_degree: u32 },     // for spherical domains
    
    // === Local Approximation ===
    MLS { degree: u8 },                         // moving least squares
}

pub enum RBFKernel {
    Gaussian { epsilon: f64 },
    Multiquadric { epsilon: f64 },
    InverseMultiquadric { epsilon: f64 },
    ThinPlateSpline,
    Polyharmonic { k: u8 },
}

pub enum Variogram {
    Spherical { range: f64, sill: f64 },
    Exponential { range: f64, sill: f64 },
    Gaussian { range: f64, sill: f64 },
}

pub enum BoundaryCondition {
    // === Coordinate-based ===
    Clamp,                              // clamp to nearest edge value
    Wrap,                               // periodic (e.g., longitude)
    Mirror,                             // reflect at boundary
    NoBoundary,                         // closed manifold (sphere)
    
    // === PDE boundary conditions ===
    Dirichlet { value: f64 },           // fixed value at boundary
    Neumann { gradient: f64 },          // fixed gradient at boundary
    Robin { alpha: f64, beta: f64 },    // α*u + β*∂u/∂n = γ
    
    // === Extrapolation ===
    Extrapolate { order: u8 },          // polynomial extrapolation
}
```

**Per-entity fields:** For fields on entities (e.g., `plate.stress`), reconstruction means **spatial interpolation between entity positions**, not interpolation across entity index. This requires:
- The entity must have a **position member signal** (e.g., `plate.centroid : Vec3<m>`)
- Lens queries at arbitrary coordinates find nearby entities and interpolate their field values

**Per-entity fields without position:**
- If entity has no position member signal, spatial reconstruction is undefined
- Lens query by coordinate → `MissingPositionForReconstruction` error
- Lens query by entity ID → returns value directly (no interpolation needed)
- Future extension: graph-based interpolation using entity adjacency relationships

> **Current status:** DSL syntax for ReconstructionHint not yet implemented. Lens only supports `NearestNeighbor`. See `continuum-dly7`.

**ValidationError — structured validation failure:**

```rust
pub struct ValidationError {
    pub kind: ValidationErrorKind,
    pub message: String,
    pub span: Span,
    pub hints: Vec<String>,
}

pub enum ValidationErrorKind {
    // Type errors
    TypeMismatch { expected: Type, found: Type },
    ShapeMismatch { expected: Shape, found: Shape },
    UnitMismatch { expected: Unit, found: Unit },
    
    // Resolution errors
    UnresolvedPath(Path),
    AmbiguousPath { path: Path, candidates: Vec<Path> },
    PathCollision { path: Path, existing: Span },
    
    // Phase/capability errors
    InvalidPhase { role: RoleId, phase: Phase },
    MissingCapability { phase: Phase, capability: Capability },
    
    // Effect errors
    EffectInConditional { 
        effect: &'static str,   // which effect kernel (emit, spawn, log)
        form: &'static str,     // which eager form (select, and, or)
    },
    UnitInExpressionPosition { span: Span },       // Unit-typed expr in pure context
    
    // Seq errors
    SeqNotConsumed { span: Span },                 // Seq<T> not immediately consumed by aggregate
    EffectInAggregate { effect: &'static str },   // effectful expression in aggregate body
    InvalidCapture { name: String },              // lambda captures disallowed binding
    
    // Kernel errors
    UnknownKernel(KernelId),
    WrongArgCount { kernel: KernelId, expected: usize, found: usize },
    
    // Config/scoping errors
    MissingConfigValue { path: Path },           // no default, not provided by scenario
    DuplicateConst { path: Path, existing: Span },
    
    // Bounds errors (runtime, but structured)
    BoundsViolation { path: Path, value: f64, bounds: Bounds },
    
    // Field/reconstruction errors
    UnsupportedReconstruction { method: String },
    MissingPositionForReconstruction { entity: EntityId },
    
    // Other
    DuplicateDefinition { path: Path, existing: Span },
    CyclicDependency { cycle: Vec<Path> },
}
```

**Outputs, inputs, payload — all just Type (KISS/DRY):**

```cdsl
type OrbitalElements {
    semi_major: Scalar<m>
    eccentricity: Scalar<>
    inclination: Scalar<rad>
}

signal body.orbit : OrbitalElements {
    resolve { ... }
}

// Output access: body.orbit.semi_major -> Scalar<m>
// Prev access:   prev.eccentricity -> Scalar<>

// Inputs declared inline become an anonymous UserType
signal body.acceleration : Vec3<m/s2> {
    : uses(gravity: Vec3<m/s2>, thrust: Vec3<N>)
    
    resolve {
        // inputs has type { gravity: Vec3<m/s2>, thrust: Vec3<N> }
        inputs.gravity + inputs.thrust / body.mass
    }
}

// Payload works the same way (apply is a statement block)
impulse apply_force {
    : payload(magnitude: Scalar<N>, direction: Vec3<>)
    
    apply {
        // Statement block — each line is a Unit-returning statement
        // payload has type { magnitude: Scalar<N>, direction: Vec3<> }
        emit(target.velocity, payload.direction * payload.magnitude / target.mass)
    }
}
```

**Shape validated at compile time:**
- `Vec2 + Vec3` → error
- `Mat3 * Vec2` → error (dimension mismatch)
- `Scalar * Vec3` → `Vec3` (broadcast)
- `signal.field` → resolves through user type fields

**Units validated at compile time:**
- `m/s + K` → error
- `m / s` → `m/s`
- `sin(radians)` → dimensionless

**Bounds tracked, validated at runtime** — on kernel types within user types, checked after resolve.

---

### 4. Single Source

One struct flows through all compiler passes. No AST → IR copying.

**Pipeline Traits (supertrait hierarchy):**

```rust
trait Named {
    fn path(&self) -> &Path;
    fn span(&self) -> &Span;
}

trait Parsed: Named {
    fn type_expr(&self) -> Option<&TypeExpr>;
    fn execution_exprs(&self) -> &[(String, Expr)];
}

trait Resolved: Parsed {
    fn output(&self) -> Option<&Type>;
    fn inputs(&self) -> &[(String, Type)];
}

trait Validated: Resolved {
    fn validation_errors(&self) -> &[ValidationError];
}

trait Compiled: Validated {
    fn executions(&self) -> &[Execution];
    fn reads(&self) -> &[Path];
}
```

**Lifecycle fields on Node:**

```rust
struct Node<I: Index = ()> {
    // ... identity, documentation, role ...
    
    // Syntax (cleared after consumption)
    type_expr: Option<TypeExpr>,          // cleared by resolution
    execution_exprs: Vec<(String, Expr)>, // cleared by compilation
    
    // Semantic (set by passes)
    output: Option<Type>,                 // set by resolution
    inputs: Vec<(String, Type)>,          // set by resolution
    validation_errors: Vec<ValidationError>, // set by validation
    executions: Vec<Execution>,           // set by compilation
    reads: Vec<Path>,                     // set by compilation
    
    role: RoleId,
    index: I,
}
```

**Clear after consumption:** Syntax fields become `None`/empty after the pass that consumes them. State is explicit:

- `type_expr.is_some()` → not yet resolved
- `type_expr.is_none() && output.is_some()` → resolved
- `execution_exprs.is_empty() && !executions.is_empty()` → compiled

**Traits are read-only.** Mutation happens on the concrete struct. Pipeline functions take `&mut Node<I>`.

**Two orthogonal trait hierarchies:**

| Hierarchy | Purpose | Traits |
|-----------|---------|--------|
| Pipeline | Data lifecycle | `Named → Parsed → Resolved → Validated → Compiled` |
| Context | Execution capabilities | `HasScoping`, `HasSignals`, `HasPrev`, `HasDt`, ... |

Pipeline traits describe what data the node has. Context traits describe what's available when running executions.

---

## Structural Declarations

Not everything is a `Node`. Some concepts define structure rather than execution.

**Common fields:** All structural declarations share `path: Path`, `span: Span`, `doc: Option<String>`. Implementation may extract a `DeclarationHeader` struct, but each has a distinct id type (EntityId, StratumId, EraId, AnalyzerId).

### Entity

An Entity declares a **namespace + index type** for per-entity primitives.

```rust
pub struct Entity {
    pub id: EntityId,
    pub path: Path,
    pub span: Span,
    pub doc: Option<String>,
}
```

Entity itself has no executions. It creates an index type that parameterizes `Node<I>`:
- `Node<()>` — global primitive
- `Node<EntityId>` — per-entity primitive (member)

Any Role can be per-entity: Signal, Field, Fracture, Operator.
Impulse and Chronicle are always global.

```cdsl
entity plate {
    member area : Scalar<m2>       // Signal per plate
    field stress : Scalar<Pa>      // Field per plate
    fracture rift { ... }          // Fracture per plate
    operator apply_friction { ... } // Operator per plate
}
```

**Entity lifecycle:**
- Entity instance count is **fixed at scenario initialization**
- **No runtime creation** (spawn) — not yet supported
- **No runtime destruction** (destroy) — not yet supported
- Instance IDs are stable throughout simulation
- `prev` is always valid (no "newborn" entity edge case)

> **Future:** Dynamic entity lifecycle tracked in `continuum-ojgp`.

### Stratum

A Stratum declares an **execution lane with cadence**.

```rust
pub struct Stratum {
    pub id: StratumId,
    pub path: Path,
    pub cadence: u32,  // stride: execute every N ticks (1 = every tick)
    pub span: Span,
    pub doc: Option<String>,
}

pub struct StratumId(Path);
```

Nodes are assigned to a stratum via `stratum: Option<StratumId>`.
Strata define *when* nodes execute, not *what* they do.

```cdsl
stratum fast { cadence: 1 }      // every tick
stratum slow { cadence: 100 }    // every 100 ticks

signal temperature : Scalar<K> {
    : stratum(fast)
    resolve { ... }
}
```

### Era

An Era declares an **execution policy regime**.

```rust
pub struct Era {
    pub id: EraId,
    pub path: Path,
    pub dt: TypedExpr,  // base timestep (may be expression)
    pub strata_policy: Vec<StratumPolicy>,
    pub transitions: Vec<EraTransition>,
    pub span: Span,
    pub doc: Option<String>,
}

pub struct StratumPolicy {
    pub stratum: StratumId,
    pub active: bool,
    pub cadence_override: Option<u32>,
}

pub struct EraTransition {
    pub target: EraId,
    pub condition: TypedExpr,  // over resolved signals, not fields
    pub span: Span,
}

pub struct EraId(Path);
```

Eras control:
- Base timestep (`dt`)
- Which strata are active
- Cadence overrides for active strata
- Transitions to other eras (signal-driven, deterministic)

```cdsl
era formation {
    : dt(1_000_000<yr>)
    : strata(tectonics: active, climate: gated)
    : transition(stable, when: mantle.temperature < 1500<K>)
}

era stable {
    : dt(1000<yr>)
    : strata(tectonics: active, climate: active)
}
```

### WarmUpPolicy

WarmUp runs tick phases repeatedly until convergence. To ensure determinism and termination:

```rust
pub struct WarmUpPolicy {
    /// Convergence predicate — warm-up ends when this returns true.
    /// Evaluated after each tick over resolved signals.
    pub converged: TypedExpr,  // must be Bool
    
    /// Maximum iterations before forced termination.
    pub max_iterations: u32,
    
    /// What to do if max_iterations reached without convergence.
    pub on_timeout: WarmUpTimeout,
}

pub enum WarmUpTimeout {
    Fault,    // emit fatal fault, halt simulation
    Warn,     // emit warning, continue with current state
}
```

```cdsl
world terra {
    warmup {
        converged: maths.abs(mantle.temperature - prev) < 0.01<K>
        max_iterations: 1000
        on_timeout: fault
    }
}
```

**Determinism:** WarmUp iteration count is part of the replayable seed. Given the same world + scenario + seed, WarmUp always runs the same number of ticks.

---

## Post-Hoc Tooling

Some primitives run outside the simulation DAG entirely.

### Analyzer

An Analyzer is a **pure observer** that runs post-hoc on field snapshots.

```rust
pub struct Analyzer {
    pub id: AnalyzerId,
    pub path: Path,
    pub span: Span,
    pub doc: Option<String>,
    
    pub requires: Vec<FieldId>,  // field dependencies
    pub compute: TypedExpr,      // produces JSON-serializable value
    pub validations: Vec<AnalyzerValidation>,
}

// AnalyzerValidation = ValidationRule (see "Types Prove Correctness" section)

pub struct AnalyzerId(Path);
```

Analyzers:
- Run after simulation (not in any tick phase)
- Access fields **through Lens** (simplified syntax since post-hoc)
- Produce JSON output
- Cannot influence causality
- Have their own context (Lens handles, stats kernels)

**Field access in Analyzers:** `field.X` returns a Lens handle with multiple access patterns:

```cdsl
analyzer example {
    : requires(fields: [elevation, temperature])
    
    compute {
        // 1. Aggregate statistics (over all samples)
        let avg = stats.mean(field.elevation)
        let max_temp = stats.max(field.temperature)
        
        // 2. Point query (with reconstruction)
        let peak_temp = field.temperature.at(lat: 45.0, lon: -122.0)
        
        // 3. Raw samples (explicit access)
        let samples = field.elevation.samples()  // Seq<(Position, Value)>
    }
}
```

**Point query coordinate system:** Inferred from Field's domain:
- Spherical surface → `.at(lat:, lon:)`
- Cartesian volume → `.at(x:, y:, z:)`

Reconstruction uses the Field's `ReconstructionHint` (Linear, Cubic, Spherical, etc.).

```cdsl
analyzer terra.elevation_stats {
    : doc "Statistical summary of elevation distribution"
    : requires(fields: [geophysics.elevation])
    
    compute {
        {
            mean: stats.mean(field.geophysics.elevation),
            std: stats.std(field.geophysics.elevation),
            min: stats.min(field.geophysics.elevation),
            max: stats.max(field.geophysics.elevation),
        }
    }
    
    validate {
        stats.min(field.geophysics.elevation) > -12000<m> : warn, "unrealistic ocean depth"
        stats.max(field.geophysics.elevation) < 12000<m> : warn, "unrealistic mountain height"
    }
}
```

---

## Execution Model (Brief)

The compiler IR (`Node<I>`, etc.) is **not** the execution DAG. Important distinctions:

### Per-Entity Nodes Are Batched

`Node<EntityId>` in the IR does NOT create one DAG node per entity instance.

```
IR Level                          DAG Level
┌─────────────────────┐           ┌─────────────────────────┐
│ Node<EntityId>      │           │ MemberSignalResolve     │
│ (plate.velocity)    │  ──────►  │ (single node, batched)  │
│                     │           │                         │
│ Represents the      │           │ Processes ALL 10k       │
│ definition          │           │ instances via LaneKernel│
└─────────────────────┘           └─────────────────────────┘
```

The DAG has O(primitives) nodes, not O(primitives × instances).

### Lane Kernels

Per-entity execution uses **Lane Kernels** with lowering strategies:

| Strategy | Population | Approach |
|----------|------------|----------|
| L1 | Medium (100-10k) | Rayon parallel over instances |
| L2 | SIMD-friendly | Vectorized SSA on arrays (SoA) |
| L3 | Complex deps | Sub-DAG per entity |
| Hybrid | 2k-10k | L1 outer, L3 inner |

### Aggregates Use Barriers

Cross-entity operations (`sum(plates, |p| p.mass)`) create **barrier nodes**:

```
MemberSignal (parallel)
        ↓
   ══════════════
   PopulationAggregate (barrier)
   ══════════════
        ↓
Dependent signals
```

All instances must complete before the aggregate resolves.

### Pairwise Operations (N-Body)

`other(entity)` and `pairs(entity)` enable N-body style interactions:

```cdsl
member body.acceleration : Vec3<m/s2> {
    resolve {
        sum(other(bodies), |o| gravity(self.pos, o.pos, o.mass))
    }
}
```

**Iteration order:**
- `other(entity)` iterates all instances except `self` in **lexical InstanceId order**
- `pairs(entity)` iterates unique pairs `(i, j)` where `i < j` in lexicographic InstanceId order
- Order is deterministic and stable across runs

**Current implementation:** O(N²) internal loop, no spatial optimization.

| N | Pairs | Notes |
|---|-------|-------|
| 1,000 | ~500k | Acceptable |
| 10,000 | ~50M | Slow |
| 100,000 | ~5B | Impractical |

> **Future:** Spatial partitioning (octree/BVH) tracked in `continuum-qk5s`.

### Determinism

- **Stable ID ordering** — all entity iteration uses lexical `InstanceId` order
- Fixed-tree reductions for aggregates (bitwise reproducible)
- Snapshot semantics (read prev, write current)

**InstanceId assignment (at scenario load):**
- Each entity type has a separate ID space starting at 0
- IDs are assigned in **scenario declaration order** (array index for lists, sorted key order for maps)
- IDs are stable for the entire simulation (no reassignment)
- `InstanceId` is a `(EntityId, u32)` tuple — entity type + instance index

```yaml
# scenario.yaml
plates:           # EntityId = "plate"
  - name: pacific # InstanceId = (plate, 0)
  - name: nazca   # InstanceId = (plate, 1)
  - name: african # InstanceId = (plate, 2)
```

This ordering is part of the determinism contract — same scenario always produces same InstanceIds.

### Per-Entity Emit Order

When multiple entity instances emit to the same signal:

```cdsl
entity plate {
    operator transfer_heat {
        : phase(Collect)
        execute {
            emit(neighbor.temperature, self.temperature * 0.1)
        }
    }
}
```

**Behavior:**
- Emissions execute in **lexical `InstanceId` order** (deterministic)
- Values **accumulate** (pushed to Vec, summed in Resolve)
- **Commutativity required** — only sum accumulation, order doesn't affect result
- **No race conditions** — Collect phase is sequential

**Key invariant:** All aggregation and iteration is based on stable `InstanceId` ordering, ensuring deterministic results regardless of implementation details.

### Cross-Stratum Reads (Staleness)

When a fast-stratum signal reads a slow-stratum signal:

```cdsl
stratum fast { stride: 1 }
stratum slow { stride: 100 }

signal fast.z : Scalar {
    : stratum(fast)
    resolve { fast.x + slow.y }  // slow.y may be 99 ticks stale
}
```

**Behavior:**
- Reads return the **most recently resolved value**
- No interpolation or extrapolation
- Value may be up to `stride - 1` ticks stale
- This is **intentional** — documented in `docs/strata.md` Section 6

**No compiler warning currently.** Cross-stratum cycles are safe (temporal buffering breaks them).

> **Future:** Optional lint for cross-stratum reads tracked in `continuum-zk7a`.

### Measure Phase Sub-Phases

Measure executes in two sequential sub-phases:

1. **Field Emission** — All Fields compute and emit values (parallelized DAG)
2. **Chronicle Observation** — All Chronicles run (parallelized, reads resolved signals)

```
Measure phase execution:
┌─────────────────────────────────────┐
│  Field Emission (DAG)               │  ← Fields only, maximum parallelism
│  ┌─────┐  ┌─────┐  ┌─────┐         │
│  │ F1  │  │ F2  │  │ F3  │  ...    │
│  └─────┘  └─────┘  └─────┘         │
├─────────────────────────────────────┤
│  Chronicle Observation              │  ← After ALL fields complete
│  ┌─────┐  ┌─────┐  ┌─────┐         │
│  │ C1  │  │ C2  │  │ C3  │  ...    │  (parallel, read-only)
│  └─────┘  └─────┘  └─────┘         │
└─────────────────────────────────────┘
```

**Why two sub-phases:**
- Chronicles cannot affect Field scheduling (observer boundary absolute)
- Field DAG is smaller and purely data-driven (better parallelism)
- Production/benchmark mode: skip Chronicle sub-phase entirely for zero observer overhead

> **Note:** Full execution/DAG spec is separate. This section clarifies that IR structure ≠ DAG structure.

---

## The Rules

### 1. No Special Cases

If Signal and Member differ only by indexing, they're the same type with different `I` and same `RoleId::Signal`.
If contexts differ only by which capabilities they have, compose them.
If roles differ only in allowed phases, they share the same `Node` struct — role is data, not type.

### 2. Capabilities, Not Dummies

Never return a dummy value for an unavailable capability.
If a phase doesn't have `prev`, the context doesn't implement `HasPrev`.
Compile error > runtime surprise.

### 3. Keywords Require Capabilities

| Keyword | Requires |
|---------|----------|
| `prev` | `HasPrev` |
| `current` | `HasCurrent` |
| `inputs` | `HasInputs` |
| `dt` | `HasDt` |
| `payload` | `HasPayload` |
| `self.x` | `HasIndex` |

No capability → compile error.

### 4. Resolution Has No Silent Fallback

```
locals → scoping hierarchy → signals → ERROR
```

Typo in path name → compile error, not silent signal reference.

### 5. Executions Are Self-Contained

```rust
struct Execution {
    name: String,
    phase: Phase,
    body: ExecutionBody,  // expression or statement block
    reads: Vec<Path>,
    
    // Traceability
    source_path: Path,
    source_kind: &'static str,
    source_span: Span,
}

/// Execution body — distinguishes pure expression blocks from effectful statement blocks
enum ExecutionBody {
    /// Pure phases (resolve, measure, assert, fracture, configure)
    /// Single expression that produces a value
    Expr(TypedExpr),
    
    /// Effect phases (collect, apply)
    /// Sequence of statements, each must be Unit-typed
    Statements(Vec<TypedExpr>),
}
```

**Expression vs Statement blocks:**
- `Expr` — used by pure phases, must return a value (not `Unit`)
- `Statements` — used by effect phases (`collect`, `apply`), each statement must be `Unit`-typed

Name and phase are explicit. No derivation tables.
Full context for error messages without backtracking.

### 6. Kernel Signatures Are Typed

```rust
pub struct KernelSignature {
    pub id: KernelId,
    pub params: Vec<KernelParam>,
    pub returns: KernelReturn,
}
```

**All kernels are deterministic.** There are no non-deterministic kernels:
- `rng.*` kernels derive randomness from `(seed, InstanceId, tick)` — fully reproducible
- No kernel depends on wall-clock time, thread scheduling, or external state
- This is enforced by kernel implementation, not by a flag

```rust

pub struct KernelParam {
    pub name: &'static str,
    pub shape: ShapeConstraint,
    pub unit: UnitConstraint,
}

pub enum ShapeConstraint {
    Exact(Shape),           // must be exactly this shape
    AnyScalar,              // any scalar
    AnyVector,              // vector of any dim
    AnyMatrix,              // matrix of any dims
    Any,                    // any shape
    SameAs(usize),          // same shape as param N
    BroadcastWith(usize),   // broadcastable with param N
    
    // Dimension-constrained shapes (for matrix ops)
    VectorDim(DimConstraint),                                 // vector with constrained dim
    MatrixDims { rows: DimConstraint, cols: DimConstraint },  // matrix with constrained dims
}

pub enum DimConstraint {
    Exact(u8),    // must be exactly this dimension
    Any,          // any dimension  
    Var(usize),   // dimension variable — Var(N) must equal all other Var(N)
}

pub enum UnitConstraint {
    Exact(Unit),            // must be exactly this unit
    Dimensionless,          // must be dimensionless
    Angle,                  // must be angle (for trig)
    Any,                    // any unit
    SameAs(usize),          // same unit as param N
}

pub struct KernelReturn {
    pub shape: ShapeDerivation,
    pub unit: UnitDerivation,
}

pub enum ShapeDerivation {
    Exact(Shape),                  // always this shape
    SameAs(usize),                 // same as param N
    FromBroadcast(usize, usize),   // broadcast result of params
    Scalar,                        // always scalar (reductions)
    
    // Derived from dimension variables
    VectorDim(DimConstraint),
    MatrixDims { rows: DimConstraint, cols: DimConstraint },
}

pub enum UnitDerivation {
    Exact(Unit),            // always this unit
    Dimensionless,          // always dimensionless
    SameAs(usize),          // same as param N
    Multiply(Vec<usize>),   // product of param units
    Divide(usize, usize),   // param N / param M
    Sqrt(usize),            // sqrt of param N unit
    Inverse(usize),         // 1 / param N unit
}
```

**Examples:**

```rust
// maths.add(a, b) → same shape, same unit
KernelSignature {
    id: KernelId { namespace: "maths", name: "add" },
    params: vec![
        KernelParam { name: "a", shape: Any, unit: Any },
        KernelParam { name: "b", shape: SameAs(0), unit: SameAs(0) },
    ],
    returns: KernelReturn { shape: SameAs(0), unit: SameAs(0) },
}

// vector.dot(a, b) → scalar, multiply units, dims must match
KernelSignature {
    id: KernelId { namespace: "vector", name: "dot" },
    params: vec![
        KernelParam { name: "a", shape: VectorDim(Var(0)), unit: Any },
        KernelParam { name: "b", shape: VectorDim(Var(0)), unit: Any },  // same dim
    ],
    returns: KernelReturn { shape: Scalar, unit: Multiply(vec![0, 1]) },
}

// maths.sin(x) → same shape, dimensionless (input must be angle)
KernelSignature {
    id: KernelId { namespace: "maths", name: "sin" },
    params: vec![
        KernelParam { name: "x", shape: Any, unit: Angle },
    ],
    returns: KernelReturn { shape: SameAs(0), unit: Dimensionless },
}

// matrix.mul(A, B) → A(m×n) * B(n×p) = (m×p), inner dims must match
KernelSignature {
    id: KernelId { namespace: "matrix", name: "mul" },
    params: vec![
        KernelParam { 
            name: "a", 
            shape: MatrixDims { rows: Var(0), cols: Var(1) },  // m×n
            unit: Any 
        },
        KernelParam { 
            name: "b", 
            shape: MatrixDims { rows: Var(1), cols: Var(2) },  // n×p (n must match)
            unit: Any 
        },
    ],
    returns: KernelReturn { 
        shape: MatrixDims { rows: Var(0), cols: Var(2) },  // m×p
        unit: Multiply(vec![0, 1]) 
    },
}
```

Wrong arg count, wrong shape, dimension mismatch, unknown kernel → compile error.
Unit propagation through kernel calls.

### 7. Path Collisions Are Forbidden

If `signal.foo` exists and `signal` has a user type field `.foo`, the compiler rejects it.

```cdsl
signal orbit : OrbitalElements     // has field .semi_major
signal orbit.semi_major : Scalar   // ERROR: collides with field access
```

Resolution is then unambiguous:
1. Try longest signal path match
2. Remaining segments are field access
3. Validate fields exist on type

No guessing. If both interpretations are valid, neither is allowed.

### 8. Aggregates Over Entities, Not Loops

No `for...in` iteration. Collections are processed declaratively via aggregates:

```cdsl
// Sum over all plates
let total_area = sum(plates, |p| p.area)

// Find maximum stress
let max_stress = max(plates, |p| p.boundary_stress)

// Transform each plate
let forces = map(plates, |p| p.velocity * p.mass)

// Reduce with custom operation
let momentum = fold(plates, Vec3::zero(), |acc, p| acc + p.velocity * p.mass)
```

**Why aggregates, not loops:**

1. **Declarative** — express *what*, not *how*
2. **Parallelizable** — no loop-carried dependencies to analyze
3. **Typed** — `sum` knows it returns same type as body, `map` returns `Seq<T>`
4. **No mutation** — no accumulator variables, no off-by-one errors
5. **Pure** — aggregate bodies must be effect-free (see below)

**Aggregate body constraints:**

| Constraint | Rule |
|------------|------|
| **Purity** | Body must be pure — no `emit`, `spawn`, `log` |
| **Capture** | Can reference: `self`, bound element, `const`/`config`, signals, outer `let` bindings |
| **No nesting** | Cannot nest aggregate over same entity inside body |

```cdsl
// OK — pure body, valid captures
sum(plates, |p| p.area * config.density)

// ERROR — effectful body
sum(plates, |p| { emit(total, p.area); p.area })

// ERROR — captures mutable state (if we had it)
let mut acc = 0
sum(plates, |p| { acc += 1; p.area })  // forbidden
```

**Aggregate signatures:**

| Aggregate | Signature | Returns |
|-----------|-----------|---------|
| `sum` | `(Entity, \|e\| Scalar) -> Scalar` | Sum of all values |
| `max` | `(Entity, \|e\| Scalar) -> Scalar` | Maximum value |
| `min` | `(Entity, \|e\| Scalar) -> Scalar` | Minimum value |
| `map` | `(Entity, \|e\| T) -> Seq<T>` | Transformed collection |
| `fold` | `(Entity, init, \|acc, e\| T) -> T` | Reduced value |
| `count` | `(Entity) -> Scalar` | Number of instances |
| `any` | `(Entity, \|e\| bool) -> bool` | True if any match |
| `all` | `(Entity, \|e\| bool) -> bool` | True if all match |

**Iteration order:**
- All aggregates iterate in **lexical `InstanceId` order** (deterministic)
- For commutative operations (`sum`, `max`, `min`), order doesn't affect result
- For `fold` with non-commutative functions, order IS the lexical InstanceId order
- `map` preserves entity order in the returned `Seq<T>`

> **Note:** `fold` and `map` are declared but not yet implemented. See `continuum-fg29`.

### 9. Syntax Is Sugar, IR Is Calls

All operators desugar to kernel calls. IR has one `Call` form, no special `Binary`, `Unary`, `If` variants.

**Arithmetic:**

| Syntax | Desugars To |
|--------|-------------|
| `a + b` | `maths.add(a, b)` |
| `a - b` | `maths.sub(a, b)` |
| `a * b` | `maths.mul(a, b)` |
| `a / b` | `maths.div(a, b)` |
| `-a` | `maths.neg(a)` |

**Comparison:**

| Syntax | Desugars To |
|--------|-------------|
| `a == b` | `compare.eq(a, b)` |
| `a != b` | `compare.ne(a, b)` |
| `a < b` | `compare.lt(a, b)` |
| `a <= b` | `compare.le(a, b)` |
| `a > b` | `compare.gt(a, b)` |
| `a >= b` | `compare.ge(a, b)` |

**Logic:**

| Syntax | Desugars To |
|--------|-------------|
| `a && b` | `logic.and(a, b)` |
| `a \|\| b` | `logic.or(a, b)` |
| `!a` | `logic.not(a)` |

**Control flow:**

| Syntax | Desugars To |
|--------|-------------|
| `if c { t } else { e }` | `logic.select(c, t, e)` |
| `if a { x } else if b { y } else { z }` | `logic.select(a, x, logic.select(b, y, z))` |

**Strict evaluation:** All operators evaluate all arguments eagerly. This includes:
- `logic.select(c, t, e)` — both `t` and `e` evaluate regardless of `c`
- `logic.and(a, b)` / `logic.or(a, b)` — both sides evaluate (no short-circuit)

**Effects forbidden under eager-evaluation forms.** Because all arguments evaluate eagerly, effectful expressions are forbidden under:

| Form | Why |
|------|-----|
| `logic.select(c, t, e)` | Both `t` and `e` execute regardless of `c` |
| `logic.and(a, b)` | Both `a` and `b` execute (no short-circuit) |
| `logic.or(a, b)` | Both `a` and `b` execute (no short-circuit) |

**Effectful = contains any effect kernel** (emit, spawn, log, etc.). The compiler checks `expr.is_pure()` recursively:

```rust
impl TypedExpr {
    fn is_pure(&self) -> bool {
        match &self.expr {
            ExprKind::Call { kernel, args } => {
                kernel.is_pure() && args.iter().all(|a| a.is_pure())
            }
            ExprKind::Let { value, body, .. } => value.is_pure() && body.is_pure(),
            // ... other cases recurse
        }
    }
}
```

**Validation rule:** When type-checking `logic.select`, `logic.and`, `logic.or`:
- Check each argument with `is_pure()`
- If not pure → `EffectInConditional` error

```cdsl
// COMPILE ERROR — emit in conditional branch
if condition { emit(target, value) } else { 0.0 }
//             ^^^^^^^^^^^^^^^^^^^^ effectful

// COMPILE ERROR — effect in short-circuit position  
should_emit && emit(target, value)  // both sides execute!

// CORRECT — conditional value, unconditional emit
let delta = if condition { value } else { 0.0 }
emit(target, delta)
```

**Error message:**
```
error: effectful expression in conditional branch
  --> world.cdsl:42:5
   |
42 |   if cond { emit(x, 1) } else { 0 }
   |             ^^^^^^^^^^ 'emit' is effectful
   |
   = note: both branches of 'if' evaluate eagerly (desugars to logic.select)
   = help: move effect outside conditional, or use statement-level control flow
```

**Constants (zero-arg calls):**

| Syntax | Desugars To |
|--------|-------------|
| `PI` | `maths.pi()` |
| `TAU` | `maths.tau()` |
| `E` | `maths.e()` |

**Why:**

1. **One code path** — all operations go through `Call`
2. **Type checking = signature lookup** — no special cases
3. **Adding operators = adding signatures** — no IR changes
4. **IR stays minimal** — fewer `ExprKind` variants

### 10. Statement Blocks for Effects

Effect phases (`collect`, `apply`) use **statement blocks**, not expression blocks.

**The problem:** With eager evaluation and `emit` returning `Unit`, nothing prevents:
```cdsl
let x = emit(a, 1) + emit(b, 2)  // nonsense — Unit + Unit
```

**The solution:** `Unit` is only legal at **statement position**.

| Phase | Form | Effects | Returns |
|-------|------|---------|---------|
| Configure | expression | no | value |
| **Collect** | **statement block** | **yes** | implicit Unit |
| Resolve | expression | no | value |
| Fracture | expression | no | value |
| Measure | expression | no | value |
| Assert | expression | no | Bool |
| **Apply (impulse)** | **statement block** | **yes** | implicit Unit |

**Statement block syntax:**
```cdsl
// Expression block (resolve) — returns a value
signal velocity : Vec3<m/s> {
    resolve { prev + inputs.acceleration * dt }
}

// Statement block (collect) — sequence of effects
operator transfer_heat {
    collect {
        emit(neighbor.temperature, self.temperature * 0.1)
        emit(self.temperature, -self.temperature * 0.1)
    }
}

// Statement block (impulse apply)
impulse apply_thrust {
    apply {
        emit(target.velocity, payload.direction * payload.magnitude / target.mass)
    }
}
```

**Rules:**
1. In expression blocks: `Unit`-typed expression anywhere → `UnitInExpressionPosition` error
2. In statement blocks: each statement must be `Unit`-typed
3. `Unit` cannot be operand to `+ - * /`, cannot be in `let` binding used as value
4. Statements are sequenced top-to-bottom (deterministic order)

**IR representation:** Statement blocks lower to `ExecutionBody::Statements(Vec<TypedExpr>)` where each `TypedExpr` has `ty: Type::Unit`. No special `do` form needed — the block structure is explicit in `ExecutionBody`.

### 11. Complete ExprKind

```rust
enum ExprKind {
    // === Literals ===
    Literal { value: f64, unit: Option<Unit> },  // 100.0<m>
    Vector(Vec<TypedExpr>),                       // [x, y, z]
    
    // === References ===
    Local(String),      // let-bound variable
    Signal(Path),       // signal reference (causal phases only)
    Field(Path),        // field reference (Analyzer/observer only)
    Config(Path),       // config from scoping
    Const(Path),        // const from scoping
    
    // === Context values (require capabilities) ===
    Prev,               // previous tick value (HasPrev)
    Current,            // just-resolved value (HasCurrent)
    Inputs,             // accumulated inputs (HasInputs)
    Dt,                 // time step (HasDt)
    Self_,              // current entity instance (HasIndex)
    Other,              // other entity instance (HasIndex, n-body)
    Payload,            // impulse payload (HasPayload)
    
    // === Binding forms (introduce scope) ===
    Let { name: String, value: Box<TypedExpr>, body: Box<TypedExpr> },
    Aggregate { op: AggregateOp, entity: EntityId, binding: String, body: Box<TypedExpr> },
    Fold { entity: EntityId, init: Box<TypedExpr>, acc: String, elem: String, body: Box<TypedExpr> },
    
    // === Calls (all operators desugar here) ===
    Call { kernel: KernelId, args: Vec<TypedExpr> },
    
    // === User types ===
    Struct { ty: UserTypeId, fields: Vec<(String, TypedExpr)> },
    FieldAccess { object: Box<TypedExpr>, field: String },  // works on user types, vectors, Prev, Self_, Payload, etc.
}

enum AggregateOp { Sum, Map, Max, Min, Count, Any, All }  // Fold is separate variant
```

**Binding forms vs Calls:**
- `Let`, `Aggregate`, and `Fold` introduce variable bindings — they're not function calls
- `Aggregate` iterates over entity instances, binding each to the parameter
- `Fold` iterates with two bindings: accumulator (`acc`) and element (`elem`)
- Everything else (arithmetic, logic, comparison, control flow) is `Call`

**Signal vs Field references:**
- `Signal(Path)` — only valid in causal phases (Collect, Resolve, Fracture) and observer phases that read signals
- `Field(Path)` — only valid in Analyzer context (post-hoc, reads field snapshots)
- Attempting to use `Field` in a causal phase → `MissingCapability` error
- This enforces the observer boundary at compile time

**FieldAccess unifies:**
- `self.area` → `FieldAccess { object: Self_, field: "area" }`
- `prev.x` → `FieldAccess { object: Prev, field: "x" }`
- `payload.value` → `FieldAccess { object: Payload, field: "value" }`
- `orbit.semi_major` → `FieldAccess { object: Signal("orbit"), field: "semi_major" }`
- `vec.x` → `FieldAccess { object: vec, field: "x" }` (vector component)

**Side effects:**
- `emit(target, value)` is a `Call` that returns `Unit`
- `Unit`-typed calls are only valid in **statement blocks** (see Rule 10)
- Effect phases (`collect`, `apply`) use `ExecutionBody::Statements`
- Pure phases use `ExecutionBody::Expr` — `Unit` anywhere in the expression tree is a compile error

---

## What This Eliminates

| Before | After |
|--------|-------|
| 13 primitive structs | `Node<I>` + `RoleId` |
| 7 empty Kind type parameters | Compile-time role registry |
| AST types + IR types (duplicate DTOs) | Single source, pipeline traits |
| 7 context structs with 90% duplication | Capability composition |
| `CompiledExpr` without types | `TypedExpr` with `Type` |
| `Binary`, `Unary`, `If` IR variants | `Call` + binding forms (`Let`, `Aggregate`, `Fold`) |
| Runtime panics on type mismatch | Compile-time shape/unit errors |
| Dummy `prev()` returning 0.0 | No `HasPrev` impl → compile error |
| Silent signal fallback on typos | Explicit resolution → error |
| Per-primitive parsing loops | Unified content handling |
| Match-as-polymorphism | Data-driven role registry |
| `emit()` anywhere in expressions | `Unit` only in statement blocks → compile error |

---

## The Test

For any proposed change, ask:

1. **Is it explicit or implicit?** — Explicit > implicit (the prime directive)
2. **Does it unify or fragment?** — One structure > many similar structures
3. **Is it single source or duplication?** — One struct through passes > AST/IR/DTO copying
4. **Is it syntax sugar or special IR?** — Sugar that desugars to `Call` > new variants (exception: binding forms)
5. **Is the constraint compile-time or runtime?** — Compile-time > runtime
6. **Is the capability present or faked?** — Present > dummy value
7. **Is the error loud or silent?** — Loud > silent
8. **Could a human misread it?** — If yes, reject it

---

## Summary

```
Explicit > Implicit      — the prime directive
Node<I> + RoleId         — execution primitives (Signal, Field, Operator, Impulse, Fracture, Chronicle)
Entity, Stratum, Era     — structural declarations (not Node, configure execution)
Analyzer                 — post-hoc tooling (outside DAG, reads fields)
Single source            — one struct through all passes, no DTO copying
Capability traits        — compose what's available, enforce at compile time
TypedExpr                — shape and unit proven correct before execution
Syntax is sugar          — operators desugar to kernel calls, IR is just Call
KernelSignature          — typed signatures with shape/unit constraints
Phase (9 total)          — 3 init (CollectConfig, Initialize, WarmUp) + 6 tick (Configure...Assert)
```

**What is what:**

| Category | Items | In DAG? |
|----------|-------|---------|
| Execution Primitives | `Node<I>` with Signal, Field, Operator, Impulse, Fracture, Chronicle | Yes |
| Structural | Entity (namespace + index), Stratum (cadence), Era (policy) | Referenced |
| Post-Hoc | Analyzer (field snapshots → JSON) | No |

**Per-entity support:**

| Role | Global | Per-Entity |
|------|--------|------------|
| Signal | ✓ | ✓ |
| Field | ✓ | ✓ |
| Operator | ✓ | ✓ |
| Fracture | ✓ | ✓ |
| Impulse | ✓ | ✗ (invocation targets entities) |
| Chronicle | ✓ | ✗ (interpretation rules) |

The compiler's job is to turn DSL into a typed, validated execution graph.
Runtime's job is to execute it.
No guessing. No fallbacks. No surprises.
