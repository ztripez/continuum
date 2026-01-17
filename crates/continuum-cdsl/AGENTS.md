# Continuum CDSL Compiler — Agent Guidelines

> **This compiler builds the foundation that everything else depends on.**  
> **One mistake here cascades through 15 phases of work.**  
> **Go slow. Go careful. Get it right.**

---

## Prime Directive: Explicit Over Implicit

**Ambiguity is a bug.** Every construct has exactly one interpretation.

- **No silent coercion** — types don't silently become other types
- **No silent fallback** — unresolved paths are errors, not assumptions
- **No silent collision** — if two things could be confused, forbid the collision
- **No inference where declaration is cheap** — write the type name, it's documentation

When there's a choice between implicit convenience and explicit clarity, **choose explicit**.

**The test:** If a human reader could misinterpret it, the compiler should reject it.

---

## Core Architecture Principles

### 1. Unified Node Structure

Everything is `Node<I>` with `RoleData`:

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
    pub output: Option<Type>,
    pub inputs: Option<Type>,
    
    // Indexing
    pub index: I,   // () for global, EntityId for per-entity
}
```

**Why this matters:**
- Member is `Node<EntityId>` with `role: RoleData::Signal`, NOT a separate type
- Invalid states are unrepresentable (Signal can't have reconstruction hint)
- Role-specific data is co-located with the role tag

### 2. Capability Composition (Not Inheritance)

Contexts are built from orthogonal capabilities:

```rust
trait HasScoping { fn config(&self, path: &Path) -> Value; }
trait HasSignals { fn signal(&self, path: &Path) -> Value; }
trait HasPrev    { fn prev(&self) -> &Value; }
trait HasCurrent { fn current(&self) -> &Value; }
trait HasInputs  { fn inputs(&self) -> f64; }
trait HasDt      { fn dt(&self) -> f64; }
trait HasPayload { fn payload(&self) -> &Value; }
trait CanEmit    { fn emit(&self, target: &Path, value: Value); }
```

**Phase contexts implement only what they provide.**  
No capability → compile error, NOT dummy value.

### 3. Types Prove Correctness

Every expression carries its type. Type errors are **compile errors**.

```rust
struct TypedExpr {
    expr: ExprKind,
    ty: Type,      // KernelType, UserType, Bool, Unit, Seq<T>
    span: Span,    // for error messages
}
```

**Type system:**
- `KernelType { shape: Shape, unit: Unit, bounds: Option<Bounds> }`
- `UserType { id: UserTypeId, fields: Vec<(String, Type)> }`
- `Bool` (distinct from Scalar)
- `Unit` (for emit, side effects — statement position only)
- `Seq<T>` (intermediate only, must be consumed by aggregate)

### 4. Single Source (No AST → IR Copying)

One struct flows through all compiler passes:

```rust
// Pipeline traits (supertrait hierarchy)
trait Named { fn path(&self) -> &Path; }
trait Parsed: Named { fn type_expr(&self) -> Option<&TypeExpr>; }
trait Resolved: Parsed { fn output(&self) -> Option<&Type>; }
trait Validated: Resolved { fn validation_errors(&self) -> &[ValidationError]; }
trait Compiled: Validated { fn executions(&self) -> &[Execution]; }
```

Syntax fields are cleared after consumption. State is explicit:
- `type_expr.is_none() && output.is_some()` → resolved
- `execution_exprs.is_empty() && !executions.is_empty()` → compiled

---

## Hard Rules (Non-Negotiable)

### Rule 1: No Special Cases

If Signal and Member differ only by indexing, they're the same type with different `I`.  
If contexts differ only by capabilities, compose them.  
If roles differ only in allowed phases, use `RoleData` — not separate types.

### Rule 2: Capabilities, Not Dummies

**Never return a dummy value for an unavailable capability.**

If a phase doesn't have `prev`, the context doesn't implement `HasPrev`.  
Using `prev` in that context → **compile error**, not `Scalar(0.0)`.

Compile error > runtime surprise.

### Rule 3: Keywords Require Capabilities

| Keyword | Requires | Error if Missing |
|---------|----------|------------------|
| `prev` | `HasPrev` | `MissingCapability` |
| `current` | `HasCurrent` | `MissingCapability` |
| `inputs` | `HasInputs` | `MissingCapability` |
| `dt` | `HasDt` | `MissingCapability` |
| `payload` | `HasPayload` | `MissingCapability` |
| `self.x` | `HasIndex` | `MissingCapability` |

### Rule 4: Resolution Has No Silent Fallback

**Resolution order:**
```
locals → scoping (config/const) → signals → ERROR
```

Typo in path name → **compile error**, not silent signal reference.

**Namespace prefixes for disambiguation:**
- `signal.X` — signal path
- `field.X` — field path
- `config.X` — config value
- `const.X` — constant
- `fn.X` — function

Bare paths allowed if unambiguous. Multiple matches → `AmbiguousReference` error.

### Rule 5: Executions Are Self-Contained

```rust
struct Execution {
    name: String,
    phase: Phase,
    body: ExecutionBody,  // Expr or Statements
    reads: Vec<Path>,
    
    // Traceability
    source_path: Path,
    source_kind: &'static str,
    source_span: Span,
}

enum ExecutionBody {
    Expr(TypedExpr),           // Pure phases: resolve, measure, assert, fracture
    Statements(Vec<TypedExpr>), // Effect phases: collect, apply (each must be Unit)
}
```

Name and phase are explicit. No derivation tables.  
Full context for error messages without backtracking.

### Rule 6: Kernel Signatures Are Typed

```rust
pub struct KernelSignature {
    pub id: KernelId,
    pub params: Vec<KernelParam>,
    pub returns: KernelReturn,
    pub purity: KernelPurity,
}

pub enum KernelPurity {
    Pure,    // No side effects, can be used anywhere
    Effect,  // Mutates state (emit, spawn) or artifacts (log)
}
```

**Purity enforcement by phase:**
- Pure phases (resolve, measure, assert) → **Pure kernels only**
- Effect phases (collect, apply) → **Pure + Effect kernels**

Effect in pure context → `EffectInPurePhase` error.

### Rule 7: Shape and Unit Are Compile-Time Checked

**Shape validation:**
- `Vec2 + Vec3` → error (dimension mismatch)
- `Mat3 * Vec2` → error (incompatible dimensions)
- `Scalar * Vec3` → `Vec3` (broadcast)

**Unit validation:**
- `m/s + K` → error (incompatible units)
- `m / s` → `m/s` (unit algebra)
- `sin(radians)` → dimensionless (checked)

**Unit kinds:**
- `Multiplicative` — standard SI (m, kg, s)
- `Affine { offset }` — temperature scales (°C, °F) — **no addition**
- `Logarithmic { base }` — dB, pH — **no arithmetic**

Mixing affine/logarithmic in addition → `AffineArithmeticForbidden` error.

### Rule 8: Bounds Are Runtime-Validated

Bounds are tracked at compile time, **validated after Resolve**:

```rust
struct Bounds {
    min: Option<f64>,  // None = unbounded below
    max: Option<f64>,  // None = unbounded above
}
```

Out-of-bounds → structured fault with path, value, bounds.

### Rule 9: Seq<T> Is Intermediate Only

`Seq<T>` is produced by `map`, consumed by aggregates:

```cdsl
let total = sum(map(plates, |p| p.area))  // OK
let s = map(plates, |p| p.area)           // ERROR: not consumed
```

**Constraints:**
1. Cannot be stored in signals
2. Cannot be returned from resolve blocks
3. Lambda bodies must be **pure** (no effects)
4. Iteration order is **deterministic** (lexical InstanceId order)

### Rule 10: Unit Is Statement Position Only

`Unit` type (from `emit`, `spawn`, `log`) is valid **only at statement position**:

```cdsl
// OK — statement block
operator apply_force {
    : phase(Collect)
    execute {
        emit(target.velocity, force / target.mass)  // Unit-typed statement
    }
}

// ERROR — expression block
signal foo : Scalar {
    resolve { emit(x, 1.0) }  // Unit in expression position
}
```

`Unit` in expression position → `UnitInExpressionPosition` error.

### Rule 11: Effects Forbidden Under Eager Conditionals

Effectful operations (`emit`, `spawn`, `log`) **cannot appear** under:
- `logic.select(cond, a, b)` — both branches always evaluate
- `logic.and(a, b)` — both operands always evaluate
- `logic.or(a, b)` — both operands always evaluate

```cdsl
// ERROR — both branches evaluate
logic.select(check, emit(x, 1.0), emit(x, 2.0))

// ERROR — both operands evaluate
logic.and(check, emit(x, 1.0))
```

Use `if` blocks (lazy evaluation) for conditional effects.

Error: `EffectInConditional { effect, form }`.

### Rule 12: Vector Field Access Is Dimension-Aware

Named component access (`.x`, `.y`, `.z`, `.w`) only valid for dimensions 2-4:

```cdsl
signal v2 : Vec2<m> { resolve { Vec2(1.0, 2.0) } }
signal v3 : Vec3<m> { resolve { Vec3(1.0, 2.0, 3.0) } }
signal v5 : Vector<5, m> { resolve { ... } }

v2.x      // OK
v3.z      // OK
v5.x      // ERROR: dim > 4, use .at(0)
v2.w      // ERROR: w not available for Vec2
```

Use `.at(index)` for arbitrary dimensions or higher-rank vectors.

Error: `NamedComponentDimMismatch { dim, component }`.

### Rule 13: other() and pairs() Scope Rules

**Type definitions:**
```rust
other(E: EntityId) -> EntityIter<E>  // all instances except self
pairs(E: EntityId) -> PairIter<E>    // unique pairs (a, b) where a.id < b.id
```

**Scope constraints:**
- `other(X)` only valid in `Node<EntityId>` context (requires `HasIndex`)
- `pairs(X)` valid in any context (global or per-entity)
- **No nesting:** `other(other(...))` forbidden
- `EntityIter<E>` / `PairIter<E>` only valid as aggregate source

Violations → `OtherOutsideEntityContext`, `NestedIteratorForbidden`, `IteratorNotConsumed`.

### Rule 14: Struct Literals Must Be Explicit

**No shorthand:**
```cdsl
// ERROR — shorthand forbidden
OrbitalElements { semi_major, eccentricity }

// OK — explicit field binding
OrbitalElements { semi_major: semi_major, eccentricity: eccentricity }

// OK — inline values
OrbitalElements { semi_major: 1000.0<m>, eccentricity: 0.1 }
```

**Type prefix required:**
```cdsl
// ERROR — type inferred
{ x: 1.0, y: 2.0 }

// OK — type stated
Vec2 { x: 1.0, y: 2.0 }
```

Violations → `ShorthandForbidden`, missing type prefix is parse error.

---

## Compilation Phases

### Phase 0: Cleanup
- Delete old compiler crates
- Create new `continuum-cdsl` crate scaffolding

### Phase 1: Foundation (5 tasks)
1. **Path and typed IDs** ✅
   - Hierarchical path representation
   - Type-safe ID wrappers (SignalId, FieldId, etc.)

2. **Unit system** (TODO)
   - `Unit { kind: UnitKind, dims: UnitDimensions }`
   - `UnitKind`: Multiplicative, Affine, Logarithmic
   - Unit algebra and validation

3. **Shape system** (TODO)
   - `Shape`: Scalar, Vector, Matrix, Tensor, Quaternion, Complex
   - Named variants for common cases, `Tensor` for rank 3+

4. **Type enum and UserType** (TODO)
   - `Type`: Kernel, User, Bool, Unit, Seq
   - `KernelType { shape, unit, bounds }`
   - `UserType { id, fields }`

5. **KernelType and Bounds** (TODO)
   - Combine shape, unit, bounds into KernelType
   - Bounds validation infrastructure

### Phase 2: Source Tracking (2 tasks)
- Span and SourceMap
- Phase and Capability enums

### Phase 3-4: AST (4 tasks)
- Node<I> unified structure
- Role system (RoleData, RoleSpec, ROLE_REGISTRY)
- Scoping, Assertion, Execution structs
- ExprKind and TypedExpr
- Untyped AST for parser output

### Phases 5-15: Parser, Resolution, Validation, Bytecode, VM

---

## When In Doubt

### Ask First
- Is this construct ambiguous to a human reader?
- Does this require type inference when declaration is cheap?
- Am I adding a silent fallback?
- Is this a special case that should be generalized?

### Fail Loudly
- Unresolved path → error, not assumption
- Type mismatch → error, not coercion
- Missing capability → error, not dummy value
- Structural violation → error, not warning

### Test Thoroughly
- Every new type gets tests
- Every validation rule gets a failing test case
- Every error message gets verified for clarity

---

## Error Message Standards

**Good error messages:**
1. State exactly what's wrong
2. Show the problematic code location
3. Explain why it's invalid
4. Suggest how to fix it

**Example:**
```
error: cannot add affine units
  --> world.cdsl:15:5
   |
15 |   let sum = t1 + t2
   |             ^^^^^^^ both operands are Affine (degC)
   |
   = note: affine units don't support addition (20°C + 30°C ≠ 50°C)
   = help: convert to Kelvin first: units.to_kelvin(t1) + units.to_kelvin(t2)
```

---

## Remember

This compiler is **infrastructure**. Everything built on top of it assumes it's correct.

**When implementing foundation types:**
- Read the spec in `.opencode/plans/compiler-manifesto.md` carefully
- Understand the why, not just the what
- Test edge cases
- Document invariants
- Verify against examples from terra world

**When extending the compiler:**
- Check if the extension violates any hard rules
- Ensure error messages are actionable
- Update tests for new validation rules
- Keep the manifesto as the single source of truth

---

**Go slow. Go careful. Get it right.**
