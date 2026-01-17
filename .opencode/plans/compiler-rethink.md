# Compiler Structure Rethink

> **Context**: The compiler does too much manual labor. This document captures the structural analysis and explores unification strategies.

---

## What We Have

### All Primitives Share These Elements

| Element | Description |
|---------|-------------|
| **Identity** | Path-based name (`terra.surface.temperature`) |
| **Location** | File + Span (source location) |
| **Documentation** | `doc`, `title`, `symbol` |
| **Stratum** | Execution lane (optional on some) |
| **Type** | The value type (most primitives) |
| **Execution Blocks** | Named code blocks (`resolve`, `measure`, etc.) |
| **Reads** | Dependencies inferred from blocks |

### Primitive Types

#### Core Simulation Primitives (with execution blocks)

| Primitive | AST Type | Compiled Type | ID Type | Execution Block(s) |
|-----------|----------|---------------|---------|-------------------|
| Signal | `SignalDef` | `CompiledSignal` | `SignalId` | `resolve`, `initial`, `warmup` |
| Member | `MemberDef` | `CompiledMember` | `MemberId` | `resolve`, `initial` |
| Field | `FieldDef` | `CompiledField` | `FieldId` | `measure` |
| Operator | `OperatorDef` | `CompiledOperator` | `OperatorId` | `collect`, `measure`, `warmup` |
| Impulse | `ImpulseDef` | `CompiledImpulse` | `ImpulseId` | `apply` |
| Fracture | `FractureDef` | `CompiledFracture` | `FractureId` | `when`, `emit` |
| Chronicle | `ChronicleDef` | `CompiledChronicle` | `ChronicleId` | `observe` |
| Analyzer | `AnalyzerDef` | `CompiledAnalyzer` | `AnalyzerId` | `compute`, `validate` |

#### Structural Primitives (no execution blocks)

| Primitive | AST Type | Compiled Type | ID Type |
|-----------|----------|---------------|---------|
| Entity | `EntityDef` | `CompiledEntity` | `EntityId` |
| Stratum | `StrataDef` | `CompiledStratum` | `StratumId` |
| Era | `EraDef` | `CompiledEra` | `EraId` |
| Function | `FnDef` | `CompiledFn` | `FnId` |
| Type | `TypeDef` | `CompiledType` | `TypeId` |

### Execution Blocks

| Block Name | Phase | Primitives Using It |
|------------|-------|---------------------|
| `resolve` | Resolve | Signal, Member |
| `initial` | Configure | Signal, Member |
| `warmup` | Warmup | Signal, Operator |
| `measure` | Measure | Field, Operator |
| `collect` | Collect | Operator |
| `apply` | Resolve | Impulse |
| `when`/`emit` | Fracture | Fracture |
| `observe` | Measure | Chronicle |
| `compute` | Post-hoc | Analyzer |

### Execution Contexts

Each block type gets a dedicated `ExecutionContext` implementation:

- `ResolverContext` - has `prev`, `inputs`, `dt`
- `AssertionContext` - `prev` returns current value
- `MeasureContext` - no `prev` or `inputs`
- `TransitionContext` - only signals/constants/config
- `WarmupContext` - `prev` is current warmup value
- `FractureExecContext` - read-only signals

---

## Pain Points

### 1. Triple Type System

Every primitive exists as three separate types:
- `SignalDef` (AST) → `CompiledSignal` (IR) → `SignalProperties` (unified)

Manual mapping between all three. Adding a field means editing 3+ structs.

### 2. Parser Duplication

Each primitive has its own `Content` enum and matching loop:

```rust
pub fn signal_def() -> impl Parser<...> {
    tok(Token::Signal)
        .ignore_then(spanned_path())
        .then(content().repeated().collect().delimited_by(...))
        .map(|(path, contents)| {
            let mut def = SignalDef { /* init all fields to None/vec![] */ };
            for content in contents {
                match content {
                    Content::Type(t) => def.ty = Some(t),
                    Content::Strata(s) => def.strata = Some(s),
                    // ... 8-15 more branches
                }
            }
            def
        })
}
```

Adding a common attribute requires editing 6+ parsers.

### 3. Lowering Duplication

`lower/mod.rs:475-750` has 300+ lines of nearly identical struct construction:

```rust
for (id, signal) in &self.signals {
    let node = CompiledNode {
        id: id.path().clone(),
        file: signal.file.clone(),
        span: signal.span.clone(),
        stratum: Some(signal.stratum.clone()),
        reads: signal.reads.clone(),
        // ... copy all properties
    };
    nodes.insert(id.path().clone(), node);
}
// Repeated 12 times
```

### 4. ExtractFromNode Boilerplate

12 nearly identical `ExtractFromNode` implementations:

```rust
impl ExtractFromNode for CompiledSignal {
    fn try_extract(path: &Path, node: &CompiledNode) -> Option<Self> {
        if let NodeKind::Signal(props) = &node.kind {
            Some(CompiledSignal {
                file: node.file.clone(),
                span: node.span.clone(),
                id: SignalId::from(path.clone()),
                // ... manually extract all fields
            })
        } else {
            None
        }
    }
}
```

### 5. ExecutionContext Duplication

6 context structs with ~90% identical trait implementations. Only differ in 2-3 methods (`prev()`, `inputs()`, `dt_scalar()`).

### 6. Field Access Friction

Lots of `.clone()` and `.map()` chains to extract nested optional fields from parameters and other structures.

---

## Existing Abstraction Attempts

### Unified Node Architecture (`unified_nodes.rs`)

**Good**: Single `CompiledNode` with `NodeKind` enum.

**Incomplete**: Still requires manual population and extraction.

### `define_id!` Macro

Good abstraction for ID types - generates all conversion impls.

### `impl_locatable!` Macro

Small macro for file/span access.

---

## Possible Directions

### A. Macro/Codegen Approach

Define primitives declaratively, generate the boilerplate:

```rust
define_primitive! {
    Signal {
        common: [doc, title, symbol, stratum, type],
        blocks: [resolve, initial, warmup],
        context: ResolverContext,
    }
}
```

Generates: AST type, IR type, parser content handling, lowering, extraction.

### B. Structural Unification

Collapse the triple type system:
- One `Primitive<K>` generic over kind
- Common fields in base struct
- Kind-specific fields in associated type

### C. Trait-Based Polymorphism

Define what varies as traits, share everything else:

```rust
trait Primitive {
    type Blocks: BlockSet;
    type Context: ExecutionContext;
    fn stratum(&self) -> Option<&StratumId>;
}
```

### D. Common Metadata Bundle

Extract shared structure:

```rust
struct PrimitiveMeta {
    id: Path,
    file: Option<PathBuf>,
    span: Span,
    doc: Option<String>,
    title: Option<String>,
    symbol: Option<String>,
    stratum: Option<StratumId>,
}
```

All primitives contain this, reducing field duplication.

---

## Questions to Resolve

1. Should we generate types or unify them?
2. How much type safety do we want to preserve? (Currently each primitive is its own type)
3. Can execution contexts be collapsed with a capability-based approach?
4. Where does the parser fit in? (AST types are closely tied to parser combinators)

---

## Refined Model

### Terminology

| Term | What it is | Examples |
|------|------------|----------|
| **Execution** | Scheduled by DAG | `resolve`, `measure`, `apply`, `initial`, `when`, `emit`, `observe`, `compute`, `validate`, `transition`, function body |
| **Scoping** | Compile-time bindings | `config { }`, `const { }` |
| **Output** | What the primitive produces | Signal → `Scalar<K>`, Function → return type |
| **Input** | What the primitive receives | `: uses(...)`, function params, impulse payload |

---

### Base Trait

Everything has identity + location + documentation:

```rust
trait Named {
    fn path(&self) -> &Path;
    fn span(&self) -> &Span;
    fn file(&self) -> Option<&PathBuf>;
    fn doc(&self) -> Option<&str>;
    fn title(&self) -> Option<&str>;
    fn symbol(&self) -> Option<&str>;
}
```

---

### Categories

```rust
trait Primitive: Named {}    // Participates in simulation (has executions)
trait Declaration: Named {}  // Structural organizers (no executions)
```

**Primitives** (9):
- Signal, Member, Field, Operator, Impulse, Fracture, Chronicle, Analyzer, Function

**Declarations** (4):
- Entity, Stratum, Era, Type

**Special**:
- Module — implicit, has scoping and assertions at world level

---

### Orthogonal Capability Traits

```rust
// Apply to Named (both Primitive and Declaration)
trait HasScoping: Named {
    fn scoping(&self) -> Option<&Scoping>;
}

trait HasAssertions: Named {
    fn assertions(&self) -> &[Assertion];
}

// Apply to Primitive only
trait HasExecutions: Primitive {
    fn executions(&self) -> &[Execution];
}

trait HasStratum: HasExecutions {  // Requires executions
    fn stratum(&self) -> &StratumId;
}

trait HasOutput: Primitive {
    fn output(&self) -> &Type;
}

trait HasInputs: Primitive {
    fn inputs(&self) -> &[Input];
}

trait HasTopology: Primitive {  // Field only
    fn topology(&self) -> &Topology;
}

trait HasRequirements: Primitive {  // Analyzer only
    fn requirements(&self) -> &Requirements;
}

// Apply to Declaration only
trait HasCount: Declaration {  // Entity
    fn count(&self) -> &CountSpec;
}

trait HasStride: Declaration {  // Stratum
    fn stride(&self) -> Option<u32>;
}

trait HasTransitions: Declaration {  // Era
    fn transitions(&self) -> &[Transition];
}

trait HasStrataConfig: Declaration {  // Era
    fn strata_config(&self) -> &StrataConfig;
}
```

---

### Design Decisions

1. **Stay flat** — no intermediate layers, just `Named` + orthogonal traits
2. **Bounds** — part of the type system, not a trait
3. **HasPayload** — collapsed into `HasInputs` (impulse payload is input)
4. **HasStratum requires HasExecutions** — can't have a stratum without something to execute
5. **Stratum and Era are odd kids** — minimal traits, just their specific capabilities

---

### Indexed Primitives (DRY Unification)

**Key insight:** Member is just Signal with an entity index. This pattern extends to Field and Fracture.

```rust
trait Index {
    fn entity(&self) -> Option<&EntityId>;
}

impl Index for () {
    fn entity(&self) -> Option<&EntityId> { None }
}

impl Index for EntityId {
    fn entity(&self) -> Option<&EntityId> { Some(self) }
}

// Generic over indexing
struct Signal<I: Index = ()> { ... }
struct Field<I: Index = ()> { ... }
struct Fracture<I: Index = ()> { ... }

// Type aliases
type GlobalSignal = Signal<()>;
type Member = Signal<EntityId>;

type GlobalField = Field<()>;
type EntityField = Field<EntityId>;

type GlobalFracture = Fracture<()>;
type EntityFracture = Fracture<EntityId>;
```

**Primitives that support indexing:**
- Signal / Member ✓
- Field ✓
- Fracture ✓

**Primitives that stay global only:**
- Operator — orchestrates, doesn't belong to entity
- Impulse — targets entities, isn't *per* entity
- Chronicle — global observation
- Analyzer — post-hoc analysis
- Function — pure computation

---

### Capability Matrix

| Thing | Category | Scoping | Assertions | Executions | Stratum | Output | Inputs | Indexed | Other |
|-------|----------|---------|------------|------------|---------|--------|--------|---------|-------|
| Module | — | ✓ | ✓ | - | - | - | - | - | - |
| Entity | Declaration | ✓ | ✓ | - | - | - | - | - | Count |
| Stratum | Declaration | - | - | - | - | - | - | - | Stride |
| Era | Declaration | - | - | - | - | - | - | - | Transitions, StrataConfig |
| Type | Declaration | ✓ | ✓ | - | - | - | - | - | - |
| Signal | Primitive | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | - |
| Field | Primitive | ✓ | - | ✓ | ✓ | ✓ | - | ✓ | Topology |
| Fracture | Primitive | ✓ | - | ✓ | ✓ | - | ✓ | ✓ | - |
| Operator | Primitive | ✓ | - | ✓ | ✓ | - | ✓ | - | - |
| Impulse | Primitive | ✓ | - | ✓ | - | - | ✓ | - | - |
| Chronicle | Primitive | ✓ | - | ✓ | - | - | - | - | - |
| Analyzer | Primitive | - | - | ✓ | - | - | - | - | Requirements |
| Function | Primitive | - | - | ✓ | - | ✓ | ✓ | - | - |

Note: "Indexed" column indicates primitives that can be generic over `Index` (global or per-entity).

---

## Final Design: Unified Node

### Core Insight

All primitives share the same shape. The differences are:
1. **Which capabilities are used** (Some vs None)
2. **Kind-specific data** (topology for fields, requirements for analyzers, etc.)
3. **Indexing** (global vs per-entity)

### The Unified Structure

```rust
struct Node<K: Kind, I: Index = ()> {
    // === Named (all nodes) ===
    path: Path,
    span: Span,
    file: Option<PathBuf>,
    doc: Option<String>,
    title: Option<String>,
    symbol: Option<String>,
    
    // === Common capabilities (Option = not all use them) ===
    scoping: Option<Scoping>,
    assertions: Vec<Assertion>,
    executions: Vec<Execution>,
    stratum: Option<StratumId>,
    output: Option<Type>,
    inputs: Vec<Input>,
    
    // === Kind-specific data ===
    kind: K,
    
    // === Indexing ===
    index: I,
}
```

### Index Trait

```rust
trait Index {
    fn entity(&self) -> Option<&EntityId>;
}

impl Index for () {
    fn entity(&self) -> Option<&EntityId> { None }
}

impl Index for EntityId {
    fn entity(&self) -> Option<&EntityId> { Some(self) }
}
```

### Kind Trait + Implementations

```rust
trait Kind {
    // Marker trait, or could have associated types/methods
}

// Primitives
struct SignalKind;
struct FieldKind { topology: Option<Topology> }
struct FractureKind;
struct OperatorKind;
struct ImpulseKind;  // payload is in `inputs`
struct ChronicleKind;
struct AnalyzerKind { requirements: Requirements }
struct FunctionKind;

// Declarations
struct EntityKind { count: CountSpec }
struct StratumKind { stride: Option<u32> }
struct EraKind { transitions: Vec<Transition>, strata_config: StrataConfig }
struct TypeKind { fields: Vec<TypeField> }
```

### Type Aliases for Ergonomics

```rust
// Primitives
type Signal = Node<SignalKind>;
type Member = Node<SignalKind, EntityId>;
type Field = Node<FieldKind>;
type EntityField = Node<FieldKind, EntityId>;
type Fracture = Node<FractureKind>;
type EntityFracture = Node<FractureKind, EntityId>;
type Operator = Node<OperatorKind>;
type Impulse = Node<ImpulseKind>;
type Chronicle = Node<ChronicleKind>;
type Analyzer = Node<AnalyzerKind>;
type Function = Node<FunctionKind>;

// Declarations
type Entity = Node<EntityKind>;
type Stratum = Node<StratumKind>;
type Era = Node<EraKind>;
type TypeDef = Node<TypeKind>;
```

### Benefits

1. **DRY** — One struct definition, one set of field handling
2. **Traits for free** — `impl<K: Kind, I: Index> Named for Node<K, I>` once
3. **Capability traits** — Can implement conditionally based on K or I
4. **Parser simplification** — Parse into common structure, vary by kind
5. **Lowering simplification** — No per-primitive loops, generic over K
6. **Type safety preserved** — `Signal` and `Field` are still distinct types

### Execution Structure

Each execution has explicit name + phase. No magic derivation.

```rust
struct Execution {
    name: String,           // "resolve", "measure", "apply", etc.
    phase: Phase,           // Explicit, not derived from name
    body: ExpressionBlock,  // The code
    reads: Vec<Path>,       // Inferred dependencies
}

enum Phase {
    Configure,  // t=0 setup (initial, warmup)
    Collect,    // gather inputs
    Resolve,    // compute new state
    Fracture,   // detect tension
    Measure,    // emit fields (observer)
    PostHoc,    // analysis after simulation
}
```

**Design decision:** Name and phase are independent. Even if `resolve` is *usually* `Phase::Resolve`, we don't assume — it's explicit. No mapping table in the type system.

**Validation:** What (name, phase) combinations are valid for which Kind is a validation concern, not a type system constraint. Keep `Kind` minimal.

### Kind Trait

Minimal — just holds kind-specific data:

```rust
trait Kind {}

// Primitives - kind-specific data only
struct SignalKind;
struct FieldKind { topology: Option<Topology> }
struct FractureKind;
struct OperatorKind;
struct ImpulseKind;
struct ChronicleKind;
struct AnalyzerKind { requirements: Requirements }
struct FunctionKind;

// Declarations - kind-specific data only
struct EntityKind { count: CountSpec }
struct StratumKind { stride: Option<u32> }
struct EraKind { transitions: Vec<Transition>, strata_config: StrataConfig }
struct TypeKind { fields: Vec<TypeField> }
```

No methods mapping names to phases. No validation logic in traits.

### Execution with Traceability

Each execution is self-contained with full context for debugging:

```rust
struct Execution {
    // What to do
    name: String,
    phase: Phase,
    body: ExpressionBlock,
    reads: Vec<Path>,
    
    // Traceability (where it came from)
    source_path: Path,
    source_kind: &'static str,  // "signal", "field", "operator", etc.
    source_span: Span,
    source_file: Option<PathBuf>,
}
```

**Why duplicate context from Node?**
- Executions are created once at compile time
- Clear ownership — execution knows its origin
- No need to carry node references through the DAG
- Error messages are self-contained: "signal terra.core.temp resolve block failed at core/thermal.cdsl:42"

**Kind trait with name:**

```rust
trait Kind {
    const KIND_NAME: &'static str;
}

impl Kind for SignalKind { const KIND_NAME: &'static str = "signal"; }
impl Kind for FieldKind { const KIND_NAME: &'static str = "field"; }
// etc.
```

---

### DAG Builder Interface

```rust
trait Schedulable {
    fn executions(&self) -> &[Execution];
    fn stratum(&self) -> Option<&StratumId>;
    fn entity(&self) -> Option<&EntityId>;
}

// Only primitives are schedulable
trait PrimitiveKind: Kind {}
trait DeclarationKind: Kind {}

impl<K: PrimitiveKind, I: Index> Schedulable for Node<K, I> { ... }
```

DAG builder only sees primitives. Declarations don't implement `Schedulable`.

---

### Scoping Hierarchy

Scoping cascades down through layers:

```
Scenario (HasScoping)
    └── Module (HasScoping)
            └── Node (HasScoping)
```

When resolving `config.some_value`:
1. Check Node's scoping
2. Check Module's scoping (walk up by path)
3. Check Scenario's scoping
4. Error if not found

**Key insight:** Module is NOT `Node<ModuleKind>`. It's a separate struct:

```rust
struct Scenario {
    scoping: Option<Scoping>,
    // scenario-specific fields
}

struct Module {
    path: Path,
    file: PathBuf,
    scoping: Option<Scoping>,
    assertions: Vec<Assertion>,
}

struct Node<K: Kind, I: Index = ()> {
    scoping: Option<Scoping>,
    // ... other fields
}
```

The commonality is `HasScoping` trait, not the struct type.

---

### Resolved Questions

1. ~~Should `Kind` have associated types for "valid capabilities"?~~ → No, too much. Validation is separate.
2. ~~How does this affect the DAG builder?~~ → Sees `Schedulable` trait, only primitives implement it.
3. ~~What about Module?~~ → Separate struct, NOT `Node<ModuleKind>`. Shares `HasScoping` trait.
4. ~~Declarations with unused fields?~~ → Keep unified `Node<K, I>`, but only `K: PrimitiveKind` implements `Schedulable`.

### Open Questions

1. How does parsing map to this unified structure?

---

## Type System Design

### Terminology

| Term | What it is | Examples |
|------|------------|----------|
| **Kernel-type** | Fundamental shapes the runtime/GPU understands | `Scalar`, `Vec<N>`, `Mat<N,M>`, `Tensor`, `Grid`, `Seq` |
| **User-type** | Groupings/bundles of kernel-types | `type OrbitalElements { semi_major: Scalar<m>, ... }` |

### Unified KernelType

```rust
struct KernelType {
    shape: Shape,
    unit: Option<Unit>,
    bounds: Option<Bounds>,
}

enum Shape {
    Scalar,
    Vector { dim: u8 },
    Matrix { rows: u8, cols: u8 },
    Tensor { rows: u8, cols: u8, constraints: Vec<TensorConstraint> },
    Grid { width: u32, height: u32, element: Box<KernelType> },
    Seq { element: Box<KernelType>, constraints: Vec<SeqConstraint> },
}
```

**Key decisions:**
- `Shape` is an enum, not a generic trait — simpler, still unified
- `unit` and `bounds` live on the type, not elsewhere
- `Grid` and `Seq` contain nested `KernelType` for element type

### FieldType (for UserType fields)

A field can be a kernel-type or a reference to another user-type:

```rust
enum FieldType {
    Kernel(KernelType),
    User(Path),  // reference to another UserType by name
}
```

### UserType

User-types are bundles of fields that can nest other user-types:

```rust
struct UserType {
    name: Path,
    fields: Vec<(String, FieldType)>,
    scoping: Option<Scoping>,
    assertions: Vec<Assertion>,
}
```

Example:
```cdsl
type Position {
    x: Scalar<m>
    y: Scalar<m>
    z: Scalar<m>
}

type OrbitalState {
    position: Position,      // UserType as field
    velocity: Vec3<m/s>,     // KernelType as field
}
```

### UserType is a Declaration

In our Node model, UserType fits as a Declaration (structural, no executions):

```rust
struct TypeKind {
    fields: Vec<(String, FieldType)>,
}

type TypeDef = Node<TypeKind>;  // Has Named + HasScoping + HasAssertions
```

### Type System Pain Points to Fix

1. **Named types don't resolve** — currently `TypeExpr::Named(_)` silently becomes `Scalar`
2. **Type checking dead code** — 470 lines in `typecheck.rs` never called
3. **Bounds decorative** — never validated at compile or runtime  
4. **Dimensional analysis unused** — `m/s + K` compiles fine
5. **AST/IR duplication** — should unify or derive one from other
6. **Hardcoded Vec2/Vec3/Vec4** — should be `Vector { dim: N }`

---

## Execution Context Design

### Current Problems

1. **7+ context structs with 90% identical code** (~600-700 lines duplicated)
2. **`prev` means different things** — previous tick vs current value vs warmup iteration
3. **Member signals completely special-cased** — own interpreter, context, value type
4. **No unified block abstraction** — 6+ wrapper types all being `{ body: Spanned<Expr> }`
5. **Phase constraints enforced by dummy values** — `MeasureContext.prev()` returns `0.0`
6. **Parameter field access is string manipulation** — no structured "my config" access

### Solution: Capability Traits

Instead of one trait with 20 methods returning dummies, use composable capability traits:

```rust
// Base — everyone has this
trait BaseContext {
    fn signal(&self, name: &str) -> Value;
    fn constant(&self, name: &str) -> Value;
    fn config(&self, name: &str) -> Value;
    fn scoping(&self) -> &Scoping;
}

// Capability traits — only implement what's available
trait HasPrev {
    fn prev(&self) -> &Value;
}

trait HasInputs {
    fn inputs(&self) -> f64;
}

trait HasDt {
    fn dt(&self) -> f64;
}

trait HasPayload {
    fn payload(&self) -> &Value;
    fn payload_field(&self, component: &str) -> Value;
}

trait CanEmit {
    fn emit_signal(&self, target: &str, value: Value);
}

trait HasIndex {
    fn self_field(&self, component: &str) -> Value;
    fn other_field(&self, component: &str) -> Value;
    fn index(&self) -> usize;
}
```

### Phase Contexts Built from Capabilities

| Phase | BaseContext | HasPrev | HasInputs | HasDt | HasPayload | CanEmit | HasIndex |
|-------|-------------|---------|-----------|-------|------------|---------|----------|
| Configure | ✓ | - | - | - | - | - | ? |
| Collect | ✓ | - | - | ✓ | - | - | - |
| Resolve | ✓ | ✓ | ✓ | ✓ | - | - | ? |
| Fracture | ✓ | - | - | ✓ | - | ✓ | ? |
| Measure | ✓ | - | - | ✓ | - | - | ? |
| Apply | ✓ | - | - | ✓ | ✓ | ✓ | - |

```rust
// Configure — minimal
struct ConfigureContext<'a> { base: &'a BaseContextData }
impl BaseContext for ConfigureContext<'_> { ... }

// Resolve — has prev, inputs, dt
struct ResolveContext<'a> {
    base: &'a BaseContextData,
    prev: Value,
    inputs: f64,
    dt: f64,
}
impl BaseContext for ResolveContext<'_> { ... }
impl HasPrev for ResolveContext<'_> { ... }
impl HasInputs for ResolveContext<'_> { ... }
impl HasDt for ResolveContext<'_> { ... }

// Measure — has dt only
struct MeasureContext<'a> {
    base: &'a BaseContextData,
    dt: f64,
}
impl BaseContext for MeasureContext<'_> { ... }
impl HasDt for MeasureContext<'_> { ... }
// No HasPrev — compile error if block uses prev!
```

### Compile-Time Enforcement

Expression compilation requires specific trait bounds:

```rust
fn compile_prev<C: HasPrev>(ctx: &C) -> CompiledExpr { ... }
fn compile_inputs<C: HasInputs>(ctx: &C) -> CompiledExpr { ... }
fn compile_signal_read<C: BaseContext>(ctx: &C, name: &str) -> CompiledExpr { ... }
```

If a measure block uses `prev`:
- **Before:** Returns dummy value `0.0` at runtime
- **After:** Compile-time error — `MeasureContext` doesn't implement `HasPrev`

### Benefits

1. **No dummy values** — capability is present or absent
2. **Compile-time errors** — can't use `prev` in measure block
3. **No code duplication** — shared BaseContextData
4. **Clear semantics** — each capability has one meaning
5. **Members unified** — just add `HasIndex` to context, no special-casing

### Indexed Context Wrapper

Indexing (per-entity) is orthogonal to phase. Use a wrapper that composes:

```rust
// Index wrapper — adds spatial capabilities to any phase context
struct Indexed<C> {
    inner: C,
    entity: EntityId,
    instance: usize,
    entity_data: EntityData,
}

// Forward all inner capabilities
impl<C: BaseContext> BaseContext for Indexed<C> {
    fn signal(&self, name: &str) -> Value { self.inner.signal(name) }
    fn constant(&self, name: &str) -> Value { self.inner.constant(name) }
    fn config(&self, name: &str) -> Value { self.inner.config(name) }
    fn scoping(&self) -> &Scoping { self.inner.scoping() }
}

impl<C: HasPrev> HasPrev for Indexed<C> {
    fn prev(&self) -> &Value { self.inner.prev() }
}

impl<C: HasDt> HasDt for Indexed<C> {
    fn dt(&self) -> f64 { self.inner.dt() }
}

// ... forward all other capabilities

// Add index-specific capabilities
impl<C> HasIndex for Indexed<C> {
    fn entity(&self) -> &EntityId { &self.entity }
    fn instance(&self) -> usize { self.instance }
    fn self_field(&self, name: &str) -> Value { ... }
    fn other_field(&self, name: &str) -> Value { ... }
}
```

### Context Type Examples

| Primitive | Index | Context Type |
|-----------|-------|--------------|
| Global Signal | `()` | `ResolveContext` |
| Member | `EntityId` | `Indexed<ResolveContext>` |
| Global Field | `()` | `MeasureContext` |
| Entity Field | `EntityId` | `Indexed<MeasureContext>` |
| Global Fracture | `()` | `FractureContext` |
| Entity Fracture | `EntityId` | `Indexed<FractureContext>` |
| Impulse | `()` | `ApplyContext` (has `HasPayload + CanEmit`) |

### DAG Executor

Generic over context, wraps with `Indexed` based on `I`:

```rust
fn execute_node<K: Kind, I: Index, C: BaseContext>(
    ctx: C,
    node: &Node<K, I>,
) {
    match I::entity() {
        Some(entity_id) => {
            // Wrap and iterate over instances
            for instance in 0..entity_count {
                let indexed_ctx = Indexed { inner: &ctx, entity: entity_id, instance, ... };
                execute_body(&indexed_ctx, &node.executions);
            }
        }
        None => {
            // Global, execute directly
            execute_body(&ctx, &node.executions);
        }
    }
}
```

---

## Path Resolution Design

### Current Problems

1. **Order-dependent fallback** — local → special → constant → config → signal (assumes signal)
2. **Greedy path parsing** — `signal.foo.x` parsed as one path, lowerer guesses where to split
3. **No validation** — typo silently becomes a signal reference
4. **Special forms scattered** — `prev`, `dt`, `self` handled ad-hoc

### Solution: Context-Driven Resolution

The expression compiler knows what's available based on capability traits.

### Keywords (Require Capabilities)

| Keyword | Requires | Description |
|---------|----------|-------------|
| `prev` | `HasPrev` | Previous tick value |
| `prev.x` | `HasPrev` | Previous tick component |
| `self.field` | `HasIndex` | Entity field access |
| `other.field` | `HasIndex` | Other instance field (n-body) |
| `inputs` | `HasInputs` | Accumulated inputs |
| `dt` | `HasDt` | Time step |
| `payload` | `HasPayload` | Impulse payload |
| `payload.field` | `HasPayload` | Impulse payload field |

If a keyword is used without the capability, **compile-time error**.

### Resolution Order (Non-Keywords)

Everything else goes through scoping:

```rust
fn resolve_scoped_path<C: BaseContext>(path: &Path, ctx: &C) -> Result<CompiledExpr, Error> {
    // 1. Locals (let bindings in scope)
    if let Some(local) = ctx.resolve_local(path) {
        return Ok(local);
    }
    
    // 2. Scoping hierarchy (config/const: Node → Module → Scenario)
    if let Some(scoped) = ctx.resolve_scoped(path) {
        return Ok(scoped);
    }
    
    // 3. Signals (must exist in world)
    if let Some(signal) = ctx.resolve_signal(path) {
        return Ok(signal);
    }
    
    // 4. Not found — compile error (no silent fallback!)
    Err(Error::UnresolvedPath(path.clone()))
}
```

### Component Access

For paths like `signal.foo.x`, the compiler:
1. Tries full path as signal — not found
2. Tries `signal.foo` as signal — found
3. Remaining `.x` is component access

```rust
fn compile_field_access(object: CompiledExpr, field: &str) -> CompiledExpr {
    match field {
        "x" | "y" | "z" | "w" => CompiledExpr::Component { object, index: component_index(field) },
        _ => CompiledExpr::FieldAccess { object, field: field.to_string() }
    }
}
```

### Key Benefits

1. **Capabilities enforced at compile time** — use `prev` in measure block → error
2. **Scoping hierarchy respected** — config walks Node → Module → Scenario
3. **No silent fallback** — unresolved path is an error
4. **Clean keyword handling** — explicit list, explicit requirements

---

## Typed Expressions Design

### Current Problem

`CompiledExpr` doesn't carry type information:
- Only `Option<Unit>` on some variants
- Type checking exists but isn't wired
- Errors at runtime with panics

### Solution: TypedExpr (Replaces CompiledExpr)

One expression type that carries full type information:

```rust
struct TypedExpr {
    expr: ExprKind,
    ty: KernelType,  // shape + unit + bounds
}

enum ExprKind {
    // Literals
    Literal(f64),
    Vector(Vec<TypedExpr>),
    
    // Variables
    Local(String),
    Signal(SignalId),
    Config(Path),
    Const(Path),
    
    // Context-dependent (validated by capabilities)
    Prev,
    PrevComponent(u8),
    Inputs,
    Dt,
    SelfField(String),
    OtherField(String),
    Payload,
    PayloadField(String),
    
    // Operations
    Binary { op: BinaryOp, left: Box<TypedExpr>, right: Box<TypedExpr> },
    Unary { op: UnaryOp, operand: Box<TypedExpr> },
    Component { object: Box<TypedExpr>, index: u8 },
    FieldAccess { object: Box<TypedExpr>, field: String },
    
    // Calls
    KernelCall { kernel: KernelId, args: Vec<TypedExpr> },
    
    // Control flow
    If { condition: Box<TypedExpr>, then_branch: Box<TypedExpr>, else_branch: Box<TypedExpr> },
    Let { name: String, value: Box<TypedExpr>, body: Box<TypedExpr> },
    
    // Entity operations
    Aggregate { op: AggregateOp, entity: EntityId, body: Box<TypedExpr> },
    // ...
}
```

### Bidirectional Type Inference

Untyped literals infer type from context:

```rust
signal.temp + 42
//     ↑         ↑
//  Scalar<K>   Scalar<?>  → infer Scalar<K>
```

```rust
fn infer_binary(op: BinaryOp, left: &Expr, right: &Expr, ctx: &TypeContext) -> Result<TypedExpr, TypeError> {
    let left_ty = try_infer_type(left, ctx);
    let right_ty = try_infer_type(right, ctx);
    
    match (left_ty, right_ty) {
        // Both known — check compatibility
        (Some(l), Some(r)) => check_and_build(op, l, r),
        
        // Left known, right unknown — propagate to right
        (Some(l), None) => {
            let right_typed = infer_with_hint(right, &l, ctx)?;
            // ...
        },
        
        // Right known, left unknown — propagate to left
        (None, Some(r)) => {
            let left_typed = infer_with_hint(left, &r, ctx)?;
            // ...
        },
        
        // Both unknown — error
        (None, None) => Err(TypeError::CannotInferType),
    }
}
```

### Binary Operation Type Rules

```rust
fn check_binary_op(op: BinaryOp, left: &KernelType, right: &KernelType) -> Result<KernelType, TypeError> {
    match (op, &left.shape, &right.shape) {
        // Scalar ⊕ Scalar → Scalar
        (Add | Sub | Mul | Div, Shape::Scalar, Shape::Scalar) => {
            let unit = combine_units(op, &left.unit, &right.unit)?;
            Ok(KernelType::scalar(unit, None))
        },
        
        // Vec ⊕ Vec → Vec (same dimension required)
        (Add | Sub, Shape::Vector { dim: a }, Shape::Vector { dim: b }) if a == b => {
            check_same_unit(&left.unit, &right.unit)?;
            Ok(KernelType::vector(*a, left.unit.clone(), None))
        },
        
        // Vec ⊕ Vec (different dimension) → Error
        (Add | Sub, Shape::Vector { dim: a }, Shape::Vector { dim: b }) => {
            Err(TypeError::DimensionMismatch { expected: *a, got: *b })
        },
        
        // Scalar * Vec → Vec (broadcast)
        (Mul, Shape::Scalar, Shape::Vector { dim }) |
        (Mul, Shape::Vector { dim }, Shape::Scalar) => {
            let unit = multiply_units(&left.unit, &right.unit);
            Ok(KernelType::vector(*dim, unit, None))
        },
        
        // Mat * Vec → Vec (transform)
        (Mul, Shape::Matrix { rows, cols }, Shape::Vector { dim }) if cols == dim => {
            Ok(KernelType::vector(*rows, right.unit.clone(), None))
        },
        
        // Comparison → Scalar (dimensionless 0/1)
        (Eq | Ne | Lt | Le | Gt | Ge, _, _) => {
            Ok(KernelType::scalar(None, None))
        },
        
        _ => Err(TypeError::InvalidOperation { op, left: left.clone(), right: right.clone() }),
    }
}
```

### Unit Combination Rules

```rust
fn combine_units(op: BinaryOp, left: &Option<Unit>, right: &Option<Unit>) -> Result<Option<Unit>, TypeError> {
    match op {
        Add | Sub => {
            // Must have same units (or one unspecified)
            match (left, right) {
                (None, None) => Ok(None),
                (Some(l), Some(r)) if l == r => Ok(Some(l.clone())),
                (Some(l), Some(r)) => Err(TypeError::UnitMismatch { left: l.clone(), right: r.clone() }),
                _ => Ok(left.clone().or(right.clone())),
            }
        },
        Mul => Ok(multiply_units(left, right)),  // m * s → m·s
        Div => Ok(divide_units(left, right)),    // m / s → m/s
        _ => Ok(None),
    }
}
```

### Bounds Validation

Bounds are tracked on `KernelType` but **not validated during expression compilation**.

Validation happens in the **assertion phase**:
- After signal resolves → check value against declared bounds
- After user type field set → check against field bounds
- Explicit assertions can reference bounds

```rust
struct KernelType {
    shape: Shape,
    unit: Option<Unit>,
    bounds: Option<Bounds>,  // tracked, validated separately
}
```

### Key Benefits

1. **DRY** — `TypedExpr` replaces `CompiledExpr`, single expression type
2. **Shape validated at compile time** — Vec2 + Vec3 is error
3. **Units validated at compile time** — m/s + K is error
4. **Bidirectional inference** — untyped literals infer from context
5. **Bounds tracked** — validated in assertion phase, not expression compilation

---

## Kernel Validation Design

### Current Problem

- Namespace existence checked at lowering
- Function existence, arg count, arg types — NOT checked
- Errors only surface at runtime

### Solution: Kernel Signature Registry (Hardcoded in Rust)

```rust
struct KernelSignature {
    namespace: String,
    name: String,
    params: Vec<KernelParam>,
    return_shape: ShapeConstraint,
    return_unit: UnitTransform,
}

struct KernelParam {
    name: String,
    shape: ShapeConstraint,
    default: Option<f64>,
}
```

### Shape Constraints (Pragmatic Polymorphism)

```rust
enum ShapeConstraint {
    Exact(Shape),        // Must be exactly this shape
    AnyVector,           // Vec2, Vec3, or Vec4
    AnyMatrix,           // Mat2, Mat3, or Mat4
    SameAs(usize),       // Same shape as param at index
}
```

### Unit Transformations

```rust
enum UnitTransform {
    Preserve,              // output = input unit
    Dimensionless,         // output has no unit
    Multiply(Vec<usize>),  // multiply units of params at indices
    Divide(usize, usize),  // divide param[0] unit by param[1] unit
    Sqrt(usize),           // sqrt of param's unit
    FromParam(usize),      // same unit as param at index
}
```

### Example Signatures

```rust
// maths.sin(x: Scalar) -> Scalar (dimensionless)
KernelSignature {
    name: "sin",
    params: vec![KernelParam { name: "x", shape: Exact(Scalar) }],
    return_shape: Exact(Scalar),
    return_unit: Dimensionless,
}

// maths.clamp(value, min, max) -> same unit as value
KernelSignature {
    name: "clamp",
    params: vec![
        KernelParam { name: "value", shape: Exact(Scalar) },
        KernelParam { name: "min", shape: Exact(Scalar) },
        KernelParam { name: "max", shape: Exact(Scalar) },
    ],
    return_shape: Exact(Scalar),
    return_unit: FromParam(0),
}

// vector.dot(a, b) -> Scalar, unit = a.unit * b.unit
KernelSignature {
    name: "dot",
    params: vec![
        KernelParam { name: "a", shape: AnyVector },
        KernelParam { name: "b", shape: SameAs(0) },
    ],
    return_shape: Exact(Scalar),
    return_unit: Multiply(vec![0, 1]),
}

// vector.normalize(v) -> same shape, dimensionless
KernelSignature {
    name: "normalize",
    params: vec![KernelParam { name: "v", shape: AnyVector }],
    return_shape: SameAs(0),
    return_unit: Dimensionless,
}

// vector.length(v) -> Scalar, same unit as input
KernelSignature {
    name: "length",
    params: vec![KernelParam { name: "v", shape: AnyVector }],
    return_shape: Exact(Scalar),
    return_unit: FromParam(0),
}
```

### Validation at Compile Time

```rust
fn validate_kernel_call(
    kernel: &KernelId,
    args: &[TypedExpr],
    registry: &KernelRegistry,
) -> Result<KernelType, TypeError> {
    // 1. Kernel exists?
    let sig = registry.get(kernel)
        .ok_or(TypeError::UnknownKernel(kernel.clone()))?;
    
    // 2. Arg count matches?
    let required = sig.params.iter().filter(|p| p.default.is_none()).count();
    if args.len() < required || args.len() > sig.params.len() {
        return Err(TypeError::ArgCountMismatch {
            kernel: kernel.clone(),
            expected: required..=sig.params.len(),
            got: args.len(),
        });
    }
    
    // 3. Arg shapes match? (resolve SameAs constraints)
    let resolved_shapes = resolve_shape_constraints(args, &sig.params)?;
    
    // 4. Compute return type
    let return_shape = resolve_return_shape(&sig.return_shape, &resolved_shapes);
    let return_unit = compute_return_unit(&sig.return_unit, args);
    
    Ok(KernelType { shape: return_shape, unit: return_unit, bounds: None })
}
```

### Key Benefits

1. **Compile-time errors** — unknown kernel, wrong arg count, wrong types
2. **Shape polymorphism** — `dot` works on Vec2, Vec3, Vec4
3. **Unit tracking** — dimensional analysis through kernel calls
4. **Signatures in code** — no external files, discoverable via IDE

---

## Assertion Design

### Key Insight: Assert = Measure + Validation

Assertions run in the **Measure phase** — they're observers that also validate.

### Assertion Context

```rust
struct AssertContext<'a> {
    base: &'a BaseContextData,
    current: Value,     // the value being asserted (just resolved)
    prev: Value,        // previous tick's resolved value
    dt: f64,            // time step
}

impl BaseContext for AssertContext<'_> { ... }
impl HasCurrent for AssertContext<'_> { ... }
impl HasPrev for AssertContext<'_> { ... }
impl HasDt for AssertContext<'_> { ... }
// Optionally Indexed<AssertContext> for member/entity assertions
```

### Assertion Capabilities

| Capability | Available | Notes |
|------------|-----------|-------|
| `BaseContext` | ✓ | signals, config, const |
| `HasCurrent` | ✓ | the just-resolved value |
| `HasPrev` | ✓ | previous tick's resolved value |
| `HasDt` | ✓ | time step |
| `HasIndex` | if indexed | for member/entity assertions |
| `HasInputs` | ✗ | not relevant |
| `HasPayload` | ✗ | not relevant |

### Bounds vs Assertions (Separate Concerns)

**Bounds** — declared on type, validated automatically:

```rust
fn validate_bounds(node: &Node<K, I>, value: &Value) -> Option<BoundsViolation> {
    if let Some(bounds) = &node.output?.bounds {
        if !bounds.contains(value) {
            return Some(BoundsViolation {
                path: node.path.clone(),
                bounds: bounds.clone(),
                actual: value.clone(),
            });
        }
    }
    None
}
```

**Assertions** — user-written conditions with custom messages:

```rust
struct Assertion {
    condition: TypedExpr,  // must evaluate to bool
    severity: Severity,
    message: String,
    span: Span,
}

enum Severity {
    Fatal,   // halt simulation immediately
    Error,   // log, mark failed, continue
    Warn,    // log, continue
}
```

### Example

```cdsl
signal core.temp {
    : Scalar<K, 100..10000>  // bounds checked automatically
    
    resolve { ... }
    
    assert {
        maths.abs(current - prev) / prev < 0.1 : warn, "temperature changing too fast"
    }
    
    assert {
        current > signal.mantle.temp : error, "core cooler than mantle"
    }
}
```

### Execution Order

1. **Resolve phase** — compute new values
2. **Bounds validation** — automatic, based on type bounds
3. **User assertions** — run in measure phase with `AssertContext`
4. **Measure phase** — fields observe state

---

## Next Steps

TBD based on direction chosen.
