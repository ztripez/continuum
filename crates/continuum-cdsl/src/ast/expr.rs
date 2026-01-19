//! Expression system for Continuum DSL
//!
//! This module defines the expression trees that live inside Node<I> execution blocks.
//! Every expression carries its type, making type errors compile errors rather than
//! runtime errors.
//!
//! # Architecture
//!
//! The expression system is built on three core types:
//!
//! 1. **[`TypedExpr`]** - Expressions with type information (after type resolution)
//! 2. **[`ExprKind`]** - The actual expression variants (literals, calls, bindings, etc)
//! 3. **[`KernelId`]** - Namespaced kernel operations (maths.add, vector.dot, etc)
//!
//! # Design Principles
//!
//! ## Explicit Over Implicit
//!
//! - **No silent coercion** - Types don't silently become other types
//! - **All operators are kernel calls** - `a + b` desugars to `maths.add(a, b)`
//! - **if/then/else is eager** - `if c { t } else { e }` → `logic.select(c, t, e)` (both branches evaluate)
//! - **No shortcuts** - Struct literals forbid field shorthand: `{x}` is invalid, must write `{x: x}`
//!
//! ## Types Prove Correctness
//!
//! Every expression carries its [`Type`], making invalid operations unrepresentable:
//!
//! ```rust,ignore
//! // Type mismatch caught at compile time, not runtime
//! let distance: Scalar<m> = ...;
//! let time: Scalar<s> = ...;
//! let speed = distance / time;  // Type: Scalar<m/s> ✓
//! let invalid = distance + time; // ERROR: unit mismatch
//! ```
//!
//! ## Seq<T> is Intermediate-Only
//!
//! [`Type::Seq`] is produced by `map` and must be consumed by an aggregate:
//!
//! ```cdsl
//! // Valid - Seq consumed by sum
//! let total = sum(plates, |p| p.area)
//!
//! // Invalid - Seq cannot be stored
//! signal masses : Seq<Scalar<kg>>  // ERROR: Seq not allowed in signals
//! ```
//!
//! # Expression Variants
//!
//! ## Literals
//!
//! - [`ExprKind::Literal`] - Numeric literals with optional units: `100.0<m>`, `3.14<>`
//! - [`ExprKind::Vector`] - Vector literals: `[x, y, z]`
//!
//! ## References
//!
//! - [`ExprKind::Local`] - Let-bound variables
//! - [`ExprKind::Signal`] - Signal references (causal phases only)
//! - [`ExprKind::Field`] - Field references (Analyzer/observer only)
//! - [`ExprKind::Config`] - Config values from scoping
//! - [`ExprKind::Const`] - Const values from scoping
//!
//! ## Context Values
//!
//! These require specific capabilities from the phase:
//!
//! - [`ExprKind::Prev`] - Previous tick value (requires `Capability::Prev`)
//! - [`ExprKind::Current`] - Just-resolved value (requires `Capability::Current`)
//! - [`ExprKind::Inputs`] - Accumulated inputs (requires `Capability::Inputs`)
//! - [`ExprKind::Dt`] - Time step (requires `Capability::Dt`)
//! - [`ExprKind::Self_`] - Current entity instance (requires `Capability::Index`)
//! - [`ExprKind::Other`] - Other entity instance (requires `Capability::Index`, n-body)
//! - [`ExprKind::Payload`] - Impulse payload (requires `Capability::Payload`)
//!
//! ## Binding Forms
//!
//! These introduce new variable bindings:
//!
//! - [`ExprKind::Let`] - Local binding: `let x = value in body`
//! - [`ExprKind::Aggregate`] - Entity iteration: `sum(plates, |p| p.mass)`
//! - [`ExprKind::Fold`] - Custom reduction: `fold(plates, 0, |acc, p| acc + p.mass)`
//!
//! ## Operations
//!
//! - [`ExprKind::Call`] - All operators desugar to kernel calls
//! - [`ExprKind::Struct`] - User type construction with explicit field names
//! - [`ExprKind::FieldAccess`] - Field access on user types, vectors, Prev, Self_, Payload
//!
//! # Kernel Operations
//!
//! All operations are kernel calls with explicit namespaces:
//!
//! | Syntax | Kernel | Namespace |
//! |--------|--------|-----------|
//! | `a + b` | `maths.add(a, b)` | maths |
//! | `a * b` | `maths.mul(a, b)` | maths |
//! | `sin(x)` | `maths.sin(x)` | maths |
//! | `dot(a, b)` | `vector.dot(a, b)` | vector |
//! | `a < b` | `compare.lt(a, b)` | compare |
//! | `a && b` | `logic.and(a, b)` | logic |
//! | `if c { t } else { e }` | `logic.select(c, t, e)` | logic |
//!
//! # Purity and Effects
//!
//! Kernels are classified by purity:
//!
//! - **Pure** - No side effects, deterministic (maths, vector, logic, compare, rng)
//! - **Effect** - Mutates state or produces artifacts (emit, spawn, destroy, log)
//!
//! Effect discipline is enforced by phase:
//!
//! | Phase | Pure | Effect | Rationale |
//! |-------|------|--------|-----------|
//! | Configure | ✓ | ✗ | Setup only |
//! | Collect | ✓ | ✓ | Emissions happen here |
//! | Resolve | ✓ | ✗ | Computing authoritative state |
//! | Fracture | ✓ | ✓ | May spawn/destroy entities |
//! | Measure | ✓ | ✗ | Fields are derived values |
//! | Assert | ✓ | ✗ | Validation only |
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::ast::{TypedExpr, ExprKind, KernelId, AggregateOp};
//! use continuum_cdsl::foundation::{Type, KernelType, Shape, Unit};
//!
//! // Simple literal
//! let hundred_meters = TypedExpr {
//!     expr: ExprKind::Literal {
//!         value: 100.0,
//!         unit: Some(Unit::meters()),
//!     },
//!     ty: Type::Kernel(KernelType {
//!         shape: Shape::Scalar,
//!         unit: Unit::meters(),
//!         bounds: None,
//!     }),
//!     span,
//! };
//!
//! // Kernel call: a + b
//! let sum = TypedExpr {
//!     expr: ExprKind::Call {
//!         kernel: KernelId { namespace: "maths", name: "add" },
//!         args: vec![a, b],
//!     },
//!     ty: Type::Kernel(result_type),
//!     span,
//! };
//!
//! // Aggregate: sum(plates, |p| p.mass)
//! let total_mass = TypedExpr {
//!     expr: ExprKind::Aggregate {
//!         op: AggregateOp::Sum,
//!         entity: EntityId::new("plate"),
//!         binding: "p".to_string(),
//!         body: Box::new(mass_expr),
//!     },
//!     ty: Type::Kernel(scalar_kg),
//!     span,
//! };
//! ```

use crate::foundation::{Path, Span, Type, UserTypeId};
use continuum_kernel_types::KernelId;

use crate::foundation::EntityId;

/// Kernel operation identifier
///
/// All operations in Continuum DSL are kernel calls. Most use namespaces
/// (`maths.*`, `vector.*`, `logic.*`, `compare.*`), but effect operations
/// are bare names (`emit`, `spawn`, `destroy`, `log`).
///
/// # Structure
///
/// - **namespace** - Category of operation (e.g., "maths", "vector", "logic", or "" for bare names)
/// - **name** - Specific operation (e.g., "add", "dot", "select", "emit")
///
/// Both fields are `&'static str` because kernel IDs are statically known at
/// compile time and come from a fixed registry.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::KernelId;
///
/// // Arithmetic: a + b → maths.add(a, b)
/// let add = KernelId { namespace: "maths", name: "add" };
///
/// // Vector operations: dot(a, b) → vector.dot(a, b)
/// let dot = KernelId { namespace: "vector", name: "dot" };
///
/// // Conditional: if c { t } else { e } → logic.select(c, t, e)
/// let select = KernelId { namespace: "logic", name: "select" };
///
/// // Effects: emit(target, value) - bare name, no namespace
/// let emit = KernelId { namespace: "", name: "emit" };
/// ```
///
/// # Namespaces
///
/// | Namespace | Operations | Purity |
/// |-----------|------------|--------|
/// | `maths` | add, sub, mul, div, sin, cos, sqrt, etc. | Pure |
/// | `vector` | dot, cross, norm, normalize, etc. | Pure |
/// | `matrix` | mul, transpose, determinant, etc. | Pure |
/// | `logic` | and, or, not, select | Pure |
/// | `compare` | lt, le, gt, ge, eq, ne | Pure |
/// | `rng` | uniform, normal, etc. (seeded) | Pure |
/// | *(bare)* | emit, spawn, destroy, log | Effect |
///
/// **NOTE:** KernelId is now imported from `continuum_kernel_types` (single source of truth)

/// Aggregate operations for entity iteration
///
/// Aggregates iterate over entity instances and combine values. All aggregates
/// produce a single result from a collection, except [`Map`] which produces
/// [`Type::Seq`] (must be consumed by another aggregate).
///
/// # Determinism
///
/// - **Iteration order** - Always lexical `InstanceId` order (deterministic)
/// - **Floating-point reductions** - Use fixed-tree reduction for bitwise stability
/// - **Fold with non-commutative function** - Order = InstanceId order
///
/// # Examples
///
/// ```cdsl
/// // Sum: sum(plates, |p| p.mass) → Scalar<kg>
/// // Map: map(plates, |p| p.velocity) → Seq<Vector<3, m/s>>
/// // Max: max(plates, |p| p.temperature) → Scalar<K>
/// // Count: count(plates, |p| p.active) → Scalar<>
/// // Any: any(plates, |p| p.fractured) → Bool
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AggregateOp {
    /// Sum all values - requires numeric type with addition
    ///
    /// Type: `Seq<T>` → `T` where `T` supports `maths.add`
    Sum,

    /// Map over all values - produces Seq<T> (must be consumed)
    ///
    /// Type: `(Entity, |e| -> T)` → `Seq<T>`
    ///
    /// **Important:** `Seq<T>` is intermediate-only and cannot be stored in
    /// signals or let bindings (unless immediately consumed by another aggregate).
    Map,

    /// Maximum value - requires ordered type
    ///
    /// Type: `Seq<T>` → `T` where `T` supports `compare.gt`
    Max,

    /// Minimum value - requires ordered type
    ///
    /// Type: `Seq<T>` → `T` where `T` supports `compare.lt`
    Min,

    /// Count matching elements - body must return Bool
    ///
    /// Type: `(Entity, |e| -> Bool)` → `Scalar<>` (dimensionless)
    Count,

    /// Any element satisfies predicate - body must return Bool
    ///
    /// Type: `(Entity, |e| -> Bool)` → `Bool`
    Any,

    /// All elements satisfy predicate - body must return Bool
    ///
    /// Type: `(Entity, |e| -> Bool)` → `Bool`
    All,
}

/// Expression variants for Continuum DSL
///
/// `ExprKind` defines all expression forms in the IR. Every expression has an
/// associated [`Type`] (wrapped in [`TypedExpr`]) and source location ([`Span`]).
///
/// # Design Principles
///
/// 1. **All operators are calls** - `a + b` desugars to `Call { kernel: maths.add, ... }`
/// 2. **Binding forms are special** - `Let`, `Aggregate`, `Fold` introduce variable bindings
/// 3. **No implicit coercion** - Types must match exactly or explicitly convert
/// 4. **Effects are explicit** - `emit`, `spawn`, `destroy` are kernel calls with `Effect` purity
///
/// # Validity Rules
///
/// - **Seq<T> intermediate-only** - `Map` produces `Seq<T>`, must be consumed by aggregate
/// - **Unit statement-only** - `Unit` type only valid in statement blocks, not pure expressions
/// - **Capability requirements** - `Prev`, `Current`, etc. require specific phase capabilities
/// - **Phase restrictions** - `Signal` only in causal phases, `Field` only in observer/Analyzer
///
/// # Examples
///
/// See individual variant documentation for usage examples.
#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    // === Literals ===
    /// Numeric literal with optional unit
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// 100.0<m>     // Scalar<m>
    /// 3.14<>       // Scalar<> (dimensionless)
    /// 273.15<K>    // Scalar<K>
    /// ```
    ///
    /// **Unit rules:**
    /// - `Some(unit)` - Literal has explicit unit (e.g., `100.0<m>`)
    /// - `None` - Unitless literal, type inference determines unit
    Literal {
        /// Numeric value
        value: f64,
        /// Optional unit annotation
        unit: Option<crate::foundation::Unit>,
    },

    /// Vector literal
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// [1.0, 2.0, 3.0]           // Vector<3, _>
    /// [x, y]                     // Vector<2, _>
    /// [0.0<m/s>, 0.0<m/s>]      // Vector<2, m/s>
    /// ```
    ///
    /// **Shape rules:**
    /// - All elements must have the same type (shape + unit)
    /// - Result dimension = number of elements
    /// - Empty vectors are invalid
    Vector(Vec<TypedExpr>),

    // === References ===
    /// Local let-bound variable reference
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// let x = 10.0<m>
    /// let y = x * 2.0  // Local("x")
    /// ```
    Local(String),

    /// Signal reference - only valid in causal phases
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// signal velocity : Vector<3, m/s>
    /// signal position : Vector<3, m> {
    ///     resolve {
    ///         prev + velocity * dt  // Signal(Path::from("velocity"))
    ///     }
    /// }
    /// ```
    ///
    /// **Phase restriction:** Compile error if used in `Measure` phase without
    /// `Capability::Signals` (only observers can read signals in Measure).
    Signal(Path),

    /// Field reference - only valid in Analyzer/observer contexts
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// analyzer check_temperature {
    ///     let temp = field(temperature)  // Field(Path::from("temperature"))
    ///     assert(temp < 1000.0<K>)
    /// }
    /// ```
    ///
    /// **Phase restriction:** Compile error if used in causal phases (Configure,
    /// Collect, Resolve, Fracture). Fields are observation-only.
    Field(Path),

    /// Config value reference
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// signal temp : Scalar<K> {
    ///     config {
    ///         initial_temp: Scalar<K> = 300.0
    ///     }
    ///     resolve {
    ///         config.initial_temp  // Config(Path::from("initial_temp"))
    ///     }
    /// }
    /// ```
    Config(Path),

    /// Const value reference
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// signal energy : Scalar<J> {
    ///     const {
    ///         BOLTZMANN: Scalar<J/K> = 1.380649e-23
    ///     }
    ///     resolve {
    ///         const.BOLTZMANN * temp  // Const(Path::from("BOLTZMANN"))
    ///     }
    /// }
    /// ```
    Const(Path),

    // === Context values (require capabilities) ===
    /// Previous tick value - requires `Capability::Prev`
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// signal position : Vector<3, m> {
    ///     resolve {
    ///         prev + velocity * dt  // Prev
    ///     }
    /// }
    /// ```
    ///
    /// **Type:** Same as parent signal's output type
    ///
    /// **Capability:** Requires `Capability::Prev` from phase
    Prev,

    /// Just-resolved current value - requires `Capability::Current`
    ///
    /// Used in post-resolution phases (Fracture, Measure) to reference the
    /// value that was just resolved.
    ///
    /// **Type:** Same as parent signal's output type
    ///
    /// **Capability:** Requires `Capability::Current` from phase
    Current,

    /// Accumulated inputs - requires `Capability::Inputs`
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// signal force : Vector<3, N> {
    ///     resolve {
    ///         inputs  // Sum of all emit(force, ...) calls
    ///     }
    /// }
    /// ```
    ///
    /// **Type:** Same as parent signal's output type
    ///
    /// **Capability:** Requires `Capability::Inputs` from phase
    Inputs,

    /// Time step - requires `Capability::Dt`
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// signal position : Vector<3, m> {
    ///     resolve {
    ///         prev + velocity * dt  // Dt
    ///     }
    /// }
    /// ```
    ///
    /// **Type:** `Scalar<s>` (seconds)
    ///
    /// **Capability:** Requires `Capability::Dt` from phase
    Dt,

    /// Current entity instance - requires `Capability::Index`
    ///
    /// Used in per-entity contexts (members) to reference the current instance.
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// member plate.mass : Scalar<kg> {
    ///     resolve {
    ///         self.density * self.area  // Self_
    ///     }
    /// }
    /// ```
    ///
    /// **Type:** User type or instance handle
    ///
    /// **Capability:** Requires `Capability::Index` from phase
    Self_,

    /// Other entity instance - requires `Capability::Index`
    ///
    /// Used in n-body interactions to reference other entity instances.
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// member body.force : Vector<3, N> {
    ///     collect {
    ///         sum(bodies, |other| {
    ///             gravity(self, other)  // Other
    ///         })
    ///     }
    /// }
    /// ```
    ///
    /// **Type:** Same as entity type
    ///
    /// **Capability:** Requires `Capability::Index` from phase
    Other,

    /// Impulse payload - requires `Capability::Payload`
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// impulse collision : CollisionData {
    ///     apply {
    ///         emit(force, payload.impulse)  // Payload
    ///     }
    /// }
    /// ```
    ///
    /// **Type:** Impulse's `payload` type
    ///
    /// **Capability:** Requires `Capability::Payload` from phase
    Payload,

    // === Binding forms (introduce scope) ===
    /// Local variable binding
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// let speed = sqrt(vx * vx + vy * vy)
    /// let kinetic = 0.5 * mass * speed * speed
    /// ```
    ///
    /// Desugars to:
    /// ```rust,ignore
    /// Let {
    ///     name: "speed",
    ///     value: Box::new(sqrt_expr),
    ///     body: Box::new(Let {
    ///         name: "kinetic",
    ///         value: Box::new(kinetic_expr),
    ///         body: Box::new(kinetic_ref),
    ///     }),
    /// }
    /// ```
    Let {
        /// Variable name
        name: String,
        /// Value to bind
        value: Box<TypedExpr>,
        /// Body expression (can reference name)
        body: Box<TypedExpr>,
    },

    /// Aggregate operation over entity instances
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// sum(plates, |p| p.mass)        // AggregateOp::Sum
    /// max(bodies, |b| b.temperature)  // AggregateOp::Max
    /// count(stars, |s| s.active)      // AggregateOp::Count
    /// ```
    ///
    /// **Iteration order:** Lexical `InstanceId` order (deterministic)
    ///
    /// **Purity:** Body must be pure (no effects like `emit`)
    Aggregate {
        /// Aggregate operation
        op: AggregateOp,
        /// Entity to iterate over
        entity: EntityId,
        /// Binding name for loop variable
        binding: String,
        /// Body expression (can reference binding)
        body: Box<TypedExpr>,
    },

    /// Custom fold/reduction over entity instances
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// fold(plates, 0.0<kg>, |total, p| total + p.mass)
    /// fold(bodies, min_temp, |min, b| {
    ///     if b.temp < min { b.temp } else { min }
    /// })
    /// ```
    ///
    /// **Iteration order:** Lexical `InstanceId` order (deterministic)
    ///
    /// **Purity:** Body must be pure
    Fold {
        /// Entity to iterate over
        entity: EntityId,
        /// Initial accumulator value
        init: Box<TypedExpr>,
        /// Accumulator binding name
        acc: String,
        /// Element binding name
        elem: String,
        /// Body expression (can reference acc and elem)
        body: Box<TypedExpr>,
    },

    // === Calls (all operators desugar here) ===
    /// Kernel operation call
    ///
    /// All operators, functions, and effects desugar to kernel calls:
    ///
    /// | Syntax | Kernel |
    /// |--------|--------|
    /// | `a + b` | `maths.add(a, b)` |
    /// | `sin(x)` | `maths.sin(x)` |
    /// | `dot(a, b)` | `vector.dot(a, b)` |
    /// | `a < b` | `compare.lt(a, b)` |
    /// | `if c { t } else { e }` | `logic.select(c, t, e)` |
    /// | `emit(x, v)` | `effect.emit(x, v)` |
    ///
    /// **Purity:** Determined by kernel signature (Pure or Effect)
    ///
    /// **Type checking:** Argument count, shapes, units, and capabilities
    /// all validated against kernel signature at compile time.
    Call {
        /// Kernel identifier
        kernel: KernelId,
        /// Argument expressions
        args: Vec<TypedExpr>,
    },

    // === User types ===
    /// User-defined struct construction
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// OrbitalElements {
    ///     semi_major: 1.5e11<m>,
    ///     eccentricity: 0.017<>,
    ///     inclination: 0.0<rad>
    /// }
    /// ```
    ///
    /// **Strict rules:**
    /// - All fields required (missing field → compile error)
    /// - No extra fields (unknown field → compile error)
    /// - Field order irrelevant (not positional)
    /// - **No shorthand** - `{x}` is forbidden, must write `{x: x}`
    Struct {
        /// User type being constructed
        ty: UserTypeId,
        /// Field name-value pairs
        fields: Vec<(String, TypedExpr)>,
    },

    /// Field access on user types, vectors, or context values
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// orbit.semi_major        // User type field
    /// velocity.x              // Vector component
    /// prev.temperature        // Prev context field
    /// payload.collision_data  // Payload field
    /// ```
    ///
    /// **Vector component rules:**
    /// - `.x/.y/.z/.w` only for `dim in 2..=4`
    /// - `.at(i)` works for any dimension (compile-time literal)
    /// - Desugars to `vector.get(obj, index)` kernel call
    FieldAccess {
        /// Object being accessed
        object: Box<TypedExpr>,
        /// Field name
        field: String,
    },
}

/// Typed expression with type and source location
///
/// Every expression in the IR carries its [`Type`], making type errors compile
/// errors rather than runtime errors. The [`Span`] provides source location
/// for error messages.
///
/// # Type Invariants
///
/// After type resolution, these invariants must hold:
///
/// 1. **Shape/unit propagation** - Kernel calls respect shape/unit constraints
/// 2. **Seq consumption** - `Seq<T>` only appears as direct argument to aggregates
/// 3. **Unit statement-only** - `Unit` type only in statement blocks
/// 4. **Capability validation** - Context values have required capabilities
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::ast::{TypedExpr, ExprKind, KernelId};
/// use continuum_cdsl::foundation::{Type, KernelType, Shape, Unit};
///
/// // Typed literal: 100.0<m>
/// let distance = TypedExpr {
///     expr: ExprKind::Literal {
///         value: 100.0,
///         unit: Some(Unit::meters()),
///     },
///     ty: Type::Kernel(KernelType {
///         shape: Shape::Scalar,
///         unit: Unit::meters(),
///         bounds: None,
///     }),
///     span,
/// };
///
/// // Typed call: sqrt(x)
/// let magnitude = TypedExpr {
///     expr: ExprKind::Call {
///         kernel: KernelId::new("maths", "sqrt"),
///         args: vec![x],
///     },
///     ty: Type::Kernel(KernelType {
///         shape: Shape::Scalar,
///         unit: Unit::meters_per_second(),
///         bounds: Some(Bounds::non_negative()),
///     }),
///     span,
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr {
    /// Expression variant
    pub expr: ExprKind,

    /// Expression type (Kernel, User, Bool, Unit, Seq)
    pub ty: Type,

    /// Source location for error messages
    pub span: Span,
}

impl TypedExpr {
    /// Create a new typed expression
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let expr = TypedExpr::new(
    ///     ExprKind::Literal { value: 10.0, unit: None },
    ///     Type::Kernel(scalar_type),
    ///     span,
    /// );
    /// ```
    pub fn new(expr: ExprKind, ty: Type, span: Span) -> Self {
        Self { expr, ty, span }
    }

    /// Check if expression is pure (no side effects)
    ///
    /// An expression is pure if it doesn't call any `Effect` kernels. This is
    /// used to enforce effect discipline in pure-only phases (Resolve, Measure).
    ///
    /// # Current Implementation
    ///
    /// **TEMPORARY HEURISTIC:** This implementation uses a namespace-based heuristic
    /// (`effect.*` is impure, all others are pure) because the kernel registry is
    /// not yet implemented (Phase 6). Once the kernel registry exists, this method
    /// must be updated to query `KernelSignature::purity` instead.
    ///
    /// **Limitation:** Unknown or unregistered kernels are conservatively treated
    /// as pure. This could allow effect kernels outside the `effect` namespace to
    /// be incorrectly classified. The kernel registry will enforce this correctly.
    ///
    /// # Returns
    ///
    /// - `true` if the expression contains no effect kernels (or uses the conservative
    ///   assumption for unknown kernels)
    /// - `false` if any `effect.*` kernel is reachable in the expression tree
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let add = TypedExpr::new(
    ///     ExprKind::Call {
    ///         kernel: KernelId::new("maths", "add"),
    ///         args: vec![a, b],
    ///     },
    ///     ty,
    ///     span,
    /// );
    /// assert!(add.is_pure());
    ///
    /// let emit = TypedExpr::new(
    ///     ExprKind::Call {
    ///         kernel: KernelId::new("", "emit"),
    ///         args: vec![target, value],
    ///     },
    ///     Type::Unit,
    ///     span,
    /// );
    /// assert!(!emit.is_pure());
    /// ```
    pub fn is_pure(&self) -> bool {
        match &self.expr {
            // Literals and references are always pure
            ExprKind::Literal { .. }
            | ExprKind::Local(_)
            | ExprKind::Signal(_)
            | ExprKind::Field(_)
            | ExprKind::Config(_)
            | ExprKind::Const(_)
            | ExprKind::Prev
            | ExprKind::Current
            | ExprKind::Inputs
            | ExprKind::Dt
            | ExprKind::Self_
            | ExprKind::Other
            | ExprKind::Payload => true,

            // Vector is pure if all elements are pure
            ExprKind::Vector(exprs) => exprs.iter().all(|e| e.is_pure()),

            // Binding forms are pure if their operands are pure
            ExprKind::Let { value, body, .. } => value.is_pure() && body.is_pure(),
            ExprKind::Aggregate { body, .. } => body.is_pure(),
            ExprKind::Fold { init, body, .. } => init.is_pure() && body.is_pure(),

            // Calls depend on kernel purity
            // Look up kernel in registry to determine purity
            ExprKind::Call { kernel, args } => {
                use super::kernel::KernelRegistry;

                let kernel_is_pure = KernelRegistry::global()
                    .get(kernel)
                    .map(|sig| sig.purity.is_pure())
                    .unwrap_or(true); // Unknown kernels assumed pure (conservative)

                kernel_is_pure && args.iter().all(|arg| arg.is_pure())
            }

            // Struct construction and field access are pure if operands are pure
            ExprKind::Struct { fields, .. } => fields.iter().all(|(_, expr)| expr.is_pure()),
            ExprKind::FieldAccess { object, .. } => object.is_pure(),
        }
    }

    /// Walk the expression tree recursively
    ///
    /// This is a generic visitor that traverses the entire `TypedExpr` tree.
    /// It ensures that any logic that needs to inspect the tree (like dependency
    /// extraction or capability validation) is centralized and doesn't miss
    /// any variants when `ExprKind` is expanded.
    pub fn walk<V: ExpressionVisitor>(&self, visitor: &mut V) {
        visitor.visit_expr(self);

        match &self.expr {
            ExprKind::Vector(exprs) => {
                for expr in exprs {
                    expr.walk(visitor);
                }
            }
            ExprKind::Let { value, body, .. } => {
                value.walk(visitor);
                body.walk(visitor);
            }
            ExprKind::Aggregate { body, .. } => {
                body.walk(visitor);
            }
            ExprKind::Fold { init, body, .. } => {
                init.walk(visitor);
                body.walk(visitor);
            }
            ExprKind::Call { args, .. } => {
                for arg in args {
                    arg.walk(visitor);
                }
            }
            ExprKind::Struct { fields, .. } => {
                for (_, expr) in fields {
                    expr.walk(visitor);
                }
            }
            ExprKind::FieldAccess { object, .. } => {
                object.walk(visitor);
            }
            // Leaf nodes
            _ => {}
        }
    }
}

/// Visitor trait for traversing typed expressions
pub trait ExpressionVisitor {
    /// Visit a typed expression node
    fn visit_expr(&mut self, expr: &TypedExpr);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_id_qualified_name() {
        let add = KernelId::new("maths", "add");
        assert_eq!(add.qualified_name(), "maths.add");

        let emit = KernelId::new("", "emit");
        assert_eq!(emit.qualified_name(), "emit"); // Bare name (empty namespace)
    }

    #[test]
    fn kernel_id_equality() {
        let add1 = KernelId::new("maths", "add");
        let add2 = KernelId::new("maths", "add");
        let mul = KernelId::new("maths", "mul");

        assert_eq!(add1, add2);
        assert_ne!(add1, mul);
    }

    #[test]
    fn aggregate_op_variants() {
        // Just verify all variants exist
        let ops = [
            AggregateOp::Sum,
            AggregateOp::Map,
            AggregateOp::Max,
            AggregateOp::Min,
            AggregateOp::Count,
            AggregateOp::Any,
            AggregateOp::All,
        ];
        assert_eq!(ops.len(), 7);
    }

    mod typed_expr_tests {
        use super::*;
        use crate::foundation::{KernelType, Shape, Unit};

        fn make_span() -> Span {
            Span::new(0, 0, 0, 0)
        }

        fn scalar_type() -> Type {
            Type::Kernel(KernelType {
                shape: Shape::Scalar,
                unit: Unit::DIMENSIONLESS,
                bounds: None,
            })
        }

        #[test]
        fn literal_is_pure() {
            let expr = TypedExpr::new(
                ExprKind::Literal {
                    value: 42.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );
            assert!(expr.is_pure());
        }

        #[test]
        fn vector_literal_is_pure() {
            let elem = TypedExpr::new(
                ExprKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );
            let expr = TypedExpr::new(
                ExprKind::Vector(vec![elem.clone(), elem.clone(), elem]),
                scalar_type(),
                make_span(),
            );
            assert!(expr.is_pure());
        }

        #[test]
        fn context_values_are_pure() {
            let contexts = vec![
                ExprKind::Prev,
                ExprKind::Current,
                ExprKind::Inputs,
                ExprKind::Dt,
                ExprKind::Self_,
                ExprKind::Other,
                ExprKind::Payload,
            ];

            for ctx in contexts {
                let expr = TypedExpr::new(ctx, scalar_type(), make_span());
                assert!(expr.is_pure(), "context value should be pure");
            }
        }

        #[test]
        fn references_are_pure() {
            let references = vec![
                ExprKind::Local("x".to_string()),
                ExprKind::Signal(Path::from_str("velocity")),
                ExprKind::Field(Path::from_str("temperature")),
                ExprKind::Config(Path::from_str("initial_temp")),
                ExprKind::Const(Path::from_str("BOLTZMANN")),
            ];

            for ref_kind in references {
                let expr = TypedExpr::new(ref_kind, scalar_type(), make_span());
                assert!(expr.is_pure(), "reference should be pure");
            }
        }

        #[test]
        fn pure_kernel_call_is_pure() {
            let a = TypedExpr::new(
                ExprKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );
            let b = TypedExpr::new(
                ExprKind::Literal {
                    value: 2.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );

            let expr = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("maths", "add"),
                    args: vec![a, b],
                },
                scalar_type(),
                make_span(),
            );
            assert!(expr.is_pure());
        }

        #[test]
        fn effect_kernel_call_is_impure() {
            let target = TypedExpr::new(
                ExprKind::Signal(Path::from_str("force")),
                scalar_type(),
                make_span(),
            );
            let value = TypedExpr::new(
                ExprKind::Literal {
                    value: 10.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );

            let expr = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("", "emit"),
                    args: vec![target, value],
                },
                Type::Unit,
                make_span(),
            );
            assert!(!expr.is_pure());
        }

        #[test]
        fn let_with_pure_body_is_pure() {
            let value = TypedExpr::new(
                ExprKind::Literal {
                    value: 10.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );
            let body = TypedExpr::new(ExprKind::Local("x".to_string()), scalar_type(), make_span());

            let expr = TypedExpr::new(
                ExprKind::Let {
                    name: "x".to_string(),
                    value: Box::new(value),
                    body: Box::new(body),
                },
                scalar_type(),
                make_span(),
            );
            assert!(expr.is_pure());
        }

        #[test]
        fn let_with_impure_value_is_impure() {
            let value = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("", "emit"),
                    args: vec![],
                },
                Type::Unit,
                make_span(),
            );
            let body = TypedExpr::new(
                ExprKind::Literal {
                    value: 0.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );

            let expr = TypedExpr::new(
                ExprKind::Let {
                    name: "x".to_string(),
                    value: Box::new(value),
                    body: Box::new(body),
                },
                scalar_type(),
                make_span(),
            );
            assert!(!expr.is_pure());
        }

        #[test]
        fn aggregate_with_pure_body_is_pure() {
            let body = TypedExpr::new(
                ExprKind::FieldAccess {
                    object: Box::new(TypedExpr::new(ExprKind::Self_, scalar_type(), make_span())),
                    field: "mass".to_string(),
                },
                scalar_type(),
                make_span(),
            );

            let expr = TypedExpr::new(
                ExprKind::Aggregate {
                    op: AggregateOp::Sum,
                    entity: EntityId::new("plate"),
                    binding: "p".to_string(),
                    body: Box::new(body),
                },
                scalar_type(),
                make_span(),
            );
            assert!(expr.is_pure());
        }

        #[test]
        fn struct_construction_with_pure_fields_is_pure() {
            use continuum_foundation::TypeId;

            let field_value = TypedExpr::new(
                ExprKind::Literal {
                    value: 1.5e11,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );

            let orbit_ty = TypeId::from("Orbit");
            let expr = TypedExpr::new(
                ExprKind::Struct {
                    ty: orbit_ty.clone(),
                    fields: vec![("semi_major".to_string(), field_value)],
                },
                Type::User(orbit_ty),
                make_span(),
            );
            assert!(expr.is_pure());
        }

        #[test]
        fn field_access_is_pure_if_object_is_pure() {
            use continuum_foundation::TypeId;

            let orbit_ty = TypeId::from("Orbit");
            let object = TypedExpr::new(
                ExprKind::Signal(Path::from_str("orbit")),
                Type::User(orbit_ty),
                make_span(),
            );

            let expr = TypedExpr::new(
                ExprKind::FieldAccess {
                    object: Box::new(object),
                    field: "semi_major".to_string(),
                },
                scalar_type(),
                make_span(),
            );
            assert!(expr.is_pure());
        }

        #[test]
        fn fold_with_pure_body_is_pure() {
            let init = TypedExpr::new(
                ExprKind::Literal {
                    value: 0.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );
            let body = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("maths", "add"),
                    args: vec![
                        TypedExpr::new(
                            ExprKind::Local("acc".to_string()),
                            scalar_type(),
                            make_span(),
                        ),
                        TypedExpr::new(
                            ExprKind::Local("elem".to_string()),
                            scalar_type(),
                            make_span(),
                        ),
                    ],
                },
                scalar_type(),
                make_span(),
            );

            let expr = TypedExpr::new(
                ExprKind::Fold {
                    entity: EntityId::new("plate"),
                    init: Box::new(init),
                    acc: "acc".to_string(),
                    elem: "elem".to_string(),
                    body: Box::new(body),
                },
                scalar_type(),
                make_span(),
            );
            assert!(expr.is_pure());
        }

        // === Impurity Propagation Tests ===

        #[test]
        fn let_with_impure_body_is_impure() {
            let value = TypedExpr::new(
                ExprKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );
            let body = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("", "emit"),
                    args: vec![],
                },
                Type::Unit,
                make_span(),
            );
            let expr = TypedExpr::new(
                ExprKind::Let {
                    name: "x".to_string(),
                    value: Box::new(value),
                    body: Box::new(body),
                },
                scalar_type(),
                make_span(),
            );
            assert!(!expr.is_pure());
        }

        #[test]
        fn aggregate_with_impure_body_is_impure() {
            let body = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("", "emit"),
                    args: vec![],
                },
                Type::Unit,
                make_span(),
            );
            let expr = TypedExpr::new(
                ExprKind::Aggregate {
                    op: AggregateOp::Sum,
                    entity: EntityId::new("plate"),
                    binding: "p".to_string(),
                    body: Box::new(body),
                },
                scalar_type(),
                make_span(),
            );
            assert!(!expr.is_pure());
        }

        #[test]
        fn fold_with_impure_init_is_impure() {
            let init = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("", "emit"),
                    args: vec![],
                },
                Type::Unit,
                make_span(),
            );
            let body = TypedExpr::new(
                ExprKind::Local("acc".to_string()),
                scalar_type(),
                make_span(),
            );
            let expr = TypedExpr::new(
                ExprKind::Fold {
                    entity: EntityId::new("plate"),
                    init: Box::new(init),
                    acc: "acc".to_string(),
                    elem: "elem".to_string(),
                    body: Box::new(body),
                },
                scalar_type(),
                make_span(),
            );
            assert!(!expr.is_pure());
        }

        #[test]
        fn fold_with_impure_body_is_impure() {
            let init = TypedExpr::new(
                ExprKind::Literal {
                    value: 0.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );
            let body = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("", "spawn"),
                    args: vec![],
                },
                Type::Unit,
                make_span(),
            );
            let expr = TypedExpr::new(
                ExprKind::Fold {
                    entity: EntityId::new("plate"),
                    init: Box::new(init),
                    acc: "acc".to_string(),
                    elem: "elem".to_string(),
                    body: Box::new(body),
                },
                scalar_type(),
                make_span(),
            );
            assert!(!expr.is_pure());
        }

        #[test]
        fn pure_kernel_with_impure_arg_is_impure() {
            let impure_arg = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("", "emit"),
                    args: vec![],
                },
                Type::Unit,
                make_span(),
            );
            let pure_lit = TypedExpr::new(
                ExprKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );
            let expr = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("maths", "add"),
                    args: vec![pure_lit, impure_arg],
                },
                scalar_type(),
                make_span(),
            );
            assert!(!expr.is_pure());
        }

        #[test]
        fn struct_with_impure_field_is_impure() {
            use continuum_foundation::TypeId;

            let field_value = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("", "log"),
                    args: vec![],
                },
                Type::Unit,
                make_span(),
            );
            let orbit_ty = TypeId::from("Orbit");
            let expr = TypedExpr::new(
                ExprKind::Struct {
                    ty: orbit_ty.clone(),
                    fields: vec![("semi_major".to_string(), field_value)],
                },
                Type::User(orbit_ty),
                make_span(),
            );
            assert!(!expr.is_pure());
        }

        #[test]
        fn field_access_with_impure_object_is_impure() {
            let object = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("", "destroy"),
                    args: vec![],
                },
                Type::Unit,
                make_span(),
            );
            let expr = TypedExpr::new(
                ExprKind::FieldAccess {
                    object: Box::new(object),
                    field: "x".to_string(),
                },
                scalar_type(),
                make_span(),
            );
            assert!(!expr.is_pure());
        }

        #[test]
        fn vector_with_impure_element_is_impure() {
            let pure = TypedExpr::new(
                ExprKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                scalar_type(),
                make_span(),
            );
            let impure = TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("", "emit"),
                    args: vec![],
                },
                Type::Unit,
                make_span(),
            );
            let expr = TypedExpr::new(
                ExprKind::Vector(vec![pure.clone(), impure, pure]),
                scalar_type(),
                make_span(),
            );
            assert!(!expr.is_pure());
        }
    }

    mod kernel_namespaces {
        use super::*;

        #[test]
        fn maths_namespace() {
            let ops = vec!["add", "sub", "mul", "div", "sin", "cos", "sqrt", "pow"];
            for op in ops {
                let kernel = KernelId::new("maths", op);
                assert_eq!(kernel.namespace, "maths");
                assert_eq!(kernel.name, op);
            }
        }

        #[test]
        fn vector_namespace() {
            let ops = vec!["dot", "cross", "norm", "normalize"];
            for op in ops {
                let kernel = KernelId::new("vector", op);
                assert_eq!(kernel.namespace, "vector");
            }
        }

        #[test]
        fn logic_namespace() {
            let ops = vec!["and", "or", "not", "select"];
            for op in ops {
                let kernel = KernelId::new("logic", op);
                assert_eq!(kernel.namespace, "logic");
            }
        }

        #[test]
        fn compare_namespace() {
            let ops = vec!["lt", "le", "gt", "ge", "eq", "ne"];
            for op in ops {
                let kernel = KernelId::new("compare", op);
                assert_eq!(kernel.namespace, "compare");
            }
        }

        #[test]
        fn effect_namespace() {
            let ops = vec!["emit", "spawn", "destroy", "log"];
            for op in ops {
                // Effect operations are bare names (no namespace)
                let kernel = KernelId::new("", op);
                assert_eq!(kernel.namespace, "");
            }
        }
    }
}
