//! Expression system for Continuum DSL
//!
//! This module defines the expression trees that live inside Node<I> execution blocks.
//! Every expression carries its type, making type errors compile errors rather than
//! runtime errors.
//!
//! # Pipeline Stage: Semantic IR (Canonical Representation)
//!
//! **This is the canonical semantic IR.** [`TypedExpr`] and [`ExprKind`] are the
//! authoritative representation of expressions used by:
//! - Type checker (produces this from untyped AST)
//! - Validation passes (consume this)
//! - DAG builder (schedules this into execution graph)
//! - Runtime (never - uses bytecode compiled from this)
//!
//! **What happened to Binary/Unary/If?**  
//! The untyped AST ([`UntypedExpr`](super::untyped::Expr)) contains syntax-only
//! variants that **desugar** during type resolution:
//! - `Binary(Add, a, b)` → `Call { kernel: maths.add, args: [a,b] }`
//! - `If(c, t, e)` → `Call { kernel: logic.select, args: [c,t,e] }`
//!
//! This IR contains **only semantic constructs**, not syntactic sugar.
//!
//! See `docs/execution/ir.md` for the full compilation pipeline.
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
//! - Time step via `dt.raw()` kernel (requires `Capability::Dt`)
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

use crate::foundation::{AggregateOp, Path, Span, Type, UserTypeId};
use continuum_kernel_types::KernelId;
// Imported for derive macros used throughout this file
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};

use crate::foundation::EntityId;

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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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

    /// String literal
    StringLiteral(String),

    /// Boolean literal (true/false)
    ///
    /// Preserved in typed AST to maintain semantic distinction from numeric literals.
    /// Type-checked to ensure Type::Bool.
    BoolLiteral(bool),

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
    Payload,

    /// Entity reference (produces Seq<Instance>)
    Entity(EntityId),

    /// Spatial lookup: nearest(entity, pos)
    Nearest {
        /// Entity to search in
        entity: EntityId,
        /// Reference position
        position: Box<TypedExpr>,
    },

    /// Spatial filter: within(entity, pos, radius)
    Within {
        /// Entity to search in
        entity: EntityId,
        /// Center position
        position: Box<TypedExpr>,
        /// Search radius
        radius: Box<TypedExpr>,
    },

    /// Spatial topology: neighbors(instance)
    ///
    /// Returns all topologically connected neighbors of the given instance.
    /// Requires the entity to have a topology defined (e.g., icosahedron_grid).
    Neighbors {
        /// Entity type (inferred from instance)
        entity: EntityId,
        /// Instance to get neighbors for
        instance: Box<TypedExpr>,
    },

    /// Filtered entity set: filter(entity, predicate)
    Filter {
        /// Entity or Seq to filter
        source: Box<TypedExpr>,
        /// Predicate expression (uses self)
        predicate: Box<TypedExpr>,
    },

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
    /// sum(entity.plates, self.mass)        // AggregateOp::Sum
    /// max(entity.bodies, self.temperature)  // AggregateOp::Max
    /// count(entity.stars, self.active)      // AggregateOp::Count
    /// ```
    ///
    /// **Iteration order:** Lexical `InstanceId` order (deterministic)
    ///
    /// **Purity:** Body must be pure (no effects like `emit`)
    Aggregate {
        /// Aggregate operation
        op: AggregateOp,
        /// Source of instances (Entity or Filtered set)
        source: Box<TypedExpr>,
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
    /// fold(entity.plates, 0.0<kg>, total + self.mass)
    /// ```
    ///
    /// **Iteration order:** Lexical `InstanceId` order (deterministic)
    ///
    /// **Purity:** Body must be pure
    Fold {
        /// Source of instances
        source: Box<TypedExpr>,
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

    /// Index access into entity instances
    ///
    /// Represents `entity.X[i]` where `i` is a runtime index expression.
    ///
    /// # Examples
    ///
    /// ```cdsl
    /// entity.terra.plate[i]    // Runtime index
    /// entity.terra.plate[0]    // Literal index
    /// ```
    ///
    /// # Type Rules
    ///
    /// - Object must resolve to an entity reference (Type::Entity)
    /// - Index must be integer type
    /// - Result type is the entity's instance handle type
    ///
    /// # Usage
    ///
    /// This is primarily used in member signal assignments:
    /// ```cdsl
    /// emit entity.terra.plate[i].omega <- torque
    /// ```
    Index {
        /// Entity being indexed
        entity: EntityId,
        /// Index expression (must be integer type)
        index: Box<TypedExpr>,
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
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
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
            | ExprKind::StringLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::Local(_)
            | ExprKind::Signal(_)
            | ExprKind::Field(_)
            | ExprKind::Config(_)
            | ExprKind::Const(_)
            | ExprKind::Prev
            | ExprKind::Current
            | ExprKind::Inputs
            | ExprKind::Self_
            | ExprKind::Other
            | ExprKind::Payload
            | ExprKind::Entity(_) => true,

            // Spatial and filter ops are pure if their components are pure
            ExprKind::Nearest { position, .. } => position.is_pure(),
            ExprKind::Within {
                position, radius, ..
            } => position.is_pure() && radius.is_pure(),
            ExprKind::Neighbors { instance, .. } => instance.is_pure(),
            ExprKind::Filter { source, predicate } => source.is_pure() && predicate.is_pure(),

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
                    .unwrap_or_else(|| {
                        panic!(
                            "BUG: Unknown kernel '{:?}' in purity check. \
                             All kernels must be registered before AST construction. \
                             This indicates the kernel was not found during resolution, \
                             which should have been caught earlier in the pipeline.",
                            kernel
                        )
                    });

                kernel_is_pure && args.iter().all(|arg| arg.is_pure())
            }

            // Struct construction and field access are pure if operands are pure
            ExprKind::Struct { fields, .. } => fields.iter().all(|(_, expr)| expr.is_pure()),
            ExprKind::FieldAccess { object, .. } => object.is_pure(),

            // Index access is pure if the index expression is pure
            ExprKind::Index { index, .. } => index.is_pure(),
        }
    }

    /// Walk the expression tree recursively using a visitor.
    ///
    /// This method implements a generic traversal of the `TypedExpr` tree,
    /// enabling centralized logic for tree inspection (e.g., dependency extraction,
    /// capability validation, or optimization passes). By using this method,
    /// traversal logic is kept in sync with the [`ExprKind`] enum definition.
    ///
    /// Traversal is depth-first: the visitor's [`ExpressionVisitor::visit_expr`]
    /// is called for the current node, then `walk` is called recursively for
    /// all child expressions.
    ///
    /// # Parameters
    /// - `visitor`: An implementation of [`ExpressionVisitor`] to apply to each node.
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
            ExprKind::Index { index, .. } => {
                index.walk(visitor);
            }
            ExprKind::Nearest { position, .. } => {
                position.walk(visitor);
            }
            ExprKind::Within {
                position, radius, ..
            } => {
                position.walk(visitor);
                radius.walk(visitor);
            }
            ExprKind::Neighbors { instance, .. } => {
                instance.walk(visitor);
            }
            ExprKind::Filter { source, predicate } => {
                source.walk(visitor);
                predicate.walk(visitor);
            }

            // Leaf nodes (explicitly listed to ensure exhaustiveness)
            ExprKind::Literal { .. }
            | ExprKind::StringLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::Local(_)
            | ExprKind::Signal(_)
            | ExprKind::Field(_)
            | ExprKind::Config(_)
            | ExprKind::Const(_)
            | ExprKind::Prev
            | ExprKind::Current
            | ExprKind::Inputs
            | ExprKind::Self_
            | ExprKind::Other
            | ExprKind::Payload
            | ExprKind::Entity(_) => {}
        }
    }
}

/// Visitor trait for traversing typed expression trees.
///
/// Implement this trait to perform analysis or transformations on the
/// [`TypedExpr`] tree. Combined with [`TypedExpr::walk`], this provides
/// a standard way to inspect expressions without duplicating traversal logic.
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::ast::{TypedExpr, ExpressionVisitor};
///
/// #[derive(Default)]
/// struct Counter { count: usize }
///
/// impl ExpressionVisitor for Counter {
///     fn visit_expr(&mut self, _expr: &TypedExpr) {
///         self.count += 1;
///     }
/// }
/// ```
pub trait ExpressionVisitor {
    /// Visit a typed expression node.
    ///
    /// This method is called for every [`TypedExpr`] node encountered during
    /// the traversal initiated by [`TypedExpr::walk`].
    fn visit_expr(&mut self, expr: &TypedExpr);
}
