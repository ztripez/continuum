//! Compile-time kernel type signatures
//!
//! This crate defines **pure type definitions** for kernel operations used during
//! DSL compilation and type checking. These types are shared between:
//! - `continuum-cdsl` (compile-time type checking)
//! - `kernel-macros` (macro code generation)
//! - `kernels/functions` (annotations on kernel functions)
//!
//! # Architecture
//!
//! - **Single source of truth**: Types defined here, populated by `#[kernel_fn]` macro
//! - **No logic**: Pure data structures, no algorithms or business logic
//! - **Distributed slice**: `KERNEL_SIGNATURES` populated at link time by macro
//!
//! # Dual Type Hierarchy Justification
//!
//! **Note**: This crate defines types that are parallel to (but separate from) the AST types
//! in `continuum-cdsl::ast::kernel`. This duplication exists for architectural reasons:
//!
//! ## Why Two Hierarchies?
//!
//! 1. **Const-compatible static initialization**:
//!    - This crate uses `&'static [KernelParam]` for parameter lists
//!    - Required for `const` initialization in distributed slice (`KERNEL_SIGNATURES`)
//!    - The `linkme` distributed slice requires all data to be `'static`
//!
//! 2. **AST requires owned, mutable structures**:
//!    - `continuum-cdsl` uses `Vec<KernelParam>` for owned storage
//!    - AST needs to build, modify, and serialize kernel registries
//!    - Requires `serde` support for persistence
//!
//! 3. **Different lifetimes**:
//!    - Compile-time signatures: `'static` lifetime (embedded in binary)
//!    - AST signatures: Owned, heap-allocated (runtime constructed)
//!
//! ## Conversion Layer
//!
//! The `continuum-cdsl::ast::kernel::KernelRegistry::convert_*` functions bridge
//! between these representations. This conversion is **one-way** (compile-time → AST)
//! and happens once during registry initialization.
//!
//! ## Trade-offs
//!
//! - **Benefit**: Clean separation of const/static vs owned/mutable concerns
//! - **Cost**: Requires maintaining parallel type definitions and conversion layer
//! - **Risk**: Type drift if new constraint variants are added to only one side
//!
//! ## Mitigation
//!
//! - Shared tests ensure both hierarchies support the same constraint vocabulary
//! - Conversion functions are centralized and exhaustive (fail to compile if variants added)
//! - Future: Consider auto-generating one from the other via proc-macro
//!
//! # Design Principles
//!
//! ## Type Safety
//!
//! All kernel calls are type-checked at compile time:
//! - Argument count must match signature
//! - Argument shapes must satisfy constraints
//! - Argument units must satisfy constraints
//! - Return type is derived from argument types
//!
//! ## Purity Enforcement
//!
//! Kernels are classified as Pure or Effect:
//! - **Pure**: No side effects, deterministic, can be used in any phase
//! - **Effect**: Mutates state (emit, spawn, destroy) or artifacts (log)
//!
//! ## Determinism
//!
//! All kernels are deterministic. There are no non-deterministic kernels:
//! - `rng.*` kernels derive randomness from `(seed, InstanceId, tick)` - fully reproducible
//! - No kernel depends on wall-clock time, thread scheduling, or external state

pub mod prelude;
pub mod shape;
pub mod unit;

pub use shape::Shape;
pub use unit::{Unit, UnitDimensions, UnitKind};

use linkme::distributed_slice;

/// Distributed slice for kernel signatures
///
/// Populated at link time by the `#[kernel_fn]` macro.
/// Each kernel function annotated with `#[kernel_fn]` emits a static
/// `KernelSignature` that gets added to this slice.
///
/// # Examples
///
/// ```rust
/// use continuum_kernel_types::KERNEL_SIGNATURES;
///
/// // Iterate over all registered kernel signatures
/// for sig in KERNEL_SIGNATURES {
///     println!("{}: {} params", sig.id.qualified_name(), sig.params.len());
/// }
/// ```
#[distributed_slice]
pub static KERNEL_SIGNATURES: [KernelSignature];

/// Kernel identifier
///
/// Uniquely identifies a kernel by namespace and name.
/// Namespaces organize kernels into logical groups.
///
/// # Namespace Convention
///
/// - `maths.*` - Arithmetic operations
/// - `vector.*` - Vector operations
/// - `matrix.*` - Matrix operations
/// - `logic.*` - Boolean operations
/// - `compare.*` - Comparison operations
/// - `rng.*` - Seeded random number generation
/// - `""` (empty) - Effect operations (emit, spawn, destroy, log) use bare names
///
/// # Examples
///
/// ```rust
/// use continuum_kernel_types::KernelId;
///
/// let add = KernelId::new("maths", "add");
/// assert_eq!(add.qualified_name(), "maths.add");
///
/// let emit = KernelId::new("", "emit");
/// assert_eq!(emit.qualified_name(), "emit");  // Bare name
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelId {
    pub namespace: &'static str,
    pub name: &'static str,
}

impl KernelId {
    /// Create a new kernel identifier
    ///
    /// # Parameters
    ///
    /// - `namespace`: Kernel namespace (empty string for bare names)
    /// - `name`: Kernel name
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_kernel_types::KernelId;
    ///
    /// let id = KernelId::new("maths", "add");
    /// ```
    pub const fn new(namespace: &'static str, name: &'static str) -> Self {
        Self { namespace, name }
    }

    /// Get the qualified name (namespace.name)
    ///
    /// For bare names (empty namespace), returns just the name.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_kernel_types::KernelId;
    ///
    /// let add = KernelId::new("maths", "add");
    /// assert_eq!(add.qualified_name(), "maths.add");
    ///
    /// let emit = KernelId::new("", "emit");
    /// assert_eq!(emit.qualified_name(), "emit");
    /// ```
    pub fn qualified_name(&self) -> String {
        if self.namespace.is_empty() {
            self.name.to_string()
        } else {
            format!("{}.{}", self.namespace, self.name)
        }
    }
}

/// Kernel signature
///
/// Defines the type signature and purity class of a kernel operation.
/// Used for compile-time type checking and purity enforcement.
///
/// # Fields
///
/// - `id`: Unique kernel identifier (namespace + name)
/// - `params`: Parameter type constraints
/// - `returns`: Return type derivation
/// - `purity`: Effect discipline (pure vs effectful)
///
/// # Examples
///
/// ```rust
/// use continuum_kernel_types::{KernelSignature, KernelParam, KernelReturn, KernelPurity};
/// use continuum_kernel_types::{ShapeConstraint, UnitConstraint, ShapeDerivation, UnitDerivation, KernelId};
///
/// // maths.add(a, b) → same shape, same unit, pure
/// let add_sig = KernelSignature {
///     id: KernelId::new("maths", "add"),
///     params: vec![
///         KernelParam { name: "a", shape: ShapeConstraint::Any, unit: UnitConstraint::Any },
///         KernelParam { name: "b", shape: ShapeConstraint::SameAs(0), unit: UnitConstraint::SameAs(0) },
///     ],
///     returns: KernelReturn {
///         shape: ShapeDerivation::SameAs(0),
///         unit: UnitDerivation::SameAs(0)
///     },
///     purity: KernelPurity::Pure,
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSignature {
    /// Kernel identifier
    pub id: KernelId,

    /// Parameter type constraints
    pub params: &'static [KernelParam],

    /// Return type derivation
    pub returns: KernelReturn,

    /// Effect discipline
    pub purity: KernelPurity,
}

/// Kernel parameter type constraint
///
/// Defines the type requirements for a kernel parameter.
/// Used during type checking to validate argument types.
///
/// # Fields
///
/// - `name`: Parameter name (for error messages)
/// - `shape`: Shape constraint (exact, any, same-as, broadcast)
/// - `unit`: Unit constraint (exact, dimensionless, angle, same-as)
#[derive(Debug, Clone, PartialEq)]
pub struct KernelParam {
    /// Parameter name
    pub name: &'static str,

    /// Shape constraint
    pub shape: ShapeConstraint,

    /// Unit constraint
    pub unit: UnitConstraint,
}

/// Shape constraint for kernel parameters
///
/// Defines what shapes are allowed for a parameter.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShapeConstraint {
    /// Must be exactly this shape
    Exact(Shape),

    /// Any scalar
    AnyScalar,

    /// Vector of any dimension
    AnyVector,

    /// Matrix of any dimensions
    AnyMatrix,

    /// Any shape
    Any,

    /// Same shape as parameter N
    SameAs(usize),

    /// Broadcastable with parameter N
    BroadcastWith(usize),

    /// Vector with constrained dimension
    VectorDim(DimConstraint),

    /// Matrix with constrained dimensions
    MatrixDims {
        rows: DimConstraint,
        cols: DimConstraint,
    },
}

/// Dimension constraint for vectors and matrices
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DimConstraint {
    /// Must be exactly this dimension
    Exact(u8),

    /// Any dimension
    Any,

    /// Dimension variable - Var(N) must equal all other Var(N)
    Var(usize),
}

/// Unit constraint for kernel parameters
#[derive(Debug, Clone, PartialEq)]
pub enum UnitConstraint {
    /// Must be exactly this unit
    Exact(Unit),

    /// Must be dimensionless
    Dimensionless,

    /// Must be angle (for trig functions)
    Angle,

    /// Any unit
    Any,

    /// Same unit as parameter N
    SameAs(usize),
}

/// Kernel return type
///
/// Defines how the return type is derived from parameter types.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelReturn {
    /// Shape derivation
    pub shape: ShapeDerivation,

    /// Unit derivation
    pub unit: UnitDerivation,
}

/// Shape derivation for kernel return type
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ShapeDerivation {
    /// Always this exact shape
    Exact(Shape),

    /// Same shape as parameter N
    SameAs(usize),

    /// Broadcast result of parameters N and M
    FromBroadcast(usize, usize),

    /// Always scalar (reductions)
    Scalar,

    /// Vector with dimension from constraint
    VectorDim(DimConstraint),

    /// Matrix with dimensions from constraints
    MatrixDims {
        rows: DimConstraint,
        cols: DimConstraint,
    },
}

/// Unit derivation for kernel return type
#[derive(Debug, Clone, PartialEq)]
pub enum UnitDerivation {
    /// Always this exact unit
    Exact(Unit),

    /// Always dimensionless
    Dimensionless,

    /// Same unit as parameter N
    SameAs(usize),

    /// Product of parameter units
    Multiply(&'static [usize]),

    /// Parameter N unit divided by parameter M unit
    Divide(usize, usize),

    /// Square root of parameter N unit
    Sqrt(usize),

    /// Inverse of parameter N unit (1 / unit)
    Inverse(usize),
}

/// Kernel purity class
///
/// Classifies kernels by their effect discipline.
/// Used to enforce phase restrictions (pure-only phases vs effect-allowed phases).
///
/// # Purity Classes
///
/// - **Pure**: No side effects, deterministic, can be used in any phase
/// - **Effect**: Mutates state (emit, spawn, destroy) or artifacts (log)
///
/// # Phase Restrictions
///
/// | Phase | Allowed |
/// |-------|---------|
/// | Configure | Pure |
/// | Collect | Pure + Effect |
/// | Resolve | **Pure only** |
/// | Fracture | Pure + Effect |
/// | Measure | **Pure only** |
/// | Assert | Pure |
/// | Apply | Pure + Effect |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelPurity {
    /// No side effects, deterministic, can be used anywhere
    Pure,

    /// Mutates state (emit, spawn, destroy) or artifacts (log)
    Effect,
}

impl KernelPurity {
    /// Check if this kernel is pure
    ///
    /// # Returns
    ///
    /// `true` if Pure, `false` if Effect
    pub fn is_pure(self) -> bool {
        matches!(self, Self::Pure)
    }
}
