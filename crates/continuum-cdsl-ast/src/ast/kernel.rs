//! Compile-time kernel type signatures
//!
//! This module defines **compile-time type signatures** for kernel operations used
//! during DSL compilation and type checking. This is separate from the **runtime
//! kernel registry** (`crates/kernel-registry`) which dispatches kernel calls during
//! simulation execution.
//!
//! # Architecture
//!
//! - **Compile-time** (this module): Type signatures for DSL validation
//! - **Runtime** (`kernel-registry`): Actual implementations populated via `#[kernel_fn]` macro
//!
//! Eventually, these should be synced by extracting type signatures from the runtime
//! registry metadata.
//!
//! # Design Principles
//!
//! ## Type Safety
//!
//! All kernel calls are type-checked at compile time:
//! - Argument count must match signature
//! - Argument shapes must satisfy constraints (exact, broadcast, dimension variables)
//! - Argument units must satisfy constraints (exact, dimensionless, angle, same-as)
//! - Return type is derived from argument types
//!
//! ## Purity Enforcement
//!
//! Kernels are classified as Pure or Effect:
//! - **Pure**: No side effects, deterministic, can be used in any phase
//! - **Effect**: Mutates state (emit, spawn, destroy) or artifacts (log)
//!
//! Phase restrictions:
//! - **Resolve, Measure, Assert**: Pure only
//! - **Collect, Fracture, Apply**: Pure + Effect
//!
//! ## Determinism
//!
//! All kernels are deterministic. There are no non-deterministic kernels:
//! - `rng.*` kernels derive randomness from `(seed, InstanceId, tick)` - fully reproducible
//! - No kernel depends on wall-clock time, thread scheduling, or external state
//!
//! # Examples
//!
//! ## Mathematical Operations
//!
//! ```cdsl
//! // maths.add: same shape, same unit, pure
//! let sum = maths.add(velocity1, velocity2)  // Vector<3, m/s> + Vector<3, m/s> → Vector<3, m/s>
//!
//! // maths.mul: same shape, multiply units, pure
//! let momentum = maths.mul(mass, velocity)  // Scalar<kg> * Vector<3, m/s> → Vector<3, kg·m/s>
//! ```
//!
//! ## Vector Operations
//!
//! ```cdsl
//! // vector.dot: scalar result, multiply units, dimensions must match
//! let energy = vector.dot(force, displacement)  // Vector<3, N> · Vector<3, m> → Scalar<N·m>
//!
//! // vector.cross: 3D only, multiply units
//! let torque = vector.cross(position, force)  // Vector<3, m> × Vector<3, N> → Vector<3, N·m>
//! ```
//!
//! ## Effect Operations
//!
//! ```cdsl
//! // emit: effect kernel, only in Collect/Apply phases
//! emit(target_signal, computed_value)  // Mutates signal inputs
//!
//! // spawn: effect kernel, only in Fracture phase
//! spawn(PlateEntity, { position, velocity })  // Creates new entity
//! ```
//!
//! # Kernel Registry
//!
//! All built-in kernels are registered in a compile-time registry (`KERNEL_REGISTRY`).
//! The registry is used for:
//! - Type-checking kernel calls during IR construction
//! - Validating purity during phase assignment
//! - Deriving return types from argument types

use crate::foundation::{Shape, Unit};
use continuum_kernel_types::{KernelId, KERNEL_SIGNATURES};

/// Declares that a kernel requires explicit `: uses()` declaration
///
/// Dangerous functions (error masking, dt-fragile) require explicit opt-in.
/// If a signal/member uses such a function without declaring the uses clause,
/// compilation fails.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::RequiresUses;
///
/// // maths.clamp requires : uses(maths.clamping)
/// let req = RequiresUses {
///     key: "clamping".to_string(),
///     hint: "Clamping silently masks out-of-range values".to_string(),
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RequiresUses {
    /// The uses key (e.g., "clamping" → `: uses(maths.clamping)`)
    pub key: String,
    /// Hint message explaining why explicit opt-in is required
    pub hint: String,
}

/// Kernel signature
///
/// Defines the type signature and purity class of a kernel operation.
/// Used for compile-time type checking and purity enforcement.
///
/// # Fields
///
/// - `id`: Unique kernel identifier
/// - `params`: Parameter type constraints
/// - `returns`: Return type derivation
/// - `purity`: Effect discipline (pure vs effectful)
/// - `requires_uses`: If Some, using this kernel requires `: uses(namespace.key)` declaration
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::{KernelSignature, KernelParam, KernelReturn, KernelPurity};
/// use continuum_cdsl::ast::{ShapeConstraint, UnitConstraint, ShapeDerivation, UnitDerivation};
/// use continuum_cdsl::ast::KernelId;
/// use continuum_kernel_types::ValueType;
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
///         unit: UnitDerivation::SameAs(0),
///         value_type: ValueType::Scalar,
///     },
///     purity: KernelPurity::Pure,
///     requires_uses: None,
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSignature {
    /// Kernel identifier
    pub id: KernelId,

    /// Parameter type constraints
    pub params: Vec<KernelParam>,

    /// Return type derivation
    pub returns: KernelReturn,

    /// Effect discipline
    pub purity: KernelPurity,

    /// If Some, using this function requires `: uses(namespace.key)` declaration
    pub requires_uses: Option<RequiresUses>,
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
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::{KernelParam, ShapeConstraint, UnitConstraint};
///
/// // Parameter that accepts any shape/unit
/// let param_any = KernelParam {
///     name: "value",
///     shape: ShapeConstraint::Any,
///     unit: UnitConstraint::Any
/// };
///
/// // Parameter that must match parameter 0
/// let param_same = KernelParam {
///     name: "other",
///     shape: ShapeConstraint::SameAs(0),
///     unit: UnitConstraint::SameAs(0)
/// };
/// ```
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
/// Constraints can be exact, categorical (any scalar/vector/matrix),
/// or relational (same-as, broadcast-with, dimension variables).
///
/// # Constraint Types
///
/// - **Exact**: Must be exactly this shape
/// - **Categorical**: Any scalar/vector/matrix
/// - **Relational**: Must match or broadcast with another parameter
/// - **Dimension-constrained**: Vector/matrix with constrained dimensions
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::{ShapeConstraint, DimConstraint};
/// use continuum_cdsl::foundation::Shape;
///
/// // Must be exactly Scalar
/// let exact = ShapeConstraint::Exact(Shape::Scalar);
///
/// // Any vector (dimension unconstrained)
/// let any_vec = ShapeConstraint::AnyVector;
///
/// // Same shape as parameter 0
/// let same = ShapeConstraint::SameAs(0);
///
/// // Vector with dimension variable 0
/// let vec_dim = ShapeConstraint::VectorDim(DimConstraint::Var(0));
/// ```
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
        /// Row dimension constraint
        rows: DimConstraint,
        /// Column dimension constraint
        cols: DimConstraint,
    },
}

/// Dimension constraint for vectors and matrices
///
/// Defines dimension requirements for vector/matrix parameters.
/// Dimension variables (Var) allow expressing constraints like
/// "these two dimensions must match" without specifying exact values.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::DimConstraint;
///
/// // Must be exactly 3
/// let exact = DimConstraint::Exact(3);
///
/// // Any dimension
/// let any = DimConstraint::Any;
///
/// // Dimension variable 0 (must match all other Var(0))
/// let var = DimConstraint::Var(0);
/// ```
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
///
/// Defines what units are allowed for a parameter.
/// Constraints can be exact, categorical (dimensionless, angle),
/// or relational (same-as).
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::UnitConstraint;
/// use continuum_cdsl::foundation::Unit;
///
/// // Must be exactly meters
/// let exact = UnitConstraint::Exact(Unit::meters());
///
/// // Must be dimensionless
/// let dimensionless = UnitConstraint::Dimensionless;
///
/// // Must be angle (rad or deg)
/// let angle = UnitConstraint::Angle;
///
/// // Same unit as parameter 0
/// let same = UnitConstraint::SameAs(0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum UnitConstraint {
    /// Must be exactly this unit (including scale)
    Exact(Unit),

    /// Must be dimensionless (any scale)
    Dimensionless,

    /// Must be angle (for trig functions)
    Angle,

    /// Any unit
    Any,

    /// Same unit as parameter N (exact match including scale)
    ///
    /// Use for operations that require exact unit match like add/subtract.
    SameAs(usize),

    /// Same dimensional type as parameter N (scale can differ)
    ///
    /// Requires matching kind (Multiplicative/Affine/Logarithmic) and
    /// dimensions, but allows scale differences. Use for operations like
    /// comparisons where dimensional compatibility matters but scale doesn't.
    ///
    /// # Examples
    ///
    /// - Comparing `1000<m>` with `1<km>` - compatible (both length)
    /// - Comparing `100<ppmv>` with `0.0001` - compatible (both dimensionless)
    /// - Comparing `5<K>` with `278<C>` - incompatible (different kinds)
    SameDimsAs(usize),
}

/// Kernel return type
///
/// Defines how the return type is derived from parameter types.
/// The return type is computed during type checking based on
/// the actual argument types and declared value category.
///
/// # Fields
///
/// - `shape`: Shape derivation rule
/// - `unit`: Unit derivation rule
/// - `value_type`: Rust value category (numeric vs boolean)
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::{KernelReturn, ShapeDerivation, UnitDerivation};
/// use continuum_kernel_types::ValueType;
///
/// // Returns same shape and unit as parameter 0
/// let same_as = KernelReturn {
///     shape: ShapeDerivation::SameAs(0),
///     unit: UnitDerivation::SameAs(0),
///     value_type: ValueType::Scalar,
/// };
///
/// // Returns scalar with product of parameter units
/// let dot_product = KernelReturn {
///     shape: ShapeDerivation::Scalar,
///     unit: UnitDerivation::Multiply(vec![0, 1]),
///     value_type: ValueType::Scalar,
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct KernelReturn {
    /// Shape derivation
    pub shape: ShapeDerivation,

    /// Unit derivation
    pub unit: UnitDerivation,

    /// Rust value type returned by the kernel.
    ///
    /// Boolean returns produce `Type::Bool` directly instead of deriving
    /// a kernel type with shape/unit.
    pub value_type: continuum_kernel_types::ValueType,
}

/// Shape derivation for kernel return type
///
/// Defines how the return shape is computed from parameter shapes.
/// Can be exact, derived from a parameter, computed from broadcast,
/// or computed from dimension variables.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::{ShapeDerivation, DimConstraint};
/// use continuum_cdsl::foundation::Shape;
///
/// // Always returns scalar
/// let scalar = ShapeDerivation::Scalar;
///
/// // Returns same shape as parameter 0
/// let same = ShapeDerivation::SameAs(0);
///
/// // Returns broadcast of parameters 0 and 1
/// let broadcast = ShapeDerivation::FromBroadcast(0, 1);
///
/// // Returns vector with dimension from variable 0
/// let vec_dim = ShapeDerivation::VectorDim(DimConstraint::Var(0));
/// ```
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
        /// Row dimension constraint
        rows: DimConstraint,
        /// Column dimension constraint
        cols: DimConstraint,
    },
}

/// Unit derivation for kernel return type
///
/// Defines how the return unit is computed from parameter units.
/// Can be exact, derived from a parameter, or computed from
/// arithmetic operations (multiply, divide, sqrt, inverse).
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::UnitDerivation;
/// use continuum_cdsl::foundation::Unit;
///
/// // Always dimensionless
/// let dimensionless = UnitDerivation::Dimensionless;
///
/// // Same unit as parameter 0
/// let same = UnitDerivation::SameAs(0);
///
/// // Product of parameter 0 and 1 units
/// let multiply = UnitDerivation::Multiply(vec![0, 1]);
///
/// // Parameter 0 unit divided by parameter 1 unit
/// let divide = UnitDerivation::Divide(0, 1);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum UnitDerivation {
    /// Always this exact unit
    Exact(Unit),

    /// Always dimensionless
    Dimensionless,

    /// Same unit as parameter N
    SameAs(usize),

    /// Product of parameter units
    Multiply(Vec<usize>),

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
///
/// # Determinism
///
/// All kernels are deterministic, including `rng.*`:
/// - `rng.*` kernels derive randomness from `(seed, InstanceId, tick)`
/// - Fully reproducible given the same seed
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::ast::KernelPurity;
///
/// // Pure kernel (maths, vector, logic, etc.)
/// let pure = KernelPurity::Pure;
///
/// // Effect kernel (emit, spawn, destroy, log)
/// let effect = KernelPurity::Effect;
///
/// assert!(pure.is_pure());
/// assert!(!effect.is_pure());
/// ```
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
    /// # Parameters
    ///
    /// None
    ///
    /// # Returns
    ///
    /// `true` if Pure, `false` if Effect
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::ast::KernelPurity;
    ///
    /// assert!(KernelPurity::Pure.is_pure());
    /// assert!(!KernelPurity::Effect.is_pure());
    /// ```
    pub fn is_pure(self) -> bool {
        matches!(self, Self::Pure)
    }
}

/// Kernel registry
///
/// Global registry of all built-in kernel signatures.
/// Used for type checking kernel calls during compilation.
///
/// # Thread Safety
///
/// The registry is initialized lazily using `OnceLock` for thread-safe initialization.
/// After first access, lookups are lock-free.
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::ast::KernelRegistry;
/// use continuum_cdsl::ast::KernelId;
///
/// let registry = KernelRegistry::global();
/// let sig = registry.get(&KernelId::new("maths", "add"));
/// assert!(sig.is_some());
/// ```
pub struct KernelRegistry {
    signatures: std::collections::HashMap<KernelId, KernelSignature>,
}

impl KernelRegistry {
    /// Create a new empty kernel registry
    ///
    /// # Parameters
    ///
    /// None
    ///
    /// # Returns
    ///
    /// Empty registry
    fn new() -> Self {
        Self {
            signatures: std::collections::HashMap::new(),
        }
    }

    /// Register a kernel signature
    ///
    /// # Parameters
    ///
    /// - `signature`: Kernel signature to register
    ///
    /// # Panics
    ///
    /// Panics if a kernel with the same ID is already registered
    fn register(&mut self, signature: KernelSignature) {
        let id = signature.id.clone();
        if self.signatures.insert(id.clone(), signature).is_some() {
            panic!("Duplicate kernel registration: {}", id.qualified_name());
        }
    }

    /// Get a kernel signature by ID
    ///
    /// # Parameters
    ///
    /// - `id`: Kernel identifier
    ///
    /// # Returns
    ///
    /// Kernel signature if found, None otherwise
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use continuum_cdsl::ast::KernelRegistry;
    /// use continuum_cdsl::ast::KernelId;
    ///
    /// let registry = KernelRegistry::global();
    /// let sig = registry.get(&KernelId::new("maths", "add"));
    /// assert!(sig.is_some());
    /// assert_eq!(sig.unwrap().params.len(), 2);
    /// ```
    pub fn get(&self, id: &KernelId) -> Option<&KernelSignature> {
        self.signatures.get(id)
    }

    /// Check if a kernel is registered
    ///
    /// # Parameters
    ///
    /// - `id`: Kernel identifier
    ///
    /// # Returns
    ///
    /// `true` if registered, `false` otherwise
    pub fn contains(&self, id: &KernelId) -> bool {
        self.signatures.contains_key(id)
    }

    /// Get a kernel signature by namespace and name.
    ///
    /// # Parameters
    ///
    /// - `namespace`: Kernel namespace (empty string for bare names)
    /// - `name`: Kernel name
    ///
    /// # Returns
    ///
    /// Kernel signature if found, None otherwise.
    pub fn get_by_name(&self, namespace: &str, name: &str) -> Option<&KernelSignature> {
        self.signatures
            .iter()
            .find(|(id, _)| id.namespace == namespace && id.name == name)
            .map(|(_, sig)| sig)
    }

    /// Get all kernel signatures matching a name prefix within a namespace
    ///
    /// # Parameters
    ///
    /// - `namespace`: Namespace to search in
    /// - `name_prefix`: Prefix to match kernel names against
    ///
    /// # Returns
    ///
    /// Vector of matching kernel signatures
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// let overloads = registry.get_overloads("vector", "length");
    /// // Returns: length_vec2, length_vec3, length_vec4, length_scalar
    /// ```
    pub fn get_overloads(&self, namespace: &str, name_prefix: &str) -> Vec<&KernelSignature> {
        self.signatures
            .iter()
            .filter(|(id, _)| id.namespace == namespace && id.name.starts_with(name_prefix))
            .map(|(_, sig)| sig)
            .collect()
    }

    /// Get all registered kernel IDs
    ///
    /// # Parameters
    ///
    /// None
    ///
    /// # Returns
    ///
    /// Iterator over kernel IDs
    pub fn ids(&self) -> impl Iterator<Item = &KernelId> {
        self.signatures.keys()
    }
}

/// Global kernel registry instance
///
/// Lazily initialized on first access.
/// Contains all built-in kernel signatures.
pub static KERNEL_REGISTRY: std::sync::OnceLock<KernelRegistry> = std::sync::OnceLock::new();

impl KernelRegistry {
    /// Get or initialize the global kernel registry
    ///
    /// # Parameters
    ///
    /// None
    ///
    /// # Returns
    ///
    /// Reference to the global kernel registry
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// use continuum_cdsl::ast::KernelRegistry;
    ///
    /// let registry = KernelRegistry::global();
    /// assert!(registry.ids().count() > 0);
    /// ```
    pub fn global() -> &'static KernelRegistry {
        KERNEL_REGISTRY.get_or_init(Self::initialize)
    }

    /// Initialize the kernel registry with all built-in kernels
    ///
    /// # Built-in Namespaces
    ///
    /// - `maths.*` - Arithmetic operations (add, sub, mul, div, sin, cos, etc.)
    /// - `vector.*` - Vector operations (dot, cross, norm, normalize, etc.)
    /// - `matrix.*` - Matrix operations (mul, transpose, inverse, etc.)
    /// - `logic.*` - Boolean operations (and, or, not, select, etc.)
    /// - `compare.*` - Comparison operations (eq, ne, lt, le, gt, ge)
    /// - `rng.*` - Seeded random number generation
    /// - `effect.*` - Effect operations (emit, spawn, destroy, log)
    fn initialize() -> KernelRegistry {
        let mut registry = KernelRegistry::new();

        // Populate from KERNEL_SIGNATURES distributed slice
        // The `#[kernel_fn]` macro emits compile-time signatures that are collected
        // by the distributed_slice macro at link time.
        for sig in KERNEL_SIGNATURES {
            // Convert from compile-time signature to AST signature
            let ast_sig = Self::convert_signature(sig);
            registry.register(ast_sig);
        }

        registry
    }

    /// Convert compile-time KernelSignature to AST KernelSignature
    fn convert_signature(sig: &continuum_kernel_types::KernelSignature) -> KernelSignature {
        KernelSignature {
            id: sig.id.clone(),
            params: sig.params.iter().map(Self::convert_param).collect(),
            returns: Self::convert_return(&sig.returns),
            purity: Self::convert_purity(sig.purity),
            requires_uses: sig.requires_uses.map(|req| RequiresUses {
                key: req.key.to_string(),
                hint: req.hint.to_string(),
            }),
        }
    }

    /// Convert compile-time KernelParam to AST KernelParam
    fn convert_param(param: &continuum_kernel_types::KernelParam) -> KernelParam {
        KernelParam {
            name: param.name,
            shape: Self::convert_shape_constraint(&param.shape),
            unit: Self::convert_unit_constraint(&param.unit),
        }
    }

    /// Convert compile-time ShapeConstraint to AST ShapeConstraint
    fn convert_shape_constraint(
        constraint: &continuum_kernel_types::ShapeConstraint,
    ) -> ShapeConstraint {
        use continuum_kernel_types::ShapeConstraint as CtShapeConstraint;
        match constraint {
            CtShapeConstraint::Exact(shape) => ShapeConstraint::Exact(shape.clone()),
            CtShapeConstraint::AnyScalar => ShapeConstraint::AnyScalar,
            CtShapeConstraint::AnyVector => ShapeConstraint::AnyVector,
            CtShapeConstraint::AnyMatrix => ShapeConstraint::AnyMatrix,
            CtShapeConstraint::Any => ShapeConstraint::Any,
            CtShapeConstraint::SameAs(idx) => ShapeConstraint::SameAs(*idx),
            CtShapeConstraint::BroadcastWith(idx) => ShapeConstraint::BroadcastWith(*idx),
            CtShapeConstraint::VectorDim(dim) => {
                ShapeConstraint::VectorDim(Self::convert_dim_constraint(dim))
            }
            CtShapeConstraint::MatrixDims { rows, cols } => ShapeConstraint::MatrixDims {
                rows: Self::convert_dim_constraint(rows),
                cols: Self::convert_dim_constraint(cols),
            },
        }
    }

    /// Convert compile-time DimConstraint to AST DimConstraint
    fn convert_dim_constraint(constraint: &continuum_kernel_types::DimConstraint) -> DimConstraint {
        use continuum_kernel_types::DimConstraint as CtDimConstraint;
        match constraint {
            CtDimConstraint::Exact(dim) => DimConstraint::Exact(*dim),
            CtDimConstraint::Any => DimConstraint::Any,
            CtDimConstraint::Var(idx) => DimConstraint::Var(*idx),
        }
    }

    /// Convert compile-time UnitConstraint to AST UnitConstraint
    fn convert_unit_constraint(
        constraint: &continuum_kernel_types::UnitConstraint,
    ) -> UnitConstraint {
        use continuum_kernel_types::UnitConstraint as CtUnitConstraint;
        match constraint {
            CtUnitConstraint::Exact(unit) => UnitConstraint::Exact(unit.clone()),
            CtUnitConstraint::Dimensionless => UnitConstraint::Dimensionless,
            CtUnitConstraint::Angle => UnitConstraint::Angle,
            CtUnitConstraint::Any => UnitConstraint::Any,
            CtUnitConstraint::SameAs(idx) => UnitConstraint::SameAs(*idx),
            CtUnitConstraint::SameDimsAs(idx) => UnitConstraint::SameDimsAs(*idx),
        }
    }

    /// Convert compile-time KernelReturn to AST KernelReturn
    fn convert_return(ret: &continuum_kernel_types::KernelReturn) -> KernelReturn {
        KernelReturn {
            shape: Self::convert_shape_derivation(&ret.shape),
            unit: Self::convert_unit_derivation(&ret.unit),
            value_type: ret.value_type,
        }
    }

    /// Convert compile-time ShapeDerivation to AST ShapeDerivation
    fn convert_shape_derivation(
        deriv: &continuum_kernel_types::ShapeDerivation,
    ) -> ShapeDerivation {
        use continuum_kernel_types::ShapeDerivation as CtShapeDerivation;
        match deriv {
            CtShapeDerivation::Exact(shape) => ShapeDerivation::Exact(shape.clone()),
            CtShapeDerivation::SameAs(idx) => ShapeDerivation::SameAs(*idx),
            CtShapeDerivation::FromBroadcast(a, b) => ShapeDerivation::FromBroadcast(*a, *b),
            CtShapeDerivation::Scalar => ShapeDerivation::Scalar,
            CtShapeDerivation::VectorDim(dim) => {
                ShapeDerivation::VectorDim(Self::convert_dim_constraint(dim))
            }
            CtShapeDerivation::MatrixDims { rows, cols } => ShapeDerivation::MatrixDims {
                rows: Self::convert_dim_constraint(rows),
                cols: Self::convert_dim_constraint(cols),
            },
        }
    }

    /// Convert compile-time UnitDerivation to AST UnitDerivation
    fn convert_unit_derivation(deriv: &continuum_kernel_types::UnitDerivation) -> UnitDerivation {
        use continuum_kernel_types::UnitDerivation as CtUnitDerivation;
        match deriv {
            CtUnitDerivation::Exact(unit) => UnitDerivation::Exact(unit.clone()),
            CtUnitDerivation::Dimensionless => UnitDerivation::Dimensionless,
            CtUnitDerivation::SameAs(idx) => UnitDerivation::SameAs(*idx),
            CtUnitDerivation::Multiply(indices) => UnitDerivation::Multiply(indices.to_vec()),
            CtUnitDerivation::Divide(a, b) => UnitDerivation::Divide(*a, *b),
            CtUnitDerivation::Sqrt(idx) => UnitDerivation::Sqrt(*idx),
            CtUnitDerivation::Inverse(idx) => UnitDerivation::Inverse(*idx),
        }
    }

    /// Convert compile-time KernelPurity to AST KernelPurity
    fn convert_purity(purity: continuum_kernel_types::KernelPurity) -> KernelPurity {
        match purity {
            continuum_kernel_types::KernelPurity::Pure => KernelPurity::Pure,
            continuum_kernel_types::KernelPurity::Effect => KernelPurity::Effect,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::foundation::Shape;

    // Import continuum-functions to ensure kernel signatures are linked
    // This populates the KERNEL_SIGNATURES distributed slice for tests
    #[allow(unused_imports)]
    use continuum_functions as _;

    #[test]
    fn kernel_purity_is_pure() {
        assert!(KernelPurity::Pure.is_pure());
        assert!(!KernelPurity::Effect.is_pure());
    }

    #[test]
    fn kernel_purity_equality() {
        assert_eq!(KernelPurity::Pure, KernelPurity::Pure);
        assert_eq!(KernelPurity::Effect, KernelPurity::Effect);
        assert_ne!(KernelPurity::Pure, KernelPurity::Effect);
    }

    #[test]
    fn shape_constraint_exact() {
        let constraint = ShapeConstraint::Exact(Shape::Scalar);
        assert_eq!(constraint, ShapeConstraint::Exact(Shape::Scalar));
    }

    #[test]
    fn shape_constraint_any() {
        let any = ShapeConstraint::Any;
        let any_scalar = ShapeConstraint::AnyScalar;
        let any_vector = ShapeConstraint::AnyVector;
        let any_matrix = ShapeConstraint::AnyMatrix;

        assert_ne!(any, any_scalar);
        assert_ne!(any, any_vector);
        assert_ne!(any, any_matrix);
    }

    #[test]
    fn shape_constraint_same_as() {
        let same0 = ShapeConstraint::SameAs(0);
        let same1 = ShapeConstraint::SameAs(1);

        assert_eq!(same0, ShapeConstraint::SameAs(0));
        assert_ne!(same0, same1);
    }

    #[test]
    fn dim_constraint_exact() {
        let dim3 = DimConstraint::Exact(3);
        let dim4 = DimConstraint::Exact(4);

        assert_eq!(dim3, DimConstraint::Exact(3));
        assert_ne!(dim3, dim4);
    }

    #[test]
    fn dim_constraint_var() {
        let var0 = DimConstraint::Var(0);
        let var1 = DimConstraint::Var(1);

        assert_eq!(var0, DimConstraint::Var(0));
        assert_ne!(var0, var1);
        assert_ne!(var0, DimConstraint::Any);
    }

    #[test]
    fn unit_constraint_dimensionless() {
        let dimensionless = UnitConstraint::Dimensionless;
        assert_eq!(dimensionless, UnitConstraint::Dimensionless);
    }

    #[test]
    fn unit_constraint_angle() {
        let angle = UnitConstraint::Angle;
        assert_eq!(angle, UnitConstraint::Angle);
        assert_ne!(angle, UnitConstraint::Dimensionless);
    }

    #[test]
    fn kernel_param_creation() {
        let param = KernelParam {
            name: "value",
            shape: ShapeConstraint::Any,
            unit: UnitConstraint::Any,
        };

        assert_eq!(param.name, "value");
        assert_eq!(param.shape, ShapeConstraint::Any);
        assert_eq!(param.unit, UnitConstraint::Any);
    }

    #[test]
    fn kernel_return_creation() {
        let ret = KernelReturn {
            shape: ShapeDerivation::Scalar,
            unit: UnitDerivation::Dimensionless,
            value_type: continuum_kernel_types::ValueType::Scalar,
        };

        assert_eq!(ret.shape, ShapeDerivation::Scalar);
        assert_eq!(ret.unit, UnitDerivation::Dimensionless);
        assert_eq!(ret.value_type, continuum_kernel_types::ValueType::Scalar);
    }

    #[test]
    fn shape_derivation_scalar() {
        let scalar = ShapeDerivation::Scalar;
        assert_eq!(scalar, ShapeDerivation::Scalar);
    }

    #[test]
    fn shape_derivation_same_as() {
        let same0 = ShapeDerivation::SameAs(0);
        let same1 = ShapeDerivation::SameAs(1);

        assert_eq!(same0, ShapeDerivation::SameAs(0));
        assert_ne!(same0, same1);
    }

    #[test]
    fn shape_derivation_broadcast() {
        let broadcast = ShapeDerivation::FromBroadcast(0, 1);
        assert_eq!(broadcast, ShapeDerivation::FromBroadcast(0, 1));
        assert_ne!(broadcast, ShapeDerivation::FromBroadcast(1, 0));
    }

    #[test]
    fn unit_derivation_dimensionless() {
        let dimensionless = UnitDerivation::Dimensionless;
        assert_eq!(dimensionless, UnitDerivation::Dimensionless);
    }

    #[test]
    fn unit_derivation_multiply() {
        let multiply = UnitDerivation::Multiply(vec![0, 1]);
        assert_eq!(multiply.clone(), multiply);
    }

    #[test]
    fn unit_derivation_divide() {
        let divide = UnitDerivation::Divide(0, 1);
        assert_eq!(divide, UnitDerivation::Divide(0, 1));
        assert_ne!(divide, UnitDerivation::Divide(1, 0));
    }

    #[test]
    fn unit_derivation_sqrt() {
        let sqrt = UnitDerivation::Sqrt(0);
        assert_eq!(sqrt, UnitDerivation::Sqrt(0));
    }

    #[test]
    fn unit_derivation_inverse() {
        let inverse = UnitDerivation::Inverse(0);
        assert_eq!(inverse, UnitDerivation::Inverse(0));
    }

    #[test]
    fn kernel_signature_creation() {
        let sig = KernelSignature {
            id: KernelId::new("maths", "add"),
            params: vec![
                KernelParam {
                    name: "a",
                    shape: ShapeConstraint::Any,
                    unit: UnitConstraint::Any,
                },
                KernelParam {
                    name: "b",
                    shape: ShapeConstraint::SameAs(0),
                    unit: UnitConstraint::SameAs(0),
                },
            ],
            returns: KernelReturn {
                shape: ShapeDerivation::SameAs(0),
                unit: UnitDerivation::SameAs(0),
                value_type: continuum_kernel_types::ValueType::Scalar,
            },
            purity: KernelPurity::Pure,
            requires_uses: None,
        };

        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.purity, KernelPurity::Pure);
    }

    #[test]
    fn registry_contains_maths_add() {
        let registry = KernelRegistry::global();
        let id = KernelId::new("maths", "add");
        assert!(registry.contains(&id));
    }

    #[test]
    fn registry_get_maths_add() {
        let registry = KernelRegistry::global();
        let id = KernelId::new("maths", "add");
        let sig = registry.get(&id);

        assert!(sig.is_some());
        let sig = sig.unwrap();
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.purity, KernelPurity::Pure);
    }

    #[test]
    fn registry_contains_vector_dot() {
        let registry = KernelRegistry::global();
        // Vector dot product is now typed by dimension
        assert!(registry.contains(&KernelId::new("vector", "dot_vec2")));
        assert!(registry.contains(&KernelId::new("vector", "dot_vec3")));
        assert!(registry.contains(&KernelId::new("vector", "dot_vec4")));
    }

    #[test]
    fn registry_get_vector_dot() {
        let registry = KernelRegistry::global();
        // Test the vec3 variant as an example
        let id = KernelId::new("vector", "dot_vec3");
        let sig = registry.get(&id);

        assert!(sig.is_some());
        let sig = sig.unwrap();
        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.purity, KernelPurity::Pure);

        // Check vector dimension constraint (VectorDim with exact dimension 3)
        assert!(matches!(sig.params[0].shape, ShapeConstraint::VectorDim(_)));
        assert!(matches!(sig.params[1].shape, ShapeConstraint::VectorDim(_)));

        // Check return is scalar
        assert_eq!(sig.returns.shape, ShapeDerivation::Scalar);
    }

    #[test]
    fn registry_emit_is_bare_name() {
        let registry = KernelRegistry::global();
        // Effect operations are bare names (no namespace)
        let id = KernelId::new("", "emit");
        let sig = registry.get(&id);

        assert!(sig.is_some());
        assert_eq!(sig.unwrap().purity, KernelPurity::Effect);
        assert_eq!(sig.unwrap().id.namespace, "");
        assert_eq!(sig.unwrap().id.name, "emit");
    }

    #[test]
    fn registry_maths_kernels_are_pure() {
        let registry = KernelRegistry::global();

        for name in &["add", "sub", "mul", "div", "sin", "cos", "abs"] {
            let id = KernelId::new("maths", name);
            let sig = registry.get(&id);
            assert!(sig.is_some(), "maths.{} not found", name);
            assert_eq!(
                sig.unwrap().purity,
                KernelPurity::Pure,
                "maths.{} should be pure",
                name
            );
        }
    }

    #[test]
    fn registry_logic_select_signature() {
        let registry = KernelRegistry::global();
        let id = KernelId::new("logic", "select");
        let sig = registry.get(&id);

        assert!(sig.is_some());
        let sig = sig.unwrap();
        assert_eq!(sig.params.len(), 3);
        assert_eq!(sig.params[0].name, "condition");
        assert_eq!(sig.params[1].name, "if_true");
        assert_eq!(sig.params[2].name, "if_false");
    }

    #[test]
    fn registry_unknown_kernel_returns_none() {
        let registry = KernelRegistry::global();
        let id = KernelId::new("unknown", "kernel");
        assert!(registry.get(&id).is_none());
    }

    #[test]
    fn registry_has_multiple_namespaces() {
        let registry = KernelRegistry::global();

        // Check each namespace has at least one kernel
        assert!(registry.contains(&KernelId::new("maths", "add")));
        assert!(registry.contains(&KernelId::new("vector", "dot_vec3")));
        assert!(registry.contains(&KernelId::new("logic", "select")));
        assert!(registry.contains(&KernelId::new("compare", "lt")));
        // Effect operations are bare names (empty namespace)
        assert!(registry.contains(&KernelId::new("", "emit")));
    }

    #[test]
    fn registry_compare_all_operations() {
        let registry = KernelRegistry::global();

        // All comparison operations should be present
        for name in &["eq", "ne", "lt", "le", "gt", "ge"] {
            let id = KernelId::new("compare", name);
            assert!(registry.contains(&id), "compare.{} not found", name);
            assert_eq!(registry.get(&id).unwrap().purity, KernelPurity::Pure);
        }
    }

    #[test]
    fn registry_compare_returns_bool_value_type() {
        use continuum_kernel_types::ValueType;

        let registry = KernelRegistry::global();
        let sig = registry
            .get(&KernelId::new("compare", "eq"))
            .expect("compare.eq not found");

        assert_eq!(sig.returns.value_type, ValueType::Bool);
        assert_eq!(sig.returns.shape, ShapeDerivation::SameAs(0));
        assert_eq!(sig.returns.unit, UnitDerivation::Dimensionless);
    }

    #[test]
    fn registry_effect_all_operations() {
        let registry = KernelRegistry::global();

        // All effect operations should be bare names
        for name in &["emit", "spawn", "destroy", "log"] {
            let id = KernelId::new("", name);
            assert!(registry.contains(&id), "{} not found (bare name)", name);
            assert_eq!(registry.get(&id).unwrap().purity, KernelPurity::Effect);
        }
    }
}
