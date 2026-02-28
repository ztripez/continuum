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
//! # Single Source of Truth
//!
//! Constraint enums (`ShapeConstraint`, `DimConstraint`, `UnitConstraint`,
//! `ShapeDerivation`, `UnitDerivation`, `KernelPurity`) are defined once in
//! `continuum-kernel-types` and re-exported here. This eliminates type drift
//! and removes the need for variant-by-variant conversion.
//!
//! The only types defined locally are those that require **owned storage**
//! (`Vec`, `String`) vs the `&'static` storage in `kernel-types`:
//! - `KernelSignature` — uses `Vec<KernelParam>` (vs `&'static [KernelParam]`)
//! - `KernelReturn` — uses `UnitDerivation` with `Vec` internally (via `Multiply`)
//! - `RequiresUses` — uses `String` fields (vs `&'static str`)
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

use continuum_kernel_types::{KernelId, KERNEL_SIGNATURES};

// Re-export constraint types from kernel-types (single source of truth).
// These were previously duplicated here with identical variants.
pub use continuum_kernel_types::{
    DimConstraint, KernelPurity, ShapeConstraint, ShapeDerivation, UnitConstraint, UnitDerivation,
    ValueType,
};

/// Declares that a kernel requires explicit `: uses()` declaration
///
/// Dangerous functions (error masking, dt-fragile) require explicit opt-in.
/// If a signal/member uses such a function without declaring the uses clause,
/// compilation fails.
///
/// This is the **owned** variant (with `String` fields) used in the AST.
/// The compile-time variant in `continuum_kernel_types::RequiresUses` uses
/// `&'static str` fields for const initialization.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl_ast::RequiresUses;
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
/// This is the **owned** variant used in the AST registry. The compile-time
/// variant in `continuum_kernel_types::KernelSignature` uses `&'static [KernelParam]`
/// for const initialization in distributed slices.
///
/// # Fields
///
/// - `id`: Unique kernel identifier
/// - `params`: Parameter type constraints (owned `Vec`)
/// - `returns`: Return type derivation
/// - `purity`: Effect discipline (pure vs effectful)
/// - `requires_uses`: If Some, using this kernel requires `: uses(namespace.key)` declaration
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl_ast::{
///     KernelSignature, KernelParam, KernelReturn, KernelPurity,
///     ShapeConstraint, UnitConstraint, ShapeDerivation, UnitDerivation, ValueType,
/// };
/// use continuum_kernel_types::KernelId;
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

    /// Parameter type constraints (owned for AST mutability)
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
/// use continuum_cdsl_ast::{KernelParam, ShapeConstraint, UnitConstraint};
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

    /// Shape constraint (from `continuum_kernel_types`)
    pub shape: ShapeConstraint,

    /// Unit constraint (from `continuum_kernel_types`)
    pub unit: UnitConstraint,
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
/// use continuum_cdsl_ast::{KernelReturn, ShapeDerivation, UnitDerivation, ValueType};
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
///     unit: UnitDerivation::Multiply(&[0, 1]),
///     value_type: ValueType::Scalar,
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct KernelReturn {
    /// Shape derivation (from `continuum_kernel_types`)
    pub shape: ShapeDerivation,

    /// Unit derivation (from `continuum_kernel_types`)
    pub unit: UnitDerivation,

    /// Rust value type returned by the kernel.
    ///
    /// Boolean returns produce `Type::Bool` directly instead of deriving
    /// a kernel type with shape/unit.
    pub value_type: ValueType,
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
/// use continuum_cdsl_ast::KernelRegistry;
/// use continuum_kernel_types::KernelId;
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
    fn new() -> Self {
        Self {
            signatures: std::collections::HashMap::new(),
        }
    }

    /// Register a kernel signature
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
    pub fn get_overloads(&self, namespace: &str, name_prefix: &str) -> Vec<&KernelSignature> {
        self.signatures
            .iter()
            .filter(|(id, _)| id.namespace == namespace && id.name.starts_with(name_prefix))
            .map(|(_, sig)| sig)
            .collect()
    }

    /// Get all registered kernel IDs
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
    /// # Returns
    ///
    /// Reference to the global kernel registry
    pub fn global() -> &'static KernelRegistry {
        KERNEL_REGISTRY.get_or_init(Self::initialize)
    }

    /// Initialize the kernel registry with all built-in kernels
    ///
    /// Populates from `KERNEL_SIGNATURES` distributed slice. The `#[kernel_fn]` macro
    /// emits compile-time signatures that are collected by `linkme` at link time.
    /// Conversion from `&'static` to owned types is minimal: only `Vec`/`String` wrapping.
    fn initialize() -> KernelRegistry {
        let mut registry = KernelRegistry::new();

        for sig in KERNEL_SIGNATURES {
            let ast_sig = Self::convert_signature(sig);
            registry.register(ast_sig);
        }

        registry
    }

    /// Convert compile-time `KernelSignature` to owned AST `KernelSignature`.
    ///
    /// Since constraint enums are now shared (single source of truth), this
    /// conversion only handles lifetime differences:
    /// - `&'static [KernelParam]` → `Vec<KernelParam>` (clone)
    /// - `&'static [usize]` → `&'static [usize]` (already static, no conversion needed)
    /// - `&'static str` → `String` (for `RequiresUses`)
    fn convert_signature(sig: &continuum_kernel_types::KernelSignature) -> KernelSignature {
        KernelSignature {
            id: sig.id.clone(),
            params: sig
                .params
                .iter()
                .map(|p| KernelParam {
                    name: p.name,
                    shape: p.shape.clone(),
                    unit: p.unit.clone(),
                })
                .collect(),
            returns: KernelReturn {
                shape: sig.returns.shape.clone(),
                unit: sig.returns.unit.clone(),
                value_type: sig.returns.value_type,
            },
            purity: sig.purity,
            requires_uses: sig.requires_uses.map(|req| RequiresUses {
                key: req.key.to_string(),
                hint: req.hint.to_string(),
            }),
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
            value_type: ValueType::Scalar,
        };

        assert_eq!(ret.shape, ShapeDerivation::Scalar);
        assert_eq!(ret.unit, UnitDerivation::Dimensionless);
        assert_eq!(ret.value_type, ValueType::Scalar);
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
        let multiply = UnitDerivation::Multiply(&[0, 1]);
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
                value_type: ValueType::Scalar,
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
