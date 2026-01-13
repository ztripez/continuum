//! Kernel Function Registry.
//!
//! Provides distributed registration for kernel functions callable from DSL
//! expressions. Kernels are mathematical operations like `sin`, `cos`, `clamp`,
//! `integrate`, and `decay` that can be invoked from resolve blocks.
//!
//! # Architecture
//!
//! The registry uses [`linkme::distributed_slice`] for compile-time registration:
//!
//! 1. Functions register themselves using the `#[kernel_fn]` attribute macro
//! 2. At link time, all registrations are collected into [`KERNELS`]
//! 3. At runtime, the registry provides lookup by name for validation and dispatch
//!
//! This allows kernel functions to be defined anywhere in the codebase (including
//! in downstream crates) while remaining discoverable by the DSL validator.
//!
//! # Kernel Types
//!
//! Kernels come in two flavors:
//!
//! - **Pure** ([`KernelImpl::Pure`]) - Takes only `Value` arguments, e.g., `sin(x)`
//! - **Dt-dependent** ([`KernelImpl::WithDt`]) - Also receives the time step, e.g., `integrate(prev, rate)`
//!
//! # Example Registration
//!
//! ```ignore
//! use continuum_kernel_macros::kernel_fn;
//! use continuum_foundation::Dt;
//!
//! #[kernel_fn(namespace = "dt")]
//! pub fn decay(value: f64, halflife: f64, dt: Dt) -> f64 {
//!     value * 0.5_f64.powf(dt / halflife)
//! }
//! ```
//!
//! # Example Lookup
//!
//! ```ignore
//! use continuum_kernel_registry::{eval_in_namespace, is_known_in};
//! use continuum_foundation::Value;
//!
//! if is_known_in("maths", "sin") {
//!     let args = vec![Value::Scalar(3.14159)];
//!     let result = eval_in_namespace("maths", "sin", &args, 0.0);
//! }
//! ```

pub use continuum_foundation::{Dt, FromValue, IntoValue, Value};
pub use linkme;

use linkme::distributed_slice;

/// Virtual register buffer for vectorized operations
///
/// This represents the data storage for vectorized kernel execution.
/// It matches the VRegBuffer type used in the IR execution engine.
#[derive(Debug, Clone)]
pub enum VRegBuffer {
    /// Array of scalar values (one per entity)
    Scalar(Vec<f64>),
    /// Array of integer values
    Integer(Vec<i64>),
    /// Array of boolean values
    Boolean(Vec<bool>),
    /// Array of Vec2 values
    Vec2(Vec<[f64; 2]>),
    /// Array of Vec3 values (one per entity)  
    Vec3(Vec<[f64; 3]>),
    /// Array of Vec4 values
    Vec4(Vec<[f64; 4]>),
    /// Uniform value (same for all entities, broadcast on demand)
    Uniform(Value),
}

impl VRegBuffer {
    /// Create a uniform scalar buffer
    pub fn uniform_scalar(value: f64) -> Self {
        VRegBuffer::Uniform(Value::Scalar(value))
    }

    /// Create a uniform buffer from any Value
    pub fn uniform(value: Value) -> Self {
        VRegBuffer::Uniform(value)
    }

    /// Get value at index as a generic Value
    pub fn get(&self, idx: usize) -> Option<Value> {
        match self {
            VRegBuffer::Scalar(v) => v.get(idx).map(|&x| Value::Scalar(x)),
            VRegBuffer::Integer(v) => v.get(idx).map(|&x| Value::Integer(x)),
            VRegBuffer::Boolean(v) => v.get(idx).map(|&x| Value::Boolean(x)),
            VRegBuffer::Vec2(v) => v.get(idx).map(|&x| Value::Vec2(x)),
            VRegBuffer::Vec3(v) => v.get(idx).map(|&x| Value::Vec3(x)),
            VRegBuffer::Vec4(v) => v.get(idx).map(|&x| Value::Vec4(x)),
            VRegBuffer::Uniform(v) => Some(v.clone()),
        }
    }

    /// Get scalar value at index (convenience)
    pub fn get_scalar(&self, idx: usize) -> Option<f64> {
        self.get(idx).and_then(|v| v.as_scalar())
    }

    /// Get as uniform value if possible
    pub fn as_uniform(&self) -> Option<&Value> {
        match self {
            VRegBuffer::Uniform(v) => Some(v),
            _ => None,
        }
    }

    /// Get as scalar slice if possible
    pub fn as_scalar_slice(&self) -> Option<&[f64]> {
        match self {
            VRegBuffer::Scalar(arr) => Some(arr),
            _ => None,
        }
    }

    /// Convert to full scalar array with given size (if scalar or uniform scalar)
    pub fn to_scalar_array(&self, size: usize) -> Option<Vec<f64>> {
        match self {
            VRegBuffer::Scalar(arr) => Some(arr.clone()),
            VRegBuffer::Uniform(Value::Scalar(v)) => Some(vec![*v; size]),
            VRegBuffer::Uniform(Value::Integer(v)) => Some(vec![*v as f64; size]),
            VRegBuffer::Integer(arr) => Some(arr.iter().map(|&x| x as f64).collect()),
            _ => None,
        }
    }
}

/// Result type for vectorized operations (re-exported from IR crate for convenience)
pub type VectorizedResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Signature for kernel functions that don't need dt
pub type PureFn = fn(&[Value]) -> Value;

/// Signature for kernel functions that need dt
pub type DtFn = fn(&[Value], Dt) -> Value;

/// Signature for vectorized kernel functions that don't need dt
/// Args: &[buffer1, buffer2, ...], population_size -> Result<VRegBuffer, Error>
pub type VectorizedPureFn = fn(&[&VRegBuffer], usize) -> VectorizedResult<VRegBuffer>;

/// Signature for vectorized kernel functions that need dt  
/// Args: &[buffer1, buffer2, ...], dt, population_size -> Result<VRegBuffer, Error>
pub type VectorizedDtFn = fn(&[&VRegBuffer], Dt, usize) -> VectorizedResult<VRegBuffer>;

/// The actual function pointer, tagged by whether it needs dt
#[derive(Clone, Copy)]
pub enum KernelImpl {
    /// Pure function: `fn(&[Value]) -> Value`
    Pure(PureFn),
    /// Dt-dependent function: `fn(&[Value], Dt) -> Value`
    WithDt(DtFn),
}

/// Vectorized implementation for high-performance entity processing
#[derive(Clone, Copy)]
pub enum VectorizedImpl {
    /// Pure vectorized function: `fn(&[&VRegBuffer], usize) -> Result<VRegBuffer, Error>`
    Pure(VectorizedPureFn),
    /// Dt-dependent vectorized function: `fn(&[&VRegBuffer], Dt, usize) -> Result<VRegBuffer, Error>`
    WithDt(VectorizedDtFn),
}

/// Descriptor for a registered vectorized kernel function
pub struct VectorizedKernelDescriptor {
    /// Namespace name (e.g., "maths", "vector", "dt")
    pub namespace: &'static str,
    /// DSL name (e.g., "integrate", "decay", "sin")
    pub name: &'static str,
    /// Vectorized implementation
    pub implementation: VectorizedImpl,
}

/// Descriptor for a registered namespace.
pub struct NamespaceDescriptor {
    /// Namespace name (e.g., "maths", "physics")
    pub name: &'static str,
}

impl KernelImpl {
    /// Evaluate the kernel function
    pub fn eval(&self, args: &[Value], dt: Dt) -> Value {
        match self {
            KernelImpl::Pure(f) => f(args),
            KernelImpl::WithDt(f) => f(args, dt),
        }
    }

    /// Check if this implementation requires dt
    pub fn requires_dt(&self) -> bool {
        matches!(self, KernelImpl::WithDt(_))
    }
}

impl VectorizedImpl {
    /// Evaluate the vectorized kernel function
    pub fn eval(
        &self,
        args: &[&VRegBuffer],
        dt: Dt,
        population: usize,
    ) -> VectorizedResult<VRegBuffer> {
        match self {
            VectorizedImpl::Pure(f) => f(args, population),
            VectorizedImpl::WithDt(f) => f(args, dt, population),
        }
    }

    /// Check if this implementation requires dt
    pub fn requires_dt(&self) -> bool {
        matches!(self, VectorizedImpl::WithDt(_))
    }
}

/// Arity specification for a kernel function
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arity {
    /// Fixed number of arguments
    Fixed(usize),
    /// Variadic (any number of arguments)
    Variadic,
}

impl Arity {
    /// Get as `Option<usize>` for compatibility
    pub fn as_option(&self) -> Option<usize> {
        match self {
            Arity::Fixed(n) => Some(*n),
            Arity::Variadic => None,
        }
    }
}

/// Descriptor for a registered kernel function
pub struct KernelDescriptor {
    /// Namespace name (e.g., "maths", "vector", "dt")
    pub namespace: &'static str,
    /// DSL name (e.g., "decay", "sin", "min")
    pub name: &'static str,
    /// Full signature string (e.g., "clamp(value, min, max) -> Scalar")
    pub signature: &'static str,
    /// Documentation string
    pub doc: &'static str,
    /// Category tag (e.g., "math", "vector", "simulation")
    pub category: &'static str,
    /// Number of arguments (excluding dt if present)
    pub arity: Arity,
    /// The scalar implementation
    pub implementation: KernelImpl,
    /// Optional vectorized implementation for high-performance entity processing
    pub vectorized_impl: Option<VectorizedImpl>,
}

impl KernelDescriptor {
    /// Check if this kernel requires dt
    pub fn requires_dt(&self) -> bool {
        self.implementation.requires_dt()
    }

    /// Evaluate the scalar kernel
    pub fn eval(&self, args: &[Value], dt: Dt) -> Value {
        self.implementation.eval(args, dt)
    }

    /// Check if this kernel has a vectorized implementation
    pub fn has_vectorized(&self) -> bool {
        self.vectorized_impl.is_some()
    }

    /// Evaluate the vectorized kernel (if available)
    pub fn eval_vectorized(
        &self,
        args: &[&VRegBuffer],
        dt: Dt,
        population: usize,
    ) -> Option<VectorizedResult<VRegBuffer>> {
        self.vectorized_impl
            .as_ref()
            .map(|impl_| impl_.eval(args, dt, population))
    }
}

/// Distributed slice collecting all kernel function registrations.
///
/// Populated at link time by the `#[kernel_fn]` macro.
#[distributed_slice]
pub static KERNELS: [KernelDescriptor];

/// Distributed slice collecting all vectorized kernel function registrations.
///
/// Populated at link time by the `#[vectorized_kernel_fn]` macro.
#[distributed_slice]
pub static VECTOR_KERNELS: [VectorizedKernelDescriptor];

/// Distributed slice collecting all namespace registrations.
///
/// Populated at link time by namespace descriptor definitions.
#[distributed_slice]
pub static NAMESPACES: [NamespaceDescriptor];

/// Get all registered kernel function names (namespace, name)
pub fn all_names() -> impl Iterator<Item = (&'static str, &'static str)> {
    KERNELS.iter().map(|k| (k.namespace, k.name))
}

/// Get all registered namespace names
pub fn namespace_names() -> impl Iterator<Item = &'static str> {
    NAMESPACES.iter().map(|n| n.name)
}

/// Look up a kernel by namespace/name
pub fn get_in_namespace(namespace: &str, name: &str) -> Option<&'static KernelDescriptor> {
    KERNELS
        .iter()
        .find(|k| k.namespace == namespace && k.name == name)
}

/// Look up a namespace by name
pub fn get_namespace(name: &str) -> Option<&'static NamespaceDescriptor> {
    NAMESPACES.iter().find(|n| n.name == name)
}

/// Check if a namespace is registered
pub fn namespace_exists(name: &str) -> bool {
    get_namespace(name).is_some()
}

/// Check if a function name is a known kernel in a namespace
pub fn is_known_in(namespace: &str, name: &str) -> bool {
    get_in_namespace(namespace, name).is_some()
}

/// Evaluate a kernel by namespace/name
pub fn eval_in_namespace(namespace: &str, name: &str, args: &[Value], dt: Dt) -> Option<Value> {
    get_in_namespace(namespace, name).map(|k| k.eval(args, dt))
}

/// Get vectorized implementation for a kernel by name
pub fn get_vectorized(namespace: &str, name: &str) -> Option<&'static VectorizedImpl> {
    VECTOR_KERNELS
        .iter()
        .find(|k| k.namespace == namespace && k.name == name)
        .map(|k| &k.implementation)
        .or_else(|| get_in_namespace(namespace, name).and_then(|k| k.vectorized_impl.as_ref()))
}

/// Check if a kernel has vectorized implementation
pub fn has_vectorized_impl(namespace: &str, name: &str) -> bool {
    get_vectorized(namespace, name).is_some()
}

/// Evaluate a vectorized kernel by name
pub fn eval_vectorized(
    namespace: &str,
    name: &str,
    args: &[&VRegBuffer],
    dt: Dt,
    population: usize,
) -> Option<VectorizedResult<VRegBuffer>> {
    get_vectorized(namespace, name).map(|impl_| impl_.eval(args, dt, population))
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_foundation::FromValue;

    // Test namespace registered via the slice directly
    #[distributed_slice(NAMESPACES)]
    static TEST_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "test" };

    // Test kernel registered via the slice directly
    #[distributed_slice(KERNELS)]
    static TEST_ABS: KernelDescriptor = KernelDescriptor {
        namespace: "test",
        name: "test_abs",
        signature: "test_abs(x) -> Scalar",
        doc: "Test absolute value",
        category: "test",
        arity: Arity::Fixed(1),
        implementation: KernelImpl::Pure(|args| {
            let val = f64::from_value(&args[0]).unwrap();
            Value::Scalar(val.abs())
        }),
        vectorized_impl: None,
    };

    #[test]
    fn test_lookup() {
        assert!(is_known_in("test", "test_abs"));
        assert!(get_in_namespace("test", "test_abs").is_some());
        assert!(!is_known_in("test", "missing"));
    }

    #[test]
    fn test_namespace_registry() {
        assert!(namespace_exists("test"));
        assert!(get_namespace("test").is_some());
        assert!(!namespace_exists("missing"));
    }

    #[test]
    fn test_eval() {
        let args = vec![Value::Scalar(-5.0)];
        let result = eval_in_namespace("test", "test_abs", &args, 1.0);
        assert_eq!(result, Some(Value::Scalar(5.0)));
    }

    #[test]
    fn test_arity() {
        let desc = get_in_namespace("test", "test_abs").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
        assert!(!desc.requires_dt());
    }

    #[test]
    fn test_vectorized_api() {
        // test_abs should not have vectorized implementation
        assert!(!has_vectorized_impl("test", "test_abs"));
        assert!(get_vectorized("test", "test_abs").is_none());

        let desc = get_in_namespace("test", "test_abs").unwrap();
        assert!(!desc.has_vectorized());
        assert!(desc.eval_vectorized(&[], 0.0, 10).is_none());
    }

    #[test]
    fn test_vreg_buffer() {
        let uniform = VRegBuffer::uniform_scalar(42.0);
        assert_eq!(uniform.as_uniform(), Some(&Value::Scalar(42.0)));
        assert_eq!(uniform.get_scalar(0), Some(42.0));
        assert_eq!(uniform.get_scalar(999), Some(42.0));

        let scalar = VRegBuffer::Scalar(vec![1.0, 2.0, 3.0]);
        // assert_eq!(scalar.as_uniform(), None); // None != None comparison issues
        assert!(scalar.as_uniform().is_none());
        assert_eq!(scalar.get_scalar(1), Some(2.0));
        assert_eq!(scalar.as_scalar_slice(), Some(&[1.0, 2.0, 3.0][..]));
    }
}
