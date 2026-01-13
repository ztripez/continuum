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
//! - **Pure** ([`KernelImpl::Pure`]) - Takes only numeric arguments, e.g., `sin(x)`
//! - **Dt-dependent** ([`KernelImpl::WithDt`]) - Also receives the time step, e.g., `integrate(prev, rate)`
//!
//! # Example Registration
//!
//! ```ignore
//! use continuum_kernel_macros::kernel_fn;
//! use continuum_foundation::Dt;
//!
//! #[kernel_fn(name = "decay")]
//! pub fn decay(value: f64, halflife: f64, dt: Dt) -> f64 {
//!     value * 0.5_f64.powf(dt / halflife)
//! }
//! ```
//!
//! # Example Lookup
//!
//! ```ignore
//! use continuum_kernel_registry::{get, eval, is_known};
//!
//! if is_known("sin") {
//!     let result = eval("sin", &[3.14159], 0.0);
//! }
//! ```

pub use continuum_foundation::Dt;
pub use linkme;

use linkme::distributed_slice;

/// Signature for kernel functions that don't need dt
pub type PureFn = fn(&[f64]) -> f64;

/// Signature for kernel functions that need dt
pub type DtFn = fn(&[f64], Dt) -> f64;

/// The actual function pointer, tagged by whether it needs dt
#[derive(Clone, Copy)]
pub enum KernelImpl {
    /// Pure function: `fn(&[f64]) -> f64`
    Pure(PureFn),
    /// Dt-dependent function: `fn(&[f64], Dt) -> f64`
    WithDt(DtFn),
}

impl KernelImpl {
    /// Evaluate the kernel function
    pub fn eval(&self, args: &[f64], dt: Dt) -> f64 {
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
    /// The implementation
    pub implementation: KernelImpl,
}

impl KernelDescriptor {
    /// Check if this kernel requires dt
    pub fn requires_dt(&self) -> bool {
        self.implementation.requires_dt()
    }

    /// Evaluate the kernel
    pub fn eval(&self, args: &[f64], dt: Dt) -> f64 {
        self.implementation.eval(args, dt)
    }
}

/// Distributed slice collecting all kernel function registrations.
///
/// Populated at link time by the `#[kernel_fn]` macro.
#[distributed_slice]
pub static KERNELS: [KernelDescriptor];

/// Get all registered kernel function names
pub fn all_names() -> impl Iterator<Item = &'static str> {
    KERNELS.iter().map(|k| k.name)
}

/// Look up a kernel by name
pub fn get(name: &str) -> Option<&'static KernelDescriptor> {
    KERNELS.iter().find(|k| k.name == name)
}

/// Check if a function name is a known kernel
pub fn is_known(name: &str) -> bool {
    get(name).is_some()
}

/// Evaluate a kernel by name
pub fn eval(name: &str, args: &[f64], dt: Dt) -> Option<f64> {
    get(name).map(|k| k.eval(args, dt))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test kernel registered via the slice directly
    #[distributed_slice(KERNELS)]
    static TEST_ABS: KernelDescriptor = KernelDescriptor {
        name: "test_abs",
        signature: "test_abs(x) -> Scalar",
        doc: "Test absolute value",
        category: "test",
        arity: Arity::Fixed(1),
        implementation: KernelImpl::Pure(|args| args[0].abs()),
    };

    #[test]
    fn test_lookup() {
        assert!(is_known("test_abs"));
        assert!(!is_known("nonexistent"));
    }

    #[test]
    fn test_eval() {
        let result = eval("test_abs", &[-5.0], 1.0);
        assert_eq!(result, Some(5.0));
    }

    #[test]
    fn test_arity() {
        let desc = get("test_abs").unwrap();
        assert_eq!(desc.arity, Arity::Fixed(1));
        assert!(!desc.requires_dt());
    }
}
