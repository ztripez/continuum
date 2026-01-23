// Allow unwrap in tests
#![cfg_attr(test, allow(clippy::unwrap_used))]

//! Continuum Kernel Functions
//!
//! Kernel functions available for use in DSL expressions.
//! Functions are registered via the `#[kernel_fn]` attribute macro.

mod compare;
mod dt;
mod effect; // Re-enabled for type checking (implementations are stubs)
mod logic;
mod math;
mod matrix;
mod quat;
mod rng;
mod stats;
/// Tensor operations - exposed for VM executor arithmetic support
pub mod tensor_ops;
mod vector;

// Re-export for convenience
pub use continuum_foundation::Dt;
pub use continuum_kernel_registry::{
    all_names, eval_in_namespace, get_in_namespace, is_known_in, namespace_names,
};

use continuum_kernel_registry::{NAMESPACES, NamespaceDescriptor};

#[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
static DT_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "dt" };

#[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
static MATHS_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "maths" };

#[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
static VECTOR_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "vector" };

#[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
static QUAT_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "quat" };

#[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
static MATRIX_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "matrix" };

#[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
static TENSOR_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "tensor" };

#[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
static RNG_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "rng" };

#[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
static STATS_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "stats" };

#[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
static LOGIC_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "logic" };

#[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
static COMPARE_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "compare" };

// Disabled pending runtime context - see effect.rs header
// #[continuum_kernel_registry::linkme::distributed_slice(NAMESPACES)]
// static EFFECT_NAMESPACE: NamespaceDescriptor = NamespaceDescriptor { name: "effect" };
