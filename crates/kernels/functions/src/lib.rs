//! Continuum Kernel Functions
//!
//! Kernel functions available for use in DSL expressions.
//! Functions are registered via the `#[kernel_fn]` attribute macro.

mod dt;
mod math;
mod matrix;
mod quat;
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
