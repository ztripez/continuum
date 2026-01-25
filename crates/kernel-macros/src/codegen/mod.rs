//! Code generation for kernel registration.
//!
//! This module contains the logic for generating registration code for kernel functions,
//! including runtime descriptors, compile-time signatures, constant aliases, and vectorized
//! implementations.

pub(crate) mod constant;
pub(crate) mod runtime;
pub(crate) mod signature;
pub(crate) mod vectorized;
