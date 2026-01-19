//! Matrix Operations
//!
//! Functions for matrix operations: identity, transpose, determinant, inverse, eigenvalues, SVD.

mod basic;
mod construction;
mod decomp;
mod projection;

pub use basic::*;
pub use construction::*;
pub use decomp::*;
pub use projection::*;

#[cfg(test)]
mod tests;
