//! Matrix Operations
//!
//! Functions for matrix operations: identity, transpose, determinant, inverse, eigenvalues, SVD.

mod basic;
mod construction;
mod decomp;
mod projection;
pub mod utils;

#[allow(unused_imports)]
pub use basic::*;
#[allow(unused_imports)]
pub use construction::*;
#[allow(unused_imports)]
pub use decomp::*;
#[allow(unused_imports)]
pub use projection::*;

#[cfg(test)]
mod tests;
