//! SSA (Static Single Assignment) Intermediate Representation
//!
//! This module provides SSA IR for signal resolve expressions, enabling
//! compiler optimizations like CSE, DCE, and vectorization.
//!
//! # Overview
//!
//! The SSA IR transforms tree-structured `CompiledExpr` into a linear sequence
//! of instructions where each value is assigned exactly once. This form enables:
//!
//! - **Common Subexpression Elimination (CSE)**: Identical computations are computed once
//! - **Dead Code Elimination (DCE)**: Unused values are removed
//! - **Constant Folding**: Constant expressions are evaluated at compile time
//! - **Vectorization**: Linear sequences map naturally to SIMD/GPU kernels
//!
//! # Example
//!
//! DSL: `clamp(prev + signal.heat * 0.5, 0.0, 1.0)`
//!
//! SSA:
//! ```text
//! %0 = LoadPrev
//! %1 = LoadSignal(heat)
//! %2 = LoadConst(0.5)
//! %3 = BinOp(Mul, %1, %2)
//! %4 = BinOp(Add, %0, %3)
//! %5 = LoadConst(0.0)
//! %6 = LoadConst(1.0)
//! %7 = Call(clamp, [%4, %5, %6])
//! Return(%7)
//! ```

mod lower;
mod types;
mod validate;

pub use lower::lower_to_ssa;
pub use types::*;
pub use validate::{validate_ssa, SsaValidationError};

#[cfg(test)]
mod tests;
