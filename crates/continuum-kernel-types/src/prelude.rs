//! Kernel type prelude for convenience imports
//!
//! This module re-exports commonly used types for kernel annotations,
//! reducing boilerplate in kernel function files.
//!
//! # Usage
//!
//! ```rust,ignore
//! use continuum_kernel_types::prelude::*;
//!
//! #[kernel_fn(
//!     namespace = "maths",
//!     purity = Pure,
//!     shape_in = [Any, SameAs(0)],
//!     unit_in = [Any, SameAs(0)],
//!     shape_out = SameAs(0),
//!     unit_out = SameAs(0)
//! )]
//! pub fn add(a: f64, b: f64) -> f64 { a + b }
//! ```

pub use crate::shape::Shape;
pub use crate::unit::{Unit, UnitDimensions, UnitKind};
pub use crate::{
    DimConstraint, KernelId, KernelParam, KernelPurity, KernelReturn, KernelSignature,
    ShapeConstraint, ShapeDerivation, UnitConstraint, UnitDerivation,
};

// Re-export enum variants for ergonomic usage
pub use crate::DimConstraint::{Any as DimAny, Exact as DimExact, Var as DimVar};
pub use crate::KernelPurity::{Effect, Pure};
pub use crate::ShapeConstraint::{
    Any, AnyMatrix, AnyScalar, AnyVector, BroadcastWith, Exact, MatrixDims, SameAs, VectorDim,
};
pub use crate::ShapeDerivation::{
    Exact as ShapeExact, FromBroadcast, MatrixDims as ShapeMatrixDims, SameAs as ShapeSameAs,
    Scalar, VectorDim as ShapeVectorDim,
};
pub use crate::UnitConstraint::{
    Angle, Any as UnitAny, Dimensionless as UnitDimensionless, Exact as UnitExact,
    SameAs as UnitSameAs,
};
pub use crate::UnitDerivation::{
    Dimensionless, Divide, Exact as UnitDerivExact, Inverse, Multiply, SameAs as UnitDerivSameAs,
    Sqrt,
};
