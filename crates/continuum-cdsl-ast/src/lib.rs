// Allow unwrap in tests
#![cfg_attr(test, allow(clippy::unwrap_used))]

//! AST types for Continuum DSL
//!
//! This crate contains all AST node definitions, type system types,
//! and foundation types used by the parser and compiler.

pub mod ast;
pub mod error;
pub mod foundation;

// Re-export commonly used types
pub use foundation::{
    AggregateOp, BinaryOp, Bounds, DeterminismPolicy, EntityId, FaultPolicy, KernelType, Path,
    Shape, SourceFile, SourceMap, Span, Type, UnaryOp, Unit, UnitDimensions, UnitKind, UserType,
    UserTypeId, WorldPolicy,
};

pub use ast::*;
