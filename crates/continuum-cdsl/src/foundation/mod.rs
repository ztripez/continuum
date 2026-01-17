//! Compiler foundation types
//!
//! These types are used throughout the compiler but are distinct from
//! runtime foundation types. They represent compile-time constructs.

pub mod path;
pub mod shape;
pub mod unit;

pub use path::Path;
pub use shape::Shape;
pub use unit::{Unit, UnitDimensions, UnitKind};

// Re-export typed IDs
pub use continuum_foundation::{
    AnalyzerId, ChronicleId, EntityId, EraId, FieldId, FnId, FractureId, ImpulseId, InstanceId,
    MemberId, OperatorId, SignalId, StratumId, TypeId,
};
