//! Execution DAG for Simulation Scheduling.
//!
//! This module defines the directed acyclic graph (DAG) structure used to
//! schedule signal resolution and operator execution within each tick.
//!
//! # Structure
//!
//! - [`DagNode`] - A single execution unit (signal resolver, operator, field emitter)
//! - [`Level`] - A set of nodes with no inter-dependencies (can execute in parallel)
//! - [`ExecutableDag`] - A complete DAG for one (phase, stratum) combination
//! - [`EraDags`] - All DAGs for a single era
//! - [`DagSet`] - All DAGs across all eras
//!
//! # Execution Model
//!
//! DAGs are organized into topological levels. Within a level, all nodes can
//! execute in parallel since they have no dependencies on each other. Levels
//! are separated by barriers—all nodes in level N must complete before any
//! node in level N+1 begins.
//!
//! # Building DAGs
//!
//! Use [`DagBuilder`] to construct DAGs. Nodes are added with their dependencies,
//! then [`DagBuilder::build`] performs topological sorting to create levels.
//! If a cycle is detected, [`CycleError`] is returned.

mod builder;
mod collection;
mod topology;
mod types;
mod verification;

#[cfg(test)]
mod tests;

// Re-export public API
pub use builder::{
    AggregateBarrier, BarrierDagBuilder, BarrierStats, DagBuilder, MemberSignalDependency,
    StratumEligibility,
};
pub use collection::{DagSet, EraDags};
pub use topology::CycleError;
pub use types::{DagNode, ExecutableDag, Level, NodeId, NodeKind};
pub use verification::{verify_barrier_semantics, BarrierViolation};
