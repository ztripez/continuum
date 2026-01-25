//! Spatial topology infrastructure for entity neighbor queries.
//!
//! This module provides spatial topology structures that enable `spatial.*` kernel queries
//! like `spatial.neighbors()` and `spatial.gradient()`. Topologies are frozen in the Configure
//! phase and represent authoritative spatial structure (not observation).
//!
//! # Topology Types
//!
//! - **IcosahedralTopology** - Spherical icosahedral grid with near-uniform cells
//!
//! # Design Principles
//!
//! - Topologies are **authoritative structure** (frozen in Configure phase)
//! - Neighbor queries are **deterministic** (stable ordering)
//! - Memory layout optimized for **cache locality** (SoA-friendly)
//! - CPU implementation for MVP (GPU optimization is future work)
//!
//! # Architecture Alignment
//!
//! Per core invariants:
//! - Topology is NOT observation (it's causal structure)
//! - Frozen in Configure phase (deterministic)
//! - Enables signal-based spatial queries (preserves signal authority)

pub mod icosahedron;

pub use icosahedron::IcosahedralTopology;

use crate::EntityIndex;

/// Spatial topology trait - provides neighbor query interface.
///
/// All topology implementations must provide deterministic neighbor queries
/// with stable ordering. Topologies are frozen at initialization (Configure phase).
pub trait SpatialTopology: Send + Sync {
    /// Get the neighbors of an entity.
    ///
    /// Returns entity indices in deterministic order.
    /// Empty slice if entity has no neighbors or index is out of bounds.
    fn neighbors(&self, entity: EntityIndex) -> &[EntityIndex];

    /// Get the total number of entities in this topology.
    fn entity_count(&self) -> usize;

    /// Validate that an entity index is within bounds.
    fn is_valid_index(&self, entity: EntityIndex) -> bool {
        entity.0 < self.entity_count()
    }
}
