//! World container for a compiled simulation.
//!
//! A [`World`] represents the complete, resolved, and verified state of a
//! simulation definition. It is produced by the compiler pipeline after
//! all resolution and validation passes have completed.

use crate::ast::declaration::{Declaration, WorldDecl};
use crate::ast::node::Node;
use crate::ast::structural::{Entity, Era, Stratum};
use crate::foundation::{EntityId, Path};
use crate::resolve::graph::DagSet;
use std::collections::HashMap;

/// A compiled and resolved Continuum world.
///
/// Contains all primitives (signals, fields, operators) and structural
/// definitions (entities, strata, eras) ready for DAG construction.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct World {
    /// World metadata and warmup policy
    pub metadata: WorldDecl,

    /// Global nodes (I = ())
    pub globals: HashMap<Path, Node<()>>,

    /// Per-entity members (I = EntityId)
    pub members: HashMap<Path, Node<EntityId>>,

    /// Entity definitions
    pub entities: HashMap<Path, Entity>,

    /// Stratum definitions
    pub strata: HashMap<Path, Stratum>,

    /// Era definitions
    pub eras: HashMap<Path, Era>,

    /// Raw declarations (preserved for reference)
    pub declarations: Vec<Declaration>,
}

impl World {
    /// Create a new empty world with metadata.
    pub fn new(metadata: WorldDecl) -> Self {
        Self {
            metadata,
            globals: HashMap::new(),
            members: HashMap::new(),
            entities: HashMap::new(),
            strata: HashMap::new(),
            eras: HashMap::new(),
            declarations: Vec::new(),
        }
    }
}

/// A fully compiled Continuum world, including execution graphs.
///
/// This is the final output of the compiler pipeline, ready to be
/// consumed by the runtime.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CompiledWorld {
    /// The resolved world structure
    // Note: We might want to optimize what we include here for serialization
    pub world: World,

    /// The compiled execution DAGs
    pub dag_set: DagSet,
}

impl CompiledWorld {
    /// Create a new compiled world from a resolved world and its DAGs.
    pub fn new(world: World, dag_set: DagSet) -> Self {
        Self { world, dag_set }
    }
}
