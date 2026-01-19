//! World container for a compiled simulation.
//!
//! A [`World`] represents the complete, resolved, and verified state of a
//! simulation definition. it is produced by the compiler pipeline after
//! all resolution and validation passes have completed.

use crate::ast::declaration::{Declaration, WorldDecl};
use crate::ast::node::Node;
use crate::ast::structural::{Entity, Era, Stratum};
use crate::foundation::{EntityId, Path};
use std::collections::HashMap;

/// A compiled and resolved Continuum world.
///
/// Contains all primitives (signals, fields, operators) and structural
/// definitions (entities, strata, eras) ready for DAG construction.
#[derive(Debug, Clone)]
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
