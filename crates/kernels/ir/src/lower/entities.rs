//! Entity lowering.
//!
//! This module handles lowering entity definitions from AST to IR.
//! Entities are pure index spaces - they define what exists but not what state
//! it has. Per-entity state is defined via member signals.

use continuum_dsl::ast::EntityDef;
use continuum_foundation::EntityId;

use crate::CompiledEntity;

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(crate) fn lower_entity(&mut self, def: &EntityDef) -> Result<(), LowerError> {
        let id = EntityId::from(def.path.node.join(".").as_str());

        // Check for duplicate entity definition
        if self.entities.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("entity.{}", id.0)));
        }

        // Count source from config path
        let count_source = def
            .count_source
            .as_ref()
            .map(|p| p.node.join(".").replace("config.", ""));

        // Count bounds
        let count_bounds = def.count_bounds.as_ref().map(|b| (b.min, b.max));

        let entity = CompiledEntity {
            id: id.clone(),
            count_source,
            count_bounds,
        };

        self.entities.insert(id, entity);
        Ok(())
    }
}
