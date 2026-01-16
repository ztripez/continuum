//! Entity lowering.
//!
//! This module handles lowering entity definitions from AST to IR.
//! Entities are pure index spaces - they define what exists but not what state
//! it has. Per-entity state is defined via member signals.

use continuum_dsl::ast::{EntityDef, Span};
use continuum_foundation::EntityId;

use crate::CompiledEntity;

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(crate) fn lower_entity(&mut self, def: &EntityDef, span: Span) -> Result<(), LowerError> {
        let id = EntityId::from(def.path.node.clone());

        // Check for duplicate entity definition
        if self.entities.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition {
                name: format!("entity.{}", id),
                file: self.file.clone(),
                span: def.path.span.clone(),
            });
        }

        // Count source from config path
        let count_source = def
            .count_source
            .as_ref()
            .map(|p| p.node.to_string().replace("config.", ""));

        // Count bounds
        let count_bounds = def.count_bounds.as_ref().map(|b| (b.min, b.max));

        let entity = CompiledEntity {
            file: self.file.clone(),
            span,
            id: id.clone(),
            doc: def.doc.clone(),
            count_source,
            count_bounds,
        };

        self.entities.insert(id, entity);
        Ok(())
    }
}
