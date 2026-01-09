//! Entity lowering.
//!
//! This module handles lowering entity definitions from AST to IR,
//! including schema fields and entity field definitions.

use continuum_dsl::ast::EntityDef;
use continuum_foundation::{EntityId, StratumId};

use crate::{CompiledEntity, CompiledEntityField, CompiledSchemaField, TopologyIr, ValueType};

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(crate) fn lower_entity(&mut self, def: &EntityDef) -> Result<(), LowerError> {
        let id = EntityId::from(def.path.node.join(".").as_str());

        // Determine stratum
        let stratum = def
            .strata
            .as_ref()
            .map(|s| StratumId::from(s.node.join(".").as_str()))
            .unwrap_or_else(|| StratumId::from("default"));

        // Count source from config path
        let count_source = def
            .count_source
            .as_ref()
            .map(|p| p.node.join(".").replace("config.", ""));

        // Count bounds
        let count_bounds = def.count_bounds.as_ref().map(|b| (b.min, b.max));

        // Schema fields
        let schema = def
            .schema
            .iter()
            .map(|f| CompiledSchemaField {
                name: f.name.node.clone(),
                value_type: self.lower_type_expr(&f.ty.node),
            })
            .collect();

        // Collect signal reads from resolve and field measure blocks
        let mut reads = Vec::new();
        let mut entity_reads = Vec::new();
        if let Some(resolve) = &def.resolve {
            self.collect_signal_refs(&resolve.body.node, &mut reads);
            self.collect_entity_refs(&resolve.body.node, &mut entity_reads);
        }
        for field in &def.fields {
            if let Some(measure) = &field.measure {
                self.collect_signal_refs(&measure.body.node, &mut reads);
                self.collect_entity_refs(&measure.body.node, &mut entity_reads);
            }
        }

        // Resolve expression
        let resolve = def.resolve.as_ref().map(|r| self.lower_expr(&r.body.node));

        // Assertions
        let assertions = def
            .assertions
            .as_ref()
            .map(|a| self.lower_assert_block(a))
            .unwrap_or_default();

        // Entity fields
        let fields = def
            .fields
            .iter()
            .map(|f| CompiledEntityField {
                name: f.name.node.clone(),
                value_type: f
                    .ty
                    .as_ref()
                    .map(|t| self.lower_type_expr(&t.node))
                    .unwrap_or(ValueType::Scalar { range: None }),
                topology: f
                    .topology
                    .as_ref()
                    .map(|t| self.lower_topology(&t.node))
                    .unwrap_or(TopologyIr::PointCloud),
                measure: f.measure.as_ref().map(|m| self.lower_expr(&m.body.node)),
            })
            .collect();

        let entity = CompiledEntity {
            id: id.clone(),
            stratum,
            count_source,
            count_bounds,
            schema,
            reads,
            entity_reads,
            resolve,
            assertions,
            fields,
        };

        self.entities.insert(id, entity);
        Ok(())
    }
}
