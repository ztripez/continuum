//! Chronicle lowering.
//!
//! This module handles lowering chronicle definitions from AST to IR.
//! Chronicles are observer-only event recorders that do not affect causality.

use continuum_dsl::ast::{self, Span};
use continuum_foundation::ChronicleId;

use crate::{CompiledChronicle, CompiledEventField, CompiledObserveHandler};

use super::{LowerError, Lowerer};

impl Lowerer {
    /// Lower a chronicle definition from AST to IR.
    ///
    /// Chronicles observe simulation state and emit events for logging and analytics.
    /// They cannot affect causality - removing all chronicles must not change simulation results.
    pub(crate) fn lower_chronicle(
        &mut self,
        def: &ast::ChronicleDef,
        span: Span,
    ) -> Result<(), LowerError> {
        let id = ChronicleId::from(def.path.node.clone());

        // Check for duplicate chronicle definition
        if self.chronicles.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition {
                name: format!("chronicle.{}", id),
                file: self.file.clone(),
                span: def.path.span.clone(),
            });
        }

        // Collect signal reads from all handlers
        let mut reads = Vec::new();
        if let Some(observe) = &def.observe {
            for handler in &observe.handlers {
                // Collect from condition
                self.collect_signal_refs(&handler.condition.node, &mut reads);
                // Collect from event fields
                for (_name, value) in &handler.event_fields {
                    self.collect_signal_refs(&value.node, &mut reads);
                }
            }
        }

        // Lower observation handlers
        let handlers = if let Some(observe) = &def.observe {
            observe
                .handlers
                .iter()
                .map(|h| CompiledObserveHandler {
                    condition: self.lower_expr(&h.condition.node),
                    event_name: h.event_name.node.to_string(),
                    event_fields: h
                        .event_fields
                        .iter()
                        .map(|(name, value)| CompiledEventField {
                            name: name.node.clone(),
                            value: self.lower_expr(&value.node),
                        })
                        .collect(),
                })
                .collect()
        } else {
            Vec::new()
        };

        let chronicle = CompiledChronicle {
            file: self.file.clone(),
            span,
            id: id.clone(),
            reads,
            handlers,
        };

        self.chronicles.insert(id, chronicle);
        Ok(())
    }
}
