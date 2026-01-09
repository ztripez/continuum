//! Event lowering (impulses and fractures).
//!
//! This module handles lowering impulse and fracture definitions from AST to IR.

use continuum_dsl::ast;
use continuum_foundation::{FractureId, ImpulseId, SignalId};

use crate::{CompiledEmit, CompiledFracture, CompiledImpulse, ValueType};

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(crate) fn lower_impulse(&mut self, def: &ast::ImpulseDef) -> Result<(), LowerError> {
        let id = ImpulseId::from(def.path.node.join(".").as_str());

        // Check for duplicate impulse definition
        if self.impulses.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("impulse.{}", id.0)));
        }

        let impulse = CompiledImpulse {
            id: id.clone(),
            payload_type: def
                .payload_type
                .as_ref()
                .map(|t| self.lower_type_expr(&t.node))
                .unwrap_or(ValueType::Scalar { range: None }),
            apply: def.apply.as_ref().map(|a| self.lower_expr(&a.body.node)),
        };

        self.impulses.insert(id, impulse);
        Ok(())
    }

    pub(crate) fn lower_fracture(&mut self, def: &ast::FractureDef) -> Result<(), LowerError> {
        let id = FractureId::from(def.path.node.join(".").as_str());

        // Check for duplicate fracture definition
        if self.fractures.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!(
                "fracture.{}",
                id.0
            )));
        }

        let mut reads = Vec::new();
        for cond in &def.conditions {
            self.collect_signal_refs(&cond.node, &mut reads);
        }
        for emit in &def.emit {
            self.collect_signal_refs(&emit.value.node, &mut reads);
        }

        let fracture = CompiledFracture {
            id: id.clone(),
            reads,
            conditions: def.conditions.iter().map(|c| self.lower_expr(&c.node)).collect(),
            emits: def
                .emit
                .iter()
                .map(|e| CompiledEmit {
                    target: SignalId::from(e.target.node.join(".").as_str()),
                    value: self.lower_expr(&e.value.node),
                })
                .collect(),
        };

        self.fractures.insert(id, fracture);
        Ok(())
    }
}
