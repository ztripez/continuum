//! Era lowering.
//!
//! This module handles lowering era definitions from AST to IR,
//! including dt conversion and transition condition lowering.

use indexmap::IndexMap;

use continuum_dsl::ast::{self, Span, StrataStateKind};
use continuum_foundation::{EraId, Path, StratumId};

use crate::{BinaryOp, CompiledEra, CompiledExpr, CompiledTransition, StratumState};

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(crate) fn lower_era(&mut self, def: &ast::EraDef, span: Span) -> Result<(), LowerError> {
        let id = EraId::from(Path::from_str(def.name.node.as_str()));

        // Check for duplicate era definition
        if self.eras.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition {
                name: format!("era.{}", id),
                file: self.file.clone(),
                span: def.name.span.clone(),
            });
        }

        // Convert dt to seconds
        let dt_seconds = def
            .dt
            .as_ref()
            .map(|dt| self.value_with_unit_to_seconds(&dt.node))
            .unwrap_or(1.0);

        // Convert strata states
        let mut strata_states = IndexMap::new();
        for state in &def.strata_states {
            let stratum_id = StratumId::from(state.strata.node.clone());
            let ir_state = match &state.state {
                StrataStateKind::Active => StratumState::Active,
                StrataStateKind::ActiveWithStride(s) => StratumState::ActiveWithStride(*s),
                StrataStateKind::Gated => StratumState::Gated,
            };
            strata_states.insert(stratum_id, ir_state);
        }

        // Convert transitions
        let transitions = def
            .transitions
            .iter()
            .map(|t| {
                // Combine all conditions with AND
                let condition = if t.conditions.is_empty() {
                    CompiledExpr::Literal(1.0, None) // always true
                } else if t.conditions.len() == 1 {
                    self.lower_expr(&t.conditions[0].node)
                } else {
                    t.conditions.iter().skip(1).fold(
                        self.lower_expr(&t.conditions[0].node),
                        |acc, cond| CompiledExpr::Binary {
                            op: BinaryOp::And,
                            left: Box::new(acc),
                            right: Box::new(self.lower_expr(&cond.node)),
                        },
                    )
                };

                CompiledTransition {
                    target_era: EraId::from(t.target.node.clone()),
                    condition,
                }
            })
            .collect();

        let era = CompiledEra {
            file: self.file.clone(),
            span,
            id: id.clone(),
            doc: def.doc.clone(),
            is_initial: def.is_initial,
            is_terminal: def.is_terminal,
            title: def.title.as_ref().map(|s| s.node.clone()),
            dt_seconds,
            strata_states,
            transitions,
        };

        self.eras.insert(id, era);
        Ok(())
    }
}
