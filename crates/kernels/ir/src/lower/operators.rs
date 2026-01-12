//! Operator and function lowering.
//!
//! This module handles lowering operator and user-defined function definitions
//! from AST to IR, including phase inference.

use std::collections::HashSet;

use continuum_dsl::ast::{self, FnDef, OperatorBody};
use continuum_foundation::{FnId, OperatorId, StratumId};

use crate::{CompiledFn, CompiledOperator, OperatorPhaseIr};

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(crate) fn lower_operator(&mut self, def: &ast::OperatorDef) -> Result<(), LowerError> {
        let id = OperatorId::from(def.path.node.clone());

        // Check for duplicate operator definition
        if self.operators.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("operator.{}", id)));
        }

        // Determine stratum
        let stratum = def
            .strata
            .as_ref()
            .map(|s| StratumId::from(s.node.clone()))
            .unwrap_or_else(|| StratumId::from("default"));

        // Validate stratum exists
        self.validate_stratum(&stratum)?;

        let phase = def
            .phase
            .as_ref()
            .map(|p| self.lower_operator_phase(&p.node))
            .or_else(|| {
                def.body.as_ref().map(|b| match b {
                    OperatorBody::Warmup(_) => OperatorPhaseIr::Warmup,
                    OperatorBody::Collect(_) => OperatorPhaseIr::Collect,
                    OperatorBody::Measure(_) => OperatorPhaseIr::Measure,
                })
            })
            .unwrap_or(OperatorPhaseIr::Collect);

        let body_expr = def.body.as_ref().map(|b| match b {
            OperatorBody::Warmup(e) | OperatorBody::Collect(e) | OperatorBody::Measure(e) => {
                &e.node
            }
        });

        let mut reads = Vec::new();
        if let Some(expr) = body_expr {
            self.collect_signal_refs(expr, &mut reads);
        }

        // Lower assertions
        let assertions = def
            .assertions
            .as_ref()
            .map(|a| self.lower_assert_block(a))
            .unwrap_or_default();

        let operator = CompiledOperator {
            id: id.clone(),
            stratum,
            phase,
            reads,
            body: body_expr.map(|e| self.lower_expr(e)),
            assertions,
        };

        self.operators.insert(id, operator);
        Ok(())
    }

    pub(crate) fn lower_fn(&mut self, def: &FnDef) -> Result<(), LowerError> {
        let id = FnId::from(def.path.node.clone());

        // Check for duplicate function definition
        if self.functions.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("fn.{}", id)));
        }

        // Collect parameter names
        let params: Vec<String> = def.params.iter().map(|p| p.name.node.clone()).collect();

        // Lower the body with parameters as locals
        let mut locals = HashSet::new();
        for param in &params {
            locals.insert(param.clone());
        }
        let body = self.lower_expr_with_locals(&def.body.node, &locals);

        let compiled_fn = CompiledFn {
            id: id.clone(),
            params,
            body,
        };

        self.functions.insert(id, compiled_fn);
        Ok(())
    }
}
