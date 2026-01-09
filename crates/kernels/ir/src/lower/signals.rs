//! Signal and field lowering.
//!
//! This module handles lowering signal and field definitions from AST to IR,
//! including dependency extraction and dt-robustness validation.

use continuum_dsl::ast;
use continuum_foundation::{FieldId, SignalId, StratumId};

use crate::{CompiledField, CompiledSignal, CompiledWarmup, TopologyIr, ValueType};

use super::{LowerError, Lowerer};

impl Lowerer {
    pub(crate) fn lower_signal(&mut self, def: &ast::SignalDef) -> Result<(), LowerError> {
        let id = SignalId::from(def.path.node.join(".").as_str());
        let signal_path = def.path.node.join(".");

        // Check for duplicate signal definition
        if self.signals.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("signal.{}", id.0)));
        }

        // Determine stratum
        let stratum = def
            .strata
            .as_ref()
            .map(|s| StratumId::from(s.node.join(".").as_str()))
            .unwrap_or_else(|| StratumId::from("default"));

        // Validate stratum exists
        self.validate_stratum(&stratum)?;

        // Process local const blocks - add to global constants with signal-prefixed keys
        for entry in &def.local_consts {
            let local_key = entry.path.node.join(".");
            // Add with full signal path prefix: signal.path.local_key
            let full_key = format!("{}.{}", signal_path, local_key);
            let value = self.literal_to_f64(&entry.value.node)?;
            self.constants.insert(full_key, value);
        }

        // Process local config blocks - add to global config with signal-prefixed keys
        for entry in &def.local_config {
            let local_key = entry.path.node.join(".");
            // Add with full signal path prefix: signal.path.local_key
            let full_key = format!("{}.{}", signal_path, local_key);
            let value = self.literal_to_f64(&entry.value.node)?;
            self.config.insert(full_key, value);
        }

        // Collect signal dependencies from resolve expression
        let mut reads = Vec::new();
        if let Some(resolve) = &def.resolve {
            self.collect_signal_refs(&resolve.body.node, &mut reads);
        }

        // Validate dt_raw usage: if resolve uses dt_raw, signal must declare it
        if let Some(resolve) = &def.resolve {
            if !def.dt_raw && self.expr_uses_dt_raw(&resolve.body.node) {
                return Err(LowerError::UndeclaredDtRawUsage(signal_path));
            }
        }

        // Lower warmup if present
        let warmup = def.warmup.as_ref().map(|w| CompiledWarmup {
            iterations: w.iterations.node,
            convergence: w.convergence.as_ref().map(|c| c.node),
            iterate: self.lower_expr(&w.iterate.node),
        });

        // Lower resolve expression
        let resolve = def.resolve.as_ref().map(|r| self.lower_expr(&r.body.node));

        // Lower assertions
        let assertions = def
            .assertions
            .as_ref()
            .map(|a| self.lower_assert_block(a))
            .unwrap_or_default();

        let signal = CompiledSignal {
            id: id.clone(),
            stratum,
            title: def.title.as_ref().map(|s| s.node.clone()),
            symbol: def.symbol.as_ref().map(|s| s.node.clone()),
            value_type: def
                .ty
                .as_ref()
                .map(|t| self.lower_type_expr(&t.node))
                .unwrap_or(ValueType::Scalar { range: None }),
            uses_dt_raw: def.dt_raw,
            reads,
            resolve,
            warmup,
            assertions,
        };

        self.signals.insert(id, signal);
        Ok(())
    }

    pub(crate) fn lower_field(&mut self, def: &ast::FieldDef) -> Result<(), LowerError> {
        let id = FieldId::from(def.path.node.join(".").as_str());

        // Check for duplicate field definition
        if self.fields.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("field.{}", id.0)));
        }

        let stratum = def
            .strata
            .as_ref()
            .map(|s| StratumId::from(s.node.join(".").as_str()))
            .unwrap_or_else(|| StratumId::from("default"));

        // Validate stratum exists
        self.validate_stratum(&stratum)?;

        let mut reads = Vec::new();
        if let Some(measure) = &def.measure {
            self.collect_signal_refs(&measure.body.node, &mut reads);
        }

        let field = CompiledField {
            id: id.clone(),
            stratum,
            title: def.title.as_ref().map(|s| s.node.clone()),
            topology: def
                .topology
                .as_ref()
                .map(|t| self.lower_topology(&t.node))
                .unwrap_or(TopologyIr::SphereSurface),
            value_type: def
                .ty
                .as_ref()
                .map(|t| self.lower_type_expr(&t.node))
                .unwrap_or(ValueType::Scalar { range: None }),
            reads,
            measure: def.measure.as_ref().map(|m| self.lower_expr(&m.body.node)),
        };

        self.fields.insert(id, field);
        Ok(())
    }
}
