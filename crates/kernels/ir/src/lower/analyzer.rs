//! Analyzer lowering.
//!
//! This module handles lowering analyzer definitions from AST to IR.
//! Analyzers are observer-only analysis queries that do not affect causality.

use continuum_dsl::ast::{self, Severity, Span};
use continuum_foundation::FieldId;

use crate::{
    CompiledAnalyzer, CompiledValidation, OutputField, OutputSchema, ValidationSeverity, ValueType,
};

use super::{LowerError, Lowerer};

impl Lowerer {
    /// Lower an analyzer definition from AST to IR.
    ///
    /// Analyzers read field data and produce structured analysis results.
    /// They cannot affect causality - removing all analyzers must not change simulation results.
    pub(crate) fn lower_analyzer(
        &mut self,
        def: &ast::AnalyzerDef,
        span: Span,
    ) -> Result<(), LowerError> {
        let id = continuum_foundation::AnalyzerId::from(def.path.node.clone());

        // Check for duplicate analyzer definition
        if self.analyzers.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition {
                name: format!("analyzer.{}", id),
                file: self.file.clone(),
                span: def.path.span.clone(),
            });
        }

        // Convert required fields from Path to FieldId
        let required_fields: Vec<FieldId> = def
            .required_fields
            .iter()
            .map(|field| FieldId::from(field.node.clone()))
            .collect();

        // Lower the compute expression
        let compute = match &def.compute {
            Some(expr) => self.lower_expr(&expr.node),
            None => {
                return Err(LowerError::Generic {
                    message: "Analyzer must have a compute block".to_string(),
                    file: self.file.clone(),
                    span: def.path.span.clone(),
                });
            }
        };

        // For now, infer output schema from a placeholder
        // In a full implementation, this would analyze the structure of the compute result
        let output_schema = OutputSchema {
            fields: vec![OutputField {
                name: "value".to_string(),
                value_type: ValueType::scalar(None, None, None),
                nested: None,
            }],
        };

        // Lower validation checks
        let mut validations = Vec::new();
        for check in &def.validations {
            let condition = self.lower_expr(&check.condition.node);
            let severity = match check.severity {
                Severity::Error => ValidationSeverity::Error,
                Severity::Warning => ValidationSeverity::Warning,
                Severity::Info => ValidationSeverity::Info,
            };
            let message = check.message.as_ref().map(|msg| msg.node.clone());

            validations.push(CompiledValidation {
                condition,
                severity,
                message,
            });
        }

        let analyzer = CompiledAnalyzer {
            file: self.file.clone(),
            span,
            id: id.clone(),
            doc: def.doc.clone(),
            required_fields,
            compute,
            output_schema,
            validations,
        };

        self.analyzers.insert(id, analyzer);
        Ok(())
    }
}
