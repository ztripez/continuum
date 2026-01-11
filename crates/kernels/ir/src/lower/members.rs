//! Member signal lowering.
//!
//! This module handles lowering member signal definitions from AST to IR.
//! Member signals are per-entity authoritative state with their own resolve blocks.

use continuum_dsl::ast;
use continuum_foundation::{EntityId, MemberId, StratumId};

use crate::{CompiledMember, ValueType};

use super::{LowerError, Lowerer};

impl Lowerer {
    /// Lower a member signal definition to CompiledMember.
    ///
    /// Member signals follow the path structure: `entity_path.signal_name`
    /// For example, `human.person.age` belongs to entity `human.person` with signal `age`.
    pub(crate) fn lower_member(&mut self, def: &ast::MemberDef) -> Result<(), LowerError> {
        let full_path = def.path.node.join(".");
        let id = MemberId::from(full_path.as_str());

        // Check for duplicate member definition
        if self.members.contains_key(&id) {
            return Err(LowerError::DuplicateDefinition(format!("member.{}", id.0)));
        }

        // Extract entity path and signal name from the full path
        // e.g., "human.person.age" -> entity "human.person", signal "age"
        let segments = &def.path.node.segments;
        if segments.len() < 2 {
            return Err(LowerError::InvalidExpression(format!(
                "member path '{}' must have at least entity.signal format",
                full_path
            )));
        }

        let entity_path = segments[..segments.len() - 1].join(".");
        let signal_name = segments.last().unwrap().clone();
        let entity_id = EntityId::from(entity_path.as_str());

        // Determine stratum
        let stratum = def
            .strata
            .as_ref()
            .map(|s| StratumId::from(s.node.join(".").as_str()))
            .unwrap_or_else(|| StratumId::from("default"));

        // Validate stratum exists
        self.validate_stratum(&stratum)?;

        // Process local config blocks - add to global config with member-prefixed keys
        for entry in &def.local_config {
            let local_key = entry.path.node.join(".");
            let full_key = format!("{}.{}", full_path, local_key);
            let value = self.literal_to_f64(&entry.value.node)?;
            self.config.insert(full_key, value);
        }

        // Collect signal dependencies from resolve and initial expressions
        let mut reads = Vec::new();
        if let Some(resolve) = &def.resolve {
            self.collect_signal_refs(&resolve.body.node, &mut reads);
        }
        if let Some(initial) = &def.initial {
            self.collect_signal_refs(&initial.body.node, &mut reads);
        }

        // Member reads (self.* references) are resolved at runtime within entity context
        // For now, we leave member_reads empty as the exact MemberIds depend on
        // which entity instance is being resolved
        let member_reads = Vec::new();

        // Detect dt_raw usage
        let uses_dt_raw = def
            .resolve
            .as_ref()
            .is_some_and(|r| self.expr_uses_dt_raw(&r.body.node));

        // Lower initial expression
        let initial = def.initial.as_ref().map(|i| self.lower_expr(&i.body.node));

        // Lower resolve expression
        let resolve = def.resolve.as_ref().map(|r| self.lower_expr(&r.body.node));

        // Lower assertions
        let assertions = def
            .assertions
            .as_ref()
            .map(|a| self.lower_assert_block(a))
            .unwrap_or_default();

        // Lower the type
        let value_type = def
            .ty
            .as_ref()
            .map(|t| self.lower_type_expr(&t.node))
            .unwrap_or(ValueType::Scalar {
                unit: None,
                dimension: None,
                range: None,
            });

        let member = CompiledMember {
            id: id.clone(),
            entity_id,
            signal_name,
            stratum,
            title: def.title.as_ref().map(|s| s.node.clone()),
            symbol: def.symbol.as_ref().map(|s| s.node.clone()),
            value_type,
            uses_dt_raw,
            reads,
            member_reads,
            initial,
            resolve,
            assertions,
        };

        self.members.insert(id, member);
        Ok(())
    }
}
