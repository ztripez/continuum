//! Dependency collection for lowering.
//!
//! This module extracts signal and entity references from expressions,
//! which is used to build the execution dependency graph.
//!
//! Uses the ExprVisitor pattern from continuum_dsl for tree traversal,
//! with specialized visitors for each reference type.

use std::collections::HashSet;

use continuum_dsl::ast::{Expr, ExprVisitor, Path};
use continuum_foundation::{EntityId, SignalId};

use super::Lowerer;

/// Visitor that collects signal references from expressions.
struct SignalRefCollector<'a> {
    refs: Vec<SignalId>,
    constants: &'a HashSet<String>,
    config: &'a HashSet<String>,
}

impl<'a> SignalRefCollector<'a> {
    fn new(constants: &'a HashSet<String>, config: &'a HashSet<String>) -> Self {
        Self {
            refs: Vec::new(),
            constants,
            config,
        }
    }

    fn add_if_new(&mut self, id: SignalId) {
        if !self.refs.contains(&id) {
            self.refs.push(id);
        }
    }
}

impl ExprVisitor for SignalRefCollector<'_> {
    fn visit_signal_ref(&mut self, path: &Path) -> bool {
        let id = SignalId::from(path.join(".").as_str());
        self.add_if_new(id);
        true
    }

    fn visit_path(&mut self, path: &Path) -> bool {
        let joined = path.join(".");
        // Only treat as signal ref if not a constant or config
        if !self.constants.contains(&joined) && !self.config.contains(&joined) {
            let id = SignalId::from(joined.as_str());
            self.add_if_new(id);
        }
        true
    }
}

/// Visitor that collects entity references from expressions.
struct EntityRefCollector {
    refs: Vec<EntityId>,
}

impl EntityRefCollector {
    fn new() -> Self {
        Self { refs: Vec::new() }
    }

    fn add_if_new(&mut self, id: EntityId) {
        if !self.refs.contains(&id) {
            self.refs.push(id);
        }
    }

    fn add_entity_path(&mut self, path: &Path) {
        let id = EntityId::from(path.join(".").as_str());
        self.add_if_new(id);
    }
}

impl ExprVisitor for EntityRefCollector {
    fn visit_entity_ref(&mut self, path: &Path) -> bool {
        self.add_entity_path(path);
        true
    }

    fn visit_other(&mut self, path: &Path) -> bool {
        self.add_entity_path(path);
        true
    }

    fn visit_pairs(&mut self, path: &Path) -> bool {
        self.add_entity_path(path);
        true
    }

    fn visit_entity_access(&mut self, entity: &Path) -> bool {
        self.add_entity_path(entity);
        true // Continue walking to visit instance
    }

    fn visit_aggregate(&mut self, _op: &continuum_dsl::ast::AggregateOp, entity: &Path) -> bool {
        self.add_entity_path(entity);
        true // Continue walking to visit body
    }

    fn visit_filter(&mut self, entity: &Path) -> bool {
        self.add_entity_path(entity);
        true // Continue walking to visit predicate
    }

    fn visit_first(&mut self, entity: &Path) -> bool {
        self.add_entity_path(entity);
        true // Continue walking to visit predicate
    }

    fn visit_nearest(&mut self, entity: &Path) -> bool {
        self.add_entity_path(entity);
        true // Continue walking to visit position
    }

    fn visit_within(&mut self, entity: &Path) -> bool {
        self.add_entity_path(entity);
        true // Continue walking to visit position and radius
    }
}

impl Lowerer {
    /// Collect signal references from an expression using visitor pattern.
    pub(crate) fn collect_signal_refs(&self, expr: &Expr, refs: &mut Vec<SignalId>) {
        // Build sets for constant/config lookup
        let constants: HashSet<String> = self.constants.keys().cloned().collect();
        let config: HashSet<String> = self.config.keys().cloned().collect();

        let mut collector = SignalRefCollector::new(&constants, &config);
        collector.walk(expr);
        refs.extend(collector.refs);
    }

    /// Collect entity references from an expression using visitor pattern.
    pub(crate) fn collect_entity_refs(&self, expr: &Expr, refs: &mut Vec<EntityId>) {
        let mut collector = EntityRefCollector::new();
        collector.walk(expr);
        refs.extend(collector.refs);
    }
}
