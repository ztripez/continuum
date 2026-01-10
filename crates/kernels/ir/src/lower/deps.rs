//! Dependency collection for lowering.
//!
//! This module extracts signal references from expressions,
//! which is used to build the execution dependency graph.
//!
//! Uses the ExprVisitor pattern from continuum_dsl for tree traversal.

use std::collections::HashSet;

use continuum_dsl::ast::{Expr, ExprVisitor, Path};
use continuum_foundation::SignalId;

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
}
