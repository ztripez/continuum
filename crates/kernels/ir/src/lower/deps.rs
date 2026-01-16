//! Dependency collection for lowering.
//!
//! This module extracts signal references from expressions,
//! which is used to build the execution dependency graph.
//!
//! Uses the ExprVisitor pattern from continuum_dsl for tree traversal.
//!
//! # Signal Path Resolution
//!
//! The parser greedily consumes dot-separated paths after `signal.`, which means
//! `signal.atmosphere.surface_temp.x` is parsed as `SignalRef("atmosphere.surface_temp.x")`.
//! However, the actual signal is `atmosphere.surface_temp` and `.x` is a field access on
//! the Vec3 value.
//!
//! To handle this, we try progressively shorter prefixes of the path until we find a
//! matching signal or member. For example:
//! - Try `atmosphere.surface_temp.x` - not a signal
//! - Try `atmosphere.surface_temp` - found! This is the dependency.

use std::collections::HashSet;

use continuum_dsl::ast::{Expr, ExprVisitor, Path};
use continuum_foundation::SignalId;

use super::Lowerer;

/// Visitor that collects signal references from expressions.
struct SignalRefCollector<'a> {
    refs: Vec<SignalId>,
    constants: &'a HashSet<String>,
    config: &'a HashSet<String>,
    /// Known signal names for path resolution
    signals: &'a HashSet<String>,
    /// Known member names for path resolution
    members: &'a HashSet<String>,
    /// Let-bound variable names (to exclude from signal refs)
    locals: HashSet<String>,
}

impl<'a> SignalRefCollector<'a> {
    fn new(
        constants: &'a HashSet<String>,
        config: &'a HashSet<String>,
        signals: &'a HashSet<String>,
        members: &'a HashSet<String>,
    ) -> Self {
        Self {
            refs: Vec::new(),
            constants,
            config,
            signals,
            members,
            locals: HashSet::new(),
        }
    }

    fn add_if_new(&mut self, id: SignalId) {
        if !self.refs.contains(&id) {
            self.refs.push(id);
        }
    }

    /// Resolve a signal path, handling field access suffixes.
    ///
    /// The parser may include field access (like `.x` on a Vec3) as part of the path.
    /// We try progressively shorter prefixes until we find a matching signal or member.
    fn resolve_signal_path(&mut self, path: &Path) {
        let segments = &path.segments;

        // Try progressively shorter prefixes until we find a match
        for len in (1..=segments.len()).rev() {
            let prefix: Vec<String> = segments[..len].to_vec();
            let prefix_str = prefix.join(".");

            // Check if this prefix is a known signal or member
            if self.signals.contains(&prefix_str) || self.members.contains(&prefix_str) {
                let id = SignalId::from(Path::new(prefix));
                self.add_if_new(id);
                return;
            }
        }

        // No matching signal/member found - add the full path anyway
        // (this preserves the existing behavior for forward references or errors)
        let id = SignalId::from(path.clone());
        self.add_if_new(id);
    }
}

impl ExprVisitor for SignalRefCollector<'_> {
    fn visit_signal_ref(&mut self, path: &Path) -> bool {
        self.resolve_signal_path(path);
        true
    }

    fn visit_path(&mut self, path: &Path) -> bool {
        let joined = path.to_string();
        // Only treat as signal ref if not a constant, config, or local variable
        if !self.constants.contains(&joined)
            && !self.config.contains(&joined)
            && !self.locals.contains(&joined)
        {
            self.resolve_signal_path(path);
        }
        true
    }

    fn visit_let(&mut self, name: &str) -> bool {
        // Track let-bound variable to exclude from signal refs
        self.locals.insert(name.to_string());
        true
    }
}

impl Lowerer {
    /// Collect signal references from an expression using visitor pattern.
    pub(crate) fn collect_signal_refs(&self, expr: &Expr, refs: &mut Vec<SignalId>) {
        // Build sets for constant/config lookup
        let constants: HashSet<String> = self.constants.keys().cloned().collect();
        let config: HashSet<String> = self.config.keys().cloned().collect();

        // Use pre-collected signal/member names for path resolution
        // (these are populated before lowering to handle forward references)
        let mut collector = SignalRefCollector::new(
            &constants,
            &config,
            &self.known_signal_names,
            &self.known_member_names,
        );
        collector.walk(expr);
        refs.extend(collector.refs);
    }
}
