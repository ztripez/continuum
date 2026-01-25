//! Validation pass for detecting orphaned strata.
//!
//! A stratum is orphaned if it is declared but never used:
//! 1. Not referenced in any era's strata list
//! 2. Not assigned to any node (signal, field, operator, etc.)

use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::StratumId;
use continuum_cdsl_ast::{Era, Node, Stratum};
use indexmap::IndexMap;
use std::collections::HashSet;

/// Validates that all declared strata are used.
///
/// Emits warnings for strata that are:
/// - Not referenced by any era
/// - Not assigned to any node
///
/// # Parameters
/// - `strata`: All declared strata
/// - `eras`: All resolved eras
/// - `global_nodes`: All global nodes (signals, fields, operators, etc.)
/// - `member_nodes`: All per-entity member nodes
///
/// # Returns
/// Vec of warnings for orphaned strata (does not fail compilation)
pub fn validate_orphaned_strata<I: continuum_cdsl_ast::Index>(
    strata: &IndexMap<continuum_cdsl_ast::foundation::Path, Stratum>,
    eras: &IndexMap<continuum_cdsl_ast::foundation::Path, Era>,
    global_nodes: &[Node<()>],
    member_nodes: &[Node<I>],
) -> Vec<CompileError> {
    let mut used_strata = HashSet::new();

    // Collect strata referenced in eras
    for era in eras.values() {
        for policy in &era.strata_policy {
            used_strata.insert(policy.stratum.clone());
        }
    }

    // Collect strata assigned to nodes
    for node in global_nodes {
        if let Some(stratum_id) = &node.stratum {
            used_strata.insert(stratum_id.clone());
        }
    }
    for node in member_nodes {
        if let Some(stratum_id) = &node.stratum {
            used_strata.insert(stratum_id.clone());
        }
    }

    // Find orphaned strata
    let mut warnings = Vec::new();
    for stratum in strata.values() {
        if !used_strata.contains(&stratum.id) {
            warnings.push(CompileError::warning(
                ErrorKind::Internal,
                stratum.span,
                format!(
                    "stratum '{}' is declared but never used (not in any era, not assigned to any node)",
                    stratum.path
                ),
            ));
        }
    }

    warnings
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl_ast::foundation::{Path, Span, StratumId};
    use continuum_cdsl_ast::{Node, RoleData, Stratum};

    /// Helper to create a test stratum
    fn make_stratum(name: &str) -> Stratum {
        Stratum {
            path: Path::from_path_str(name),
            id: StratumId::new(name),
            span: Span::zero(0),
            attributes: vec![],
            cadence: Some(1),
            doc: None,
        }
    }

    #[test]
    fn test_orphaned_stratum() {
        let mut strata = IndexMap::new();
        let stratum = make_stratum("unused");
        strata.insert(stratum.path.clone(), stratum);

        let eras = IndexMap::new();

        let empty_members: &[Node<continuum_cdsl_ast::foundation::EntityId>] = &[];
        let warnings = validate_orphaned_strata(&strata, &eras, &[], empty_members);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0]
            .message
            .contains("stratum 'unused' is declared but never used"));
    }

    #[test]
    fn test_stratum_used_by_node() {
        let mut strata = IndexMap::new();
        let stratum = make_stratum("main");
        strata.insert(stratum.path.clone(), stratum);

        let eras = IndexMap::new();

        let mut node = Node::new(
            Path::from_path_str("test.signal"),
            Span::zero(0),
            RoleData::Signal,
            (),
        );
        node.stratum = Some(StratumId::new("main"));

        let empty_members: &[Node<continuum_cdsl_ast::foundation::EntityId>] = &[];
        let warnings = validate_orphaned_strata(&strata, &eras, &[node], empty_members);
        assert!(warnings.is_empty());
    }
}
