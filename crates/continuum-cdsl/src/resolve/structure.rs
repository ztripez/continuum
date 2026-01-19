//! Structure validation for CDSL AST.
//!
//! Validates structural invariants that must hold across the entire AST:
//!
//! 1. **Cycle detection** - No circular dependencies between nodes
//! 2. **Collision detection** - No namespace conflicts in paths
//!
//! # What This Pass Does
//!
//! ## Cycle Detection
//!
//! Detects circular dependencies that would cause infinite loops during execution:
//! - Signal A depends on Signal B which depends on Signal A
//! - Field X reads from Field Y which reads from Field X
//! - Member dependencies across entity boundaries
//!
//! Uses depth-first search to detect cycles in the dependency graph built from
//! `reads` fields on Node<I>.
//!
//! ## Collision Detection
//!
//! Detects name conflicts that create ambiguity:
//! - `signal.x` collides with `signal` field named `x`
//! - Struct field names colliding with member access paths
//! - Parent path shadowing child declarations
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Res → Type Res → Type Validation → Structure Validation → Compilation
//!                                                                  ^^^^^^^^^^^^^^^
//!                                                                   YOU ARE HERE
//! ```
//!
//! This pass runs after all type and capability validation passes complete.
//! It validates the entire AST structure, not individual expressions.
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::resolve::structure::{validate_cycles, validate_collisions};
//! use continuum_cdsl::ast::Node;
//!
//! let nodes: Vec<Node<()>> = parse_world(source);
//!
//! // Check for cycles
//! let cycle_errors = validate_cycles(&nodes);
//! if !cycle_errors.is_empty() {
//!     println!("Circular dependencies detected!");
//! }
//!
//! // Check for collisions
//! let collision_errors = validate_collisions(&nodes);
//! if !collision_errors.is_empty() {
//!     println!("Name collisions detected!");
//! }
//! ```

use crate::ast::Node;
use crate::error::{CompileError, ErrorKind};
use crate::foundation::Path;
use std::collections::{HashMap, HashSet};

/// Validates that the AST contains no circular dependencies.
///
/// Builds a dependency graph from node `reads` fields and performs depth-first
/// search to detect cycles. Each node's dependencies come from expressions that
/// reference other signals/fields via paths.
///
/// # Errors
///
/// Returns [`ErrorKind::CyclicDependency`] for each detected cycle, with a
/// message describing the cycle path.
///
/// # Examples
///
/// ```rust,ignore
/// // signal a { resolve { Prev(b) } }
/// // signal b { resolve { Prev(a) } }
/// // → Cycle detected: a → b → a
///
/// let errors = validate_cycles(&nodes);
/// assert!(!errors.is_empty());
/// ```
pub fn validate_cycles<I: crate::ast::Index>(nodes: &[Node<I>]) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Build dependency graph: path → list of paths it depends on
    let mut graph: HashMap<Path, Vec<Path>> = HashMap::new();
    for node in nodes {
        graph.insert(node.path.clone(), node.reads.clone());
    }

    // Track visited nodes during DFS
    let mut visited = HashSet::new();
    let mut rec_stack = HashSet::new();

    // Perform DFS from each node
    for node in nodes {
        if !visited.contains(&node.path) {
            if let Some(cycle) =
                detect_cycle_dfs(&node.path, &graph, &mut visited, &mut rec_stack, Vec::new())
            {
                errors.push(CompileError::new(
                    ErrorKind::CyclicDependency,
                    node.span,
                    format_cycle_error(&cycle),
                ));
            }
        }
    }

    errors
}

/// Depth-first search to detect cycles in dependency graph.
///
/// Returns Some(cycle_path) if a cycle is detected, None otherwise.
fn detect_cycle_dfs(
    current: &Path,
    graph: &HashMap<Path, Vec<Path>>,
    visited: &mut HashSet<Path>,
    rec_stack: &mut HashSet<Path>,
    mut path: Vec<Path>,
) -> Option<Vec<Path>> {
    visited.insert(current.clone());
    rec_stack.insert(current.clone());
    path.push(current.clone());

    if let Some(deps) = graph.get(current) {
        for dep in deps {
            if !visited.contains(dep) {
                if let Some(cycle) = detect_cycle_dfs(dep, graph, visited, rec_stack, path.clone())
                {
                    return Some(cycle);
                }
            } else if rec_stack.contains(dep) {
                // Found cycle - construct cycle path
                let cycle_start = path.iter().position(|p| p == dep).unwrap();
                let mut cycle = path[cycle_start..].to_vec();
                cycle.push(dep.clone()); // Close the cycle
                return Some(cycle);
            }
        }
    }

    rec_stack.remove(current);
    None
}

/// Formats a cycle error message showing the dependency chain.
fn format_cycle_error(cycle: &[Path]) -> String {
    let cycle_str = cycle
        .iter()
        .map(|p| p.to_string())
        .collect::<Vec<_>>()
        .join(" → ");
    format!("Circular dependency detected: {}", cycle_str)
}

/// Validates that the AST contains no path collisions.
///
/// Detects namespace conflicts where paths overlap in ways that create ambiguity:
/// - A node at path `foo.bar` collides with a node at path `foo` with field `bar`
/// - Parent paths shadowing child declarations
///
/// # Errors
///
/// Returns [`ErrorKind::PathCollision`] for each detected collision, with a
/// message describing the conflicting paths.
///
/// # Examples
///
/// ```rust,ignore
/// // signal velocity { ... }  // path: "velocity"
/// // field velocity { ... }   // path: "velocity"
/// // → Collision: both declare "velocity"
///
/// let errors = validate_collisions(&nodes);
/// assert!(!errors.is_empty());
/// ```
pub fn validate_collisions<I: crate::ast::Index>(nodes: &[Node<I>]) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Build path index: path → (node_index, span)
    let mut path_index: HashMap<Path, (usize, crate::foundation::Span)> = HashMap::new();

    for (idx, node) in nodes.iter().enumerate() {
        if let Some((_other_idx, other_span)) = path_index.get(&node.path) {
            // Direct collision - same path declared twice
            errors.push(
                CompileError::new(
                    ErrorKind::PathCollision,
                    node.span,
                    format!("Path '{}' is declared multiple times", node.path),
                )
                .with_label(*other_span, "first declared here".to_string()),
            );
        } else {
            path_index.insert(node.path.clone(), (idx, node.span));
        }

        // Check for parent/child collisions
        // If we have "foo.bar.baz", check if "foo" or "foo.bar" exist as separate declarations
        check_parent_collisions(&node.path, &path_index, node.span, &mut errors);
    }

    errors
}

/// Checks if a path collides with any of its parent paths.
fn check_parent_collisions(
    path: &Path,
    path_index: &HashMap<Path, (usize, crate::foundation::Span)>,
    span: crate::foundation::Span,
    errors: &mut Vec<CompileError>,
) {
    // For path "a.b.c", check if "a" or "a.b" exist as separate nodes
    let segments = path.segments();
    for i in 1..segments.len() {
        let parent = Path::new(segments[..i].iter().map(|s| s.to_string()).collect());
        if let Some((_, parent_span)) = path_index.get(&parent) {
            errors.push(
                CompileError::new(
                    ErrorKind::PathCollision,
                    span,
                    format!("Path '{}' collides with parent path '{}'", path, parent),
                )
                .with_label(*parent_span, "parent path declared here".to_string()),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::RoleData;
    use crate::foundation::Span;

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    fn make_signal(path: &str, reads: Vec<&str>) -> Node<()> {
        let mut node = Node::new(Path::from_str(path), test_span(), RoleData::Signal, ());
        node.reads = reads.iter().map(|s| Path::from_str(s)).collect();
        node
    }

    #[test]
    fn test_no_cycles_in_acyclic_graph() {
        // a → b → c (no cycle)
        let nodes = vec![
            make_signal("a", vec!["b"]),
            make_signal("b", vec!["c"]),
            make_signal("c", vec![]),
        ];

        let errors = validate_cycles(&nodes);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_detects_simple_cycle() {
        // a → b → a
        let nodes = vec![make_signal("a", vec!["b"]), make_signal("b", vec!["a"])];

        let errors = validate_cycles(&nodes);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::CyclicDependency);
        assert!(errors[0].message.contains("a"));
        assert!(errors[0].message.contains("b"));
    }

    #[test]
    fn test_detects_self_cycle() {
        // a → a
        let nodes = vec![make_signal("a", vec!["a"])];

        let errors = validate_cycles(&nodes);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::CyclicDependency);
    }

    #[test]
    fn test_detects_longer_cycle() {
        // a → b → c → a
        let nodes = vec![
            make_signal("a", vec!["b"]),
            make_signal("b", vec!["c"]),
            make_signal("c", vec!["a"]),
        ];

        let errors = validate_cycles(&nodes);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::CyclicDependency);
    }

    #[test]
    fn test_no_collision_for_unique_paths() {
        let nodes = vec![
            make_signal("a", vec![]),
            make_signal("b", vec![]),
            make_signal("c.d", vec![]),
        ];

        let errors = validate_collisions(&nodes);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_detects_duplicate_path() {
        let nodes = vec![make_signal("a", vec![]), make_signal("a", vec![])];

        let errors = validate_collisions(&nodes);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::PathCollision);
        assert!(errors[0].message.contains("declared multiple times"));
    }

    #[test]
    fn test_detects_parent_child_collision() {
        // "a" and "a.b" should collide
        let nodes = vec![make_signal("a", vec![]), make_signal("a.b", vec![])];

        let errors = validate_collisions(&nodes);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::PathCollision);
        assert!(errors[0].message.contains("collides with parent"));
    }
}
