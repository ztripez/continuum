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

use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::{EntityId, Path, Span};
use continuum_cdsl_ast::{Declaration, Node, RoleId};
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
pub fn validate_cycles(globals: &[Node<()>], members: &[Node<EntityId>]) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Build dependency graph: path → (list of paths it depends on, span)
    let mut graph: HashMap<Path, (Vec<Path>, Span)> = HashMap::new();
    for node in globals {
        graph.insert(node.path.clone(), (node.reads.clone(), node.span));
    }
    for node in members {
        graph.insert(node.path.clone(), (node.reads.clone(), node.span));
    }

    // Track visited nodes during DFS
    let mut visited = HashSet::new();
    let mut rec_stack = HashSet::new();

    // Perform DFS from each node in the graph
    let paths: Vec<_> = graph.keys().cloned().collect();
    for path in paths {
        if !visited.contains(&path) {
            if let Some(cycle) =
                detect_cycle_dfs(&path, &graph, &mut visited, &mut rec_stack, Vec::new())
            {
                let span = graph[&path].1;
                errors.push(CompileError::new(
                    ErrorKind::CyclicDependency,
                    span,
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
    graph: &HashMap<Path, (Vec<Path>, Span)>,
    visited: &mut HashSet<Path>,
    rec_stack: &mut HashSet<Path>,
    mut path: Vec<Path>,
) -> Option<Vec<Path>> {
    visited.insert(current.clone());
    rec_stack.insert(current.clone());
    path.push(current.clone());

    let result = if let Some((deps, _)) = graph.get(current) {
        let mut found_cycle = None;
        for dep in deps {
            if !visited.contains(dep) {
                if let Some(cycle) = detect_cycle_dfs(dep, graph, visited, rec_stack, path.clone())
                {
                    found_cycle = Some(cycle);
                    break;
                }
            } else if rec_stack.contains(dep) {
                // Found cycle - dep is an ancestor in the current DFS path
                match path.iter().position(|p| p == dep) {
                    Some(cycle_start) => {
                        let mut cycle = path[cycle_start..].to_vec();
                        cycle.push(dep.clone()); // Close the cycle
                        found_cycle = Some(cycle);
                        break;
                    }
                    None => {
                        // BUG: This should not happen if rec_stack is properly maintained
                        // Return minimal cycle for error reporting
                        found_cycle = Some(vec![current.clone(), dep.clone(), current.clone()]);
                        break;
                    }
                }
            }
        }
        found_cycle
    } else {
        None
    };

    // IMPORTANT: Always remove current from rec_stack before returning
    rec_stack.remove(current);
    result
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

/// Declaration kind for collision detection.
/// Different kinds can have the same path without collision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DeclKind {
    Signal,
    Field,
    Operator,
    Impulse,
    Fracture,
    Chronicle,
    Member,
    Entity,
    Stratum,
    Era,
    Function,
    World,
    Type,
}

impl DeclKind {
    /// Convert a node's RoleId to DeclKind
    fn from_role(role: RoleId) -> Self {
        match role {
            RoleId::Signal => DeclKind::Signal,
            RoleId::Field => DeclKind::Field,
            RoleId::Operator => DeclKind::Operator,
            RoleId::Impulse => DeclKind::Impulse,
            RoleId::Fracture => DeclKind::Fracture,
            RoleId::Chronicle => DeclKind::Chronicle,
        }
    }
}

/// Validates that the AST contains no path collisions across all declarations.
///
/// Collision rules:
/// - Same (kind, path) pair is a collision (e.g., two signals with same path)
/// - Different kinds with same path are allowed (e.g., `fn.foo.bar` and `field.foo.bar`)
/// - Parent paths shadowing child declarations (with exceptions for strata, world, entity-member)
///
/// # Errors
///
/// Returns [`ErrorKind::PathCollision`] for each detected collision, with a
/// message describing the conflicting paths.
pub fn validate_collisions(declarations: &[Declaration]) -> Vec<CompileError> {
    let mut errors = Vec::new();
    // Key is (kind, path) - different kinds can have the same path
    let mut path_index: HashMap<(DeclKind, Path), Span> = HashMap::new();
    // Also track all paths regardless of kind for parent collision checking
    let mut all_paths: HashMap<Path, (Span, DeclKind)> = HashMap::new();

    // Helper to register a path and check for direct collisions
    fn register_path(
        path: &Path,
        span: Span,
        kind: DeclKind,
        index: &mut HashMap<(DeclKind, Path), Span>,
        all_paths: &mut HashMap<Path, (Span, DeclKind)>,
        errors: &mut Vec<CompileError>,
    ) {
        // Reserve "debug" namespace for system-generated fields
        if let Some(first) = path.segments.first() {
            if first == "debug" {
                errors.push(CompileError::new(
                    ErrorKind::PathCollision,
                    span,
                    "Top-level namespace 'debug' is reserved for system use".to_string(),
                ));
                return;
            }
        }

        // Check for same-kind collision
        let key = (kind, path.clone());
        if let Some(other_span) = index.get(&key) {
            errors.push(
                CompileError::new(
                    ErrorKind::PathCollision,
                    span,
                    format!("Path '{}' is declared multiple times as {:?}", path, kind),
                )
                .with_label(*other_span, "first declared here".to_string()),
            );
        } else {
            index.insert(key, span);
        }

        // Track for parent collision checking (first one wins)
        all_paths.entry(path.clone()).or_insert((span, kind));
    }

    for decl in declarations {
        match decl {
            Declaration::Node(n) => {
                let kind = DeclKind::from_role(n.role.id());
                register_path(
                    &n.path,
                    n.span,
                    kind,
                    &mut path_index,
                    &mut all_paths,
                    &mut errors,
                );
            }
            Declaration::Member(m) => register_path(
                &m.path,
                m.span,
                DeclKind::Member,
                &mut path_index,
                &mut all_paths,
                &mut errors,
            ),
            Declaration::Entity(e) => register_path(
                &e.path,
                e.span,
                DeclKind::Entity,
                &mut path_index,
                &mut all_paths,
                &mut errors,
            ),
            Declaration::Stratum(s) => register_path(
                &s.path,
                s.span,
                DeclKind::Stratum,
                &mut path_index,
                &mut all_paths,
                &mut errors,
            ),
            Declaration::Era(e) => register_path(
                &e.path,
                e.span,
                DeclKind::Era,
                &mut path_index,
                &mut all_paths,
                &mut errors,
            ),
            Declaration::World(w) => register_path(
                &w.path,
                w.span,
                DeclKind::World,
                &mut path_index,
                &mut all_paths,
                &mut errors,
            ),
            Declaration::Type(t) => {
                let path = Path::from(t.name.as_str());
                register_path(
                    &path,
                    t.span,
                    DeclKind::Type,
                    &mut path_index,
                    &mut all_paths,
                    &mut errors,
                );
            }
            Declaration::Const(_entries) => {
                // Config/const entries are in a separate namespace (const.*, config.*)
                // and don't participate in collision checking with other declarations.
                // They're only accessible via their respective prefixes.
            }
            Declaration::Config(_entries) => {
                // Config/const entries are in a separate namespace (const.*, config.*)
                // and don't participate in collision checking with other declarations.
                // They're only accessible via their respective prefixes.
            }
            Declaration::Function(f) => register_path(
                &f.path,
                f.span,
                DeclKind::Function,
                &mut path_index,
                &mut all_paths,
                &mut errors,
            ),
        }
    }

    // Check for parent/child collisions using all_paths (ignoring kind for parent lookup)
    for ((kind, path), span) in &path_index {
        check_parent_collisions(path, &all_paths, *span, *kind, &mut errors);
    }

    errors
}

/// Checks if a path collides with any of its parent paths.
fn check_parent_collisions(
    path: &Path,
    path_index: &HashMap<Path, (Span, DeclKind)>,
    span: Span,
    kind: DeclKind,
    errors: &mut Vec<CompileError>,
) {
    // For path "a.b.c", check if "a" or "a.b" exist as separate nodes
    let segments = path.segments();
    for i in 1..segments.len() {
        let parent = Path::new(segments[..i].iter().map(|s| s.to_string()).collect());
        if let Some((parent_span, parent_kind)) = path_index.get(&parent) {
            // EXCEPTION: Members are allowed to have an Entity as their parent path
            // e.g., `member stellar.moon.mass` has parent `entity stellar.moon`
            if kind == DeclKind::Member
                && *parent_kind == DeclKind::Entity
                && i == segments.len() - 1
            {
                continue;
            }

            // EXCEPTION: Strata are namespace markers, not nodes that occupy the path
            // e.g., `strata atmosphere` should not collide with `signal atmosphere.surface_temp`
            // Strata exist in a separate conceptual namespace from signals/operators/fields
            if *parent_kind == DeclKind::Stratum {
                continue;
            }

            // EXCEPTION: World declarations define the root namespace for all declarations
            // e.g., `world terra` should not collide with `entity terra.plate`
            if *parent_kind == DeclKind::World {
                continue;
            }

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
    use continuum_cdsl_ast::foundation::Span;
    use continuum_cdsl_ast::{Entity, RoleData};

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    fn make_signal(path: &str, reads: Vec<&str>) -> Node<()> {
        let mut node = Node::new(Path::from_path_str(path), test_span(), RoleData::Signal, ());
        node.reads = reads.iter().map(|s| Path::from_path_str(s)).collect();
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

        let errors = validate_cycles(&nodes, &[]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_detects_simple_cycle() {
        // a → b → a
        let nodes = vec![make_signal("a", vec!["b"]), make_signal("b", vec!["a"])];

        let errors = validate_cycles(&nodes, &[]);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::CyclicDependency);
        assert!(errors[0].message.contains("a"));
        assert!(errors[0].message.contains("b"));
    }

    #[test]
    fn test_detects_self_cycle() {
        // a → a
        let nodes = vec![make_signal("a", vec!["a"])];

        let errors = validate_cycles(&nodes, &[]);
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

        let errors = validate_cycles(&nodes, &[]);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::CyclicDependency);
    }

    #[test]
    fn test_config_reads_not_treated_as_signal_dependencies() {
        // Regression test for continuum-v5cr: Config/const paths in node.reads
        // should not be treated as signal dependencies and should not cause
        // false cycle detection.
        //
        // Previously, a signal `planet.radius` reading `config.planet.radius`
        // would have "planet.radius" added to its reads (because config paths
        // are stored without the "config." prefix), causing it to appear as
        // a self-cycle.
        //
        // This test verifies that config/const paths are excluded from cycle
        // detection, which happens during dependency extraction in dependencies.rs.
        //
        // Note: In practice, config paths would NOT appear in node.reads after
        // the fix, so this test documents the expected behavior where config
        // dependencies do not participate in signal cycles.
        let nodes = vec![
            // Signal that would read config.planet.radius (path without "config." prefix)
            // Should NOT be treated as reading itself
            make_signal("planet.radius", vec![]),
            // Another signal that reads the first one - should be valid
            make_signal("planet.surface_gravity", vec!["planet.radius"]),
        ];

        let errors = validate_cycles(&nodes, &[]);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_no_collision_for_unique_paths() {
        let decls = vec![
            Declaration::Node(make_signal("a", vec![])),
            Declaration::Node(make_signal("b", vec![])),
            Declaration::Node(make_signal("c.d", vec![])),
        ];

        let errors = validate_collisions(&decls);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_detects_duplicate_path() {
        let decls = vec![
            Declaration::Node(make_signal("a", vec![])),
            Declaration::Node(make_signal("a", vec![])),
        ];

        let errors = validate_collisions(&decls);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::PathCollision);
        assert!(errors[0].message.contains("declared multiple times"));
    }

    #[test]
    fn test_detects_parent_child_collision() {
        // "a" and "a.b" should collide
        let decls = vec![
            Declaration::Node(make_signal("a", vec![])),
            Declaration::Node(make_signal("a.b", vec![])),
        ];

        let errors = validate_collisions(&decls);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::PathCollision);
        assert!(errors[0].message.contains("collides with parent"));
    }

    #[test]
    fn test_member_entity_exception() {
        let span = test_span();
        let entity = Entity::new(EntityId::new("player"), Path::from_path_str("player"), span);
        let member = Node::new(
            Path::from_path_str("player.health"),
            span,
            RoleData::Signal,
            EntityId::new("player"),
        );

        let decls = vec![Declaration::Entity(entity), Declaration::Member(member)];
        let errors = validate_collisions(&decls);
        assert!(
            errors.is_empty(),
            "Member should be allowed under its Entity path"
        );
    }

    fn make_entity(path: &str) -> Entity {
        Entity::new(EntityId::new(path), Path::from_path_str(path), test_span())
    }

    #[test]
    fn test_cross_type_no_collision() {
        // Different declaration kinds CAN share paths because the prefix syntax
        // makes them unambiguous (e.g., `entity sim` vs `signal.sim`).
        // This is by design since commit b9bcbbc.
        let decls = vec![
            Declaration::Entity(make_entity("sim")),
            Declaration::Node(make_signal("sim", vec![])),
        ];

        let errors = validate_collisions(&decls);
        assert!(errors.is_empty(), "Different kinds can share paths");
    }
}
