//! Entity member flattening and normalization.
//!
//! This module handles transformation of entity-nested member syntax into flat member
//! declarations. It runs early in the resolution pipeline (before desugaring) to ensure
//! member expressions are properly processed.
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Flatten Members → Desugar → Name Resolution → ...
//!         ^^^^^^^^^^^^^^^^
//!         entities.rs
//! ```
//!
//! # Key Operations
//!
//! - **Path Construction**: Combines entity.path + member.path into full member paths
//! - **Stratum Inheritance**: Members inherit entity's stratum if not explicitly specified
//! - **Declaration Flattening**: Nested members become standalone `Declaration::Member` entries

use crate::error::CompileError;
use crate::resolve::attributes::{extract_single_identifier, extract_single_path, has_attribute};
use continuum_cdsl_ast::foundation::Path;
use continuum_cdsl_ast::{Attribute, Declaration, Expr, UntypedKind};

/// Extracts nested members from entity declarations into standalone member declarations.
///
/// Transforms entity-nested member syntax into flat member declarations by:
/// 1. Extracting each member from `entity.members` vector
/// 2. Building full member path: `entity.path` + `member.path` (e.g., `["plate", "mass"]`)
/// 3. Inheriting entity's stratum attribute if member lacks explicit stratum
/// 4. Creating standalone `Declaration::Member` entries with full paths
///
/// This pass MUST run BEFORE desugaring to ensure member expressions are properly desugared.
///
/// # Stratum Inheritance
///
/// If entity has `:stratum(thermal)` and member lacks stratum attribute:
/// - Attribute `:stratum(thermal)` is injected into member's attribute list
/// - Member now inherits entity's execution lane policy
/// - Member's explicit stratum attribute always takes precedence (no override)
///
/// # Parameters
///
/// * `declarations` - Input declaration list containing entities with nested members
/// * `errors` - Error accumulator for compilation diagnostics (currently unused)
///
/// # Returns
///
/// Flattened declaration list where:
/// - Entities remain as-is (members field preserved for reference)
/// - Nested members become standalone `Declaration::Member` with full paths
/// - All other declarations pass through unchanged
///
/// # Examples
///
/// Input DSL:
/// ```cdsl
/// entity plate {
///     : stratum(thermal)
///     signal temp { resolve { 273.0 } }
/// }
/// ```
///
/// Output AST:
/// - `Declaration::Entity(path: "plate", members: [...])`
/// - `Declaration::Member(path: "plate.temp", stratum: "thermal", ...)`
///
/// # Pipeline Position
///
/// ```text
/// Parse → Flatten Members → Desugar → Name Resolution → ...
///         ^^^^^^^^^^^^^^^^
///         YOU ARE HERE
/// ```
pub(crate) fn flatten_entity_members(
    declarations: Vec<Declaration>,
    errors: &mut Vec<CompileError>,
) -> Vec<Declaration> {
    let mut flattened = Vec::new();

    for decl in declarations {
        if let Declaration::Entity(ref entity) = decl {
            // Keep the entity declaration
            flattened.push(decl.clone());

            // Extract stratum from entity attributes (if present)
            let entity_stratum =
                extract_single_path(&entity.attributes, "stratum", entity.span, errors);

            // Flatten each nested member
            for member in &entity.members {
                let mut flattened_member = member.clone();

                // Build full path: entity.path + member.path
                let mut full_segments = entity.path.segments.clone();
                full_segments.extend(member.path.segments.clone());
                flattened_member.path = Path::new(full_segments);

                // Entity stratum provides default: inherit if member doesn't have one
                // This allows entity-level convenience while permitting member-level specificity
                if !has_attribute(&flattened_member.attributes, "stratum") {
                    if let Some(ref stratum_path) = entity_stratum {
                        // Add entity's stratum attribute to member
                        let stratum_attr = Attribute {
                            name: "stratum".to_string(),
                            args: vec![Expr::new(
                                UntypedKind::Signal(stratum_path.clone()),
                                member.span,
                            )],
                            span: member.span,
                        };
                        flattened_member.attributes.push(stratum_attr);
                    }
                }

                // Add as member declaration
                flattened.push(Declaration::Member(flattened_member));
            }
        } else {
            // Keep non-entity declarations as-is
            flattened.push(decl);
        }
    }

    flattened
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl_ast::foundation::{EntityId, Span};
    use continuum_cdsl_ast::{Entity, Expr, Node, RoleData};

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    #[test]
    fn test_flatten_entity_members_basic() {
        let span = test_span();
        let mut entity = Entity::new(
            EntityId::new("particle"),
            Path::from_path_str("particle"),
            span,
        );

        let member = Node::new(
            Path::from_path_str("mass"),
            span,
            RoleData::Signal,
            EntityId::new("particle"),
        );
        entity.members.push(member);

        let declarations = vec![Declaration::Entity(entity)];
        let mut errors = Vec::new();
        let flattened = flatten_entity_members(declarations, &mut errors);

        assert_eq!(flattened.len(), 2);
        assert!(matches!(flattened[0], Declaration::Entity(_)));

        if let Declaration::Member(member) = &flattened[1] {
            assert_eq!(member.path, Path::from_path_str("particle.mass"));
        } else {
            panic!("Expected Declaration::Member");
        }

        assert!(errors.is_empty());
    }

    #[test]
    fn test_flatten_entity_members_stratum_inheritance() {
        let span = test_span();
        let mut entity = Entity::new(
            EntityId::new("particle"),
            Path::from_path_str("particle"),
            span,
        );

        entity.attributes.push(Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                continuum_cdsl_ast::UntypedKind::Signal(Path::from_path_str("fast")),
                span,
            )],
            span,
        });

        let member = Node::new(
            Path::from_path_str("mass"),
            span,
            RoleData::Signal,
            EntityId::new("particle"),
        );
        entity.members.push(member);

        let declarations = vec![Declaration::Entity(entity)];
        let mut errors = Vec::new();
        let flattened = flatten_entity_members(declarations, &mut errors);

        if let Declaration::Member(member) = &flattened[1] {
            assert!(has_attribute(&member.attributes, "stratum"));
            let mut test_errors = Vec::new();
            let stratum =
                extract_single_identifier(&member.attributes, "stratum", span, &mut test_errors);
            assert_eq!(stratum, Some("fast".to_string()));
            assert!(test_errors.is_empty());
        } else {
            panic!("Expected Declaration::Member");
        }
    }

    #[test]
    fn test_flatten_entity_members_member_stratum_overrides_entity() {
        // Test that member-level stratum takes precedence over entity-level (specificity model)
        let span = test_span();
        let mut entity = Entity::new(
            EntityId::new("particle"),
            Path::from_path_str("particle"),
            span,
        );

        entity.attributes.push(Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                continuum_cdsl_ast::UntypedKind::Signal(Path::from_path_str("fast")),
                span,
            )],
            span,
        });

        let mut member = Node::new(
            Path::from_path_str("mass"),
            span,
            RoleData::Signal,
            EntityId::new("particle"),
        );
        member.attributes.push(Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                continuum_cdsl_ast::UntypedKind::Signal(Path::from_path_str("slow")),
                span,
            )],
            span,
        });
        entity.members.push(member);

        let declarations = vec![Declaration::Entity(entity)];
        let mut errors = Vec::new();
        let flattened = flatten_entity_members(declarations, &mut errors);

        // Member stratum should be preserved (member wins over entity)
        if let Declaration::Member(member) = &flattened[1] {
            assert!(has_attribute(&member.attributes, "stratum"));
            let mut test_errors = Vec::new();
            let stratum =
                extract_single_path(&member.attributes, "stratum", span, &mut test_errors);
            assert_eq!(stratum, Some(Path::from_path_str("slow")));
            assert!(test_errors.is_empty());
        } else {
            panic!("Expected Declaration::Member");
        }
    }

    #[test]
    fn test_flatten_entity_members_inherits_entity_stratum() {
        // Test that member without stratum inherits from entity (default/convenience model)
        let span = test_span();
        let mut entity = Entity::new(
            EntityId::new("particle"),
            Path::from_path_str("particle"),
            span,
        );

        entity.attributes.push(Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                continuum_cdsl_ast::UntypedKind::Signal(Path::from_path_str("fast")),
                span,
            )],
            span,
        });

        // Member WITHOUT stratum attribute
        let member = Node::new(
            Path::from_path_str("mass"),
            span,
            RoleData::Signal,
            EntityId::new("particle"),
        );
        entity.members.push(member);

        let declarations = vec![Declaration::Entity(entity)];
        let mut errors = Vec::new();
        let flattened = flatten_entity_members(declarations, &mut errors);

        // Member should inherit entity's stratum
        if let Declaration::Member(member) = &flattened[1] {
            assert!(has_attribute(&member.attributes, "stratum"));
            let mut test_errors = Vec::new();
            let stratum =
                extract_single_path(&member.attributes, "stratum", span, &mut test_errors);
            assert_eq!(stratum, Some(Path::from_path_str("fast")));
            assert!(test_errors.is_empty());
        } else {
            panic!("Expected Declaration::Member");
        }
    }

    #[test]
    fn test_flatten_entity_members_empty_entity() {
        let span = test_span();
        let entity = Entity::new(
            EntityId::new("particle"),
            Path::from_path_str("particle"),
            span,
        );

        let declarations = vec![Declaration::Entity(entity)];
        let mut errors = Vec::new();
        let flattened = flatten_entity_members(declarations, &mut errors);

        assert_eq!(flattened.len(), 1);
        assert!(matches!(flattened[0], Declaration::Entity(_)));
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_stratum_malformed_no_args() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "stratum".to_string(),
            args: vec![],
            span,
        }];

        let mut errors = Vec::new();
        let stratum = extract_single_path(&attrs, "stratum", span, &mut errors);

        assert_eq!(stratum, None);
        assert_eq!(errors.len(), 1);
        assert!(errors[0]
            .message
            .contains(":stratum attribute expects exactly 1 argument"));
    }

    #[test]
    fn test_extract_stratum_malformed_wrong_type() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                continuum_cdsl_ast::UntypedKind::Literal {
                    value: 42.0,
                    unit: None,
                },
                span,
            )],
            span,
        }];

        let mut errors = Vec::new();
        let stratum = extract_single_path(&attrs, "stratum", span, &mut errors);

        assert_eq!(stratum, None);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("must be a path or identifier"));
        assert!(errors[0].message.contains("literal value"));
    }

    #[test]
    fn test_extract_stratum_no_attribute() {
        let span = test_span();
        let attrs = vec![];

        let mut errors = Vec::new();
        let stratum = extract_single_path(&attrs, "stratum", span, &mut errors);

        assert_eq!(stratum, None);
        assert!(errors.is_empty()); // No attribute is not an error
    }

    #[test]
    fn test_flatten_entity_members_no_stratum_anywhere() {
        // Test that members without stratum and entity without stratum both have no stratum attribute
        let span = test_span();
        let mut entity = Entity::new(
            EntityId::new("particle"),
            Path::from_path_str("particle"),
            span,
        );

        // Add two members without stratum attributes
        let member1 = Node::new(
            Path::from_path_str("mass"),
            span,
            RoleData::Signal,
            EntityId::new("particle"),
        );
        entity.members.push(member1);

        let member2 = Node::new(
            Path::from_path_str("velocity"),
            span,
            RoleData::Signal,
            EntityId::new("particle"),
        );
        entity.members.push(member2);

        let declarations = vec![Declaration::Entity(entity)];
        let mut errors = Vec::new();
        let flattened = flatten_entity_members(declarations, &mut errors);

        assert_eq!(flattened.len(), 3); // 1 Entity + 2 Members
        assert!(errors.is_empty());

        // Check that both members have no stratum attribute
        for i in 1..3 {
            if let Declaration::Member(member) = &flattened[i] {
                assert!(!has_attribute(&member.attributes, "stratum"));
            } else {
                panic!("Expected Declaration::Member at index {}", i);
            }
        }
    }

    #[test]
    fn test_flatten_entity_members_mixed_stratum_attributes() {
        // Test members with mixed stratum behavior: inherit, override, and override again
        let span = test_span();
        let mut entity = Entity::new(
            EntityId::new("particle"),
            Path::from_path_str("particle"),
            span,
        );

        // Entity has default stratum
        entity.attributes.push(Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("default")),
                span,
            )],
            span,
        });

        // Member1: no stratum (should inherit "default")
        let member1 = Node::new(
            Path::from_path_str("mass"),
            span,
            RoleData::Signal,
            EntityId::new("particle"),
        );
        entity.members.push(member1);

        // Member2: explicit stratum "fast" (should override)
        let mut member2 = Node::new(
            Path::from_path_str("velocity"),
            span,
            RoleData::Signal,
            EntityId::new("particle"),
        );
        member2.attributes.push(Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("fast")),
                span,
            )],
            span,
        });
        entity.members.push(member2);

        // Member3: explicit stratum "slow" (should override)
        let mut member3 = Node::new(
            Path::from_path_str("position"),
            span,
            RoleData::Signal,
            EntityId::new("particle"),
        );
        member3.attributes.push(Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("slow")),
                span,
            )],
            span,
        });
        entity.members.push(member3);

        let declarations = vec![Declaration::Entity(entity)];
        let mut errors = Vec::new();
        let flattened = flatten_entity_members(declarations, &mut errors);

        assert_eq!(flattened.len(), 4); // 1 Entity + 3 Members
        assert!(errors.is_empty());

        // Check member1: inherited "default"
        if let Declaration::Member(member) = &flattened[1] {
            assert_eq!(member.path, Path::from_path_str("particle.mass"));
            assert!(has_attribute(&member.attributes, "stratum"));
            let mut test_errors = Vec::new();
            let stratum =
                extract_single_path(&member.attributes, "stratum", span, &mut test_errors);
            assert_eq!(stratum, Some(Path::from_path_str("default")));
            assert!(test_errors.is_empty());
        } else {
            panic!("Expected Declaration::Member at index 1");
        }

        // Check member2: overridden to "fast"
        if let Declaration::Member(member) = &flattened[2] {
            assert_eq!(member.path, Path::from_path_str("particle.velocity"));
            assert!(has_attribute(&member.attributes, "stratum"));
            let mut test_errors = Vec::new();
            let stratum =
                extract_single_path(&member.attributes, "stratum", span, &mut test_errors);
            assert_eq!(stratum, Some(Path::from_path_str("fast")));
            assert!(test_errors.is_empty());
        } else {
            panic!("Expected Declaration::Member at index 2");
        }

        // Check member3: overridden to "slow"
        if let Declaration::Member(member) = &flattened[3] {
            assert_eq!(member.path, Path::from_path_str("particle.position"));
            assert!(has_attribute(&member.attributes, "stratum"));
            let mut test_errors = Vec::new();
            let stratum =
                extract_single_path(&member.attributes, "stratum", span, &mut test_errors);
            assert_eq!(stratum, Some(Path::from_path_str("slow")));
            assert!(test_errors.is_empty());
        } else {
            panic!("Expected Declaration::Member at index 3");
        }
    }
}
