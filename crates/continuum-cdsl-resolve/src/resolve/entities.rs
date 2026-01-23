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
                extract_stratum_from_attributes(&entity.attributes, entity.span, errors);

            // Flatten each nested member
            for member in &entity.members {
                let mut flattened_member = member.clone();

                // Build full path: entity.path + member.path
                let mut full_segments = entity.path.segments.clone();
                full_segments.extend(member.path.segments.clone());
                flattened_member.path = Path::new(full_segments);

                // Inherit stratum if member doesn't have one
                if !has_stratum_attribute(&flattened_member.attributes) {
                    if let Some(ref stratum_path) = entity_stratum {
                        // Add :stratum(name) attribute to member
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

/// Extracts stratum path from `:stratum(name)` attribute expression.
///
/// Handles both pre-resolution expression variants since this runs BEFORE name resolution:
/// - `Local("thermal")` → converts to `Path::from_path_str("thermal")`
/// - `Signal(Path)` → returns path directly
///
/// This function is phase-boundary safe: it operates on untyped AST before the type
/// resolution pass, extracting only syntactic information from attribute arguments.
///
/// # Fail-Hard Behavior
///
/// This function follows the fail-hard principle:
/// - Malformed `:stratum` attributes emit errors immediately (wrong argument type, missing args)
/// - Returns `None` only when no `:stratum` attribute exists (legitimate absence)
/// - Never silently discards invalid input
///
/// # Parameters
///
/// * `attrs` - Attribute slice to search for `:stratum` attribute
/// * `context_span` - Span of the entity/member for error reporting
/// * `errors` - Error accumulator for malformed attributes
///
/// # Returns
///
/// - `Some(Path)`: Stratum path extracted from valid `:stratum(...)` attribute
/// - `None`: No stratum attribute found (legitimate - not an error)
///
/// # Errors
///
/// Emits `CompileError` for:
/// - `:stratum` attribute with no arguments
/// - `:stratum` attribute with non-path argument (e.g., `:stratum(42)`)
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl_ast::{Attribute, Expr, UntypedKind};
/// use continuum_foundation::Path;
///
/// let attrs = vec![Attribute {
///     name: "stratum".to_string(),
///     args: vec![Expr::new(UntypedKind::Local("thermal".to_string()), span)],
///     span,
/// }];
///
/// let mut errors = Vec::new();
/// let stratum = extract_stratum_from_attributes(&attrs, span, &mut errors);
/// assert_eq!(stratum, Some(Path::from_path_str("thermal")));
/// assert!(errors.is_empty());
/// ```
///
/// # Why Local and Signal?
///
/// Parser emits `Local("name")` for bare identifiers, `Signal(path)` for dotted paths.
/// Type resolution hasn't run yet, so we accept both syntactic forms.
fn extract_stratum_from_attributes(
    attrs: &[Attribute],
    context_span: continuum_cdsl_ast::foundation::Span,
    errors: &mut Vec<CompileError>,
) -> Option<Path> {
    let stratum_attr = attrs.iter().find(|a| a.name == "stratum")?;

    // Check if attribute has arguments
    let arg = match stratum_attr.args.first() {
        Some(arg) => arg,
        None => {
            errors.push(CompileError::new(
                crate::error::ErrorKind::Syntax,
                stratum_attr.span,
                ":stratum attribute requires a path argument (e.g., :stratum(thermal))".to_string(),
            ));
            return None;
        }
    };

    // Extract path from argument expression
    match &arg.kind {
        UntypedKind::Signal(path) => Some(path.clone()),
        UntypedKind::Local(name) => Some(Path::from_path_str(name)),
        _ => {
            errors.push(
                CompileError::new(
                    crate::error::ErrorKind::Syntax,
                    arg.span,
                    format!(
                        ":stratum argument must be a path, found {}",
                        describe_expr_kind(&arg.kind)
                    ),
                )
                .with_label(
                    context_span,
                    "in this entity or member declaration".to_string(),
                ),
            );
            None
        }
    }
}

/// Describes an expression kind for error messages.
fn describe_expr_kind(kind: &UntypedKind) -> &'static str {
    match kind {
        UntypedKind::Literal { .. } => "literal value",
        UntypedKind::BoolLiteral(_) => "boolean",
        UntypedKind::StringLiteral(_) => "string",
        UntypedKind::Vector(_) => "vector",
        UntypedKind::Signal(_) => "signal path",
        UntypedKind::Field(_) => "field path",
        UntypedKind::Config(_) => "config path",
        UntypedKind::Const(_) => "const path",
        UntypedKind::Local(_) => "local variable",
        UntypedKind::Binary { .. } => "binary operation",
        UntypedKind::Unary { .. } => "unary operation",
        UntypedKind::Call { .. } | UntypedKind::KernelCall { .. } => "function call",
        UntypedKind::Let { .. } => "let binding",
        UntypedKind::If { .. } => "if expression",
        UntypedKind::Aggregate { .. } => "aggregate expression",
        UntypedKind::Fold { .. } => "fold expression",
        UntypedKind::Struct { .. } => "struct literal",
        UntypedKind::FieldAccess { .. } => "field access",
        UntypedKind::Entity(_) => "entity reference",
        UntypedKind::Nearest { .. } => "nearest query",
        UntypedKind::Within { .. } => "within query",
        UntypedKind::Filter { .. } => "filter query",
        UntypedKind::OtherInstances(_) => "other instances",
        UntypedKind::PairsInstances(_) => "pairs instances",
        UntypedKind::Prev => "prev",
        UntypedKind::Current => "current",
        UntypedKind::Inputs => "inputs",
        UntypedKind::Self_ => "self",
        UntypedKind::Other => "other",
        UntypedKind::Payload => "payload",
        UntypedKind::ParseError(_) => "parse error",
    }
}

/// Returns true if attribute list contains a `:stratum` attribute.
///
/// Used to detect explicit stratum declarations on members to avoid
/// overriding with entity-level stratum inheritance.
///
/// # Parameters
///
/// * `attrs` - Attribute slice to search
///
/// # Returns
///
/// - `true`: At least one `:stratum` attribute present (regardless of arguments)
/// - `false`: No stratum attribute found
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl_ast::Attribute;
///
/// let attrs = vec![Attribute {
///     name: "stratum".to_string(),
///     args: vec![],
///     span,
/// }];
///
/// assert!(has_stratum_attribute(&attrs));
/// ```
fn has_stratum_attribute(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| a.name == "stratum")
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
            assert!(has_stratum_attribute(&member.attributes));
            let mut test_errors = Vec::new();
            let stratum =
                extract_stratum_from_attributes(&member.attributes, span, &mut test_errors);
            assert_eq!(stratum, Some(Path::from_path_str("fast")));
            assert!(test_errors.is_empty());
        } else {
            panic!("Expected Declaration::Member");
        }
    }

    #[test]
    fn test_flatten_entity_members_member_stratum_override() {
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

        if let Declaration::Member(member) = &flattened[1] {
            let mut test_errors = Vec::new();
            let stratum =
                extract_stratum_from_attributes(&member.attributes, span, &mut test_errors);
            assert_eq!(stratum, Some(Path::from_path_str("slow")));
            assert!(test_errors.is_empty());

            let stratum_count = member
                .attributes
                .iter()
                .filter(|a| a.name == "stratum")
                .count();
            assert_eq!(stratum_count, 1);
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
        let stratum = extract_stratum_from_attributes(&attrs, span, &mut errors);

        assert_eq!(stratum, None);
        assert_eq!(errors.len(), 1);
        assert!(errors[0]
            .message
            .contains(":stratum attribute requires a path argument"));
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
        let stratum = extract_stratum_from_attributes(&attrs, span, &mut errors);

        assert_eq!(stratum, None);
        assert_eq!(errors.len(), 1);
        assert!(errors[0]
            .message
            .contains(":stratum argument must be a path"));
        assert!(errors[0].message.contains("literal value"));
    }

    #[test]
    fn test_extract_stratum_no_attribute() {
        let span = test_span();
        let attrs = vec![];

        let mut errors = Vec::new();
        let stratum = extract_stratum_from_attributes(&attrs, span, &mut errors);

        assert_eq!(stratum, None);
        assert!(errors.is_empty()); // No attribute is not an error
    }
}
