//! Stratum resolution for CDSL AST.
//!
//! Resolves stratum assignments from node attributes and populates `Node.stratum`.
//!
//! # What This Pass Does
//!
//! ## Stratum Assignment
//!
//! Extracts `:stratum(name)` attributes from nodes and validates against world's
//! stratum declarations:
//! - Validates stratum name exists
//! - Populates `node.stratum` field
//! - Provides default stratum if not specified (when world has single stratum)
//! - Errors if no stratum specified and world has multiple strata
//!
//! ## Cadence Resolution
//!
//! Extracts cadence from `:stride(N)` or `:cadence(N)` attributes on `Stratum`:
//! - Validates cadence is positive integer
//! - Defaults to 1 if not specified
//! - Populates `Stratum.cadence` field
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Name Res → Type Res → Validation → Stratum Resolution → Era Resolution → Block Compilation
//!                                                 ^^^^^^^^^^^^^^^^^
//!                                                  YOU ARE HERE
//! ```
//!
//! This pass runs after validation (Phase 12) and before era resolution (Phase 12.5-B).
//! It's part of Phase 12.5 execution prerequisites for execution DAG construction.
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use continuum_cdsl::resolve::strata;
//! use continuum_cdsl::ast::{Node, Stratum};
//!
//! // After parsing and type resolution
//! let mut nodes: Vec<Node<_>> = parsed_ast.nodes;
//! let mut strata: Vec<Stratum> = parsed_ast.strata;
//!
//! // Phase 12.5-A: Resolve stratum assignments and cadences
//!
//! // Step 1: Resolve cadences for all strata
//! match strata::resolve_cadences(&mut strata) {
//!     Ok(()) => {
//!         // All strata have cadence assigned (default 1 or explicit)
//!         for stratum in &strata {
//!             assert!(stratum.cadence.is_some());
//!             assert!(stratum.cadence.unwrap() > 0);
//!         }
//!     }
//!     Err(cadence_errors) => {
//!         // Handle invalid cadence values (zero, negative, non-integer)
//!         return Err(cadence_errors);
//!     }
//! }
//!
//! // Step 2: Resolve stratum assignments for all nodes
//! match strata::resolve_strata(&mut nodes, &strata) {
//!     Ok(()) => {
//!         // All nodes have stratum assigned
//!         for node in &nodes {
//!             assert!(node.stratum.is_some());
//!         }
//!     }
//!     Err(stratum_errors) => {
//!         // Handle undefined strata, ambiguous assignments, invalid attributes
//!         return Err(stratum_errors);
//!     }
//! }
//!
//! // Ready for Phase 12.5-B: Era resolution
//! let stratum_ids: Vec<StratumId> = strata.iter().map(|s| s.id.clone()).collect();
//! // Pass stratum_ids to era resolution for validation...
//! ```

use crate::ast::{Expr, Node, Stratum, UntypedKind};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::StratumId;
use std::collections::HashMap;

/// Extract identifier string from expression (for attribute arguments).
///
/// Attributes like `:stratum(thermal)` have expressions as arguments.
/// This helper extracts the identifier string for simple cases.
fn extract_identifier(expr: &Expr) -> Option<String> {
    match &expr.kind {
        UntypedKind::Signal(path)
        | UntypedKind::Field(path)
        | UntypedKind::Config(path)
        | UntypedKind::Const(path) => path.last().map(|s| s.to_string()),
        _ => None,
    }
}

/// Resolve stratum assignments for all nodes.
///
/// Populates `node.stratum` by extracting `:stratum(name)` from attributes.
///
/// # Errors
///
/// Returns [`ErrorKind::UndefinedName`] if stratum name doesn't exist.
/// Returns [`ErrorKind::AmbiguousName`] if no stratum specified and multiple exist.
///
/// # Examples
///
/// ```rust,ignore
/// let mut nodes = parse_nodes(source);
/// let strata = parse_strata(source);
///
/// resolve_strata(&mut nodes, &strata)?;
///
/// // All nodes now have stratum assigned
/// for node in &nodes {
///     assert!(node.stratum.is_some());
/// }
/// ```
pub fn resolve_strata<I: crate::ast::Index>(
    nodes: &mut [Node<I>],
    strata: &[Stratum],
) -> Result<(), Vec<CompileError>> {
    let mut errors = Vec::new();

    // Build stratum lookup map: name → StratumId
    let mut stratum_map: HashMap<String, StratumId> = HashMap::new();
    for stratum in strata {
        let name = stratum.path.last().unwrap_or("").to_string();
        stratum_map.insert(name, stratum.id.clone());
    }

    // Determine default stratum (if world has exactly one)
    let default_stratum = if strata.len() == 1 {
        Some(strata[0].id.clone())
    } else {
        None
    };

    // Resolve stratum for each node
    for node in nodes.iter_mut() {
        // Extract :stratum(name) attribute
        let stratum_attr = node.attributes.iter().find(|attr| attr.name == "stratum");

        match stratum_attr {
            Some(attr) => {
                // Extract stratum name from attribute value
                // Attribute format: :stratum(name) where name is an identifier
                if attr.args.len() != 1 {
                    errors.push(CompileError::new(
                        ErrorKind::InvalidCapability,
                        node.span,
                        format!(
                            "stratum attribute expects exactly one argument, got {}",
                            attr.args.len()
                        ),
                    ));
                    continue;
                }

                match extract_identifier(&attr.args[0]) {
                    Some(stratum_name) => match stratum_map.get(&stratum_name) {
                        Some(stratum_id) => {
                            node.stratum = Some(stratum_id.clone());
                        }
                        None => {
                            errors.push(
                                CompileError::new(
                                    ErrorKind::UndefinedName,
                                    node.span,
                                    format!("undefined stratum '{}'", stratum_name),
                                )
                                .with_note(format!(
                                    "available strata: {}",
                                    stratum_map
                                        .keys()
                                        .map(|s| format!("'{}'", s))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                )),
                            );
                        }
                    },
                    None => {
                        errors.push(CompileError::new(
                            ErrorKind::InvalidCapability,
                            node.span,
                            "stratum attribute argument must be an identifier".to_string(),
                        ));
                    }
                }
            }
            None => {
                // No explicit stratum attribute
                match default_stratum.as_ref() {
                    Some(stratum_id) => {
                        // Use default stratum
                        node.stratum = Some(stratum_id.clone());
                    }
                    None => {
                        // Multiple strata exist, must specify explicitly
                        errors.push(
                            CompileError::new(
                                ErrorKind::AmbiguousName,
                                node.span,
                                format!(
                                    "node '{}' must specify stratum (world has multiple strata)",
                                    node.path
                                ),
                            )
                            .with_note(format!(
                                "available strata: {}",
                                stratum_map
                                    .keys()
                                    .map(|s| format!("'{}'", s))
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            ))
                            .with_note("add :stratum(name) attribute to specify".to_string()),
                        );
                    }
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Resolve cadence for all strata from attributes.
///
/// Extracts `:stride(N)` or `:cadence(N)` and populates `Stratum.cadence`.
///
/// # Errors
///
/// Returns [`ErrorKind::InvalidCapability`] if cadence is not a positive integer.
///
/// # Examples
///
/// ```rust,ignore
/// let mut strata = parse_strata(source);
///
/// resolve_cadences(&mut strata)?;
///
/// // All strata now have cadence resolved
/// for stratum in &strata {
///     assert!(stratum.cadence.is_some());
///     assert!(stratum.cadence.unwrap() > 0);
/// }
/// ```
pub fn resolve_cadences(strata: &mut [Stratum]) -> Result<(), Vec<CompileError>> {
    let mut errors = Vec::new();

    for stratum in strata.iter_mut() {
        // Look for :stride(N) or :cadence(N) attribute
        let cadence_attr = stratum
            .attributes
            .iter()
            .find(|attr| attr.name == "stride" || attr.name == "cadence");

        match cadence_attr {
            Some(attr) => {
                if attr.args.len() != 1 {
                    errors.push(CompileError::new(
                        ErrorKind::InvalidCapability,
                        stratum.span,
                        format!(
                            "{} attribute expects exactly one argument, got {}",
                            attr.name,
                            attr.args.len()
                        ),
                    ));
                    continue;
                }

                // Extract numeric value from literal expression
                if let UntypedKind::Literal { value, .. } = attr.args[0].kind {
                    let cadence = value as u32;
                    if cadence > 0 && (cadence as f64 - value).abs() < 1e-9 {
                        stratum.cadence = Some(cadence);
                    } else {
                        errors.push(CompileError::new(
                            ErrorKind::InvalidCapability,
                            stratum.span,
                            format!("{} must be a positive integer, got {}", attr.name, value),
                        ));
                    }
                } else {
                    errors.push(CompileError::new(
                        ErrorKind::InvalidCapability,
                        stratum.span,
                        format!("{} must be a numeric literal", attr.name),
                    ));
                }
            }
            None => {
                // Default cadence is 1 (execute every tick)
                stratum.cadence = Some(1);
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Attribute, RoleData};
    use crate::foundation::{Path, Span};

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    fn make_stratum(name: &str, attributes: Vec<Attribute>) -> Stratum {
        let path = Path::from_str(name);
        let mut stratum = Stratum::new(StratumId::new(name), path, test_span());
        stratum.attributes = attributes;
        stratum
    }

    fn make_node(path: &str, attributes: Vec<Attribute>) -> Node<()> {
        let mut node = Node::new(Path::from_str(path), test_span(), RoleData::Signal, ());
        node.attributes = attributes;
        node
    }

    fn make_attr(name: &str, arg_names: Vec<&str>) -> Attribute {
        use crate::foundation::Path;

        let span = test_span();
        let args = arg_names
            .into_iter()
            .map(|name| Expr::new(UntypedKind::Signal(Path::from_str(name)), span))
            .collect();

        Attribute {
            name: name.to_string(),
            args,
            span,
        }
    }

    fn make_attr_numeric(name: &str, value: f64) -> Attribute {
        let span = test_span();
        Attribute {
            name: name.to_string(),
            args: vec![Expr::new(UntypedKind::Literal { value, unit: None }, span)],
            span,
        }
    }

    #[test]
    fn test_resolve_single_stratum_default() {
        let strata = vec![make_stratum("fast", vec![])];
        let mut nodes = vec![make_node("signal.temp", vec![])];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_ok());
        assert_eq!(nodes[0].stratum, Some(StratumId::new("fast")));
    }

    #[test]
    fn test_resolve_explicit_stratum() {
        let strata = vec![make_stratum("fast", vec![]), make_stratum("slow", vec![])];
        let mut nodes = vec![make_node(
            "signal.temp",
            vec![make_attr("stratum", vec!["fast"])],
        )];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_ok());
        assert_eq!(nodes[0].stratum, Some(StratumId::new("fast")));
    }

    #[test]
    fn test_resolve_multiple_strata_requires_explicit() {
        let strata = vec![make_stratum("fast", vec![]), make_stratum("slow", vec![])];
        let mut nodes = vec![make_node("signal.temp", vec![])];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::AmbiguousName);
    }

    #[test]
    fn test_resolve_undefined_stratum() {
        let strata = vec![make_stratum("fast", vec![])];
        let mut nodes = vec![make_node(
            "signal.temp",
            vec![make_attr("stratum", vec!["nonexistent"])],
        )];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::UndefinedName);
    }

    #[test]
    fn test_resolve_cadence_default() {
        let mut strata = vec![make_stratum("fast", vec![])];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_ok());
        assert_eq!(strata[0].cadence, Some(1));
    }

    #[test]
    fn test_resolve_cadence_explicit() {
        let mut strata = vec![make_stratum(
            "slow",
            vec![make_attr_numeric("stride", 10.0)],
        )];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_ok());
        assert_eq!(strata[0].cadence, Some(10));
    }

    #[test]
    fn test_resolve_cadence_invalid_zero() {
        let mut strata = vec![make_stratum("bad", vec![make_attr_numeric("stride", 0.0)])];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
    }

    #[test]
    fn test_resolve_cadence_invalid_non_literal() {
        // Use an identifier expression instead of a numeric literal
        let mut strata = vec![make_stratum(
            "bad",
            vec![make_attr("stride", vec!["not_a_literal"])],
        )];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
    }

    // ===== Edge Case Tests =====

    #[test]
    fn test_resolve_no_strata_no_nodes() {
        let mut nodes: Vec<Node<()>> = vec![];
        let strata: Vec<Stratum> = vec![];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_no_nodes_with_strata() {
        let mut nodes: Vec<Node<()>> = vec![];
        let strata = vec![make_stratum("fast", vec![])];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_ok());
    }

    #[test]
    fn test_resolve_multiple_nodes_single_stratum() {
        let strata = vec![make_stratum("default", vec![])];
        let mut nodes = vec![
            make_node("signal1", vec![]),
            make_node("signal2", vec![]),
            make_node("signal3", vec![]),
        ];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_ok());

        // All nodes should have the default stratum assigned
        for node in &nodes {
            assert_eq!(node.stratum, Some(StratumId::new("default")));
        }
    }

    #[test]
    fn test_resolve_multiple_nodes_multiple_strata_explicit() {
        let strata = vec![make_stratum("fast", vec![]), make_stratum("slow", vec![])];

        let mut nodes = vec![
            make_node("signal1", vec![make_attr("stratum", vec!["fast"])]),
            make_node("signal2", vec![make_attr("stratum", vec!["slow"])]),
            make_node("signal3", vec![make_attr("stratum", vec!["fast"])]),
        ];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_ok());

        assert_eq!(nodes[0].stratum, Some(StratumId::new("fast")));
        assert_eq!(nodes[1].stratum, Some(StratumId::new("slow")));
        assert_eq!(nodes[2].stratum, Some(StratumId::new("fast")));
    }

    #[test]
    fn test_resolve_stratum_attribute_wrong_arg_count_zero() {
        let strata = vec![make_stratum("fast", vec![])];
        let mut nodes = vec![make_node("signal1", vec![make_attr("stratum", vec![])])];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
        assert!(errors[0].message.contains("exactly one argument"));
    }

    #[test]
    fn test_resolve_stratum_attribute_wrong_arg_count_multiple() {
        let strata = vec![make_stratum("fast", vec![])];
        let mut nodes = vec![make_node(
            "signal1",
            vec![make_attr("stratum", vec!["fast", "slow"])],
        )];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
        assert!(errors[0].message.contains("exactly one argument"));
    }

    #[test]
    fn test_resolve_stratum_attribute_non_identifier() {
        use crate::ast::UntypedKind;

        let strata = vec![make_stratum("fast", vec![])];

        // Create attribute with literal expression (not an identifier)
        let span = test_span();
        let invalid_arg = Expr::new(
            UntypedKind::Literal {
                value: 42.0,
                unit: None,
            },
            span,
        );
        let attr = Attribute {
            name: "stratum".to_string(),
            args: vec![invalid_arg],
            span,
        };

        let mut nodes = vec![make_node("signal1", vec![attr])];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
        assert!(errors[0].message.contains("must be an identifier"));
    }

    // ===== Cadence Edge Case Tests =====

    #[test]
    fn test_resolve_cadence_negative() {
        let mut strata = vec![make_stratum(
            "bad",
            vec![make_attr_numeric("cadence", -5.0)],
        )];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
        assert!(errors[0].message.contains("positive integer"));
    }

    #[test]
    fn test_resolve_cadence_float() {
        let mut strata = vec![make_stratum("bad", vec![make_attr_numeric("cadence", 1.5)])];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
        assert!(errors[0].message.contains("positive integer"));
    }

    #[test]
    fn test_resolve_cadence_large_value() {
        let mut strata = vec![make_stratum(
            "slow",
            vec![make_attr_numeric("stride", 1000.0)],
        )];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_ok());
        assert_eq!(strata[0].cadence, Some(1000));
    }

    #[test]
    fn test_resolve_cadence_multiple_strata_mixed() {
        let mut strata = vec![
            make_stratum("fast", vec![]), // default 1
            make_stratum("medium", vec![make_attr_numeric("stride", 5.0)]), // explicit 5
            make_stratum("slow", vec![make_attr_numeric("cadence", 10.0)]), // explicit 10
        ];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_ok());
        assert_eq!(strata[0].cadence, Some(1));
        assert_eq!(strata[1].cadence, Some(5));
        assert_eq!(strata[2].cadence, Some(10));
    }

    #[test]
    fn test_resolve_cadence_stride_vs_cadence_same_behavior() {
        let mut strata_stride = vec![make_stratum("test", vec![make_attr_numeric("stride", 7.0)])];
        let mut strata_cadence = vec![make_stratum(
            "test",
            vec![make_attr_numeric("cadence", 7.0)],
        )];

        let result1 = resolve_cadences(&mut strata_stride);
        let result2 = resolve_cadences(&mut strata_cadence);

        assert!(result1.is_ok());
        assert!(result2.is_ok());
        assert_eq!(strata_stride[0].cadence, strata_cadence[0].cadence);
        assert_eq!(strata_stride[0].cadence, Some(7));
    }

    #[test]
    fn test_resolve_cadence_attribute_wrong_arg_count() {
        let mut strata = vec![make_stratum(
            "bad",
            vec![make_attr("cadence", vec!["arg1", "arg2"])],
        )];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
        assert!(errors[0].message.contains("exactly one argument"));
    }

    // ===== Error Accumulation Tests =====

    #[test]
    fn test_resolve_multiple_undefined_strata() {
        let strata = vec![make_stratum("valid", vec![])];
        let mut nodes = vec![
            make_node("signal1", vec![make_attr("stratum", vec!["missing1"])]),
            make_node("signal2", vec![make_attr("stratum", vec!["missing2"])]),
            make_node("signal3", vec![make_attr("stratum", vec!["missing3"])]),
        ];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 3);

        for error in &errors {
            assert_eq!(error.kind, ErrorKind::UndefinedName);
            assert!(error.message.contains("undefined stratum"));
            assert!(error.notes.iter().any(|n| n.contains("available strata")));
        }
    }

    #[test]
    fn test_resolve_multiple_nodes_missing_stratum() {
        let strata = vec![make_stratum("fast", vec![]), make_stratum("slow", vec![])];
        let mut nodes = vec![
            make_node("signal1", vec![]),
            make_node("signal2", vec![]),
            make_node("signal3", vec![]),
        ];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 3);

        for error in &errors {
            assert_eq!(error.kind, ErrorKind::AmbiguousName);
            assert!(error.message.contains("must specify stratum"));
            assert!(error.notes.iter().any(|n| n.contains("add :stratum(name)")));
        }
    }

    #[test]
    fn test_resolve_cadence_multiple_errors() {
        let mut strata = vec![
            make_stratum("bad1", vec![make_attr_numeric("stride", 0.0)]), // zero
            make_stratum("bad2", vec![make_attr_numeric("cadence", -1.0)]), // negative
            make_stratum("bad3", vec![make_attr("stride", vec!["not_num"])]), // non-literal
        ];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 3);

        for error in &errors {
            assert_eq!(error.kind, ErrorKind::InvalidCapability);
        }
    }

    // ===== Mixed Valid and Invalid Cases =====

    #[test]
    fn test_resolve_mixed_valid_invalid_nodes() {
        let strata = vec![make_stratum("fast", vec![]), make_stratum("slow", vec![])];
        let mut nodes = vec![
            make_node("signal1", vec![make_attr("stratum", vec!["fast"])]), // valid
            make_node("signal2", vec![make_attr("stratum", vec!["missing"])]), // invalid
            make_node("signal3", vec![make_attr("stratum", vec!["slow"])]), // valid
        ];

        let result = resolve_strata(&mut nodes, &strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        // Only signal2 should have an error
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("missing"));

        // Valid nodes should still be assigned
        assert_eq!(nodes[0].stratum, Some(StratumId::new("fast")));
        assert_eq!(nodes[2].stratum, Some(StratumId::new("slow")));
    }

    #[test]
    fn test_resolve_mixed_valid_invalid_cadences() {
        let mut strata = vec![
            make_stratum("good", vec![make_attr_numeric("stride", 5.0)]), // valid
            make_stratum("bad", vec![make_attr_numeric("cadence", 0.0)]), // invalid
            make_stratum("default", vec![]),                              // valid (default)
        ];

        let result = resolve_cadences(&mut strata);
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("positive integer"));

        // Valid strata should still have cadence assigned
        assert_eq!(strata[0].cadence, Some(5));
        assert_eq!(strata[2].cadence, Some(1));
    }
}
