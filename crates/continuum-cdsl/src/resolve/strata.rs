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
//! Parse → Name Res → Type Res → Validation → Stratum Resolution → Block Compilation
//!                                                 ^^^^^^^^^^^^^^^^^
//!                                                  YOU ARE HERE
//! ```
//!
//! This pass runs after validation (Phase 12) and before execution block
//! compilation (Phase 12.5-D). It's part of Phase 12.5 prerequisites for
//! execution DAG construction.

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
}
