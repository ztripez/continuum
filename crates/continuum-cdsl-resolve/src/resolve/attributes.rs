//! Common utilities for extracting and validating attribute values.
//!
//! This module provides reusable functions for attribute extraction patterns used
//! throughout the resolution pipeline. Centralizes error handling and follows the
//! One Truth doctrine by eliminating duplicated extraction logic.
//!
//! # Common Patterns
//!
//! - **Single path extraction**: `:stratum(thermal)`, `:integrator(rk4)`
//! - **Multiple path extraction**: `:uses(a, b, c)`
//! - **Identifier extraction**: Convert expression to string identifier
//! - **Existence checks**: Simple attribute presence testing
//!
//! # Design Principles
//!
//! - **Fail-hard**: Invalid attributes emit errors immediately
//! - **DRY**: Single implementation for each extraction pattern
//! - **Clear errors**: Descriptive messages with context

use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::{Path, Span};
use continuum_cdsl_ast::{Attribute, Expr, UntypedKind};

/// Extracts a single path from an attribute with exactly one argument.
///
/// Use this for attributes like `:stratum(thermal)` or `:integrator(rk4)` that
/// expect exactly one path-like argument.
///
/// # Fail-Hard Behavior
///
/// Emits errors for:
/// - Wrong number of arguments (not exactly 1)
/// - Non-path argument types (e.g., literals, operations)
/// - Returns `None` only when attribute doesn't exist (legitimate absence)
///
/// # Parameters
///
/// * `attrs` - Attribute slice to search
/// * `attr_name` - Name of attribute to extract (e.g., "stratum", "integrator")
/// * `context_span` - Span for error context (entity/node/stratum location)
/// * `errors` - Error accumulator
///
/// # Returns
///
/// - `Some(Path)`: Successfully extracted path
/// - `None`: Attribute not found (not an error) OR error emitted
///
/// # Examples
///
/// ```rust,ignore
/// let attrs = vec![Attribute {
///     name: "stratum".to_string(),
///     args: vec![Expr::local("thermal", span)],
///     span,
/// }];
///
/// let mut errors = Vec::new();
/// let path = extract_single_path(&attrs, "stratum", span, &mut errors);
/// assert_eq!(path, Some(Path::from_path_str("thermal")));
/// assert!(errors.is_empty());
/// ```
pub fn extract_single_path(
    attrs: &[Attribute],
    attr_name: &str,
    context_span: Span,
    errors: &mut Vec<CompileError>,
) -> Option<Path> {
    let attr = attrs.iter().find(|a| a.name == attr_name)?;

    // Validate argument count
    if attr.args.len() != 1 {
        errors.push(CompileError::new(
            ErrorKind::Syntax,
            attr.span,
            format!(
                ":{} attribute expects exactly 1 argument, got {}",
                attr_name,
                attr.args.len()
            ),
        ));
        return None;
    }

    // Extract path from argument
    extract_path_from_expr(&attr.args[0], attr_name, context_span, errors)
}

/// Extracts a single identifier string from an attribute with exactly one argument.
///
/// Similar to `extract_single_path` but returns the last segment as a string
/// rather than the full path. Use for attributes like `:integrator(rk4)` where
/// you need just the method name.
///
/// # Parameters
///
/// * `attrs` - Attribute slice to search
/// * `attr_name` - Name of attribute to extract
/// * `context_span` - Span for error context
/// * `errors` - Error accumulator
///
/// # Returns
///
/// - `Some(String)`: Successfully extracted identifier
/// - `None`: Attribute not found OR error emitted
///
/// # Examples
///
/// ```rust,ignore
/// let attrs = vec![Attribute {
///     name: "integrator".to_string(),
///     args: vec![Expr::signal(Path::from_path_str("rk4"), span)],
///     span,
/// }];
///
/// let mut errors = Vec::new();
/// let method = extract_single_identifier(&attrs, "integrator", span, &mut errors);
/// assert_eq!(method, Some("rk4".to_string()));
/// ```
pub fn extract_single_identifier(
    attrs: &[Attribute],
    attr_name: &str,
    context_span: Span,
    errors: &mut Vec<CompileError>,
) -> Option<String> {
    extract_single_path(attrs, attr_name, context_span, errors)
        .and_then(|path| path.last().map(|s| s.to_string()))
}

/// Extracts multiple paths from an attribute with zero or more arguments.
///
/// Use this for attributes like `:uses(maths.clamping, dt.raw)` that accept
/// multiple path arguments.
///
/// # Fail-Hard Behavior
///
/// Emits errors for:
/// - Non-path arguments (emits error, continues extracting valid ones)
/// - Returns empty Vec when attribute doesn't exist (legitimate)
///
/// # Parameters
///
/// * `attrs` - Attribute slice to search
/// * `attr_name` - Name of attribute to extract
/// * `context_span` - Span for error context
/// * `errors` - Error accumulator
///
/// # Returns
///
/// Vector of successfully extracted paths. May be empty if attribute absent or all arguments invalid.
///
/// # Examples
///
/// ```rust,ignore
/// let attrs = vec![Attribute {
///     name: "uses".to_string(),
///     args: vec![
///         Expr::signal(Path::from_path_str("maths.clamping"), span),
///         Expr::signal(Path::from_path_str("dt.raw"), span),
///     ],
///     span,
/// }];
///
/// let mut errors = Vec::new();
/// let paths = extract_multiple_paths(&attrs, "uses", span, &mut errors);
/// assert_eq!(paths.len(), 2);
/// assert!(errors.is_empty());
/// ```
pub fn extract_multiple_paths(
    attrs: &[Attribute],
    attr_name: &str,
    context_span: Span,
    errors: &mut Vec<CompileError>,
) -> Vec<Path> {
    let mut paths = Vec::new();

    for attr in attrs.iter().filter(|a| a.name == attr_name) {
        for arg in &attr.args {
            if let Some(path) = extract_path_from_expr(arg, attr_name, context_span, errors) {
                paths.push(path);
            }
        }
    }

    paths
}

/// Returns true if attribute with given name exists in the attribute list.
///
/// Simple existence check with no validation of arguments.
///
/// # Examples
///
/// ```rust,ignore
/// let attrs = vec![Attribute {
///     name: "stratum".to_string(),
///     args: vec![],
///     span,
/// }];
///
/// assert!(has_attribute(&attrs, "stratum"));
/// assert!(!has_attribute(&attrs, "integrator"));
/// ```
pub fn has_attribute(attrs: &[Attribute], attr_name: &str) -> bool {
    attrs.iter().any(|a| a.name == attr_name)
}

/// Extracts topology expression from `:topology(...)` attribute.
///
/// Parses topology declarations like `:topology(icosahedron_grid { subdivisions: 5 })`.
///
/// # Fail-Hard Behavior
///
/// Emits errors for:
/// - Wrong number of arguments (not exactly 1)
/// - Non-struct argument (must be struct literal)
/// - Unknown topology type (not icosahedron_grid)
/// - Missing or invalid `subdivisions` field
/// - Returns `None` only when attribute doesn't exist (legitimate absence)
///
/// # Parameters
///
/// * `attrs` - Attribute slice to search
/// * `context_span` - Span for error context (entity location)
/// * `errors` - Error accumulator
///
/// # Returns
///
/// - `Some(TopologyExpr)`: Successfully parsed topology
/// - `None`: Attribute not found (not an error) OR error emitted
///
/// # Examples
///
/// ```cdsl
/// entity cell {
///     :topology(icosahedron_grid { subdivisions: 5 })
///     member elevation : Scalar<m>
/// }
/// ```
pub fn extract_topology(
    attrs: &[Attribute],
    context_span: Span,
    errors: &mut Vec<CompileError>,
) -> Option<continuum_cdsl_ast::TopologyExpr> {
    use continuum_cdsl_ast::TopologyExpr;

    let attr = attrs.iter().find(|a| a.name == "topology")?;

    // Validate argument count
    if attr.args.len() != 1 {
        errors.push(CompileError::new(
            ErrorKind::Syntax,
            attr.span,
            format!(
                ":topology attribute expects exactly 1 argument (topology expression), got {}",
                attr.args.len()
            ),
        ));
        return None;
    }

    let arg = &attr.args[0];

    // Extract struct expression
    let UntypedKind::Struct {
        ty: type_path,
        fields,
    } = &arg.kind
    else {
        errors.push(CompileError::new(
            ErrorKind::Syntax,
            arg.span,
            format!(
                ":topology argument must be a topology expression (e.g., icosahedron_grid {{ ... }}), found {}",
                describe_expr_kind(&arg.kind)
            ),
        ).with_label(context_span, "in this :topology attribute".to_string()));
        return None;
    };

    // Check topology type
    let type_name = type_path.last()?;
    match type_name.as_ref() {
        "icosahedron_grid" => {
            // Extract subdivisions field
            let subdivisions_field = fields.iter().find(|(name, _)| name == "subdivisions");
            let subdivisions = match subdivisions_field {
                Some((_, expr)) => {
                    // Extract literal value
                    match &expr.kind {
                        UntypedKind::Literal { value, .. } => {
                            let sub = *value as u32;
                            if (*value as f64 - sub as f64).abs() > 1e-10 {
                                errors.push(CompileError::new(
                                    ErrorKind::Syntax,
                                    expr.span,
                                    format!(
                                        "subdivisions must be a non-negative integer, got {}",
                                        value
                                    ),
                                ));
                                return None;
                            }
                            sub
                        }
                        _ => {
                            errors.push(CompileError::new(
                                ErrorKind::Syntax,
                                expr.span,
                                format!(
                                    "subdivisions must be a literal integer, found {}",
                                    describe_expr_kind(&expr.kind)
                                ),
                            ));
                            return None;
                        }
                    }
                }
                None => {
                    errors.push(CompileError::new(
                        ErrorKind::Syntax,
                        arg.span,
                        "icosahedron_grid topology requires 'subdivisions' field".to_string(),
                    ));
                    return None;
                }
            };

            Some(TopologyExpr::IcosahedronGrid {
                subdivisions,
                span: arg.span,
            })
        }
        other => {
            errors.push(CompileError::new(
                ErrorKind::Syntax,
                arg.span,
                format!(
                    "unknown topology type '{}' (supported: icosahedron_grid)",
                    other
                ),
            ));
            None
        }
    }
}

/// Extracts a path from an expression, handling various syntactic forms.
///
/// Handles expressions that can represent paths:
/// - `Local("thermal")` → `Path::from_path_str("thermal")`
/// - `Signal(path)` → `path`
/// - `Field(path)` → `path`
/// - `Config(path)` → `path`
/// - `Const(path)` → `path`
///
/// # Parameters
///
/// * `expr` - Expression to extract path from
/// * `attr_name` - Attribute name for error messages
/// * `context_span` - Context span for error reporting
/// * `errors` - Error accumulator
///
/// # Returns
///
/// - `Some(Path)`: Successfully extracted path
/// - `None`: Expression is not path-like (error emitted)
fn extract_path_from_expr(
    expr: &Expr,
    attr_name: &str,
    context_span: Span,
    errors: &mut Vec<CompileError>,
) -> Option<Path> {
    match &expr.kind {
        UntypedKind::Signal(path)
        | UntypedKind::Field(path)
        | UntypedKind::Config(path)
        | UntypedKind::Const(path) => Some(path.clone()),
        UntypedKind::Local(name) => Some(Path::from_path_str(name)),
        // FieldAccess chains like `maths.clamping` - convert to path via as_path()
        // This handles cases like `: uses(maths.clamping)` where the dotted syntax
        // is parsed as field access but should be treated as a path
        UntypedKind::FieldAccess { .. } => expr.as_path(),
        _ => {
            errors.push(
                CompileError::new(
                    ErrorKind::Syntax,
                    expr.span,
                    format!(
                        ":{} argument must be a path or identifier, found {}",
                        attr_name,
                        describe_expr_kind(&expr.kind)
                    ),
                )
                .with_label(context_span, format!("in this :{} attribute", attr_name)),
            );
            None
        }
    }
}

/// Describes an expression kind for error messages.
///
/// Returns user-friendly descriptions for all expression types.
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
        UntypedKind::Neighbors { .. } => "neighbors query",
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

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl_ast::foundation::Span;

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    #[test]
    fn test_extract_single_path_success() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(UntypedKind::Local("thermal".to_string()), span)],
            span,
        }];

        let mut errors = Vec::new();
        let path = extract_single_path(&attrs, "stratum", span, &mut errors);

        assert_eq!(path, Some(Path::from_path_str("thermal")));
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_single_path_wrong_arg_count() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "stratum".to_string(),
            args: vec![],
            span,
        }];

        let mut errors = Vec::new();
        let path = extract_single_path(&attrs, "stratum", span, &mut errors);

        assert_eq!(path, None);
        assert_eq!(errors.len(), 1);
        assert!(errors[0]
            .message
            .contains(":stratum attribute expects exactly 1 argument"));
    }

    #[test]
    fn test_extract_single_path_wrong_type() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                UntypedKind::Literal {
                    value: 42.0,
                    unit: None,
                },
                span,
            )],
            span,
        }];

        let mut errors = Vec::new();
        let path = extract_single_path(&attrs, "stratum", span, &mut errors);

        assert_eq!(path, None);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("must be a path or identifier"));
        assert!(errors[0].message.contains("literal value"));
    }

    #[test]
    fn test_extract_single_path_missing() {
        let span = test_span();
        let attrs = vec![];

        let mut errors = Vec::new();
        let path = extract_single_path(&attrs, "stratum", span, &mut errors);

        assert_eq!(path, None);
        assert!(errors.is_empty()); // Missing attribute is not an error
    }

    #[test]
    fn test_extract_single_identifier() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "integrator".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("methods.rk4")),
                span,
            )],
            span,
        }];

        let mut errors = Vec::new();
        let method = extract_single_identifier(&attrs, "integrator", span, &mut errors);

        assert_eq!(method, Some("rk4".to_string()));
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_multiple_paths() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "uses".to_string(),
            args: vec![
                Expr::new(
                    UntypedKind::Signal(Path::from_path_str("maths.clamping")),
                    span,
                ),
                Expr::new(UntypedKind::Signal(Path::from_path_str("dt.raw")), span),
            ],
            span,
        }];

        let mut errors = Vec::new();
        let paths = extract_multiple_paths(&attrs, "uses", span, &mut errors);

        assert_eq!(paths.len(), 2);
        assert_eq!(paths[0], Path::from_path_str("maths.clamping"));
        assert_eq!(paths[1], Path::from_path_str("dt.raw"));
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_multiple_paths_with_errors() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "uses".to_string(),
            args: vec![
                Expr::new(
                    UntypedKind::Signal(Path::from_path_str("maths.clamping")),
                    span,
                ),
                Expr::new(
                    UntypedKind::Literal {
                        value: 42.0,
                        unit: None,
                    },
                    span,
                ),
                Expr::new(UntypedKind::Signal(Path::from_path_str("dt.raw")), span),
            ],
            span,
        }];

        let mut errors = Vec::new();
        let paths = extract_multiple_paths(&attrs, "uses", span, &mut errors);

        assert_eq!(paths.len(), 2); // Two valid paths extracted
        assert_eq!(errors.len(), 1); // One error for invalid argument
        assert!(errors[0].message.contains("must be a path or identifier"));
    }

    #[test]
    fn test_has_attribute() {
        let span = test_span();
        let attrs = vec![
            Attribute {
                name: "stratum".to_string(),
                args: vec![],
                span,
            },
            Attribute {
                name: "uses".to_string(),
                args: vec![],
                span,
            },
        ];

        assert!(has_attribute(&attrs, "stratum"));
        assert!(has_attribute(&attrs, "uses"));
        assert!(!has_attribute(&attrs, "integrator"));
    }

    #[test]
    fn test_extract_topology_icosahedron_grid() {
        use continuum_cdsl_ast::TopologyExpr;

        let span = test_span();
        let attrs = vec![Attribute {
            name: "topology".to_string(),
            args: vec![Expr::new(
                UntypedKind::Struct {
                    ty: Path::from_path_str("icosahedron_grid"),
                    fields: vec![(
                        "subdivisions".to_string(),
                        Expr::new(
                            UntypedKind::Literal {
                                value: 5.0,
                                unit: None,
                            },
                            span,
                        ),
                    )],
                },
                span,
            )],
            span,
        }];

        let mut errors = Vec::new();
        let topology = extract_topology(&attrs, span, &mut errors);

        assert!(errors.is_empty(), "Expected no errors, got: {:?}", errors);
        assert!(topology.is_some());

        if let Some(TopologyExpr::IcosahedronGrid { subdivisions, .. }) = topology {
            assert_eq!(subdivisions, 5);
        } else {
            panic!("Expected IcosahedronGrid, got: {:?}", topology);
        }
    }

    #[test]
    fn test_extract_topology_missing_subdivisions() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "topology".to_string(),
            args: vec![Expr::new(
                UntypedKind::Struct {
                    ty: Path::from_path_str("icosahedron_grid"),
                    fields: vec![],
                },
                span,
            )],
            span,
        }];

        let mut errors = Vec::new();
        let topology = extract_topology(&attrs, span, &mut errors);

        assert_eq!(topology, None);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("requires 'subdivisions' field"));
    }

    #[test]
    fn test_extract_topology_invalid_type() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "topology".to_string(),
            args: vec![Expr::new(
                UntypedKind::Struct {
                    ty: Path::from_path_str("cartesian_grid"),
                    fields: vec![],
                },
                span,
            )],
            span,
        }];

        let mut errors = Vec::new();
        let topology = extract_topology(&attrs, span, &mut errors);

        assert_eq!(topology, None);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("unknown topology type"));
        assert!(errors[0].message.contains("cartesian_grid"));
    }

    #[test]
    fn test_extract_topology_not_struct() {
        let span = test_span();
        let attrs = vec![Attribute {
            name: "topology".to_string(),
            args: vec![Expr::new(
                UntypedKind::Local("icosahedron_grid".to_string()),
                span,
            )],
            span,
        }];

        let mut errors = Vec::new();
        let topology = extract_topology(&attrs, span, &mut errors);

        assert_eq!(topology, None);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("must be a topology expression"));
    }
}
