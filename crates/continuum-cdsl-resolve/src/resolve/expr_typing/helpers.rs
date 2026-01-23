//! Helper functions for expression type resolution.
//!
//! This module provides common utilities for error reporting, symbol lookup,
//! and typing of specific expression kinds (literals, structs, calls, etc.)
//! used by the main [`type_expression`] entry point.

use super::context::TypingContext;
use super::derivation::derive_return_type;
use super::type_expression;
use crate::error::{CompileError, ErrorKind};
use crate::resolve::units::resolve_unit_expr;
use continuum_cdsl_ast::foundation::{AggregateOp, KernelType, Path, Shape, Type, Unit};
use continuum_cdsl_ast::{Expr, ExprKind, TypedExpr, UntypedKind};
use std::collections::HashMap;

// === Error Helpers ===

/// Creates an undefined name error.
///
/// # Parameters
/// - `span`: Source location where the undefined name was used.
/// - `name`: The name that was not found.
/// - `kind`: The kind of symbol expected (e.g., "local variable", "signal").
pub fn err_undefined(
    span: continuum_cdsl_ast::foundation::Span,
    name: &str,
    kind: &str,
) -> Vec<CompileError> {
    vec![CompileError::new(
        ErrorKind::UndefinedName,
        span,
        format!("{} '{}' not found", kind, name),
    )]
}

/// Creates a type mismatch error.
///
/// # Parameters
/// - `span`: Source location of the expression with the wrong type.
/// - `expected`: Description of the expected type.
/// - `found`: Description of the actual type found.
pub fn err_type_mismatch(
    span: continuum_cdsl_ast::foundation::Span,
    expected: &str,
    found: &str,
) -> Vec<CompileError> {
    vec![CompileError::new(
        ErrorKind::TypeMismatch,
        span,
        format!("type mismatch: expected {}, found {}", expected, found),
    )]
}

/// Creates an internal compiler error.
///
/// # Parameters
/// - `span`: Source location related to the error.
/// - `message`: Descriptive message about the internal failure.
pub fn err_internal(
    span: continuum_cdsl_ast::foundation::Span,
    message: impl Into<String>,
) -> Vec<CompileError> {
    vec![CompileError::new(ErrorKind::Internal, span, message.into())]
}

// === Lookup Helpers ===

/// Looks up a path in a type registry and returns its type.
///
/// # Parameters
/// - `_ctx`: The typing context (reserved for future use).
/// - `path`: The hierarchical path to look up.
/// - `span`: Source location for error reporting.
/// - `registry_name`: Name of the registry for error messages.
/// - `registry`: The map of paths to types.
///
/// # Errors
/// Returns [`ErrorKind::UndefinedName`] if the path is not found in the registry.
pub fn lookup_path_type(
    _ctx: &TypingContext,
    path: &Path,
    span: continuum_cdsl_ast::foundation::Span,
    registry_name: &str,
    registry: &HashMap<Path, Type>,
) -> Result<Type, Vec<CompileError>> {
    registry
        .get(path)
        .cloned()
        .ok_or_else(|| err_undefined(span, &path.to_string(), registry_name))
}

/// Ensures a context-specific type is available and returns it.
///
/// # Parameters
/// - `span`: Source location for error reporting.
/// - `name`: Name of the context value (e.g., "prev", "self").
/// - `ty`: The optional type from the context.
///
/// # Errors
/// Returns [`ErrorKind::Internal`] if the type is not available in the current context.
pub fn require_context_type(
    span: continuum_cdsl_ast::foundation::Span,
    name: &str,
    ty: &Option<Type>,
) -> Result<Type, Vec<CompileError>> {
    ty.clone().ok_or_else(|| {
        err_internal(
            span,
            format!("{} not available: no context type provided", name),
        )
    })
}

// === Typing Helpers ===

/// Resolves the type of a numeric literal with optional unit syntax.
///
/// # Parameters
/// - `span`: Source location of the literal.
/// - `value`: The numeric value of the literal.
/// - `unit`: Optional unit expression syntax from the parser.
///
/// # Returns
/// A tuple containing the typed expression kind and the resolved [`Type`].
pub fn type_literal(
    span: continuum_cdsl_ast::foundation::Span,
    value: f64,
    unit: Option<&continuum_cdsl_ast::UnitExpr>,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let resolved_unit = match unit {
        Some(unit_expr) => resolve_unit_expr(Some(unit_expr), span).map_err(|e| vec![e])?,
        None => Unit::DIMENSIONLESS,
    };

    let kernel_type = KernelType {
        shape: Shape::Scalar,
        unit: resolved_unit,
        bounds: None,
    };

    Ok((
        ExprKind::Literal {
            value,
            unit: Some(resolved_unit),
        },
        Type::Kernel(kernel_type),
    ))
}

/// Resolves a field access expression (e.g., `obj.field`).
///
/// Supports accessing fields on user-defined structs and component access on vectors.
///
/// # Parameters
/// - `ctx`: The typing context.
/// - `object`: The expression being accessed.
/// - `field`: The name of the field to access.
/// - `span`: Source location of the access.
/// Attempts to interpret an expression as a bare path reference.
///
/// Attempts to extract a dot-separated path from an expression chain for bare path resolution.
///
/// Uses the canonical [`Expr::as_path()`] method from the AST, but filters out keyword expressions
/// (prev, current, dt, etc.) since those should not be treated as bare signal/field paths.
///
/// This function enables bare path resolution by checking if an expression like `foo.bar.baz`
/// could be a signal or field path before typing it as nested field accesses.
///
/// # Arguments
///
/// * `expr` - The expression to attempt path extraction from
///
/// # Returns
///
/// * `Some(Path)` - If the expression is a pure chain of `Local`/`FieldAccess` starting with a Local
/// * `None` - If the expression is a keyword, literal, function call, or other non-path expression
///
/// # Examples
///
/// ```text
/// Local("foo")                              → Some(Path("foo"))
/// FieldAccess { Local("foo"), "bar" }       → Some(Path("foo.bar"))
/// FieldAccess { FieldAccess {...}, "baz" }  → Some(Path("foo.bar.baz"))
/// Prev                                      → None (keyword, not a bare path)
/// Literal { value: 42.0 }                   → None (not a path)
/// FunctionCall { ... }                      → None (not a path)
/// ```
///
/// # Used For
///
/// This enables the bare signal/field reference feature, allowing DSL code to write:
/// ```cdsl
/// core.temp            // Instead of signal.core.temp
/// atmosphere.pressure  // Instead of signal.atmosphere.pressure
/// ```
fn try_extract_path(expr: &Expr) -> Option<Path> {
    use continuum_cdsl_ast::UntypedKind;

    // Use the canonical as_path() from AST
    let path = expr.as_path()?;

    // Filter out keyword expressions - they should not be treated as bare paths
    // Only accept paths that start with a Local identifier
    match &expr.kind {
        UntypedKind::Local(_) => Some(path),
        UntypedKind::FieldAccess { object, .. } => {
            // Check if the root is a Local (not a keyword)
            if matches!(get_root_kind(object), UntypedKind::Local(_)) {
                Some(path)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Helper to get the root expression kind from a FieldAccess chain.
fn get_root_kind(expr: &Expr) -> &UntypedKind {
    match &expr.kind {
        UntypedKind::FieldAccess { object, .. } => get_root_kind(object),
        other => other,
    }
}

pub fn type_field_access(
    ctx: &TypingContext,
    object: &Expr,
    field: &str,
    span: continuum_cdsl_ast::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    // Before typing as a field access, check if this might be a bare signal reference.
    // This allows `core.temp` to resolve as `signal.core.temp` when `core` is not a local variable.
    if let Some(mut path) = try_extract_path(object) {
        path.segments.push(field.to_string());

        // Check if this path matches a signal
        if let Some(signal_type) = ctx.signal_types.get(&path) {
            return Ok((ExprKind::Signal(path), signal_type.clone()));
        }

        // Also check field types (for bare field references)
        if let Some(field_type) = ctx.field_types.get(&path) {
            return Ok((ExprKind::Field(path), field_type.clone()));
        }

        // Path doesn't match a signal or field, continue with normal field access typing
    }

    let typed_object = type_expression(object, ctx)?;

    let field_type = match &typed_object.ty {
        Type::User(type_id) => {
            let user_type = ctx.type_table.get_by_id(type_id).ok_or_else(|| {
                err_internal(
                    span,
                    format!("user type {:?} not found in type table", type_id),
                )
            })?;

            user_type.field(field).cloned().ok_or_else(|| {
                err_undefined(
                    span,
                    field,
                    &format!("field on type '{}'", user_type.name()),
                )
            })?
        }
        Type::Kernel(kt) => match &kt.shape {
            Shape::Vector { dim } => {
                let component_index = match field {
                    "x" => Some(0),
                    "y" => Some(1),
                    "z" => Some(2),
                    "w" => Some(3),
                    _ => None,
                };

                match component_index {
                    Some(idx) if idx < *dim as usize => Type::Kernel(KernelType {
                        shape: Shape::Scalar,
                        unit: kt.unit,
                        bounds: None,
                    }),
                    Some(_) => {
                        return Err(err_undefined(
                            span,
                            field,
                            &format!("component out of bounds for vector of dimension {}", dim),
                        ));
                    }
                    None => {
                        return Err(err_undefined(span, field, "invalid vector component"));
                    }
                }
            }
            _ => {
                return Err(err_internal(
                    span,
                    "field access on non-struct, non-vector type",
                ));
            }
        },
        _ => {
            return Err(err_internal(
                span,
                format!("field access not supported on type {:?}", typed_object.ty),
            ));
        }
    };

    Ok((
        ExprKind::FieldAccess {
            object: Box::new(typed_object),
            field: field.to_string(),
        },
        field_type,
    ))
}

/// Resolves the type of a vector literal (e.g., `[1.0, 2.0, 3.0]`).
///
/// All elements must have consistent scalar kernel types with matching units.
///
/// # Parameters
/// - `ctx`: The typing context.
/// - `elements`: The list of component expressions.
/// - `span`: Source location of the literal.
pub fn type_vector(
    ctx: &TypingContext,
    elements: &[Expr],
    span: continuum_cdsl_ast::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    if elements.is_empty() {
        return Err(vec![CompileError::new(
            ErrorKind::TypeMismatch,
            span,
            "vector literal cannot be empty".to_string(),
        )]);
    }

    if elements.len() > 4 {
        return Err(vec![CompileError::new(
            ErrorKind::TypeMismatch,
            span,
            format!(
                "vector literal has {} elements, maximum is 4",
                elements.len()
            ),
        )]);
    }

    let typed_elements: Result<Vec<_>, _> = elements
        .iter()
        .map(|elem| type_expression(elem, ctx))
        .collect();
    let typed_elements = typed_elements?;

    let mut unit = None;
    for (i, elem) in typed_elements.iter().enumerate() {
        match &elem.ty {
            Type::Kernel(kt) => {
                if kt.shape != Shape::Scalar {
                    return Err(vec![CompileError::new(
                        ErrorKind::TypeMismatch,
                        elem.span,
                        format!(
                            "vector element {} has shape {:?}, expected Scalar",
                            i, kt.shape
                        ),
                    )]);
                }

                if let Some(ref expected_unit) = unit {
                    if kt.unit != *expected_unit {
                        return Err(vec![CompileError::new(
                            ErrorKind::TypeMismatch,
                            elem.span,
                            format!(
                                "vector element {} has unit {}, expected {}",
                                i, kt.unit, expected_unit
                            ),
                        )]);
                    }
                } else {
                    unit = Some(kt.unit);
                }
            }
            _ => {
                return Err(vec![CompileError::new(
                    ErrorKind::TypeMismatch,
                    elem.span,
                    format!("vector element {} has non-kernel type {:?}", i, elem.ty),
                )]);
            }
        }
    }

    let dim = elements.len() as u8;
    let unit = unit.ok_or_else(|| err_internal(span, "vector literal unit resolution failed"))?;

    Ok((
        ExprKind::Vector(typed_elements),
        Type::Kernel(KernelType {
            shape: Shape::Vector { dim },
            unit,
            bounds: None,
        }),
    ))
}

/// Resolves the type of a let binding expression.
///
/// # Parameters
/// - `ctx`: The current typing context.
/// - `name`: The name of the variable being bound.
/// - `value`: The expression whose value is being bound.
/// - `body`: The expression in which the binding is in scope.
/// - `_span`: Source location (unused).
pub fn type_let(
    ctx: &TypingContext,
    name: &str,
    value: &Expr,
    body: &Expr,
    _span: continuum_cdsl_ast::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_value = type_expression(value, ctx)?;
    let extended_ctx = ctx.with_binding(name.to_string(), typed_value.ty.clone());
    let typed_body = type_expression(body, &extended_ctx)?;
    let ty = typed_body.ty.clone();

    Ok((
        ExprKind::Let {
            name: name.to_string(),
            value: Box::new(typed_value),
            body: Box::new(typed_body),
        },
        ty,
    ))
}

/// Resolves the type of a struct construction expression.
///
/// # Parameters
/// - `ctx`: The typing context.
/// - `ty_path`: Path to the struct type definition.
/// - `fields`: List of field names and their initialization expressions.
/// - `span`: Source location of the construction.
pub fn type_struct(
    ctx: &TypingContext,
    ty_path: &Path,
    fields: &[(String, Expr)],
    span: continuum_cdsl_ast::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let type_id = ctx
        .type_table
        .get_id(ty_path)
        .ok_or_else(|| err_undefined(span, &ty_path.to_string(), "type"))?
        .clone();

    let user_type = ctx
        .type_table
        .get(ty_path)
        .ok_or_else(|| err_internal(span, format!("type '{}' not found in table", ty_path)))?;

    let mut typed_fields = Vec::new();
    let mut seen_fields = std::collections::HashSet::new();

    for (field_name, field_expr) in fields {
        if !seen_fields.insert(field_name.clone()) {
            return Err(vec![CompileError::new(
                ErrorKind::TypeMismatch,
                field_expr.span,
                format!("field '{}' specified multiple times", field_name),
            )]);
        }

        let typed_expr = type_expression(field_expr, ctx)?;
        let expected_type = user_type.field(field_name).ok_or_else(|| {
            err_undefined(
                field_expr.span,
                field_name,
                &format!("field on type '{}'", ty_path),
            )
        })?;

        if &typed_expr.ty != expected_type {
            return Err(vec![CompileError::new(
                ErrorKind::TypeMismatch,
                field_expr.span,
                format!(
                    "field '{}' has type {:?}, expected {:?}",
                    field_name, typed_expr.ty, expected_type
                ),
            )]);
        }

        typed_fields.push((field_name.clone(), typed_expr));
    }

    for (declared_field, _) in user_type.fields() {
        if !fields.iter().any(|(name, _)| name == declared_field) {
            return Err(vec![CompileError::new(
                ErrorKind::TypeMismatch,
                span,
                format!("missing field '{}'", declared_field),
            )]);
        }
    }

    Ok((
        ExprKind::Struct {
            ty: type_id.clone(),
            fields: typed_fields,
        },
        Type::User(type_id),
    ))
}

/// Resolves the type of an aggregate operation over an entity (e.g., `sum`, `count`).
///
/// # Parameters
/// - `ctx`: The typing context.
/// - `op`: The aggregate operation being performed.
/// - `source`: The expression producing the sequence to iterate.
/// - `binding`: Binding name for the instance in the body.
/// - `body`: Expression evaluated for each instance.
/// - `span`: Source location.
pub fn type_aggregate(
    ctx: &TypingContext,
    op: &AggregateOp,
    source: &Expr,
    binding: &str,
    body: &Expr,
    span: continuum_cdsl_ast::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_source = type_expression(source, ctx)?;

    let element_ty = match &typed_source.ty {
        Type::Seq(inner) => *inner.clone(),
        _ => {
            return Err(err_type_mismatch(
                source.span,
                "Seq<T>",
                &format!("{:?}", typed_source.ty),
            ));
        }
    };

    let extended_ctx = ctx.with_binding(binding.to_string(), element_ty);
    let typed_body = type_expression(body, &extended_ctx)?;

    let aggregate_ty = match op {
        AggregateOp::Map => Type::Seq(Box::new(typed_body.ty.clone())),
        AggregateOp::Sum | AggregateOp::Product | AggregateOp::Mean => typed_body.ty.clone(),
        AggregateOp::Max | AggregateOp::Min => typed_body.ty.clone(),
        AggregateOp::Count => Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        }),
        AggregateOp::Any | AggregateOp::All | AggregateOp::None => Type::Bool,
        AggregateOp::First => typed_body.ty.clone(),
    };

    Ok((
        ExprKind::Aggregate {
            op: *op,
            source: Box::new(typed_source),
            binding: binding.to_string(),
            body: Box::new(typed_body),
        },
        aggregate_ty,
    ))
}

/// Resolves the type of a custom fold/reduction over an entity.
///
/// # Parameters
/// - `ctx`: The typing context.
/// - `source`: The expression producing the sequence to iterate.
/// - `init`: Initial value for the accumulator.
/// - `acc`: Binding name for the accumulator in the body.
/// - `elem`: Binding name for the entity instance in the body.
/// - `body`: The reduction expression.
/// - `span`: Source location for error reporting.
pub fn type_fold(
    ctx: &TypingContext,
    source: &Expr,
    init: &Expr,
    acc: &str,
    elem: &str,
    body: &Expr,
    span: continuum_cdsl_ast::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_source = type_expression(source, ctx)?;
    let typed_init = type_expression(init, ctx)?;

    let element_ty = match &typed_source.ty {
        Type::Seq(inner) => *inner.clone(),
        _ => {
            return Err(err_type_mismatch(
                source.span,
                "Seq<T>",
                &format!("{:?}", typed_source.ty),
            ));
        }
    };

    let mut extended_ctx = ctx.with_binding(acc.to_string(), typed_init.ty.clone());
    extended_ctx
        .local_bindings
        .insert(elem.to_string(), element_ty);

    let typed_body = type_expression(body, &extended_ctx)?;

    if typed_body.ty != typed_init.ty {
        return Err(vec![CompileError::new(
            ErrorKind::TypeMismatch,
            span,
            format!(
                "fold body type {:?} does not match accumulator type {:?}",
                typed_body.ty, typed_init.ty
            ),
        )]);
    }

    Ok((
        ExprKind::Fold {
            source: Box::new(typed_source),
            init: Box::new(typed_init),
            acc: acc.to_string(),
            elem: elem.to_string(),
            body: Box::new(typed_body.clone()),
        },
        typed_body.ty,
    ))
}

/// Resolves the type of a function or kernel call (e.g., `sin(x)`).
///
/// # Parameters
/// - `ctx`: The typing context.
/// - `func`: Path to the function or kernel.
/// - `args`: Argument expressions.
/// - `span`: Source location of the call.
pub fn type_call(
    ctx: &TypingContext,
    func: &Path,
    args: &[Expr],
    span: continuum_cdsl_ast::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_args: Vec<TypedExpr> = args
        .iter()
        .map(|arg| type_expression(arg, ctx))
        .collect::<Result<Vec<_>, _>>()?;

    let segments = func.segments();
    if segments.is_empty() {
        return Err(err_internal(span, "kernel path is empty"));
    }

    if segments.len() > 2 {
        return Err(vec![CompileError::new(
            ErrorKind::Syntax,
            span,
            format!("kernel path '{}' must be namespace.name or bare name", func),
        )]);
    }

    let (namespace, name) = if segments.len() == 1 {
        ("", segments[0].as_str())
    } else {
        (segments[0].as_str(), segments[1].as_str())
    };

    let sig = ctx
        .kernel_registry
        .get_by_name(namespace, name)
        .ok_or_else(|| err_undefined(span, &func.to_string(), "kernel"))?;

    let kernel_id = sig.id.clone();
    let return_type = derive_return_type(sig, &typed_args, span)?;

    Ok((
        ExprKind::Call {
            kernel: kernel_id,
            args: typed_args,
        },
        return_type,
    ))
}

/// Resolves the type of a call to a specific kernel ID (desugared from operators).
///
/// # Parameters
/// - `ctx`: The typing context.
/// - `kernel`: The specific kernel ID being called.
/// - `args`: Argument expressions.
/// - `span`: Source location of the call.
pub fn type_as_kernel_call(
    ctx: &TypingContext,
    kernel: &continuum_kernel_types::KernelId,
    args: &[Expr],
    span: continuum_cdsl_ast::foundation::Span,
) -> Result<(ExprKind, Type), Vec<CompileError>> {
    let typed_args: Vec<TypedExpr> = args
        .iter()
        .map(|arg| type_expression(arg, ctx))
        .collect::<Result<Vec<_>, _>>()?;

    let sig = ctx
        .kernel_registry
        .get(kernel)
        .ok_or_else(|| err_internal(span, format!("unknown kernel: {:?}", kernel)))?;

    let return_type = derive_return_type(sig, &typed_args, span)?;

    Ok((
        ExprKind::Call {
            kernel: kernel.clone(),
            args: typed_args,
        },
        return_type,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl_ast::foundation::{Path, Span};
    use continuum_cdsl_ast::{Expr, UntypedKind};

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    fn make_local(name: &str) -> Expr {
        Expr::new(UntypedKind::Local(name.to_string()), test_span())
    }

    fn make_field_access(object: Expr, field: &str) -> Expr {
        Expr::new(
            UntypedKind::FieldAccess {
                object: Box::new(object),
                field: field.to_string(),
            },
            test_span(),
        )
    }

    fn make_keyword(kind: UntypedKind) -> Expr {
        Expr::new(kind, test_span())
    }

    #[test]
    fn test_try_extract_path_local() {
        let expr = make_local("foo");
        let path = try_extract_path(&expr);
        assert_eq!(path, Some(Path::from_path_str("foo")));
    }

    #[test]
    fn test_try_extract_path_field_access_simple() {
        let expr = make_field_access(make_local("foo"), "bar");
        let path = try_extract_path(&expr);
        assert_eq!(path, Some(Path::from_path_str("foo.bar")));
    }

    #[test]
    fn test_try_extract_path_field_access_nested() {
        let expr = make_field_access(
            make_field_access(make_local("foo"), "bar"),
            "baz",
        );
        let path = try_extract_path(&expr);
        assert_eq!(path, Some(Path::from_path_str("foo.bar.baz")));
    }

    #[test]
    fn test_try_extract_path_keyword_prev() {
        let expr = make_keyword(UntypedKind::Prev);
        let path = try_extract_path(&expr);
        assert_eq!(path, None, "Keywords should not be treated as bare paths");
    }

    #[test]
    fn test_try_extract_path_keyword_current() {
        let expr = make_keyword(UntypedKind::Current);
        let path = try_extract_path(&expr);
        assert_eq!(path, None);
    }

    #[test]
    fn test_try_extract_path_keyword_dt() {
        let expr = make_keyword(UntypedKind::Dt);
        let path = try_extract_path(&expr);
        assert_eq!(path, None);
    }

    #[test]
    fn test_try_extract_path_keyword_collected() {
        let expr = make_keyword(UntypedKind::Collected);
        let path = try_extract_path(&expr);
        assert_eq!(path, None);
    }

    #[test]
    fn test_try_extract_path_keyword_field_access() {
        // prev.temperature should not be treated as bare path
        let expr = make_field_access(make_keyword(UntypedKind::Prev), "temperature");
        let path = try_extract_path(&expr);
        assert_eq!(path, None, "Field access on keyword should not be bare path");
    }

    #[test]
    fn test_try_extract_path_literal() {
        let expr = Expr::new(
            UntypedKind::Literal {
                value: 42.0,
                unit: None,
            },
            test_span(),
        );
        let path = try_extract_path(&expr);
        assert_eq!(path, None, "Literals are not paths");
    }

    #[test]
    fn test_try_extract_path_bool_literal() {
        let expr = Expr::new(UntypedKind::BoolLiteral(true), test_span());
        let path = try_extract_path(&expr);
        assert_eq!(path, None);
    }

    #[test]

    #[test]
    fn test_get_root_kind_local() {
        let expr = make_local("foo");
        let root = get_root_kind(&expr);
        assert!(matches!(root, UntypedKind::Local(_)));
    }

    #[test]
    fn test_get_root_kind_field_access() {
        let expr = make_field_access(make_local("foo"), "bar");
        let root = get_root_kind(&expr);
        assert!(matches!(root, UntypedKind::Local(_)));
    }

    #[test]
    fn test_get_root_kind_nested_field_access() {
        let expr = make_field_access(
            make_field_access(make_local("foo"), "bar"),
            "baz",
        );
        let root = get_root_kind(&expr);
        assert!(matches!(root, UntypedKind::Local(_)));
    }

    #[test]
    fn test_get_root_kind_keyword_field_access() {
        let expr = make_field_access(make_keyword(UntypedKind::Prev), "temp");
        let root = get_root_kind(&expr);
        assert!(matches!(root, UntypedKind::Prev));
    }
}
