//! uses() declaration validation
//!
//! Validates that dangerous kernel functions and capabilities have required
//! `: uses()` declarations. Enforces Principle 7: Fail Hard, Never Mask Errors.
//!
//! # Dangerous Categories
//!
//! - `maths.clamping` - clamp, saturate, wrap (silent error masking)
//! - `dt.raw` - raw timestep access (dt-fragile code)
//!
//! # What This Pass Does
//!
//! 1. **Extract declared uses** from `: uses(...)` attributes on nodes
//! 2. **Walk all execution blocks** to find dangerous kernel calls and dt usage
//! 3. **Validate** that all required uses are declared
//! 4. **Emit errors** with helpful hints for missing declarations
//!
//! # Pipeline Position
//!
//! ```text
//! ... → Era Resolution → Uses Validation → Execution Block Compilation → DAG
//!                           ^^^^^^^^^^^
//!                          YOU ARE HERE
//! ```
//!
//! This pass runs after type resolution (so we have TypedExpr) and before
//! execution DAG construction.
//!
//! Validates both compiled execution blocks (TypedExpr) and untyped blocks
//! (warmup, when, observe) using parallel traversal logic.
//!
//! # Examples
//!
//! ```cdsl
//! signal albedo : Scalar<1> {
//!     : uses(maths.clamping)  // Required!
//!     
//!     resolve {
//!         maths.clamp(prev + delta, 0.0, 1.0)  // OK - declared
//!     }
//! }
//!
//! signal bad_albedo : Scalar<1> {
//!     resolve {
//!         maths.clamp(prev + delta, 0.0, 1.0)  // ERROR - missing uses
//!     }
//! }
//! ```

use crate::ast::{Attribute, ExprKind, Index, KernelRegistry, Node, TypedExpr};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{Path, Span};
use std::collections::HashSet;

/// Information about a required uses declaration
#[derive(Debug, Clone)]
struct RequiredUse {
    /// The uses key (e.g., "maths.clamping", "dt.raw")
    key: String,
    /// Source span where the dangerous usage occurs
    span: Span,
    /// The function/feature name that triggered this
    source: String,
    /// Hint message for the error
    hint: String,
}

/// Extract uses declarations from node attributes
///
/// Parses `: uses(maths.clamping, dt.raw)` into a set of keys.
/// Emits errors for invalid arguments that cannot be parsed.
///
/// # Example
///
/// ```rust,ignore
/// // : uses(maths.clamping, dt.raw)
/// let attrs = vec![
///     Attribute {
///         name: "uses".to_string(),
///         args: vec![
///             Expr::Signal(Path::from_str("maths.clamping")),
///             Expr::Signal(Path::from_str("dt.raw")),
///         ],
///         span,
///     },
/// ];
/// let mut errors = Vec::new();
/// let uses = extract_uses_declarations(&attrs, &mut errors);
/// assert!(uses.contains("maths.clamping"));
/// assert!(uses.contains("dt.raw"));
/// assert!(errors.is_empty());
/// ```
fn extract_uses_declarations(
    attrs: &[Attribute],
    errors: &mut Vec<CompileError>,
) -> HashSet<String> {
    let mut uses = HashSet::new();

    for attr in attrs {
        if attr.name == "uses" {
            // Each arg should be a path like maths.clamping or dt.raw
            for arg in &attr.args {
                match extract_uses_key_from_expr(arg) {
                    Some(key) => {
                        uses.insert(key);
                    }
                    None => {
                        // Invalid argument - emit error
                        errors.push(CompileError::new(
                            ErrorKind::Internal,
                            arg.span,
                            format!(
                                "invalid uses() argument: expected path like 'maths.clamping' or 'dt.raw', got {:?}",
                                arg.kind
                            ),
                        ));
                    }
                }
            }
        }
    }

    uses
}

/// Extract uses key from attribute argument expression
///
/// Handles various expression forms that might represent a uses key:
/// - Signal/Config/Const path expressions
/// - Direct path strings
fn extract_uses_key_from_expr(expr: &crate::ast::Expr) -> Option<String> {
    use crate::ast::UntypedKind;

    match &expr.kind {
        // : uses(maths.clamping) - parsed as Signal path
        UntypedKind::Signal(path) => Some(path.to_string()),
        // : uses(dt.raw) - might be parsed as Config
        UntypedKind::Config(path) => Some(path.to_string()),
        // Fall back to const
        UntypedKind::Const(path) => Some(path.to_string()),
        // Field access might be used
        UntypedKind::Field(path) => Some(path.to_string()),
        _ => None,
    }
}

/// Walk typed expression tree collecting required uses from kernel calls and dt usage
///
/// Recursively scans the expression tree for:
/// - Kernel calls with `requires_uses` in their signature
/// - Raw `dt` access (ExprKind::Dt)
fn collect_required_uses(
    expr: &TypedExpr,
    registry: &KernelRegistry,
    required: &mut Vec<RequiredUse>,
) {
    match &expr.expr {
        // Kernel call - check if it requires uses
        ExprKind::Call { kernel, args } => {
            // Look up kernel signature in registry
            if let Some(signature) = registry.get(kernel) {
                if let Some(req) = &signature.requires_uses {
                    required.push(RequiredUse {
                        key: format!("{}.{}", signature.id.namespace, req.key),
                        span: expr.span,
                        source: signature.id.qualified_name(),
                        hint: req.hint.clone(),
                    });
                }
            }

            // Recurse into arguments
            for arg in args {
                collect_required_uses(arg, registry, required);
            }
        }

        // Raw dt access - requires dt.raw declaration
        ExprKind::Dt => {
            required.push(RequiredUse {
                key: "dt.raw".to_string(),
                span: expr.span,
                source: "dt".to_string(),
                hint: "Raw dt access makes code dt-fragile. Use dt-robust operators (dt.integrate, dt.decay, dt.relax) instead. If raw dt is physically correct (e.g., Energy = Power × dt), declare : uses(dt.raw)".to_string(),
            });
        }

        // Recurse into subexpressions
        ExprKind::Let { value, body, .. } => {
            collect_required_uses(value, registry, required);
            collect_required_uses(body, registry, required);
        }

        ExprKind::Struct { fields, .. } => {
            for (_, field_expr) in fields {
                collect_required_uses(field_expr, registry, required);
            }
        }

        ExprKind::FieldAccess { object, .. } => {
            collect_required_uses(object, registry, required);
        }

        ExprKind::Vector(elements) => {
            for elem in elements {
                collect_required_uses(elem, registry, required);
            }
        }

        ExprKind::Aggregate { body, .. } => {
            collect_required_uses(body, registry, required);
        }

        ExprKind::Fold { init, body, .. } => {
            collect_required_uses(init, registry, required);
            collect_required_uses(body, registry, required);
        }

        // Leaf nodes - no recursion needed
        ExprKind::Literal { .. }
        | ExprKind::Signal(_)
        | ExprKind::Field(_)
        | ExprKind::Config(_)
        | ExprKind::Const(_)
        | ExprKind::Local(_)
        | ExprKind::Prev
        | ExprKind::Current
        | ExprKind::Inputs
        | ExprKind::Self_
        | ExprKind::Payload
        | ExprKind::Other => {}
    }
}

/// Walk untyped expression tree collecting required uses from kernel calls and dt usage
///
/// Similar to `collect_required_uses` but works on untyped `Expr` from parser.
/// Used for warmup, when, and observe blocks that haven't been compiled to TypedExpr yet.
///
/// Recursively scans the expression tree for:
/// - Explicit kernel calls (Call with namespaced function)
/// - KernelCall nodes (desugared operators)
/// - Raw `dt` access
fn collect_required_uses_untyped(
    expr: &crate::ast::Expr,
    registry: &KernelRegistry,
    required: &mut Vec<RequiredUse>,
) {
    use crate::ast::UntypedKind;

    match &expr.kind {
        // Explicit call - might be a kernel call like maths.clamp(...)
        UntypedKind::Call { func, args } => {
            // Check if this is a namespaced kernel call
            let path_str = func.to_string();
            if let Some((namespace, name)) = path_str.split_once('.') {
                // Need to iterate through registry to find matching kernel
                // (can't use KernelId::new since it requires &'static str)
                for kernel_id in registry.ids() {
                    if kernel_id.namespace == namespace && kernel_id.name == name {
                        if let Some(signature) = registry.get(kernel_id) {
                            if let Some(req) = &signature.requires_uses {
                                required.push(RequiredUse {
                                    key: format!("{}.{}", signature.id.namespace, req.key),
                                    span: expr.span,
                                    source: signature.id.qualified_name(),
                                    hint: req.hint.clone(),
                                });
                            }
                        }
                        break;
                    }
                }
            }

            // Recurse into arguments
            for arg in args {
                collect_required_uses_untyped(arg, registry, required);
            }
        }

        // KernelCall - desugared operators (Binary/Unary → KernelCall)
        UntypedKind::KernelCall { kernel, args } => {
            if let Some(signature) = registry.get(kernel) {
                if let Some(req) = &signature.requires_uses {
                    required.push(RequiredUse {
                        key: format!("{}.{}", signature.id.namespace, req.key),
                        span: expr.span,
                        source: signature.id.qualified_name(),
                        hint: req.hint.clone(),
                    });
                }
            }

            // Recurse into arguments
            for arg in args {
                collect_required_uses_untyped(arg, registry, required);
            }
        }

        // Binary/Unary operators - these desugar to kernel calls
        // We need to check them even though they're not yet desugared
        UntypedKind::Binary { left, right, .. } => {
            collect_required_uses_untyped(left, registry, required);
            collect_required_uses_untyped(right, registry, required);
        }

        UntypedKind::Unary { operand, .. } => {
            collect_required_uses_untyped(operand, registry, required);
        }

        // Raw dt access - requires dt.raw declaration
        UntypedKind::Dt => {
            required.push(RequiredUse {
                key: "dt.raw".to_string(),
                span: expr.span,
                source: "dt".to_string(),
                hint: "Raw dt access makes code dt-fragile. Use dt-robust operators (dt.integrate, dt.decay, dt.relax) instead. If raw dt is physically correct (e.g., Energy = Power × dt), declare : uses(dt.raw)".to_string(),
            });
        }

        // Recurse into subexpressions
        UntypedKind::Let { value, body, .. } => {
            collect_required_uses_untyped(value, registry, required);
            collect_required_uses_untyped(body, registry, required);
        }

        UntypedKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            collect_required_uses_untyped(condition, registry, required);
            collect_required_uses_untyped(then_branch, registry, required);
            collect_required_uses_untyped(else_branch, registry, required);
        }

        UntypedKind::Struct { fields, .. } => {
            for (_, field_expr) in fields {
                collect_required_uses_untyped(field_expr, registry, required);
            }
        }

        UntypedKind::FieldAccess { object, .. } => {
            collect_required_uses_untyped(object, registry, required);
        }

        UntypedKind::Vector(elements) => {
            for elem in elements {
                collect_required_uses_untyped(elem, registry, required);
            }
        }

        UntypedKind::Aggregate { body, .. } => {
            collect_required_uses_untyped(body, registry, required);
        }

        UntypedKind::Fold { init, body, .. } => {
            collect_required_uses_untyped(init, registry, required);
            collect_required_uses_untyped(body, registry, required);
        }

        // Leaf nodes - no recursion needed
        UntypedKind::Literal { .. }
        | UntypedKind::BoolLiteral(_)
        | UntypedKind::Signal(_)
        | UntypedKind::Field(_)
        | UntypedKind::Config(_)
        | UntypedKind::Const(_)
        | UntypedKind::Local(_)
        | UntypedKind::Prev
        | UntypedKind::Current
        | UntypedKind::Inputs
        | UntypedKind::Self_
        | UntypedKind::Other
        | UntypedKind::Payload
        | UntypedKind::ParseError(_) => {}
    }
}

/// Validate uses declarations for a single node
///
/// Checks all execution blocks (resolve, collect, warmup, when, observe, assertions)
/// for dangerous function usage and validates against declared uses.
fn validate_node_uses<I: Index>(node: &Node<I>, registry: &KernelRegistry) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Extract declared uses from attributes (emits errors for invalid arguments)
    let declared = extract_uses_declarations(&node.attributes, &mut errors);

    // Collect required uses from all execution-related blocks
    let mut required = Vec::new();

    // 1. Execution blocks (resolve, collect, measure, etc.)
    //    These are compiled TypedExpr after Phase 12.5-D
    for execution in &node.executions {
        collect_required_uses(&execution.body, registry, &mut required);
    }

    // 2. Warmup block (iterate expression)
    //    Contains untyped Expr - use untyped traversal
    if let Some(warmup) = &node.warmup {
        collect_required_uses_untyped(&warmup.iterate, registry, &mut required);
    }

    // 3. When block (condition expressions for fractures)
    //    Contains untyped Expr - use untyped traversal
    if let Some(when) = &node.when {
        for condition in &when.conditions {
            collect_required_uses_untyped(condition, registry, &mut required);
        }
    }

    // 4. Observe block (when clauses for chronicles)
    //    Contains untyped Expr in conditions - use untyped traversal
    if let Some(observe) = &node.observe {
        for when_clause in &observe.when_clauses {
            collect_required_uses_untyped(&when_clause.condition, registry, &mut required);
            // Note: emit_block contains Stmt, which we don't validate yet
            // (would need to extract expressions from statements)
        }
    }

    // Validate: for each required use, check if it's declared
    for req in required {
        if !declared.contains(&req.key) {
            errors.push(CompileError::new(
                ErrorKind::MissingUsesDeclaration,
                req.span,
                format!(
                    "function '{}' requires : uses({}) declaration. {}",
                    req.source, req.key, req.hint
                ),
            ));
        }
    }

    errors
}

/// Validate uses declarations for all nodes
///
/// Entry point for the validation pass. Call this with all nodes after type
/// resolution.
///
/// # Examples
///
/// ```rust,ignore
/// let errors = validate_uses(&nodes, &kernel_registry);
/// if !errors.is_empty() {
///     // Report compilation errors
/// }
/// ```
pub fn validate_uses<I: Index>(nodes: &[Node<I>], registry: &KernelRegistry) -> Vec<CompileError> {
    let mut all_errors = Vec::new();

    for node in nodes {
        let errors = validate_node_uses(node, registry);
        all_errors.extend(errors);
    }

    all_errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Execution, RoleData};
    use crate::foundation::{Phase, Type};

    fn make_span() -> Span {
        Span::new(0, 0, 10, 1)
    }

    fn make_attr_uses(keys: Vec<&str>) -> Attribute {
        use crate::ast::{Expr, UntypedKind};

        Attribute {
            name: "uses".to_string(),
            args: keys
                .iter()
                .map(|k| Expr::new(UntypedKind::Signal(Path::from_str(k)), make_span()))
                .collect(),
            span: make_span(),
        }
    }

    #[test]
    fn test_extract_uses_declarations_single() {
        let attrs = vec![make_attr_uses(vec!["maths.clamping"])];
        let mut errors = Vec::new();
        let uses = extract_uses_declarations(&attrs, &mut errors);

        assert_eq!(uses.len(), 1);
        assert!(uses.contains("maths.clamping"));
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_uses_declarations_multiple() {
        let attrs = vec![make_attr_uses(vec!["maths.clamping", "dt.raw"])];
        let mut errors = Vec::new();
        let uses = extract_uses_declarations(&attrs, &mut errors);

        assert_eq!(uses.len(), 2);
        assert!(uses.contains("maths.clamping"));
        assert!(uses.contains("dt.raw"));
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_uses_declarations_empty() {
        let attrs = vec![];
        let mut errors = Vec::new();
        let uses = extract_uses_declarations(&attrs, &mut errors);

        assert!(uses.is_empty());
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_uses_declarations_invalid_argument() {
        use crate::ast::{Expr, UntypedKind};

        // : uses(42) - invalid, not a path
        let attrs = vec![Attribute {
            name: "uses".to_string(),
            args: vec![Expr::new(
                UntypedKind::Literal {
                    value: 42.0,
                    unit: None,
                },
                make_span(),
            )],
            span: make_span(),
        }];

        let mut errors = Vec::new();
        let uses = extract_uses_declarations(&attrs, &mut errors);

        // Invalid argument should be ignored but error emitted
        assert!(uses.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::Internal);
        assert!(errors[0].message.contains("invalid uses() argument"));
    }

    #[test]
    fn test_collect_dt_usage() {
        let span = make_span();
        let expr = TypedExpr::new(ExprKind::Dt, Type::Bool, span);

        let registry = KernelRegistry::global();
        let mut required = Vec::new();
        collect_required_uses(&expr, registry, &mut required);

        assert_eq!(required.len(), 1);
        assert_eq!(required[0].key, "dt.raw");
        assert_eq!(required[0].source, "dt");
    }

    #[test]
    fn test_validate_node_missing_maths_clamping() {
        let span = make_span();
        let path = Path::from_str("test.signal");

        // Create node with clamp usage but no uses declaration
        let mut node = Node::new(path, span, RoleData::Signal, ());

        let value = TypedExpr::new(ExprKind::Prev, Type::Bool, span);
        let min = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            Type::Bool,
            span,
        );
        let max = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::Bool,
            span,
        );

        let clamp_call = TypedExpr::new(
            ExprKind::Call {
                kernel: crate::ast::KernelId::new("maths", "clamp"),
                args: vec![value, min, max],
            },
            Type::Bool,
            span,
        );

        node.executions
            .push(Execution::new(Phase::Resolve, clamp_call, vec![], span));

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingUsesDeclaration);
        assert!(errors[0].message.contains("maths.clamping"));
        assert!(errors[0].message.contains("maths.clamp"));
    }

    #[test]
    fn test_validate_node_with_maths_clamping_declared() {
        let span = make_span();
        let path = Path::from_str("test.signal");

        // Create node with clamp usage AND uses declaration
        let mut node = Node::new(path, span, RoleData::Signal, ());
        node.attributes.push(make_attr_uses(vec!["maths.clamping"]));

        let value = TypedExpr::new(ExprKind::Prev, Type::Bool, span);
        let min = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            Type::Bool,
            span,
        );
        let max = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::Bool,
            span,
        );

        let clamp_call = TypedExpr::new(
            ExprKind::Call {
                kernel: crate::ast::KernelId::new("maths", "clamp"),
                args: vec![value, min, max],
            },
            Type::Bool,
            span,
        );

        node.executions
            .push(Execution::new(Phase::Resolve, clamp_call, vec![], span));

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_nested_expression_finds_dt_in_call() {
        let span = make_span();

        // maths.add(prev, dt) - kernel call containing Dt
        let dt_expr = TypedExpr::new(ExprKind::Dt, Type::Bool, span);
        let prev_expr = TypedExpr::new(ExprKind::Prev, Type::Bool, span);

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: crate::ast::KernelId::new("maths", "add"),
                args: vec![prev_expr, dt_expr],
            },
            Type::Bool,
            span,
        );

        let registry = KernelRegistry::global();
        let mut required = Vec::new();
        collect_required_uses(&call, &registry, &mut required);

        assert_eq!(required.len(), 1);
        assert_eq!(required[0].key, "dt.raw");
    }

    #[test]
    fn test_clamp_requires_maths_clamping() {
        let span = make_span();

        // maths.clamp(value, min, max) call
        let value = TypedExpr::new(ExprKind::Prev, Type::Bool, span);
        let min = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            Type::Bool,
            span,
        );
        let max = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::Bool,
            span,
        );

        let clamp_call = TypedExpr::new(
            ExprKind::Call {
                kernel: crate::ast::KernelId::new("maths", "clamp"),
                args: vec![value, min, max],
            },
            Type::Bool,
            span,
        );

        let registry = KernelRegistry::global();
        let mut required = Vec::new();
        collect_required_uses(&clamp_call, &registry, &mut required);

        // Should require maths.clamping
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].key, "maths.clamping");
        assert_eq!(required[0].source, "maths.clamp");
        assert!(required[0].hint.contains("out-of-bounds"));
    }

    #[test]
    fn test_saturate_requires_maths_clamping() {
        let span = make_span();

        // maths.saturate(value) call
        let value = TypedExpr::new(ExprKind::Prev, Type::Bool, span);

        let saturate_call = TypedExpr::new(
            ExprKind::Call {
                kernel: crate::ast::KernelId::new("maths", "saturate"),
                args: vec![value],
            },
            Type::Bool,
            span,
        );

        let registry = KernelRegistry::global();
        let mut required = Vec::new();
        collect_required_uses(&saturate_call, &registry, &mut required);

        // Should require maths.clamping
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].key, "maths.clamping");
        assert_eq!(required[0].source, "maths.saturate");
        assert!(required[0].hint.contains("silently clamps"));
    }

    #[test]
    fn test_wrap_requires_maths_clamping() {
        let span = make_span();

        // maths.wrap(value, min, max) call
        let value = TypedExpr::new(ExprKind::Prev, Type::Bool, span);
        let min = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            Type::Bool,
            span,
        );
        let max = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::Bool,
            span,
        );

        let wrap_call = TypedExpr::new(
            ExprKind::Call {
                kernel: crate::ast::KernelId::new("maths", "wrap"),
                args: vec![value, min, max],
            },
            Type::Bool,
            span,
        );

        let registry = KernelRegistry::global();
        let mut required = Vec::new();
        collect_required_uses(&wrap_call, &registry, &mut required);

        // Should require maths.clamping
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].key, "maths.clamping");
        assert_eq!(required[0].source, "maths.wrap");
        assert!(required[0].hint.contains("wraps out-of-range"));
    }

    #[test]
    fn test_multiple_violations_all_reported() {
        let span = make_span();
        let path = Path::from_str("test.signal");

        // Create node with BOTH clamp AND dt usage but NO declarations
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // clamp(dt, 0.0, 1.0) - triggers both maths.clamping AND dt.raw
        let dt_expr = TypedExpr::new(ExprKind::Dt, Type::Bool, span);
        let min = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            Type::Bool,
            span,
        );
        let max = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::Bool,
            span,
        );

        let clamp_call = TypedExpr::new(
            ExprKind::Call {
                kernel: crate::ast::KernelId::new("maths", "clamp"),
                args: vec![dt_expr, min, max],
            },
            Type::Bool,
            span,
        );

        node.executions
            .push(Execution::new(Phase::Resolve, clamp_call, vec![], span));

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        // Should report BOTH violations
        assert_eq!(
            errors.len(),
            2,
            "Expected 2 errors: dt.raw + maths.clamping"
        );

        let error_messages: Vec<_> = errors.iter().map(|e| e.message.as_str()).collect();

        // Check that both violations are present
        let has_dt_raw = error_messages.iter().any(|m| m.contains("dt.raw"));
        let has_maths_clamping = error_messages.iter().any(|m| m.contains("maths.clamping"));

        assert!(
            has_dt_raw,
            "Expected error about dt.raw, got: {:?}",
            error_messages
        );
        assert!(
            has_maths_clamping,
            "Expected error about maths.clamping, got: {:?}",
            error_messages
        );
    }

    #[test]
    fn test_warmup_block_missing_dt_raw() {
        use crate::ast::{Expr, UntypedKind, WarmupBlock};

        let span = make_span();
        let path = Path::from_str("test.signal");

        // Create node with warmup block using dt
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // warmup { iterate { prev + dt } }
        let prev_expr = Expr::new(UntypedKind::Prev, span);
        let dt_expr = Expr::new(UntypedKind::Dt, span);
        let add_expr = Expr::binary(crate::ast::BinaryOp::Add, prev_expr, dt_expr, span);

        node.warmup = Some(WarmupBlock {
            attrs: vec![],
            iterate: add_expr,
            span,
        });

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingUsesDeclaration);
        assert!(errors[0].message.contains("dt.raw"));
    }

    #[test]
    fn test_warmup_block_with_dt_raw_declared() {
        use crate::ast::{Expr, UntypedKind, WarmupBlock};

        let span = make_span();
        let path = Path::from_str("test.signal");

        let mut node = Node::new(path, span, RoleData::Signal, ());
        node.attributes.push(make_attr_uses(vec!["dt.raw"]));

        // warmup { iterate { prev + dt } }
        let prev_expr = Expr::new(UntypedKind::Prev, span);
        let dt_expr = Expr::new(UntypedKind::Dt, span);
        let add_expr = Expr::binary(crate::ast::BinaryOp::Add, prev_expr, dt_expr, span);

        node.warmup = Some(WarmupBlock {
            attrs: vec![],
            iterate: add_expr,
            span,
        });

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_warmup_block_missing_maths_clamping() {
        use crate::ast::{Expr, UntypedKind, WarmupBlock};

        let span = make_span();
        let path = Path::from_str("test.signal");

        let mut node = Node::new(path, span, RoleData::Signal, ());

        // warmup { iterate { maths.clamp(prev, 0.0, 1.0) } }
        let clamp_path = Path::from_str("maths.clamp");
        let prev_expr = Expr::new(UntypedKind::Prev, span);
        let min_expr = Expr::new(
            UntypedKind::Literal {
                value: 0.0,
                unit: None,
            },
            span,
        );
        let max_expr = Expr::new(
            UntypedKind::Literal {
                value: 1.0,
                unit: None,
            },
            span,
        );
        let clamp_call = Expr::new(
            UntypedKind::Call {
                func: clamp_path,
                args: vec![prev_expr, min_expr, max_expr],
            },
            span,
        );

        node.warmup = Some(WarmupBlock {
            attrs: vec![],
            iterate: clamp_call,
            span,
        });

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingUsesDeclaration);
        assert!(errors[0].message.contains("maths.clamping"));
    }

    #[test]
    fn test_when_block_missing_dt_raw() {
        use crate::ast::{Expr, UntypedKind, WhenBlock};

        let span = make_span();
        let path = Path::from_str("test.fracture");

        let mut node = Node::new(path, span, RoleData::Fracture, ());

        // when { dt > 0.1 }
        let dt_expr = Expr::new(UntypedKind::Dt, span);
        let threshold = Expr::new(
            UntypedKind::Literal {
                value: 0.1,
                unit: None,
            },
            span,
        );
        let condition = Expr::binary(crate::ast::BinaryOp::Gt, dt_expr, threshold, span);

        node.when = Some(WhenBlock {
            conditions: vec![condition],
            span,
        });

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingUsesDeclaration);
        assert!(errors[0].message.contains("dt.raw"));
    }

    #[test]
    fn test_observe_block_missing_maths_clamping() {
        use crate::ast::{Expr, ObserveBlock, ObserveWhen, UntypedKind};

        let span = make_span();
        let path = Path::from_str("test.chronicle");

        let mut node = Node::new(path, span, RoleData::Chronicle, ());

        // observe {
        //     when maths.saturate(signal.value) > 0.9 {
        //         emit event.high_value
        //     }
        // }
        let signal_path = Path::from_str("signal.value");
        let signal_expr = Expr::new(UntypedKind::Signal(signal_path), span);
        let saturate_path = Path::from_str("maths.saturate");
        let saturate_call = Expr::new(
            UntypedKind::Call {
                func: saturate_path,
                args: vec![signal_expr],
            },
            span,
        );
        let threshold = Expr::new(
            UntypedKind::Literal {
                value: 0.9,
                unit: None,
            },
            span,
        );
        let condition = Expr::binary(crate::ast::BinaryOp::Gt, saturate_call, threshold, span);

        node.observe = Some(ObserveBlock {
            when_clauses: vec![ObserveWhen {
                condition,
                emit_block: vec![],
                span,
            }],
            span,
        });

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingUsesDeclaration);
        assert!(errors[0].message.contains("maths.clamping"));
    }
}
