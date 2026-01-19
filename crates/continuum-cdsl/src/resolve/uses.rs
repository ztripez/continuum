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
/// let uses = extract_uses_declarations(&attrs);
/// assert!(uses.contains("maths.clamping"));
/// assert!(uses.contains("dt.raw"));
/// ```
fn extract_uses_declarations(attrs: &[Attribute]) -> HashSet<String> {
    let mut uses = HashSet::new();

    for attr in attrs {
        if attr.name == "uses" {
            // Each arg should be a path like maths.clamping or dt.raw
            for arg in &attr.args {
                if let Some(key) = extract_uses_key_from_expr(arg) {
                    uses.insert(key);
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

/// Validate uses declarations for a single node
///
/// Checks all execution blocks (resolve, collect, warmup, when, observe, assertions)
/// for dangerous function usage and validates against declared uses.
fn validate_node_uses<I: Index>(node: &Node<I>, registry: &KernelRegistry) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Extract declared uses from attributes
    let declared = extract_uses_declarations(&node.attributes);

    // Collect required uses from all execution-related blocks
    let mut required = Vec::new();

    // 1. Execution blocks (resolve, collect, measure, etc.)
    //    These are compiled TypedExpr after Phase 12.5-D
    for execution in &node.executions {
        collect_required_uses(&execution.body, registry, &mut required);
    }

    // TODO: Warmup, When, Observe blocks contain untyped Expr
    // They need to be compiled to TypedExpr first, then we can validate them
    // For now, we only validate compiled execution blocks

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
        let uses = extract_uses_declarations(&attrs);

        assert_eq!(uses.len(), 1);
        assert!(uses.contains("maths.clamping"));
    }

    #[test]
    fn test_extract_uses_declarations_multiple() {
        let attrs = vec![make_attr_uses(vec!["maths.clamping", "dt.raw"])];
        let uses = extract_uses_declarations(&attrs);

        assert_eq!(uses.len(), 2);
        assert!(uses.contains("maths.clamping"));
        assert!(uses.contains("dt.raw"));
    }

    #[test]
    fn test_extract_uses_declarations_empty() {
        let attrs = vec![];
        let uses = extract_uses_declarations(&attrs);

        assert!(uses.is_empty());
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
}
