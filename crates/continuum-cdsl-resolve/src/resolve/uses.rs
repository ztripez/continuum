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

use continuum_cdsl_ast::{
    Attribute, ExecutionBody, ExprKind, ExpressionVisitor, Index, KernelRegistry, Node,
    StatementVisitor, TypedExpr,
};

use crate::error::{CompileError, ErrorKind};
use crate::resolve::attributes::extract_multiple_paths;
use continuum_cdsl_ast::foundation::Span;
use std::collections::HashSet;

/// Hint message for raw dt usage violations
const DT_RAW_HINT: &str = "Raw dt access makes code dt-fragile. Use dt-robust operators \
    (dt.integrate, dt.decay, dt.relax) instead. If raw dt is physically correct \
    (e.g., Energy = Power × dt), declare : uses(dt.raw)";

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
///             Expr::Signal(Path::from_path_str("maths.clamping")),
///             Expr::Signal(Path::from_path_str("dt.raw")),
///         ],
///         span,
///     },
/// ];
/// let mut errors = Vec::new();
/// let uses = extract_uses_declarations(&attrs, node.span, &mut errors);
/// assert!(uses.contains("maths.clamping"));
/// assert!(uses.contains("dt.raw"));
/// assert!(errors.is_empty());
/// ```
fn extract_uses_declarations(
    attrs: &[Attribute],
    context_span: Span,
    errors: &mut Vec<CompileError>,
) -> HashSet<String> {
    extract_multiple_paths(attrs, "uses", context_span, errors)
        .into_iter()
        .map(|p| p.to_string())
        .collect()
}

/// Walk typed expression tree collecting required uses from kernel calls and dt usage
fn collect_required_uses(
    expr: &TypedExpr,
    registry: &KernelRegistry,
    required: &mut Vec<RequiredUse>,
) {
    let mut visitor = RequiredUsesVisitor { registry, required };
    expr.walk(&mut visitor);
}

/// Walk compiled statement collecting required uses from its expressions
fn collect_required_uses_typed_stmt(
    stmt: &continuum_cdsl_ast::TypedStmt,
    registry: &KernelRegistry,
    required: &mut Vec<RequiredUse>,
) {
    let mut visitor = RequiredUsesVisitor { registry, required };
    visitor.visit_stmt(stmt);
}

/// Visitor that collects required uses from a typed expression tree and statements
struct RequiredUsesVisitor<'a> {
    registry: &'a KernelRegistry,
    required: &'a mut Vec<RequiredUse>,
}

impl<'a> StatementVisitor for RequiredUsesVisitor<'a> {
    fn visit_expr(&mut self, expr: &TypedExpr) {
        // Recursively walk the full expression tree
        expr.walk(self);
    }
}

impl<'a> ExpressionVisitor for RequiredUsesVisitor<'a> {
    fn visit_expr(&mut self, expr: &TypedExpr) {
        match &expr.expr {
            // Kernel call - check if it requires uses
            ExprKind::Call { kernel, .. } => {
                if let Some(signature) = self.registry.get(kernel) {
                    if let Some(req) = &signature.requires_uses {
                        self.required.push(RequiredUse {
                            key: format!("{}.{}", signature.id.namespace, req.key),
                            span: expr.span,
                            source: signature.id.qualified_name(),
                            hint: req.hint.clone(),
                        });
                    }
                }
            }

            // Other leaf nodes and containers that don't require specific uses
            ExprKind::Literal { .. }
            | ExprKind::StringLiteral(_)
            | ExprKind::BoolLiteral(_)
            | ExprKind::Vector(_)
            | ExprKind::Local(_)
            | ExprKind::Signal(_)
            | ExprKind::Field(_)
            | ExprKind::Config(_)
            | ExprKind::Const(_)
            | ExprKind::Prev
            | ExprKind::Current
            | ExprKind::Inputs
            | ExprKind::Self_
            | ExprKind::Other
            | ExprKind::Payload
            | ExprKind::Entity(_)
            | ExprKind::Let { .. }
            | ExprKind::Aggregate { .. }
            | ExprKind::Fold { .. }
            | ExprKind::Filter { .. }
            | ExprKind::Nearest { .. }
            | ExprKind::Within { .. }
            | ExprKind::Neighbors { .. }
            | ExprKind::Struct { .. }
            | ExprKind::FieldAccess { .. } => {}
        }
    }
}

/// Walk untyped statement collecting required uses from its expressions
fn collect_required_uses_untyped_stmt(
    stmt: &continuum_cdsl_ast::Stmt<continuum_cdsl_ast::Expr>,
    registry: &KernelRegistry,
    required: &mut Vec<RequiredUse>,
) {
    use continuum_cdsl_ast::Stmt;
    match stmt {
        Stmt::Let { value, .. } => collect_required_uses_untyped(value, registry, required),
        Stmt::SignalAssign { value, .. } => {
            collect_required_uses_untyped(value, registry, required)
        }
        Stmt::FieldAssign {
            position, value, ..
        } => {
            collect_required_uses_untyped(position, registry, required);
            collect_required_uses_untyped(value, registry, required);
        }
        Stmt::Assert { condition, .. } => {
            collect_required_uses_untyped(condition, registry, required)
        }
        Stmt::EmitEvent { fields, .. } => {
            for (_, expr) in fields {
                collect_required_uses_untyped(expr, registry, required);
            }
        }
        Stmt::Expr(expr) => collect_required_uses_untyped(expr, registry, required),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
            ..
        } => {
            collect_required_uses_untyped(condition, registry, required);
            for stmt in then_branch {
                collect_required_uses_untyped_stmt(stmt, registry, required);
            }
            for stmt in else_branch {
                collect_required_uses_untyped_stmt(stmt, registry, required);
            }
        }
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
    expr: &continuum_cdsl_ast::Expr,
    registry: &KernelRegistry,
    required: &mut Vec<RequiredUse>,
) {
    use continuum_cdsl_ast::UntypedKind;

    match &expr.kind {
        // Explicit call - might be a kernel call like maths.clamp(...)
        UntypedKind::Call { func, args } => {
            // Check if this is a namespaced kernel call
            // Note: Non-kernel calls (user-defined functions) won't match any kernel_id,
            // which is expected behavior. Only built-in kernels have requires_uses constraints.
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
        // Note: Currently, Binary and Unary operators don't have requires_uses constraints
        // (only explicit kernel calls do). We recurse into operands but don't validate the
        // operator itself. If future operators need validation (e.g., a hypothetical `clamp`
        // operator), they should be desugared to KernelCall form before this pass runs.
        UntypedKind::Binary { left, right, .. } => {
            collect_required_uses_untyped(left, registry, required);
            collect_required_uses_untyped(right, registry, required);
        }

        UntypedKind::Unary { operand, .. } => {
            collect_required_uses_untyped(operand, registry, required);
        }

        // Recurse into subexpressions
        UntypedKind::Let { value, body, .. } => {
            collect_required_uses_untyped(value, registry, required);
            collect_required_uses_untyped(body, registry, required);
        }

        // Note: TypedExpr doesn't have an If variant - it's desugared to logic.select during
        // type resolution. Untyped traversal handles If because warmup/when/observe blocks
        // are validated before desugaring occurs.
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

        UntypedKind::Aggregate { source, body, .. } => {
            collect_required_uses_untyped(source, registry, required);
            collect_required_uses_untyped(body, registry, required);
        }

        UntypedKind::Fold {
            source, init, body, ..
        } => {
            collect_required_uses_untyped(source, registry, required);
            collect_required_uses_untyped(init, registry, required);
            collect_required_uses_untyped(body, registry, required);
        }

        UntypedKind::Filter { source, predicate } => {
            collect_required_uses_untyped(source, registry, required);
            collect_required_uses_untyped(predicate, registry, required);
        }

        UntypedKind::Nearest { position, .. } => {
            collect_required_uses_untyped(position, registry, required);
        }

        UntypedKind::Within {
            position, radius, ..
        } => {
            collect_required_uses_untyped(position, registry, required);
            collect_required_uses_untyped(radius, registry, required);
        }

        UntypedKind::Neighbors { instance } => {
            collect_required_uses_untyped(instance, registry, required);
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
        | UntypedKind::Entity(_)
        | UntypedKind::OtherInstances(_)
        | UntypedKind::PairsInstances(_)
        | UntypedKind::ParseError(_)
        | UntypedKind::StringLiteral(_) => {}
    }
}

/// Validate uses declarations for a single node
///
/// Checks all execution blocks (resolve, collect, warmup, when, observe, assertions)
/// for dangerous function usage and validates against declared uses.
fn validate_node_uses<I: Index>(node: &Node<I>, registry: &KernelRegistry) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Extract declared uses from attributes (emits errors for invalid arguments)
    let declared = extract_uses_declarations(&node.attributes, node.span, &mut errors);

    // Collect required uses from all execution-related blocks
    let mut required = Vec::new();

    // 1. Execution blocks (resolve, collect, measure, etc.)
    //    These are compiled TypedExpr after Phase 12.5-D
    for execution in &node.executions {
        match &execution.body {
            ExecutionBody::Expr(expr) => collect_required_uses(expr, registry, &mut required),
            ExecutionBody::Statements(stmts) => {
                for stmt in stmts {
                    collect_required_uses_typed_stmt(stmt, registry, &mut required);
                }
            }
        }
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
            for stmt in &when_clause.emit_block {
                collect_required_uses_untyped_stmt(stmt, registry, &mut required);
            }
        }
    }

    // 5. Assertions
    //    Contains TypedExpr conditions
    for assertion in &node.assertions {
        collect_required_uses(&assertion.condition, registry, &mut required);
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
    use continuum_cdsl_ast::foundation::{Path, Phase, Type};
    use continuum_cdsl_ast::{Execution, RoleData};

    // Force linking of continuum-functions crate so kernel signatures are available
    // via distributed slice (KERNEL_SIGNATURES)
    #[allow(unused)]
    use continuum_functions as _;

    fn make_span() -> Span {
        Span::new(0, 0, 10, 1)
    }

    fn make_attr_uses(keys: Vec<&str>) -> Attribute {
        use continuum_cdsl_ast::{Expr, UntypedKind};

        Attribute {
            name: "uses".to_string(),
            args: keys
                .iter()
                .map(|k| Expr::new(UntypedKind::Signal(Path::from_path_str(k)), make_span()))
                .collect(),
            span: make_span(),
        }
    }

    #[test]
    fn test_extract_uses_declarations_single() {
        let attrs = vec![make_attr_uses(vec!["maths.clamping"])];
        let mut errors = Vec::new();
        let uses = extract_uses_declarations(&attrs, make_span(), &mut errors);

        assert_eq!(uses.len(), 1);
        assert!(uses.contains("maths.clamping"));
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_uses_declarations_multiple() {
        let attrs = vec![make_attr_uses(vec!["maths.clamping", "dt.raw"])];
        let mut errors = Vec::new();
        let uses = extract_uses_declarations(&attrs, make_span(), &mut errors);

        assert_eq!(uses.len(), 2);
        assert!(uses.contains("maths.clamping"));
        assert!(uses.contains("dt.raw"));
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_uses_declarations_empty() {
        let attrs = vec![];
        let mut errors = Vec::new();
        let uses = extract_uses_declarations(&attrs, make_span(), &mut errors);

        assert!(uses.is_empty());
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_uses_declarations_invalid_argument() {
        use continuum_cdsl_ast::{Expr, UntypedKind};

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
        let uses = extract_uses_declarations(&attrs, make_span(), &mut errors);

        // Invalid argument should be ignored but error emitted
        assert!(uses.is_empty());
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::Syntax);
        assert!(errors[0].message.contains("must be a path or identifier"));
    }

    #[test]
    fn test_collect_dt_usage() {
        let span = make_span();
        // dt is now accessed via kernel namespace (e.g. dt.raw())
        let kernel_id = continuum_kernel_types::KernelId::new("dt", "raw");
        let expr = TypedExpr::new(
            ExprKind::Call {
                kernel: kernel_id,
                args: vec![],
            },
            Type::Bool,
            span,
        );

        let registry = KernelRegistry::global();
        let mut required = Vec::new();
        collect_required_uses(&expr, registry, &mut required);

        assert_eq!(required.len(), 1);
        assert_eq!(required[0].key, "dt.raw");
        assert_eq!(required[0].source, "dt.raw");
    }

    #[test]
    fn test_validate_node_missing_maths_clamping() {
        let span = make_span();
        let path = Path::from_path_str("test.signal");

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
                kernel: continuum_cdsl_ast::KernelId::new("maths", "clamp"),
                args: vec![value, min, max],
            },
            Type::Bool,
            span,
        );

        node.executions.push(Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(clamp_call),
            vec![],
            vec![],
            vec![],
            span,
        ));

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
        let path = Path::from_path_str("test.signal");

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
                kernel: continuum_cdsl_ast::KernelId::new("maths", "clamp"),
                args: vec![value, min, max],
            },
            Type::Bool,
            span,
        );

        node.executions.push(Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(clamp_call),
            vec![],
            vec![],
            vec![],
            span,
        ));

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_nested_expression_finds_dt_in_call() {
        let span = make_span();

        // maths.add(prev, dt.raw()) - kernel call containing dt access
        let kernel_id = continuum_kernel_types::KernelId::new("dt", "raw");
        let dt_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: kernel_id,
                args: vec![],
            },
            Type::Bool,
            span,
        );
        let prev_expr = TypedExpr::new(ExprKind::Prev, Type::Bool, span);

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: continuum_cdsl_ast::KernelId::new("maths", "add"),
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
                kernel: continuum_cdsl_ast::KernelId::new("maths", "clamp"),
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
                kernel: continuum_cdsl_ast::KernelId::new("maths", "saturate"),
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
                kernel: continuum_cdsl_ast::KernelId::new("maths", "wrap"),
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
        let path = Path::from_path_str("test.signal");

        // Create node with BOTH clamp AND dt usage but NO declarations
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // clamp(dt.raw(), 0.0, 1.0) - triggers both maths.clamping AND dt.raw
        let kernel_id = continuum_kernel_types::KernelId::new("dt", "raw");
        let dt_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: kernel_id,
                args: vec![],
            },
            Type::Bool,
            span,
        );
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
                kernel: continuum_cdsl_ast::KernelId::new("maths", "clamp"),
                args: vec![dt_expr, min, max],
            },
            Type::Bool,
            span,
        );

        node.executions.push(Execution::new(
            "resolve".to_string(),
            Phase::Resolve,
            ExecutionBody::Expr(clamp_call),
            vec![],
            vec![],
            vec![],
            span,
        ));

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
        use continuum_cdsl_ast::{Expr, UntypedKind, WarmupBlock};

        let span = make_span();
        let path = Path::from_path_str("test.signal");

        // Create node with warmup block using dt
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // warmup { iterate { prev + dt } }
        let prev_expr = Expr::new(UntypedKind::Prev, span);
        let dt_expr = Expr::new(
            UntypedKind::Call {
                func: Path::from_path_str("dt.raw"),
                args: vec![],
            },
            span,
        );
        let add_expr = Expr::binary(continuum_cdsl_ast::BinaryOp::Add, prev_expr, dt_expr, span);

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
        use continuum_cdsl_ast::{Expr, UntypedKind, WarmupBlock};

        let span = make_span();
        let path = Path::from_path_str("test.signal");

        let mut node = Node::new(path, span, RoleData::Signal, ());
        node.attributes.push(make_attr_uses(vec!["dt.raw"]));

        // warmup { iterate { prev + dt } }
        let prev_expr = Expr::new(UntypedKind::Prev, span);
        let dt_expr = Expr::new(
            UntypedKind::Call {
                func: Path::from_path_str("dt.raw"),
                args: vec![],
            },
            span,
        );
        let add_expr = Expr::binary(continuum_cdsl_ast::BinaryOp::Add, prev_expr, dt_expr, span);

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
        use continuum_cdsl_ast::{Expr, UntypedKind, WarmupBlock};

        let span = make_span();
        let path = Path::from_path_str("test.signal");

        let mut node = Node::new(path, span, RoleData::Signal, ());

        // warmup { iterate { maths.clamp(prev, 0.0, 1.0) } }
        let clamp_path = Path::from_path_str("maths.clamp");
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
        use continuum_cdsl_ast::{Expr, UntypedKind, WhenBlock};

        let span = make_span();
        let path = Path::from_path_str("test.fracture");

        let mut node = Node::new(path, span, RoleData::Fracture, ());

        // when { dt > 0.1 }
        let dt_expr = Expr::new(
            UntypedKind::Call {
                func: Path::from_path_str("dt.raw"),
                args: vec![],
            },
            span,
        );
        let threshold = Expr::new(
            UntypedKind::Literal {
                value: 0.1,
                unit: None,
            },
            span,
        );
        let condition = Expr::binary(continuum_cdsl_ast::BinaryOp::Gt, dt_expr, threshold, span);

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
        use continuum_cdsl_ast::{Expr, ObserveBlock, ObserveWhen, UntypedKind};

        let span = make_span();
        let path = Path::from_path_str("test.chronicle");

        let mut node = Node::new(path, span, RoleData::Chronicle, ());

        // observe {
        //     when maths.saturate(signal.value) > 0.9 {
        //         emit event.high_value
        //     }
        // }
        let signal_path = Path::from_path_str("signal.value");
        let signal_expr = Expr::new(UntypedKind::Signal(signal_path), span);
        let saturate_path = Path::from_path_str("maths.saturate");
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
        let condition = Expr::binary(
            continuum_cdsl_ast::BinaryOp::Gt,
            saturate_call,
            threshold,
            span,
        );

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

    #[test]
    fn test_warmup_nested_let_with_dt() {
        use continuum_cdsl_ast::{Expr, UntypedKind, WarmupBlock};

        let span = make_span();
        let path = Path::from_path_str("test.signal");

        // Create node with warmup block using nested let with dt
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // warmup { iterate { let x = dt in x + prev } }
        let dt_value = Expr::new(
            UntypedKind::Call {
                func: Path::from_path_str("dt.raw"),
                args: vec![],
            },
            span,
        );
        let x_local = Expr::new(UntypedKind::Local("x".to_string()), span);
        let prev_expr = Expr::new(UntypedKind::Prev, span);
        let body = Expr::binary(continuum_cdsl_ast::BinaryOp::Add, x_local, prev_expr, span);
        let let_expr = Expr::new(
            UntypedKind::Let {
                name: "x".to_string(),
                value: Box::new(dt_value),
                body: Box::new(body),
            },
            span,
        );

        node.warmup = Some(WarmupBlock {
            attrs: vec![],
            iterate: let_expr,
            span,
        });

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        // Should detect dt.raw required in nested Let value
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingUsesDeclaration);
        assert!(errors[0].message.contains("dt.raw"));
    }

    #[test]
    fn test_when_if_with_dt_in_condition() {
        use continuum_cdsl_ast::{Expr, UntypedKind, WhenBlock};

        let span = make_span();
        let path = Path::from_path_str("test.fracture");

        // Create node with when block using if expression with dt
        let mut node = Node::new(path, span, RoleData::Fracture, ());

        // when { if dt > 0.1 { true } else { false } }
        let dt_expr = Expr::new(
            UntypedKind::Call {
                func: Path::from_path_str("dt.raw"),
                args: vec![],
            },
            span,
        );
        let threshold = Expr::new(
            UntypedKind::Literal {
                value: 0.1,
                unit: None,
            },
            span,
        );
        let condition = Expr::binary(continuum_cdsl_ast::BinaryOp::Gt, dt_expr, threshold, span);
        let true_branch = Expr::new(UntypedKind::BoolLiteral(true), span);
        let false_branch = Expr::new(UntypedKind::BoolLiteral(false), span);

        let if_expr = Expr::new(
            UntypedKind::If {
                condition: Box::new(condition),
                then_branch: Box::new(true_branch),
                else_branch: Box::new(false_branch),
            },
            span,
        );

        node.when = Some(WhenBlock {
            conditions: vec![if_expr],
            span,
        });

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        // Should detect dt.raw required in If condition
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingUsesDeclaration);
        assert!(errors[0].message.contains("dt.raw"));
    }

    #[test]
    fn test_warmup_nested_clamp_in_binary() {
        use continuum_cdsl_ast::{Expr, UntypedKind, WarmupBlock};

        let span = make_span();
        let path = Path::from_path_str("test.signal");

        // Create node with warmup block using clamp nested in binary expression
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // warmup { iterate { prev + maths.clamp(delta, -1, 1) } }
        let prev_expr = Expr::new(UntypedKind::Prev, span);
        let delta = Expr::new(UntypedKind::Local("delta".to_string()), span);
        let min = Expr::new(
            UntypedKind::Literal {
                value: -1.0,
                unit: None,
            },
            span,
        );
        let max = Expr::new(
            UntypedKind::Literal {
                value: 1.0,
                unit: None,
            },
            span,
        );
        let clamp_path = Path::from_path_str("maths.clamp");
        let clamp_call = Expr::new(
            UntypedKind::Call {
                func: clamp_path,
                args: vec![delta, min, max],
            },
            span,
        );
        let add_expr = Expr::binary(
            continuum_cdsl_ast::BinaryOp::Add,
            prev_expr,
            clamp_call,
            span,
        );

        node.warmup = Some(WarmupBlock {
            attrs: vec![],
            iterate: add_expr,
            span,
        });

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        // Should detect maths.clamping required in nested clamp call
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingUsesDeclaration);
        assert!(errors[0].message.contains("maths.clamping"));
    }

    #[test]
    fn test_untyped_kernel_call_with_dt() {
        use continuum_cdsl_ast::{Expr, KernelId, UntypedKind, WarmupBlock};

        let span = make_span();
        let path = Path::from_path_str("test.signal");
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // Desugared form: maths.add(prev, dt) as KernelCall
        let dt_expr = Expr::new(
            UntypedKind::Call {
                func: Path::from_path_str("dt.raw"),
                args: vec![],
            },
            span,
        );
        let prev_expr = Expr::new(UntypedKind::Prev, span);
        let kernel_call = Expr::new(
            UntypedKind::KernelCall {
                kernel: KernelId::new("maths", "add"),
                args: vec![prev_expr, dt_expr],
            },
            span,
        );

        node.warmup = Some(WarmupBlock {
            attrs: vec![],
            iterate: kernel_call,
            span,
        });

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        // Should detect dt.raw in KernelCall args
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingUsesDeclaration);
        assert!(errors[0].message.contains("dt.raw"));
    }

    #[test]
    fn test_unary_operator_with_dt() {
        use continuum_cdsl_ast::{Expr, UntypedKind, WarmupBlock};

        let span = make_span();
        let path = Path::from_path_str("test.signal");
        let mut node = Node::new(path, span, RoleData::Signal, ());

        // -dt.raw() (dt is now a pure namespace)
        let dt_expr = Expr::new(
            UntypedKind::Call {
                func: Path::from_path_str("dt.raw"),
                args: vec![],
            },
            span,
        );
        let neg_expr = Expr::unary(continuum_cdsl_ast::UnaryOp::Neg, dt_expr, span);

        node.warmup = Some(WarmupBlock {
            attrs: vec![],
            iterate: neg_expr,
            span,
        });

        let registry = KernelRegistry::global();
        let errors = validate_node_uses(&node, &registry);

        // Should detect dt.raw in Unary operand
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingUsesDeclaration);
        assert!(errors[0].message.contains("dt.raw"));
    }

    #[test]
    fn test_typed_let_with_nested_dt() {
        let span = make_span();

        // let x = dt in x + prev (TypedExpr form)
        let dt_expr = TypedExpr::new(
            ExprKind::Call {
                kernel: continuum_cdsl_ast::KernelId::new("dt", "raw"),
                args: vec![],
            },
            Type::Bool,
            span,
        );
        let local_x = TypedExpr::new(ExprKind::Local("x".to_string()), Type::Bool, span);
        let prev_expr = TypedExpr::new(ExprKind::Prev, Type::Bool, span);
        let kernel_id = continuum_cdsl_ast::KernelId::new("maths", "add");
        let body = TypedExpr::new(
            ExprKind::Call {
                kernel: kernel_id,
                args: vec![local_x, prev_expr],
            },
            Type::Bool,
            span,
        );
        let let_expr = TypedExpr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(dt_expr),
                body: Box::new(body),
            },
            Type::Bool,
            span,
        );

        let registry = KernelRegistry::global();
        let mut required = Vec::new();
        collect_required_uses(&let_expr, &registry, &mut required);

        // Should detect dt.raw in Let value
        assert_eq!(required.len(), 1);
        assert_eq!(required[0].key, "dt.raw");
    }
}
