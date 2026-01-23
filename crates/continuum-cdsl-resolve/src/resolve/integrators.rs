//! integrator() attribute validation
//!
//! Validates that signals using dt.raw declare their integration method
//! using `: integrator(method)` for author intent verification.
//!
//! # What This Pass Does
//!
//! 1. **Extract integrator declarations** from `: integrator(method)` attributes
//! 2. **Walk all execution blocks** to find dt.integrate_* kernel calls
//! 3. **Validate consistency** between declared method and actual usage
//! 4. **Emit errors** for mismatches (fail loudly, never mask errors)
//!
//! # Valid Integrator Methods
//!
//! - `euler` - First-order forward Euler integration
//! - `rk4` - Fourth-order Runge-Kutta integration
//! - `verlet` - Velocity Verlet integration (symplectic)
//! - `symplectic_euler` - Symplectic Euler integration
//!
//! # Pipeline Position
//!
//! ```text
//! ... → Uses Validation → Integrator Validation → Execution Block Compilation → DAG
//!                             ^^^^^^^^^^^^^^^^^^^
//!                              YOU ARE HERE
//! ```
//!
//! This pass runs after uses validation and before DAG construction.
//!
//! # Examples
//!
//! ```cdsl
//! signal position : Vec3<m> {
//!     : uses(dt.raw)
//!     : integrator(rk4)  // Declare expected method
//!     
//!     resolve {
//!         dt.integrate_rk4(prev, velocity)  // OK - matches declaration
//!     }
//! }
//!
//! signal bad_position : Vec3<m> {
//!     : uses(dt.raw)
//!     : integrator(euler)  // Declares euler
//!     
//!     resolve {
//!         dt.integrate_rk4(prev, velocity)  // ERROR - mismatch!
//!     }
//! }
//! ```

use crate::error::{CompileError, ErrorKind};
use crate::resolve::attributes::extract_single_identifier;
use continuum_cdsl_ast::foundation::Span;
use continuum_cdsl_ast::{
    Attribute, ExecutionBody, ExprKind, ExpressionVisitor, Index, KernelRegistry, Node,
    StatementVisitor, TypedExpr,
};

/// Valid integrator method names
const VALID_INTEGRATORS: &[&str] = &["euler", "rk4", "verlet", "symplectic_euler"];

/// Information about an integration kernel call
#[derive(Debug, Clone)]
struct IntegrationCall {
    /// The specific method used (e.g., "rk4" from dt.integrate_rk4)
    method: String,
    /// Source span where the call occurs
    span: Span,
    /// Full kernel name for error messages
    kernel_name: String,
}

/// Extract integrator declaration from node attributes
///
/// Parses `: integrator(method)` into the method string.
/// Emits errors for:
/// - Multiple integrator declarations
/// - Invalid integrator method names
/// - Wrong number of arguments
///
/// # Example
///
/// ```rust,ignore
/// // : integrator(rk4)
/// let attrs = vec![
///     Attribute {
///         name: "integrator".to_string(),
///         args: vec![Expr::Signal(Path::from_path_str("rk4"))],
///         span,
///     },
/// ];
/// let mut errors = Vec::new();
/// let integrator = extract_integrator_declaration(&attrs, node_span, &mut errors);
/// assert_eq!(integrator, Some("rk4".to_string()));
/// assert!(errors.is_empty());
/// ```
fn extract_integrator_declaration(
    attrs: &[Attribute],
    node_span: Span,
    errors: &mut Vec<CompileError>,
) -> Option<String> {
    // Check for duplicate attributes
    let integrator_attrs: Vec<_> = attrs.iter().filter(|a| a.name == "integrator").collect();

    if integrator_attrs.len() > 1 {
        for attr in integrator_attrs.iter().skip(1) {
            errors.push(CompileError::new(
                ErrorKind::Conflict,
                attr.span,
                "duplicate :integrator attribute; declare only one integration method".to_string(),
            ));
        }
        // Continue with first attribute
    }

    // Extract method using common utility
    let method = extract_single_identifier(attrs, "integrator", node_span, errors)?;

    // Validate it's a known integrator
    if !VALID_INTEGRATORS.contains(&method.as_str()) {
        // Find the attribute span for precise error location
        let attr_span = attrs
            .iter()
            .find(|a| a.name == "integrator")
            .and_then(|a| a.args.first())
            .map(|arg| arg.span)
            .unwrap_or(node_span);

        errors.push(CompileError::new(
            ErrorKind::InvalidCapability,
            attr_span,
            format!(
                "unknown integrator method '{}'. Valid methods: {}",
                method,
                VALID_INTEGRATORS.join(", ")
            ),
        ));
        return None;
    }

    Some(method)
}

/// Walk typed expression tree collecting integration kernel calls
fn collect_integration_calls(expr: &TypedExpr, calls: &mut Vec<IntegrationCall>) {
    let mut visitor = IntegrationCallVisitor { calls };
    expr.walk(&mut visitor);
}

/// Walk compiled statement collecting integration calls from its expressions
fn collect_integration_calls_typed_stmt(
    stmt: &continuum_cdsl_ast::TypedStmt,
    calls: &mut Vec<IntegrationCall>,
) {
    let mut visitor = IntegrationCallVisitor { calls };
    visitor.visit_stmt(stmt);
}

/// Visitor that collects integration kernel calls from a typed expression tree and statements
struct IntegrationCallVisitor<'a> {
    calls: &'a mut Vec<IntegrationCall>,
}

impl<'a> StatementVisitor for IntegrationCallVisitor<'a> {
    fn visit_expr(&mut self, expr: &TypedExpr) {
        // Recursively walk the full expression tree
        expr.walk(self);
    }
}

impl<'a> ExpressionVisitor for IntegrationCallVisitor<'a> {
    fn visit_expr(&mut self, expr: &TypedExpr) {
        if let ExprKind::Call { kernel, .. } = &expr.expr {
            // Check if this is a dt.integrate_* kernel
            if kernel.namespace == "dt" && kernel.name.starts_with("integrate_") {
                // Extract method from kernel name (e.g., "integrate_rk4" -> "rk4")
                let method = kernel.name.strip_prefix("integrate_").unwrap();

                // Skip generic dt.integrate() - allowed regardless of hint
                if method.is_empty() {
                    return;
                }

                self.calls.push(IntegrationCall {
                    method: method.to_string(),
                    span: expr.span,
                    kernel_name: kernel.qualified_name(),
                });
            }
        }
    }
}

/// Validate integrator declarations for a single node
///
/// Checks:
/// 1. If integration kernels are used, integrator hint must be present (error)
/// 2. If integrator hint is present, it must match usage (error)
/// 3. Generic dt.integrate() calls are allowed regardless of hint
fn validate_node_integrator<I: Index>(
    node: &mut Node<I>,
    _registry: &KernelRegistry,
) -> Vec<CompileError> {
    let mut errors = Vec::new();

    // Extract declared integrator from attributes
    let declared = extract_integrator_declaration(&node.attributes, node.span, &mut errors);

    // Populate node field for downstream use
    node.integrator = declared.clone();

    // Collect integration calls from all execution blocks
    let mut calls = Vec::new();

    // Walk execution blocks (resolve, collect, measure, etc.)
    for execution in &node.executions {
        match &execution.body {
            ExecutionBody::Expr(expr) => collect_integration_calls(expr, &mut calls),
            ExecutionBody::Statements(stmts) => {
                for stmt in stmts {
                    collect_integration_calls_typed_stmt(stmt, &mut calls);
                }
            }
        }
    }

    // Check assertions
    for assertion in &node.assertions {
        collect_integration_calls(&assertion.condition, &mut calls);
    }

    // Validate consistency
    if let Some(declared_method) = declared {
        // If integrator is declared, all specific integration calls must match
        for call in &calls {
            if call.method != declared_method {
                errors.push(CompileError::new(
                    ErrorKind::IntegratorMismatch,
                    call.span,
                    format!(
                        "integrator mismatch: declared : integrator({}) but using {}. \
                         Change declaration to match usage or use dt.integrate() for generic integration",
                        declared_method, call.kernel_name
                    ),
                ));
            }
        }
    } else if !calls.is_empty() {
        // Integrator hint missing but specific methods are used
        // Require explicit integrator declarations for all specific dt.integrate_* usage
        let first_call = &calls[0];
        errors.push(CompileError::new(
            ErrorKind::MissingIntegratorHint,
            first_call.span,
            format!(
                "missing required : integrator({}) declaration for {}. \
                 Integration method must be explicitly declared to document author intent",
                first_call.method, first_call.kernel_name
            ),
        ));
    }

    errors
}

/// Validate integrator declarations for all nodes
///
/// Entry point for the validation pass. Call this with all nodes after execution
/// block compilation.
///
/// # Examples
///
/// ```rust,ignore
/// let errors = validate_integrators(&mut nodes, &kernel_registry);
/// if !errors.is_empty() {
///     // Report compilation errors
/// }
/// ```
pub fn validate_integrators<I: Index>(
    nodes: &mut [Node<I>],
    registry: &KernelRegistry,
) -> Vec<CompileError> {
    let mut all_errors = Vec::new();

    for node in nodes {
        let errors = validate_node_integrator(node, registry);
        all_errors.extend(errors);
    }

    all_errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl_ast::foundation::{Path, Phase, Shape, Type, Unit};
    use continuum_cdsl_ast::{Execution, Expr, RoleData, UntypedKind};
    use continuum_kernel_types::KernelId;

    fn make_span() -> Span {
        Span::new(0, 0, 0, 0)
    }

    fn make_node() -> Node<()> {
        Node::new(
            Path::from_path_str("test.signal"),
            make_span(),
            RoleData::Signal,
            (),
        )
    }

    #[test]
    fn test_extract_valid_integrator() {
        let attrs = vec![Attribute {
            name: "integrator".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("rk4")),
                make_span(),
            )],
            span: make_span(),
        }];

        let mut errors = Vec::new();
        let integrator = extract_integrator_declaration(&attrs, make_span(), &mut errors);

        assert_eq!(integrator, Some("rk4".to_string()));
        assert!(errors.is_empty());
    }

    #[test]
    fn test_extract_invalid_integrator() {
        let attrs = vec![Attribute {
            name: "integrator".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("invalid_method")),
                make_span(),
            )],
            span: make_span(),
        }];

        let mut errors = Vec::new();
        let integrator = extract_integrator_declaration(&attrs, make_span(), &mut errors);

        assert_eq!(integrator, None);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("unknown integrator method"));
    }

    #[test]
    fn test_duplicate_integrator_declaration() {
        let attrs = vec![
            Attribute {
                name: "integrator".to_string(),
                args: vec![Expr::new(
                    UntypedKind::Signal(Path::from_path_str("euler")),
                    make_span(),
                )],
                span: make_span(),
            },
            Attribute {
                name: "integrator".to_string(),
                args: vec![Expr::new(
                    UntypedKind::Signal(Path::from_path_str("rk4")),
                    make_span(),
                )],
                span: make_span(),
            },
        ];

        let mut errors = Vec::new();
        extract_integrator_declaration(&attrs, make_span(), &mut errors);

        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("duplicate"));
    }

    #[test]
    fn test_integrator_mismatch_detected() {
        let registry = KernelRegistry::global();
        let mut node = make_node();

        // Declare euler
        node.attributes.push(Attribute {
            name: "integrator".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("euler")),
                make_span(),
            )],
            span: make_span(),
        });

        // But use rk4
        node.executions.push(Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Expr(TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("dt", "integrate_rk4"),
                    args: vec![],
                },
                Type::kernel(Shape::Scalar, Unit::dimensionless(), None),
                make_span(),
            )),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span: make_span(),
        });

        let errors = validate_node_integrator(&mut node, &registry);

        assert_eq!(errors.len(), 1);
        assert!(errors[0].message.contains("integrator mismatch"));
        assert!(errors[0].message.contains("euler"));
        assert!(errors[0].message.contains("integrate_rk4"));
    }

    #[test]
    fn test_integrator_match_passes() {
        let registry = KernelRegistry::global();
        let mut node = make_node();

        // Declare rk4
        node.attributes.push(Attribute {
            name: "integrator".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("rk4")),
                make_span(),
            )],
            span: make_span(),
        });

        // Use rk4
        node.executions.push(Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Expr(TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("dt", "integrate_rk4"),
                    args: vec![],
                },
                Type::kernel(Shape::Scalar, Unit::dimensionless(), None),
                make_span(),
            )),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span: make_span(),
        });

        let errors = validate_node_integrator(&mut node, &registry);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_missing_hint_warning() {
        let registry = KernelRegistry::global();
        let mut node = make_node();

        // No integrator declaration

        // But use rk4
        node.executions.push(Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Expr(TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("dt", "integrate_rk4"),
                    args: vec![],
                },
                Type::kernel(Shape::Scalar, Unit::dimensionless(), None),
                make_span(),
            )),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span: make_span(),
        });

        let errors = validate_node_integrator(&mut node, &registry);

        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::MissingIntegratorHint);
        assert!(errors[0].message.contains("missing required"));
    }

    // Helper to make execution with integration call
    fn make_execution(method: &str) -> Execution {
        // Leak string for test (acceptable in test code)
        let method_name: &'static str = Box::leak(format!("integrate_{}", method).into_boxed_str());
        Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Expr(TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("dt", method_name),
                    args: vec![],
                },
                Type::kernel(Shape::Scalar, Unit::dimensionless(), None),
                make_span(),
            )),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span: make_span(),
        }
    }

    #[test]
    fn test_integrator_wrong_arg_count_zero() {
        let attrs = vec![Attribute {
            name: "integrator".to_string(),
            args: vec![],
            span: make_span(),
        }];

        let mut errors = Vec::new();
        let integrator = extract_integrator_declaration(&attrs, make_span(), &mut errors);

        assert_eq!(integrator, None);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
        assert!(errors[0].message.contains("expects 1 argument, got 0"));
    }

    #[test]
    fn test_integrator_wrong_arg_count_multiple() {
        let attrs = vec![Attribute {
            name: "integrator".to_string(),
            args: vec![
                Expr::new(UntypedKind::Signal(Path::from_path_str("rk4")), make_span()),
                Expr::new(
                    UntypedKind::Signal(Path::from_path_str("euler")),
                    make_span(),
                ),
            ],
            span: make_span(),
        }];

        let mut errors = Vec::new();
        let integrator = extract_integrator_declaration(&attrs, make_span(), &mut errors);

        assert_eq!(integrator, None);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
        assert!(errors[0].message.contains("expects 1 argument, got 2"));
    }

    #[test]
    fn test_multiple_calls_same_method() {
        let registry = KernelRegistry::global();
        let mut node = make_node();

        node.attributes.push(Attribute {
            name: "integrator".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("rk4")),
                make_span(),
            )],
            span: make_span(),
        });

        // Two calls to rk4
        node.executions.push(make_execution("rk4"));
        node.executions.push(make_execution("rk4"));

        let errors = validate_node_integrator(&mut node, &registry);
        assert!(
            errors.is_empty(),
            "Multiple calls to same method should pass"
        );
    }

    #[test]
    fn test_multiple_calls_different_methods() {
        let registry = KernelRegistry::global();
        let mut node = make_node();

        node.attributes.push(Attribute {
            name: "integrator".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("rk4")),
                make_span(),
            )],
            span: make_span(),
        });

        // Use both rk4 and euler
        node.executions.push(make_execution("rk4"));
        node.executions.push(make_execution("euler"));

        let errors = validate_node_integrator(&mut node, &registry);

        // Should report mismatch for euler call only
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::IntegratorMismatch);
        assert!(errors[0].message.contains("euler"));
    }

    #[test]
    fn test_generic_integrate_skipped() {
        let registry = KernelRegistry::global();
        let mut node = make_node();

        node.attributes.push(Attribute {
            name: "integrator".to_string(),
            args: vec![Expr::new(
                UntypedKind::Signal(Path::from_path_str("rk4")),
                make_span(),
            )],
            span: make_span(),
        });

        // Use generic dt.integrate (no specific method suffix)
        node.executions.push(Execution {
            name: "resolve".to_string(),
            phase: Phase::Resolve,
            body: ExecutionBody::Expr(TypedExpr::new(
                ExprKind::Call {
                    kernel: KernelId::new("dt", "integrate"),
                    args: vec![],
                },
                Type::kernel(Shape::Scalar, Unit::dimensionless(), None),
                make_span(),
            )),
            reads: vec![],
            temporal_reads: vec![],
            emits: vec![],
            span: make_span(),
        });

        let errors = validate_node_integrator(&mut node, &registry);

        // Generic integrate should be allowed regardless of hint
        assert!(
            errors.is_empty(),
            "Generic dt.integrate() should be allowed with any hint"
        );
    }
}
