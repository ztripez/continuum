//! Effect validation for kernel purity in phase contexts.
//!
//! Validates that kernel calls respect purity restrictions based on execution phase.
//! Effect kernels (emit, spawn, destroy, log) may only be called in phases that allow
//! side effects.
//!
//! # Phase Purity Rules
//!
//! - **Pure-only phases**: Configure, Resolve, Measure, Assert
//!   - May only call Pure kernels
//!   - Effect kernels cause validation errors
//!
//! - **Effect-allowed phases**: Collect, Fracture
//!   - May call both Pure and Effect kernels
//!   - No restrictions
//!
//! # What This Pass Does
//!
//! 1. **Phase detection** - Determines execution phase from role and context
//! 2. **Kernel scanning** - Recursively finds all kernel calls in expressions
//! 3. **Purity validation** - Checks that Effect kernels are only called in effect-allowed phases
//!
//! # What This Pass Does NOT Do
//!
//! - **No type checking** - Types must already be validated
//! - **No kernel existence checking** - Kernels must be in registry
//! - **No capability validation** - This is separate (Phase 12: capability validation)
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Res → Type Res → Type Val → Effect Val → ...
//!                                                       ^^^^^^^^^^
//!                                                     YOU ARE HERE
//! ```
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::resolve::effects::{validate_effect_purity, EffectContext};
//! use continuum_foundation::Phase;
//!
//! let ctx = EffectContext { phase: Phase::Resolve };
//! let errors = validate_effect_purity(&typed_expr, &ctx, kernel_registry);
//!
//! if !errors.is_empty() {
//!     eprintln!("Effect kernel called in pure-only phase!");
//! }
//! ```

use crate::ast::{ExprKind, KernelRegistry, TypedExpr};
use crate::error::{CompileError, ErrorKind};
use continuum_foundation::Phase;

/// Context for validating effect purity in expressions.
///
/// Carries the execution phase information needed to determine whether
/// effect kernels are allowed in the current context.
///
/// **Effect kernels** are kernels that produce side effects such as `emit`,
/// `spawn`, `destroy`, and `log`. **Pure kernels** are deterministic,
/// side-effect-free operations like `maths.*`, `vector.*`, `logic.*`.
///
/// Effect kernels may only be called in effect-allowed phases (Collect, Fracture).
/// Pure kernels may be called in any phase.
///
/// # Parameters
///
/// - `phase`: The execution phase this expression will run in.
///
/// # Examples
///
/// ```rust
/// use continuum_cdsl::resolve::effects::EffectContext;
/// use continuum_foundation::Phase;
///
/// let ctx = EffectContext { phase: Phase::Collect };
/// assert!(ctx.allows_effects());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct EffectContext {
    /// Execution phase for this expression
    pub phase: Phase,
}

impl EffectContext {
    /// Create a new effect context for the given phase.
    ///
    /// # Parameters
    ///
    /// - `phase`: The execution phase.
    ///
    /// # Returns
    ///
    /// A new `EffectContext` for the specified phase.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::resolve::effects::EffectContext;
    /// use continuum_foundation::Phase;
    ///
    /// let ctx = EffectContext::new(Phase::Resolve);
    /// assert!(!ctx.allows_effects());
    /// ```
    pub fn new(phase: Phase) -> Self {
        Self { phase }
    }

    /// Check if this phase allows effect kernels.
    ///
    /// **Effect kernels** (emit, spawn, destroy, log) produce side effects and are
    /// only allowed in Collect and Fracture phases. **Pure kernels** (maths.*, vector.*,
    /// logic.*) are deterministic, side-effect-free operations allowed in all phases.
    ///
    /// Effect-allowed phases: Collect, Fracture  
    /// Pure-only phases: Configure, Resolve, Measure, Assert
    ///
    /// # Returns
    ///
    /// `true` if effect kernels are allowed, `false` if only pure kernels are allowed.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use continuum_cdsl::resolve::effects::EffectContext;
    /// use continuum_foundation::Phase;
    ///
    /// assert!(EffectContext::new(Phase::Collect).allows_effects());
    /// assert!(EffectContext::new(Phase::Fracture).allows_effects());
    /// assert!(!EffectContext::new(Phase::Resolve).allows_effects());
    /// assert!(!EffectContext::new(Phase::Configure).allows_effects());
    /// assert!(!EffectContext::new(Phase::Measure).allows_effects());
    /// ```
    pub fn allows_effects(&self) -> bool {
        matches!(self.phase, Phase::Collect | Phase::Fracture)
    }
}

/// Validates that all kernel calls in an expression respect purity restrictions for the current phase.
///
/// **Effect kernels** (emit, spawn, destroy, log) produce side effects and may only be
/// called in Collect and Fracture phases. **Pure kernels** (maths.*, vector.*, logic.*)
/// are deterministic operations allowed in all phases.
///
/// Recursively scans the expression tree for kernel calls and validates that:
/// - Pure kernels are allowed in all phases
/// - Effect kernels are only called in effect-allowed phases (Collect, Fracture)
///
/// # Parameters
///
/// - `expr`: The typed expression to validate.
/// - `ctx`: The effect context (execution phase).
/// - `registry`: Kernel registry for looking up kernel purity.
///
/// # Returns
///
/// Vector of validation errors (empty if all kernel calls are valid).
/// Returns `ErrorKind::EffectInPureContext` when effect kernels are called in pure-only phases.
/// Returns `ErrorKind::UnknownKernel` defensively when kernel is not in registry
/// (should have been caught by type validation).
///
/// # Examples
///
/// ```rust,ignore
/// use continuum_cdsl::resolve::effects::{validate_effect_purity, EffectContext};
/// use continuum_cdsl::ast::KernelRegistry;
/// use continuum_foundation::Phase;
///
/// let ctx = EffectContext::new(Phase::Resolve);
/// let registry = KernelRegistry::global();
/// let errors = validate_effect_purity(&expr, &ctx, registry);
///
/// assert!(errors.is_empty(), "Pure-only phase cannot call effect kernels");
/// ```
pub fn validate_effect_purity(
    expr: &TypedExpr,
    ctx: &EffectContext,
    registry: &KernelRegistry,
) -> Vec<CompileError> {
    let mut errors = Vec::new();
    scan_for_effect_violations(expr, ctx, registry, &mut errors);
    errors
}

/// Recursively scans an expression for effect kernel violations.
///
/// This is the internal recursive function that traverses the expression tree
/// and accumulates errors when effect kernels are called in pure-only phases.
///
/// # Parameters
///
/// - `expr`: Current expression being scanned.
/// - `ctx`: Effect context (execution phase).
/// - `registry`: Kernel registry for purity lookups.
/// - `errors`: Accumulator for validation errors.
fn scan_for_effect_violations(
    expr: &TypedExpr,
    ctx: &EffectContext,
    registry: &KernelRegistry,
    errors: &mut Vec<CompileError>,
) {
    match &expr.expr {
        ExprKind::Call { kernel, args } => {
            // Validate this kernel call
            let Some(signature) = registry.get(&kernel) else {
                // Unknown kernel - this should have been caught by type validation,
                // but fail loudly here as a defensive check
                errors.push(CompileError::new(
                    ErrorKind::UnknownKernel,
                    expr.span,
                    format!(
                        "unknown kernel {} in effect validation (should have been caught by type validation)",
                        kernel.qualified_name()
                    ),
                ));
                // Still validate arguments even though kernel is unknown
                for arg in args {
                    scan_for_effect_violations(&arg, ctx, registry, errors);
                }
                return;
            };

            // Check purity restrictions
            if !signature.purity.is_pure() && !ctx.allows_effects() {
                errors.push(CompileError::new(
                    ErrorKind::EffectInPureContext,
                    expr.span,
                    format!(
                        "effect kernel {} cannot be called in {:?} phase (pure-only context)",
                        kernel.qualified_name(),
                        ctx.phase
                    ),
                ));
            }

            // Recursively validate arguments
            for arg in args {
                scan_for_effect_violations(&arg, ctx, registry, errors);
            }
        }

        ExprKind::Let { value, body, .. } => {
            scan_for_effect_violations(&value, ctx, registry, errors);
            scan_for_effect_violations(&body, ctx, registry, errors);
        }

        ExprKind::Struct { fields, .. } => {
            for (_name, field_expr) in fields {
                scan_for_effect_violations(&field_expr, ctx, registry, errors);
            }
        }

        ExprKind::FieldAccess { object, .. } => {
            scan_for_effect_violations(&object, ctx, registry, errors);
        }

        ExprKind::Vector(elements) => {
            for elem in elements {
                scan_for_effect_violations(&elem, ctx, registry, errors);
            }
        }

        ExprKind::Aggregate { body, .. } => {
            scan_for_effect_violations(&body, ctx, registry, errors);
        }

        ExprKind::Fold { init, body, .. } => {
            scan_for_effect_violations(&init, ctx, registry, errors);
            scan_for_effect_violations(&body, ctx, registry, errors);
        }

        // Leaf nodes - no kernel calls possible
        ExprKind::Literal { .. }
        | ExprKind::Local(_)
        | ExprKind::Signal(_)
        | ExprKind::Field(_)
        | ExprKind::Config(_)
        | ExprKind::Const(_)
        | ExprKind::Prev
        | ExprKind::Current
        | ExprKind::Inputs
        | ExprKind::Dt
        | ExprKind::Self_
        | ExprKind::Other
        | ExprKind::Payload => {
            // No recursive scanning needed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::KernelId;
    use crate::foundation::{Shape, Span, Type, Unit};

    fn test_span() -> Span {
        Span::new(0, 10, 20, 1)
    }

    #[test]
    fn test_effect_context_allows_effects() {
        assert!(EffectContext::new(Phase::Collect).allows_effects());
        assert!(EffectContext::new(Phase::Fracture).allows_effects());
        assert!(!EffectContext::new(Phase::Configure).allows_effects());
        assert!(!EffectContext::new(Phase::Resolve).allows_effects());
        assert!(!EffectContext::new(Phase::Measure).allows_effects());
    }

    #[test]
    fn test_pure_kernel_in_pure_phase() {
        let registry = KernelRegistry::global();
        let ctx = EffectContext::new(Phase::Resolve);

        // maths.abs is pure, should be allowed in Resolve
        let arg = TypedExpr::new(
            ExprKind::Literal {
                value: -5.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "abs"),
                args: vec![arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_effect_purity(&call, &ctx, registry);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_pure_kernel_in_effect_phase() {
        let registry = KernelRegistry::global();
        let ctx = EffectContext::new(Phase::Collect);

        // maths.abs is pure, should be allowed in Collect too
        let arg = TypedExpr::new(
            ExprKind::Literal {
                value: -5.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "abs"),
                args: vec![arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_effect_purity(&call, &ctx, registry);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_effect_kernel_in_effect_phase() {
        let registry = KernelRegistry::global();
        let ctx = EffectContext::new(Phase::Collect);

        // emit is an effect kernel, should be allowed in Collect
        let signal_arg = TypedExpr::new(
            ExprKind::Local("signal_id".to_string()),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let value_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![signal_arg, value_arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_effect_purity(&call, &ctx, registry);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_effect_kernel_in_pure_phase() {
        let registry = KernelRegistry::global();
        let ctx = EffectContext::new(Phase::Resolve);

        // emit is an effect kernel (bare name), should NOT be allowed in Resolve
        let signal_arg = TypedExpr::new(
            ExprKind::Local("signal_id".to_string()),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let value_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![signal_arg, value_arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_effect_purity(&call, &ctx, registry);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::EffectInPureContext);
        assert!(errors[0].message.contains("emit"));
        assert!(errors[0].message.contains("Resolve"));
    }

    #[test]
    fn test_nested_effect_kernel_violation() {
        let registry = KernelRegistry::global();
        let ctx = EffectContext::new(Phase::Measure);

        // Nested: let x = emit(...) in x
        let signal_arg = TypedExpr::new(
            ExprKind::Local("signal_id".to_string()),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let value_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let emit_call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![signal_arg, value_arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let let_expr = TypedExpr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(emit_call),
                body: Box::new(TypedExpr::new(
                    ExprKind::Local("x".to_string()),
                    Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                    test_span(),
                )),
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_effect_purity(&let_expr, &ctx, registry);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::EffectInPureContext);
    }

    #[test]
    fn test_effect_kernel_in_fracture_phase() {
        let registry = KernelRegistry::global();
        let ctx = EffectContext::new(Phase::Fracture);

        // emit is an effect kernel, should be allowed in Fracture
        let signal_arg = TypedExpr::new(
            ExprKind::Local("signal_id".to_string()),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let value_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![signal_arg, value_arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_effect_purity(&call, &ctx, registry);
        assert!(errors.is_empty());
    }

    #[test]
    fn test_effect_kernel_in_configure_phase() {
        let registry = KernelRegistry::global();
        let ctx = EffectContext::new(Phase::Configure);

        // emit is an effect kernel, should NOT be allowed in Configure
        let signal_arg = TypedExpr::new(
            ExprKind::Local("signal_id".to_string()),
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let value_arg = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![signal_arg, value_arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_effect_purity(&call, &ctx, registry);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::EffectInPureContext);
        assert!(errors[0].message.contains("Configure"));
    }

    #[test]
    fn test_unknown_kernel_defensive_check() {
        let registry = KernelRegistry::global();
        let ctx = EffectContext::new(Phase::Resolve);

        // Unknown kernel should trigger defensive error
        let arg = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("unknown", "bogus"),
                args: vec![arg],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_effect_purity(&call, &ctx, registry);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ErrorKind::UnknownKernel);
        assert!(errors[0].message.contains("unknown.bogus"));
        assert!(
            errors[0]
                .message
                .contains("should have been caught by type validation")
        );
    }

    #[test]
    fn test_unknown_kernel_still_scans_nested_args() {
        let registry = KernelRegistry::global();
        let ctx = EffectContext::new(Phase::Resolve);

        // Nested effect call inside arg of unknown kernel
        let nested_effect = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![
                    TypedExpr::new(
                        ExprKind::Local("signal_id".to_string()),
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                    TypedExpr::new(
                        ExprKind::Literal {
                            value: 1.0,
                            unit: None,
                        },
                        Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
                        test_span(),
                    ),
                ],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let call = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("unknown", "bogus"),
                args: vec![nested_effect],
            },
            Type::kernel(Shape::Scalar, Unit::DIMENSIONLESS, None),
            test_span(),
        );

        let errors = validate_effect_purity(&call, &ctx, registry);
        // Should have both: unknown kernel AND nested effect violation
        assert_eq!(errors.len(), 2);
        assert!(errors.iter().any(|e| e.kind == ErrorKind::UnknownKernel));
        assert!(
            errors
                .iter()
                .any(|e| e.kind == ErrorKind::EffectInPureContext)
        );
    }
}
