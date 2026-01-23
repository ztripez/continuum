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
//! 1. **Phase detection** - Determines execution phase from operator metadata
//! 2. **Kernel scanning** - Recursively finds all kernel calls in expressions
//! 3. **Purity validation** - Checks that Effect kernels are only called in effect-allowed phases
//!
//! # What This Pass Does NOT Do
//!
//! - **No type checking** - Types must already be validated
//! - **No kernel existence checking** - Kernels must be in registry
//! - **No capability validation** - This is a separate validation pass
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

use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::{ExprKind, KernelRegistry, TypedExpr};
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
    pub fn new(phase: Phase) -> Self {
        Self { phase }
    }

    /// Check if this phase allows effect kernels.
    ///
    /// **Effect kernels** (emit, spawn, destroy, log) produce side effects and are
    /// only allowed in Collect and Fracture phases. **Pure kernels** (maths.*, vector.*,
    /// logic.*) are deterministic, side-effect-free operations allowed in all phases.
    pub fn allows_effects(&self) -> bool {
        matches!(self.phase, Phase::Collect | Phase::Fracture)
    }
}

/// Validates that all kernel calls in an expression respect purity restrictions for the current phase.
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
                // Unknown kernel - caught by type validation
                errors.push(CompileError::new(
                    ErrorKind::UnknownKernel,
                    expr.span,
                    format!(
                        "unknown kernel {} in effect validation",
                        kernel.qualified_name()
                    ),
                ));
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

        ExprKind::Aggregate { source, body, .. } => {
            scan_for_effect_violations(&source, ctx, registry, errors);
            scan_for_effect_violations(&body, ctx, registry, errors);
        }

        ExprKind::Fold {
            source, init, body, ..
        } => {
            scan_for_effect_violations(&source, ctx, registry, errors);
            scan_for_effect_violations(&init, ctx, registry, errors);
            scan_for_effect_violations(&body, ctx, registry, errors);
        }

        ExprKind::Filter { source, predicate } => {
            scan_for_effect_violations(&source, ctx, registry, errors);
            scan_for_effect_violations(&predicate, ctx, registry, errors);
        }

        ExprKind::Nearest { position, .. } => {
            scan_for_effect_violations(&position, ctx, registry, errors);
        }

        ExprKind::Within {
            position, radius, ..
        } => {
            scan_for_effect_violations(&position, ctx, registry, errors);
            scan_for_effect_violations(&radius, ctx, registry, errors);
        }

        // Leaf nodes - no kernel calls possible
        ExprKind::Literal { .. }
        | ExprKind::StringLiteral(_)
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
        | ExprKind::Entity(_) => {
            // No recursive scanning needed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl_ast::foundation::{Shape, Span, Type, Unit};
    use continuum_cdsl_ast::KernelId;

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
    fn test_effect_kernel_in_pure_phase() {
        let registry = KernelRegistry::global();
        let ctx = EffectContext::new(Phase::Resolve);

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
    }
}
