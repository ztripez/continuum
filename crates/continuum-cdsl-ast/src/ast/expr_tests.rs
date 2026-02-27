//! Tests for the expression system (ExprKind, TypedExpr, purity checking).
//!
//! Validates:
//! - KernelId construction and equality
//! - AggregateOp variants
//! - Purity checking across all ExprKind variants
//! - Impurity propagation through compound expressions
//! - Kernel namespace verification

use crate::ast::expr::{ExprKind, TypedExpr};
use crate::foundation::{AggregateOp, EntityId, Path, Span, Type};
use continuum_kernel_types::KernelId;

// Import continuum-functions to ensure kernel signatures are linked
#[allow(unused_imports)]
use continuum_functions as _;

#[test]
fn kernel_id_qualified_name() {
    let add = KernelId::new("maths", "add");
    assert_eq!(add.qualified_name(), "maths.add");

    let emit = KernelId::new("", "emit");
    assert_eq!(emit.qualified_name(), "emit"); // Bare name (empty namespace)
}

#[test]
fn kernel_id_equality() {
    let add1 = KernelId::new("maths", "add");
    let add2 = KernelId::new("maths", "add");
    let mul = KernelId::new("maths", "mul");

    assert_eq!(add1, add2);
    assert_ne!(add1, mul);
}

#[test]
fn aggregate_op_variants() {
    // Just verify all variants exist
    let ops = [
        AggregateOp::Sum,
        AggregateOp::Map,
        AggregateOp::Max,
        AggregateOp::Min,
        AggregateOp::Count,
        AggregateOp::Any,
        AggregateOp::All,
    ];
    assert_eq!(ops.len(), 7);
}

mod typed_expr_tests {
    use super::*;
    use crate::foundation::{KernelType, Shape, Unit};

    fn make_span() -> Span {
        Span::new(0, 0, 0, 0)
    }

    fn scalar_type() -> Type {
        Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::DIMENSIONLESS,
            bounds: None,
        })
    }

    #[test]
    fn literal_is_pure() {
        let expr = TypedExpr::new(
            ExprKind::Literal {
                value: 42.0,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );
        assert!(expr.is_pure());
    }

    #[test]
    fn vector_literal_is_pure() {
        let elem = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );
        let expr = TypedExpr::new(
            ExprKind::Vector(vec![elem.clone(), elem.clone(), elem]),
            scalar_type(),
            make_span(),
        );
        assert!(expr.is_pure());
    }

    #[test]
    fn context_values_are_pure() {
        let contexts = vec![
            ExprKind::Prev,
            ExprKind::Current,
            ExprKind::Inputs,
            ExprKind::Self_,
            ExprKind::Other,
            ExprKind::Payload,
        ];

        for ctx in contexts {
            let expr = TypedExpr::new(ctx, scalar_type(), make_span());
            assert!(expr.is_pure(), "context value should be pure");
        }
    }

    #[test]
    fn references_are_pure() {
        let references = vec![
            ExprKind::Local("x".to_string()),
            ExprKind::Signal(Path::from_path_str("velocity")),
            ExprKind::Field(Path::from_path_str("temperature")),
            ExprKind::Config(Path::from_path_str("initial_temp")),
            ExprKind::Const(Path::from_path_str("BOLTZMANN")),
        ];

        for ref_kind in references {
            let expr = TypedExpr::new(ref_kind, scalar_type(), make_span());
            assert!(expr.is_pure(), "reference should be pure");
        }
    }

    #[test]
    fn pure_kernel_call_is_pure() {
        let a = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );
        let b = TypedExpr::new(
            ExprKind::Literal {
                value: 2.0,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );

        let expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![a, b],
            },
            scalar_type(),
            make_span(),
        );
        assert!(expr.is_pure());
    }

    #[test]
    fn effect_kernel_call_is_impure() {
        let target = TypedExpr::new(
            ExprKind::Signal(Path::from_path_str("force")),
            scalar_type(),
            make_span(),
        );
        let value = TypedExpr::new(
            ExprKind::Literal {
                value: 10.0,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );

        let expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![target, value],
            },
            Type::Unit,
            make_span(),
        );
        assert!(!expr.is_pure());
    }

    #[test]
    fn let_with_pure_body_is_pure() {
        let value = TypedExpr::new(
            ExprKind::Literal {
                value: 10.0,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );
        let body = TypedExpr::new(ExprKind::Local("x".to_string()), scalar_type(), make_span());

        let expr = TypedExpr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(value),
                body: Box::new(body),
            },
            scalar_type(),
            make_span(),
        );
        assert!(expr.is_pure());
    }

    #[test]
    fn let_with_impure_value_is_impure() {
        let value = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![],
            },
            Type::Unit,
            make_span(),
        );
        let body = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );

        let expr = TypedExpr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(value),
                body: Box::new(body),
            },
            scalar_type(),
            make_span(),
        );
        assert!(!expr.is_pure());
    }

    #[test]
    fn aggregate_with_pure_body_is_pure() {
        let span = make_span();
        let body = TypedExpr::new(
            ExprKind::FieldAccess {
                object: Box::new(TypedExpr::new(ExprKind::Self_, scalar_type(), span)),
                field: "mass".to_string(),
            },
            scalar_type(),
            span,
        );

        let expr = TypedExpr::new(
            ExprKind::Aggregate {
                op: AggregateOp::Sum,
                source: Box::new(TypedExpr::new(
                    ExprKind::Entity(EntityId::new("plate")),
                    scalar_type(), // Dummy type
                    span,
                )),
                binding: "p".to_string(),
                body: Box::new(body),
            },
            scalar_type(),
            span,
        );
        assert!(expr.is_pure());
    }

    #[test]
    fn struct_construction_with_pure_fields_is_pure() {
        use continuum_foundation::TypeId;

        let field_value = TypedExpr::new(
            ExprKind::Literal {
                value: 1.5e11,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );

        let orbit_ty = TypeId::from("Orbit");
        let expr = TypedExpr::new(
            ExprKind::Struct {
                ty: orbit_ty.clone(),
                fields: vec![("semi_major".to_string(), field_value)],
            },
            Type::User(orbit_ty),
            make_span(),
        );
        assert!(expr.is_pure());
    }

    #[test]
    fn field_access_is_pure_if_object_is_pure() {
        use continuum_foundation::TypeId;

        let orbit_ty = TypeId::from("Orbit");
        let object = TypedExpr::new(
            ExprKind::Signal(Path::from_path_str("orbit")),
            Type::User(orbit_ty),
            make_span(),
        );

        let expr = TypedExpr::new(
            ExprKind::FieldAccess {
                object: Box::new(object),
                field: "semi_major".to_string(),
            },
            scalar_type(),
            make_span(),
        );
        assert!(expr.is_pure());
    }

    #[test]
    fn fold_with_pure_body_is_pure() {
        let span = make_span();
        let init = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            scalar_type(),
            span,
        );
        let body = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![
                    TypedExpr::new(ExprKind::Local("acc".to_string()), scalar_type(), span),
                    TypedExpr::new(ExprKind::Local("elem".to_string()), scalar_type(), span),
                ],
            },
            scalar_type(),
            span,
        );

        let expr = TypedExpr::new(
            ExprKind::Fold {
                source: Box::new(TypedExpr::new(
                    ExprKind::Entity(EntityId::new("plate")),
                    scalar_type(), // Dummy type
                    span,
                )),
                init: Box::new(init),
                acc: "acc".to_string(),
                elem: "elem".to_string(),
                body: Box::new(body),
            },
            scalar_type(),
            span,
        );
        assert!(expr.is_pure());
    }

    // === Impurity Propagation Tests ===

    #[test]
    fn let_with_impure_body_is_impure() {
        let value = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );
        let body = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![],
            },
            Type::Unit,
            make_span(),
        );
        let expr = TypedExpr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(value),
                body: Box::new(body),
            },
            scalar_type(),
            make_span(),
        );
        assert!(!expr.is_pure());
    }

    #[test]
    fn aggregate_with_impure_body_is_impure() {
        let span = make_span();
        let body = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![],
            },
            Type::Unit,
            span,
        );
        let expr = TypedExpr::new(
            ExprKind::Aggregate {
                op: AggregateOp::Sum,
                source: Box::new(TypedExpr::new(
                    ExprKind::Entity(EntityId::new("plate")),
                    scalar_type(), // Dummy type
                    span,
                )),
                binding: "p".to_string(),
                body: Box::new(body),
            },
            scalar_type(),
            span,
        );
        assert!(!expr.is_pure());
    }

    #[test]
    fn fold_with_impure_init_is_impure() {
        let span = make_span();
        let init = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![],
            },
            Type::Unit,
            span,
        );
        let body = TypedExpr::new(ExprKind::Local("acc".to_string()), scalar_type(), span);
        let expr = TypedExpr::new(
            ExprKind::Fold {
                source: Box::new(TypedExpr::new(
                    ExprKind::Entity(EntityId::new("plate")),
                    scalar_type(), // Dummy type
                    span,
                )),
                init: Box::new(init),
                acc: "acc".to_string(),
                elem: "elem".to_string(),
                body: Box::new(body),
            },
            scalar_type(),
            span,
        );
        assert!(!expr.is_pure());
    }

    #[test]
    fn fold_with_impure_body_is_impure() {
        let span = make_span();
        let init = TypedExpr::new(
            ExprKind::Literal {
                value: 0.0,
                unit: None,
            },
            scalar_type(),
            span,
        );
        let body = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "spawn"),
                args: vec![],
            },
            Type::Unit,
            span,
        );
        let expr = TypedExpr::new(
            ExprKind::Fold {
                source: Box::new(TypedExpr::new(
                    ExprKind::Entity(EntityId::new("plate")),
                    scalar_type(), // Dummy type
                    span,
                )),
                init: Box::new(init),
                acc: "acc".to_string(),
                elem: "elem".to_string(),
                body: Box::new(body),
            },
            scalar_type(),
            span,
        );
        assert!(!expr.is_pure());
    }

    #[test]
    fn pure_kernel_with_impure_arg_is_impure() {
        let impure_arg = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![],
            },
            Type::Unit,
            make_span(),
        );
        let pure_lit = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );
        let expr = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("maths", "add"),
                args: vec![pure_lit, impure_arg],
            },
            scalar_type(),
            make_span(),
        );
        assert!(!expr.is_pure());
    }

    #[test]
    fn struct_with_impure_field_is_impure() {
        use continuum_foundation::TypeId;

        let field_value = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "log"),
                args: vec![],
            },
            Type::Unit,
            make_span(),
        );
        let orbit_ty = TypeId::from("Orbit");
        let expr = TypedExpr::new(
            ExprKind::Struct {
                ty: orbit_ty.clone(),
                fields: vec![("semi_major".to_string(), field_value)],
            },
            Type::User(orbit_ty),
            make_span(),
        );
        assert!(!expr.is_pure());
    }

    #[test]
    fn field_access_with_impure_object_is_impure() {
        let object = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "destroy"),
                args: vec![],
            },
            Type::Unit,
            make_span(),
        );
        let expr = TypedExpr::new(
            ExprKind::FieldAccess {
                object: Box::new(object),
                field: "x".to_string(),
            },
            scalar_type(),
            make_span(),
        );
        assert!(!expr.is_pure());
    }

    #[test]
    fn vector_with_impure_element_is_impure() {
        let pure = TypedExpr::new(
            ExprKind::Literal {
                value: 1.0,
                unit: None,
            },
            scalar_type(),
            make_span(),
        );
        let impure = TypedExpr::new(
            ExprKind::Call {
                kernel: KernelId::new("", "emit"),
                args: vec![],
            },
            Type::Unit,
            make_span(),
        );
        let expr = TypedExpr::new(
            ExprKind::Vector(vec![pure.clone(), impure, pure]),
            scalar_type(),
            make_span(),
        );
        assert!(!expr.is_pure());
    }
}

mod kernel_namespaces {
    use super::*;

    #[test]
    fn maths_namespace() {
        let ops = vec!["add", "sub", "mul", "div", "sin", "cos", "sqrt", "pow"];
        for op in ops {
            let kernel = KernelId::new("maths", op);
            assert_eq!(kernel.namespace, "maths");
            assert_eq!(kernel.name, op);
        }
    }

    #[test]
    fn vector_namespace() {
        let ops = vec!["dot", "cross", "norm", "normalize"];
        for op in ops {
            let kernel = KernelId::new("vector", op);
            assert_eq!(kernel.namespace, "vector");
        }
    }

    #[test]
    fn logic_namespace() {
        let ops = vec!["and", "or", "not", "select"];
        for op in ops {
            let kernel = KernelId::new("logic", op);
            assert_eq!(kernel.namespace, "logic");
        }
    }

    #[test]
    fn compare_namespace() {
        let ops = vec!["lt", "le", "gt", "ge", "eq", "ne"];
        for op in ops {
            let kernel = KernelId::new("compare", op);
            assert_eq!(kernel.namespace, "compare");
        }
    }

    #[test]
    fn effect_namespace() {
        let ops = vec!["emit", "spawn", "destroy", "log"];
        for op in ops {
            // Effect operations are bare names (no namespace)
            let kernel = KernelId::new("", op);
            assert_eq!(kernel.namespace, "");
        }
    }
}
