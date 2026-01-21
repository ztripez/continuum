use super::*;
use crate::ast::Expr;
use crate::ast::KernelRegistry;
use crate::ast::UntypedKind;
use crate::foundation::{KernelType, Shape, Type, Unit};
use crate::foundation::{Path, Span, UserType};
use crate::resolve::types::TypeTable;
use continuum_foundation::Phase;
use continuum_foundation::TypeId;
use std::collections::HashMap;

fn make_context<'a>() -> TypingContext<'a> {
    let type_table = Box::leak(Box::new(TypeTable::new()));
    let kernel_registry = KernelRegistry::global();
    let signal_types = Box::leak(Box::new(HashMap::new()));
    let field_types = Box::leak(Box::new(HashMap::new()));
    let config_types = Box::leak(Box::new(HashMap::new()));
    let const_types = Box::leak(Box::new(HashMap::new()));

    TypingContext::new(
        type_table,
        kernel_registry,
        signal_types,
        field_types,
        config_types,
        const_types,
    )
}

/// Create a typing context with pre-registered user types
#[allow(dead_code)]
fn make_context_with_types<'a>(types: &[(&str, &[(&str, Type)])]) -> TypingContext<'a> {
    let type_table = Box::leak(Box::new({
        let mut table = TypeTable::new();
        for (type_name, fields) in types {
            let type_id = TypeId::from(*type_name);
            let user_type = UserType::new(
                type_id.clone(),
                Path::from(*type_name),
                fields
                    .iter()
                    .map(|(name, ty)| (name.to_string(), ty.clone()))
                    .collect(),
            );
            table.register(user_type);
        }
        table
    }));

    let kernel_registry = KernelRegistry::global();
    let signal_types = Box::leak(Box::new(HashMap::new()));
    let field_types = Box::leak(Box::new(HashMap::new()));
    let config_types = Box::leak(Box::new(HashMap::new()));
    let const_types = Box::leak(Box::new(HashMap::new()));

    TypingContext::new(
        type_table,
        kernel_registry,
        signal_types,
        field_types,
        config_types,
        const_types,
    )
}

#[test]
fn test_type_literal_dimensionless() {
    let ctx = make_context();
    let expr = Expr::new(
        UntypedKind::Literal {
            value: 42.0,
            unit: None,
        },
        Span::new(0, 0, 10, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    assert!(matches!(typed.ty, Type::Kernel(_)));
}

#[test]
fn test_type_call_invalid_depth() {
    let ctx = make_context();
    let expr = Expr::new(
        UntypedKind::Call {
            func: Path::from_path_str("maths.vector.add"),
            args: vec![],
        },
        Span::new(0, 0, 10, 1),
    );

    let result = type_expression(&expr, &ctx);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert_eq!(errors[0].kind, crate::error::ErrorKind::Syntax);
    assert!(errors[0]
        .message
        .contains("must be namespace.name or bare name"));
}

#[test]
fn test_type_call_undefined_namespace() {
    let ctx = make_context();
    let expr = Expr::new(
        UntypedKind::Call {
            func: Path::from_path_str("unknown_ns.func"),
            args: vec![],
        },
        Span::new(0, 0, 10, 1),
    );

    let result = type_expression(&expr, &ctx);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert_eq!(errors[0].kind, crate::error::ErrorKind::UndefinedName);
}

#[test]
fn test_type_bool_literal() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::BoolLiteral(true), Span::new(0, 0, 10, 1));

    let typed = type_expression(&expr, &ctx).unwrap();
    assert!(matches!(typed.ty, Type::Bool));
}

#[test]
fn test_type_dt() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::Dt, Span::new(0, 0, 10, 1));

    let typed = type_expression(&expr, &ctx).unwrap();
    match &typed.ty {
        Type::Kernel(kt) => {
            assert_eq!(kt.shape, Shape::Scalar);
            assert_eq!(kt.unit, Unit::seconds());
            assert_eq!(kt.bounds, None);
        }
        _ => panic!("Expected Kernel type, got {:?}", typed.ty),
    }
}

#[test]
fn test_type_local_not_in_scope() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::Local("x".to_string()), Span::new(0, 0, 10, 1));

    let result = type_expression(&expr, &ctx);
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
}

#[test]
fn test_type_prev_with_node_output() {
    let ctx = make_context().with_phase(Phase::Resolve);
    let output_type = Type::Kernel(KernelType {
        shape: Shape::Scalar,
        unit: Unit::meters(),
        bounds: None,
    });
    let ctx = ctx.with_execution_context(None, None, Some(output_type.clone()), None, None);

    let expr = Expr::new(UntypedKind::Prev, Span::new(0, 0, 4, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, output_type);
}

#[test]
fn test_type_prev_without_node_output() {
    let ctx = make_context().with_phase(Phase::Resolve);
    let expr = Expr::new(UntypedKind::Prev, Span::new(0, 0, 4, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::Internal));
    assert!(errors[0].message.contains("prev"));
}

#[test]
fn test_type_prev_in_wrong_phase() {
    let ctx = make_context().with_phase(Phase::Measure);
    let output_type = Type::Kernel(KernelType {
        shape: Shape::Scalar,
        unit: Unit::meters(),
        bounds: None,
    });
    let ctx = ctx.with_execution_context(None, None, Some(output_type.clone()), None, None);

    let expr = Expr::new(UntypedKind::Prev, Span::new(0, 0, 4, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert_eq!(errors[0].kind, ErrorKind::InvalidCapability);
    assert!(errors[0]
        .message
        .contains("may only be used in Resolve phase"));
}

#[test]
fn test_type_current_with_node_output() {
    let ctx = make_context();
    let output_type = Type::Kernel(KernelType {
        shape: Shape::Vector { dim: 3 },
        unit: Unit::seconds(),
        bounds: None,
    });
    let ctx = ctx.with_execution_context(None, None, Some(output_type.clone()), None, None);

    let expr = Expr::new(UntypedKind::Current, Span::new(0, 0, 7, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, output_type);
}

#[test]
fn test_type_current_without_node_output() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::Current, Span::new(0, 0, 7, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::Internal));
    assert!(errors[0].message.contains("current"));
}

#[test]
fn test_type_inputs_with_inputs_type() {
    let ctx = make_context();
    let inputs_type = Type::Kernel(KernelType {
        shape: Shape::Scalar,
        unit: Unit::kilograms(),
        bounds: None,
    });
    let ctx = ctx.with_execution_context(None, None, None, Some(inputs_type.clone()), None);

    let expr = Expr::new(UntypedKind::Inputs, Span::new(0, 0, 6, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, inputs_type);
}

#[test]
fn test_type_inputs_without_inputs_type() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::Inputs, Span::new(0, 0, 6, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::Internal));
    assert!(errors[0].message.contains("inputs"));
}

#[test]
fn test_type_payload_with_payload_type() {
    let ctx = make_context();
    let payload_type = Type::User(TypeId::from("ImpulseData"));
    let ctx = ctx.with_execution_context(None, None, None, None, Some(payload_type.clone()));

    let expr = Expr::new(UntypedKind::Payload, Span::new(0, 0, 7, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, payload_type);
}

#[test]
fn test_type_payload_without_payload_type() {
    let ctx = make_context();
    let expr = Expr::new(UntypedKind::Payload, Span::new(0, 0, 7, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::Internal));
    assert!(errors[0].message.contains("payload"));
}

#[test]
fn test_type_config_found() {
    let mut ctx = make_context();
    let config_type = Type::Kernel(KernelType {
        shape: Shape::Scalar,
        unit: Unit::dimensionless(),
        bounds: None,
    });

    let path = Box::leak(Box::new(Path::from("gravity")));
    let config_types = Box::leak(Box::new({
        let mut map = HashMap::new();
        map.insert(path.clone(), config_type.clone());
        map
    }));
    ctx.config_types = config_types;

    let expr = Expr::new(UntypedKind::Config(path.clone()), Span::new(0, 0, 7, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, config_type);
}

#[test]
fn test_type_config_not_found() {
    let ctx = make_context();
    let path = Path::from("unknown_config");
    let expr = Expr::new(UntypedKind::Config(path), Span::new(0, 0, 14, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    assert!(errors[0].message.contains("config"));
}

#[test]
fn test_type_const_found() {
    let mut ctx = make_context();
    let const_type = Type::Kernel(KernelType {
        shape: Shape::Scalar,
        unit: Unit::meters(),
        bounds: None,
    });

    let path = Box::leak(Box::new(Path::from("PI")));
    let const_types = Box::leak(Box::new({
        let mut map = HashMap::new();
        map.insert(path.clone(), const_type.clone());
        map
    }));
    ctx.const_types = const_types;

    let expr = Expr::new(UntypedKind::Const(path.clone()), Span::new(0, 0, 2, 1));
    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, const_type);
}

#[test]
fn test_type_const_not_found() {
    let ctx = make_context();
    let path = Path::from("UNKNOWN");
    let expr = Expr::new(UntypedKind::Const(path), Span::new(0, 0, 7, 1));
    let errors = type_expression(&expr, &ctx).unwrap_err();
    assert_eq!(errors.len(), 1);
    assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    assert!(errors[0].message.contains("const"));
}

#[test]
fn test_type_vector_2d() {
    let ctx = make_context();
    let expr = Expr::new(
        UntypedKind::Vector(vec![
            Expr::new(
                UntypedKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
            Expr::new(
                UntypedKind::Literal {
                    value: 2.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            ),
        ]),
        Span::new(0, 0, 10, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    match &typed.ty {
        Type::Kernel(kt) => {
            assert_eq!(kt.shape, Shape::Vector { dim: 2 });
            assert_eq!(kt.unit, Unit::dimensionless());
        }
        _ => panic!("Expected Kernel type"),
    }
}

#[test]
fn test_type_struct_valid_construction() {
    let type_table = Box::leak(Box::new({
        let mut table = TypeTable::new();
        let type_id = TypeId::from("Position");
        let user_type = UserType::new(
            type_id.clone(),
            Path::from("Position"),
            vec![
                (
                    "x".to_string(),
                    Type::Kernel(KernelType {
                        shape: Shape::Scalar,
                        unit: Unit::dimensionless(),
                        bounds: None,
                    }),
                ),
                (
                    "y".to_string(),
                    Type::Kernel(KernelType {
                        shape: Shape::Scalar,
                        unit: Unit::dimensionless(),
                        bounds: None,
                    }),
                ),
            ],
        );
        table.register(user_type);
        table
    }));

    let ctx = TypingContext::new(
        type_table,
        KernelRegistry::global(),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
        Box::leak(Box::new(HashMap::new())),
    );

    let expr = Expr::new(
        UntypedKind::Struct {
            ty: Path::from("Position"),
            fields: vec![
                (
                    "x".to_string(),
                    Expr::new(
                        UntypedKind::Literal {
                            value: 10.0,
                            unit: None,
                        },
                        Span::new(0, 0, 4, 1),
                    ),
                ),
                (
                    "y".to_string(),
                    Expr::new(
                        UntypedKind::Literal {
                            value: 20.0,
                            unit: None,
                        },
                        Span::new(0, 0, 4, 1),
                    ),
                ),
            ],
        },
        Span::new(0, 0, 30, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    assert_eq!(typed.ty, Type::User(TypeId::from("Position")));
}

#[test]
fn test_type_aggregate_map() {
    let ctx = make_context();
    let entity = continuum_foundation::EntityId::new("Plate");
    let expr = Expr::new(
        UntypedKind::Aggregate {
            op: crate::ast::AggregateOp::Map,
            entity: entity.clone(),
            binding: "p".to_string(),
            body: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 1.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            )),
        },
        Span::new(0, 0, 10, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    match &typed.ty {
        Type::Seq(inner) => {
            assert!(matches!(**inner, Type::Kernel(_)));
        }
        _ => panic!("Expected Seq type, got {:?}", typed.ty),
    }
}

#[test]
fn test_type_fold_valid() {
    let ctx = make_context();
    let entity = continuum_foundation::EntityId::new("Plate");
    let expr = Expr::new(
        UntypedKind::Fold {
            entity: entity.clone(),
            init: Box::new(Expr::new(
                UntypedKind::Literal {
                    value: 0.0,
                    unit: None,
                },
                Span::new(0, 0, 3, 1),
            )),
            acc: "acc".to_string(),
            elem: "p".to_string(),
            body: Box::new(Expr::new(
                UntypedKind::Local("acc".to_string()),
                Span::new(0, 0, 3, 1),
            )),
        },
        Span::new(0, 0, 10, 1),
    );

    let typed = type_expression(&expr, &ctx).unwrap();
    assert!(matches!(typed.ty, Type::Kernel(_)));
}
