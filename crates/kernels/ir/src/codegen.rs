//! Bytecode Code Generation
//!
//! This module converts IR expressions ([`CompiledExpr`]) into executable
//! bytecode for the Continuum virtual machine.
//!
//! # Overview
//!
//! The code generator performs a straightforward recursive translation from
//! IR expression nodes to VM expression nodes, then delegates to the VM
//! compiler to produce actual bytecode.
//!
//! # Expression Mapping
//!
//! Most IR expression types map directly to VM equivalents:
//!
//! | IR Type | VM Type |
//! |---------|---------|
//! | `Literal(f64)` | `Literal(f64)` |
//! | `Prev` | `Prev` |
//! | `DtRaw` | `DtRaw` |
//! | `Signal(id)` | `Signal(name)` |
//! | `Binary { op, left, right }` | `Binary { op, left, right }` |
//! | etc. | etc. |
//!
//! # Entity Expressions
//!
//! Entity-related expressions (`SelfField`, `Aggregate`, `Filter`, etc.) are
//! **not** converted to bytecode. They return placeholder `Literal(0.0)` values
//! because entity operations require runtime access to entity storage, which
//! the bytecode VM does not have. Entity execution uses a separate interpreter.
//!
//! # Usage
//!
//! ```ignore
//! use continuum_ir::codegen::compile;
//!
//! let expr = CompiledExpr::Binary {
//!     op: BinaryOpIr::Add,
//!     left: Box::new(CompiledExpr::Prev),
//!     right: Box::new(CompiledExpr::Literal(1.0)),
//! };
//!
//! let bytecode = compile(&expr);
//! // Execute with: continuum_vm::execute(&bytecode, &context)
//! ```
//!

use continuum_vm::BytecodeChunk;
use continuum_vm::bytecode::ReductionOp;
use continuum_vm::compiler::{BinaryOp, Expr, UnaryOp};

use crate::{AggregateOpIr, BinaryOpIr, CompiledExpr, UnaryOpIr};

/// Converts an IR aggregate operator to its VM equivalent.
fn convert_reduction_op(op: AggregateOpIr) -> ReductionOp {
    match op {
        AggregateOpIr::Sum => ReductionOp::Sum,
        AggregateOpIr::Product => ReductionOp::Product,
        AggregateOpIr::Min => ReductionOp::Min,
        AggregateOpIr::Max => ReductionOp::Max,
        AggregateOpIr::Mean => ReductionOp::Mean,
        AggregateOpIr::Count => ReductionOp::Count,
        AggregateOpIr::Any => ReductionOp::Any,
        AggregateOpIr::All => ReductionOp::All,
        AggregateOpIr::None => ReductionOp::None,
    }
}

/// Converts an IR binary operator to its VM equivalent.
///
/// This is a direct 1:1 mapping as both representations use the same
/// operator semantics.
fn convert_binary_op(op: BinaryOpIr) -> BinaryOp {
    match op {
        BinaryOpIr::Add => BinaryOp::Add,
        BinaryOpIr::Sub => BinaryOp::Sub,
        BinaryOpIr::Mul => BinaryOp::Mul,
        BinaryOpIr::Div => BinaryOp::Div,
        BinaryOpIr::Pow => BinaryOp::Pow,
        BinaryOpIr::Eq => BinaryOp::Eq,
        BinaryOpIr::Ne => BinaryOp::Ne,
        BinaryOpIr::Lt => BinaryOp::Lt,
        BinaryOpIr::Le => BinaryOp::Le,
        BinaryOpIr::Gt => BinaryOp::Gt,
        BinaryOpIr::Ge => BinaryOp::Ge,
        BinaryOpIr::And => BinaryOp::And,
        BinaryOpIr::Or => BinaryOp::Or,
    }
}

/// Converts an IR unary operator to its VM equivalent.
fn convert_unary_op(op: UnaryOpIr) -> UnaryOp {
    match op {
        UnaryOpIr::Neg => UnaryOp::Neg,
        UnaryOpIr::Not => UnaryOp::Not,
    }
}

/// Recursively converts a [`CompiledExpr`] to a VM [`Expr`].
///
/// This function handles the translation of all expression types. Most types
/// map directly to VM equivalents, but entity expressions are not supported
/// and return placeholder values.
///
/// # Field Access
///
/// Field access on signals (e.g., `signal.pos.x`) is converted to
/// `SignalComponent` for vector component extraction.
fn convert_expr(expr: &CompiledExpr) -> Expr {
    match expr {
        CompiledExpr::Literal(v, _) => Expr::Literal(*v),
        CompiledExpr::Prev => Expr::Prev,
        CompiledExpr::DtRaw => Expr::DtRaw,
        CompiledExpr::SimTime => Expr::SimTime,
        CompiledExpr::Collected => Expr::Collected,
        CompiledExpr::Signal(id) => Expr::Signal(id.to_string()),
        CompiledExpr::Const(name, _) => Expr::Const(name.clone()),
        CompiledExpr::Config(name, _) => Expr::Config(name.clone()),
        CompiledExpr::Binary { op, left, right } => Expr::Binary {
            op: convert_binary_op(*op),
            left: Box::new(convert_expr(left)),
            right: Box::new(convert_expr(right)),
        },
        CompiledExpr::Unary { op, operand } => Expr::Unary {
            op: convert_unary_op(*op),
            operand: Box::new(convert_expr(operand)),
        },
        CompiledExpr::Call { function, args } => Expr::Call {
            function: function.clone(),
            args: args.iter().map(convert_expr).collect(),
        },
        CompiledExpr::KernelCall {
            namespace,
            function,
            args,
        } => {
            // Kernel functions are converted to regular calls for the VM.
            // The VM dispatches by name; the distinction is kept in IR for
            // semantic analysis and potential GPU dispatch in the future.
            Expr::Call {
                function: format!("{}.{}", namespace, function),
                args: args.iter().map(convert_expr).collect(),
            }
        }
        CompiledExpr::If {
            condition,
            then_branch,
            else_branch,
        } => Expr::If {
            condition: Box::new(convert_expr(condition)),
            then_branch: Box::new(convert_expr(then_branch)),
            else_branch: Box::new(convert_expr(else_branch)),
        },
        CompiledExpr::Let { name, value, body } => Expr::Let {
            name: name.clone(),
            value: Box::new(convert_expr(value)),
            body: Box::new(convert_expr(body)),
        },
        CompiledExpr::Local(name) => Expr::Local(name.clone()),
        CompiledExpr::FieldAccess { object, field } => {
            // Handle field access on signals, prev, and collected (e.g., signal.x, prev.y, collected.z)
            match object.as_ref() {
                CompiledExpr::Signal(id) => {
                    // Convert to SignalComponent for vector component access
                    Expr::SignalComponent(id.to_string(), field.clone())
                }
                CompiledExpr::Prev => {
                    // Convert to PrevComponent for vector prev component access
                    Expr::PrevComponent(field.clone())
                }
                CompiledExpr::Collected => {
                    // Convert to CollectedComponent for vector inputs component access
                    Expr::CollectedComponent(field.clone())
                }
                CompiledExpr::SelfField(_) => {
                    // Nested field access on SelfField should be simplified to just SelfField with component
                    Expr::SelfField(field.clone())
                }
                CompiledExpr::EntityAccess {
                    entity, instance, ..
                } => {
                    // entity[0].field
                    Expr::EntityAccess {
                        entity: entity.to_string(),
                        instance: instance.to_string(),
                        field: field.clone(),
                    }
                }
                CompiledExpr::First { entity, predicate } => Expr::First {
                    entity: entity.to_string(),
                    predicate: Box::new(convert_expr(predicate)),
                    field: field.clone(),
                },
                CompiledExpr::Nearest { entity, position } => Expr::Nearest {
                    entity: entity.to_string(),
                    position: Box::new(convert_expr(position)),
                    field: field.clone(),
                },
                CompiledExpr::Payload => Expr::PayloadField(field.clone()),
                other => {
                    panic!(
                        "Nested field access on {:?} not supported in bytecode compiler",
                        other
                    );
                }
            }
        }

        CompiledExpr::SelfField(field) => Expr::SelfField(field.clone()),
        CompiledExpr::EntityAccess {
            entity,
            instance,
            field,
        } => Expr::EntityAccess {
            entity: entity.to_string(),
            instance: instance.to_string(),
            field: field.clone(),
        },
        CompiledExpr::Aggregate { op, entity, body } => Expr::Aggregate {
            op: convert_reduction_op(*op),
            entity: entity.to_string(),
            body: Box::new(convert_expr(body)),
        },
        CompiledExpr::Other { entity, body } => Expr::Other {
            entity: entity.to_string(),
            body: Box::new(convert_expr(body)),
        },
        CompiledExpr::Pairs { entity, body } => Expr::Pairs {
            entity: entity.to_string(),
            body: Box::new(convert_expr(body)),
        },
        CompiledExpr::Filter {
            entity,
            predicate,
            body,
        } => Expr::Filter {
            entity: entity.to_string(),
            predicate: Box::new(convert_expr(predicate)),
            body: Box::new(convert_expr(body)),
        },
        CompiledExpr::First { entity, predicate } => Expr::First {
            entity: entity.to_string(),
            predicate: Box::new(convert_expr(predicate)),
            field: String::new(),
        },
        CompiledExpr::Nearest { entity, position } => Expr::Nearest {
            entity: entity.to_string(),
            position: Box::new(convert_expr(position)),
            field: String::new(),
        },
        CompiledExpr::Within {
            entity,
            position,
            radius,
            body,
        } => Expr::Within {
            entity: entity.to_string(),
            position: Box::new(convert_expr(position)),
            radius: Box::new(convert_expr(radius)),
            op: ReductionOp::Sum, // Default to sum for within(..), DSL might refine this
            body: Box::new(convert_expr(body)),
        },

        CompiledExpr::Payload => Expr::Payload,
        CompiledExpr::PayloadField(field) => Expr::PayloadField(field.clone()),
        CompiledExpr::EmitSignal { target, value } => Expr::EmitSignal {
            target: target.to_string(),
            value: Box::new(convert_expr(value)),
        },
    }
}

/// Compiles a [`CompiledExpr`] to executable bytecode.
///
/// This is the main entry point for bytecode generation. It converts the IR
/// expression to a VM expression tree, then delegates to the VM compiler to
/// produce the final bytecode chunk.
///
/// # Returns
///
/// A [`BytecodeChunk`] that can be executed by the VM with an execution context
/// providing signal values, constants, config, and kernel functions.
///
/// # Example
///
/// ```ignore
/// let expr = CompiledExpr::Binary {
///     op: BinaryOpIr::Add,
///     left: Box::new(CompiledExpr::Prev),
///     right: Box::new(CompiledExpr::Literal(1.0)),
/// };
///
/// let chunk = compile(&expr);
/// let result = continuum_vm::execute(&chunk, &my_context);
/// ```
pub fn compile(expr: &CompiledExpr) -> BytecodeChunk {
    let vm_expr = convert_expr(expr);
    continuum_vm::compile_expr(&vm_expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_foundation::{SignalId, Value};
    use continuum_vm::{ExecutionContext, execute};

    struct TestContext;

    impl ExecutionContext for TestContext {
        fn prev(&self) -> Value {
            Value::Scalar(100.0)
        }
        fn dt_scalar(&self) -> f64 {
            0.1
        }
        fn sim_time(&self) -> Value {
            Value::Scalar(10.0)
        }
        fn inputs(&self) -> Value {
            Value::Scalar(5.0)
        }
        fn signal(&self, name: &str) -> Value {
            match name {
                "temp" => Value::Scalar(25.0),
                _ => Value::Scalar(0.0),
            }
        }
        fn constant(&self, name: &str) -> Value {
            match name {
                "PI" => Value::Scalar(std::f64::consts::PI),
                _ => Value::Scalar(0.0),
            }
        }
        fn config(&self, name: &str) -> Value {
            match name {
                "scale" => Value::Scalar(2.0),
                _ => Value::Scalar(0.0),
            }
        }
        fn call_kernel(&self, name: &str, args: &[Value]) -> Value {
            match name {
                "abs" => args
                    .first()
                    .and_then(|v| v.as_scalar())
                    .map(|v| Value::Scalar(v.abs()))
                    .unwrap_or(Value::Scalar(0.0)),
                _ => Value::Scalar(0.0),
            }
        }

        fn self_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn entity_field(&self, _entity: &str, _instance: &str, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn other_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn entity_instances(&self, _entity: &str) -> Vec<String> {
            Vec::new()
        }
        fn set_current_entity(&mut self, _entity: Option<String>) {}
        fn set_self_instance(&mut self, _instance: Option<String>) {}
        fn set_other_instance(&mut self, _instance: Option<String>) {}
        fn payload(&self) -> Value {
            Value::Scalar(0.0)
        }
        fn payload_field(&self, _component: &str) -> Value {
            Value::Scalar(0.0)
        }
        fn emit_signal(&self, _target: &str, _value: Value) {}
    }

    #[test]
    fn test_compile_literal() {
        let expr = CompiledExpr::Literal(42.0, None);
        let chunk = compile(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(42.0));
    }

    #[test]
    fn test_compile_binary() {
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Literal(10.0, None)),
            right: Box::new(CompiledExpr::Literal(32.0, None)),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(42.0));
    }

    #[test]
    fn test_compile_signal() {
        let expr = CompiledExpr::Signal(SignalId::from("temp"));
        let chunk = compile(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(25.0));
    }

    #[test]
    fn test_compile_call() {
        let expr = CompiledExpr::Call {
            function: "abs".to_string(),
            args: vec![CompiledExpr::Literal(-5.0, None)],
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(5.0));
    }

    #[test]
    fn test_compile_if() {
        let expr = CompiledExpr::If {
            condition: Box::new(CompiledExpr::Binary {
                op: BinaryOpIr::Gt,
                left: Box::new(CompiledExpr::Signal(SignalId::from("temp"))),
                right: Box::new(CompiledExpr::Literal(20.0, None)),
            }),
            then_branch: Box::new(CompiledExpr::Literal(100.0, None)),
            else_branch: Box::new(CompiledExpr::Literal(0.0, None)),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(100.0)); // temp (25) > 20, so 100
    }

    #[test]
    fn test_compile_prev() {
        let expr = CompiledExpr::Prev;
        let chunk = compile(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(100.0));
    }

    #[test]
    fn test_compile_dt() {
        let expr = CompiledExpr::DtRaw;
        let chunk = compile(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(0.1));
    }

    #[test]
    fn test_compile_complex() {
        // prev + abs(temp - 30) * scale = 100 + abs(25-30) * 2 = 100 + 5*2 = 110
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Prev),
            right: Box::new(CompiledExpr::Binary {
                op: BinaryOpIr::Mul,
                left: Box::new(CompiledExpr::Call {
                    function: "abs".to_string(),
                    args: vec![CompiledExpr::Binary {
                        op: BinaryOpIr::Sub,
                        left: Box::new(CompiledExpr::Signal(SignalId::from("temp"))),
                        right: Box::new(CompiledExpr::Literal(30.0, None)),
                    }],
                }),
                right: Box::new(CompiledExpr::Config("scale".to_string(), None)),
            }),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(110.0));
    }

    #[test]
    fn test_compile_let() {
        // let a = 10.0
        // let b = 20.0
        // a + b = 30.0
        let expr = CompiledExpr::Let {
            name: "a".to_string(),
            value: Box::new(CompiledExpr::Literal(10.0, None)),
            body: Box::new(CompiledExpr::Let {
                name: "b".to_string(),
                value: Box::new(CompiledExpr::Literal(20.0, None)),
                body: Box::new(CompiledExpr::Binary {
                    op: BinaryOpIr::Add,
                    left: Box::new(CompiledExpr::Local("a".to_string())),
                    right: Box::new(CompiledExpr::Local("b".to_string())),
                }),
            }),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &mut TestContext);
        assert_eq!(result, Value::Scalar(30.0));
    }
}
