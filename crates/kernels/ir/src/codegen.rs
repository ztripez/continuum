//! Bytecode code generation
//!
//! Converts CompiledExpr to VM bytecode.

use continuum_vm::compiler::{BinaryOp, Expr, UnaryOp};
use continuum_vm::BytecodeChunk;

use crate::{BinaryOpIr, CompiledExpr, UnaryOpIr};

/// Convert IR binary operator to VM binary operator
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

/// Convert IR unary operator to VM unary operator
fn convert_unary_op(op: UnaryOpIr) -> UnaryOp {
    match op {
        UnaryOpIr::Neg => UnaryOp::Neg,
        UnaryOpIr::Not => UnaryOp::Not,
    }
}

/// Convert CompiledExpr to VM Expr
fn convert_expr(expr: &CompiledExpr) -> Expr {
    match expr {
        CompiledExpr::Literal(v) => Expr::Literal(*v),
        CompiledExpr::Prev => Expr::Prev,
        CompiledExpr::DtRaw => Expr::DtRaw,
        CompiledExpr::SumInputs => Expr::SumInputs,
        CompiledExpr::Signal(id) => Expr::Signal(id.0.clone()),
        CompiledExpr::Const(name) => Expr::Const(name.clone()),
        CompiledExpr::Config(name) => Expr::Config(name.clone()),
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
            // Handle field access on signals (e.g., signal.x, signal.y, signal.z)
            match object.as_ref() {
                CompiledExpr::Signal(id) => {
                    // Convert to SignalComponent for vector component access
                    Expr::SignalComponent(id.0.clone(), field.clone())
                }
                _ => {
                    // Nested field access not yet supported - return 0
                    Expr::Literal(0.0)
                }
            }
        }
    }
}

/// Compile a CompiledExpr to bytecode
pub fn compile(expr: &CompiledExpr) -> BytecodeChunk {
    let vm_expr = convert_expr(expr);
    continuum_vm::compile_expr(&vm_expr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CompiledExpr;
    use continuum_foundation::SignalId;
    use continuum_vm::{execute, ExecutionContext};

    struct TestContext;

    impl ExecutionContext for TestContext {
        fn prev(&self) -> f64 {
            100.0
        }
        fn dt(&self) -> f64 {
            0.1
        }
        fn inputs(&self) -> f64 {
            5.0
        }
        fn signal(&self, name: &str) -> f64 {
            match name {
                "temp" => 25.0,
                _ => 0.0,
            }
        }
        fn constant(&self, name: &str) -> f64 {
            match name {
                "PI" => std::f64::consts::PI,
                _ => 0.0,
            }
        }
        fn config(&self, name: &str) -> f64 {
            match name {
                "scale" => 2.0,
                _ => 0.0,
            }
        }
        fn call_kernel(&self, name: &str, args: &[f64]) -> f64 {
            match name {
                "abs" => args.first().map(|v| v.abs()).unwrap_or(0.0),
                _ => 0.0,
            }
        }
    }

    #[test]
    fn test_compile_literal() {
        let expr = CompiledExpr::Literal(42.0);
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_compile_binary() {
        let expr = CompiledExpr::Binary {
            op: BinaryOpIr::Add,
            left: Box::new(CompiledExpr::Literal(10.0)),
            right: Box::new(CompiledExpr::Literal(32.0)),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 42.0);
    }

    #[test]
    fn test_compile_signal() {
        let expr = CompiledExpr::Signal(SignalId::from("temp"));
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 25.0);
    }

    #[test]
    fn test_compile_call() {
        let expr = CompiledExpr::Call {
            function: "abs".to_string(),
            args: vec![CompiledExpr::Literal(-5.0)],
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 5.0);
    }

    #[test]
    fn test_compile_if() {
        let expr = CompiledExpr::If {
            condition: Box::new(CompiledExpr::Binary {
                op: BinaryOpIr::Gt,
                left: Box::new(CompiledExpr::Signal(SignalId::from("temp"))),
                right: Box::new(CompiledExpr::Literal(20.0)),
            }),
            then_branch: Box::new(CompiledExpr::Literal(100.0)),
            else_branch: Box::new(CompiledExpr::Literal(0.0)),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 100.0); // temp (25) > 20, so 100
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
                        right: Box::new(CompiledExpr::Literal(30.0)),
                    }],
                }),
                right: Box::new(CompiledExpr::Config("scale".to_string())),
            }),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 110.0);
    }

    #[test]
    fn test_compile_let() {
        // let a = 10.0
        // let b = 20.0
        // a + b = 30.0
        let expr = CompiledExpr::Let {
            name: "a".to_string(),
            value: Box::new(CompiledExpr::Literal(10.0)),
            body: Box::new(CompiledExpr::Let {
                name: "b".to_string(),
                value: Box::new(CompiledExpr::Literal(20.0)),
                body: Box::new(CompiledExpr::Binary {
                    op: BinaryOpIr::Add,
                    left: Box::new(CompiledExpr::Local("a".to_string())),
                    right: Box::new(CompiledExpr::Local("b".to_string())),
                }),
            }),
        };
        let chunk = compile(&expr);
        let result = execute(&chunk, &TestContext);
        assert_eq!(result, 30.0);
    }
}
