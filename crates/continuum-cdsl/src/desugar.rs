//! Desugaring pass - converts syntax sugar to kernel calls
//!
//! This module implements the desugaring transformation that converts:
//! - Binary operators (`+`, `-`, `*`, `/`, etc.) → `maths.*` kernel calls
//! - Unary operators (`-`, `!`) → `maths.neg` / `logic.not` kernel calls
//! - Comparison operators (`<`, `>`, `==`, etc.) → `compare.*` kernel calls
//! - Logical operators (`&&`, `||`) → `logic.and` / `logic.or` kernel calls
//! - If-expressions (`if c { t } else { e }`) → `logic.select(c, t, e)` kernel call
//!
//! # Design
//!
//! Desugaring happens **before type resolution**. It transforms `Expr` (untyped AST)
//! into simpler `Expr` with only `KernelCall` nodes for operations.
//!
//! This separation allows:
//! - Type resolution to work with a smaller set of expression variants
//! - Kernel registry to handle all operations uniformly
//! - New operators to be added via kernel signatures, not AST changes
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Resolution → Type Resolution → Validation → Compilation
//!           ^^^^^^
//!           YOU ARE HERE
//! ```
//!
//! **Integration status:** Not yet wired into the compilation pipeline.
//! Desugaring must run:
//! - **After:** Parser produces untyped AST
//! - **Before:** name resolution and typing
//! - **Before:** uses validation on typed expressions

use crate::ast::{
    Attribute, BinaryOp, BlockBody, Declaration, Entity, EraDecl, Expr, Index, KernelId, Node,
    ObserveBlock, ObserveWhen, Stmt, Stratum, UnaryOp, UntypedKind as ExprKind, WarmupBlock,
    WhenBlock, WorldDecl,
};
use crate::foundation::Span;

/// Construct a kernel call expression
fn kernel_call(kernel: KernelId, args: Vec<Expr>, span: Span) -> Expr {
    Expr {
        kind: ExprKind::KernelCall { kernel, args },
        span,
    }
}

/// Get the kernel ID for a binary operator
fn binary_kernel(op: BinaryOp) -> KernelId {
    match op {
        BinaryOp::Add => KernelId::new("maths", "add"),
        BinaryOp::Sub => KernelId::new("maths", "sub"),
        BinaryOp::Mul => KernelId::new("maths", "mul"),
        BinaryOp::Div => KernelId::new("maths", "div"),
        BinaryOp::Mod => KernelId::new("maths", "mod"),
        BinaryOp::Pow => KernelId::new("maths", "pow"),
        BinaryOp::Eq => KernelId::new("compare", "eq"),
        BinaryOp::Ne => KernelId::new("compare", "ne"),
        BinaryOp::Lt => KernelId::new("compare", "lt"),
        BinaryOp::Le => KernelId::new("compare", "le"),
        BinaryOp::Gt => KernelId::new("compare", "gt"),
        BinaryOp::Ge => KernelId::new("compare", "ge"),
        BinaryOp::And => KernelId::new("logic", "and"),
        BinaryOp::Or => KernelId::new("logic", "or"),
    }
}

/// Get the kernel ID for a unary operator
fn unary_kernel(op: UnaryOp) -> KernelId {
    match op {
        UnaryOp::Neg => KernelId::new("maths", "neg"),
        UnaryOp::Not => KernelId::new("logic", "not"),
    }
}

/// Desugar an expression, converting operators to kernel calls
///
/// Recursively transforms operator syntax into explicit kernel calls, preserving
/// all other expression forms. This is a pure syntax transformation that does not
/// perform type resolution or semantic validation.
///
/// # Invariants
///
/// - **Span preservation** - The returned expression preserves the original span
/// - **No type resolution** - Works on untyped AST, does not resolve types
/// - **No semantic validation** - Does not check kernel existence or signatures
/// - **Recursive descent** - All nested expressions are desugared
/// - **Idempotent** - Calling `desugar_expr` twice produces same result (no operators left)
///
/// # Transformations
///
/// - `Binary { op, left, right }` → `KernelCall { kernel: op.kernel(), args: [left, right] }`
/// - `Unary { op, operand }` → `KernelCall { kernel: op.kernel(), args: [operand] }`
/// - `If { condition, then_branch, else_branch }` → `KernelCall { kernel: logic.select, args: [cond, then, else] }`
pub fn desugar_expr(expr: Expr) -> Expr {
    let span = expr.span;

    match expr.kind {
        ExprKind::Binary { op, left, right } => kernel_call(
            binary_kernel(op),
            vec![desugar_expr(*left), desugar_expr(*right)],
            span,
        ),

        ExprKind::Unary { op, operand } => {
            kernel_call(unary_kernel(op), vec![desugar_expr(*operand)], span)
        }

        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => kernel_call(
            KernelId::new("logic", "select"),
            vec![
                desugar_expr(*condition),
                desugar_expr(*then_branch),
                desugar_expr(*else_branch),
            ],
            span,
        ),

        ExprKind::Let { name, value, body } => Expr {
            kind: ExprKind::Let {
                name,
                value: Box::new(desugar_expr(*value)),
                body: Box::new(desugar_expr(*body)),
            },
            span,
        },

        ExprKind::Vector(elements) => Expr {
            kind: ExprKind::Vector(elements.into_iter().map(desugar_expr).collect()),
            span,
        },

        ExprKind::Call { func, args } => Expr {
            kind: ExprKind::Call {
                func,
                args: args.into_iter().map(desugar_expr).collect(),
            },
            span,
        },

        ExprKind::KernelCall { kernel, args } => Expr {
            kind: ExprKind::KernelCall {
                kernel,
                args: args.into_iter().map(desugar_expr).collect(),
            },
            span,
        },

        ExprKind::Aggregate {
            op,
            source,
            binding,
            body,
        } => Expr {
            kind: ExprKind::Aggregate {
                op,
                source: Box::new(desugar_expr(*source)),
                binding,
                body: Box::new(desugar_expr(*body)),
            },
            span,
        },

        ExprKind::Fold {
            source,
            init,
            acc,
            elem,
            body,
        } => Expr {
            kind: ExprKind::Fold {
                source: Box::new(desugar_expr(*source)),
                init: Box::new(desugar_expr(*init)),
                acc,
                elem,
                body: Box::new(desugar_expr(*body)),
            },
            span,
        },

        ExprKind::Nearest { entity, position } => Expr {
            kind: ExprKind::Nearest {
                entity,
                position: Box::new(desugar_expr(*position)),
            },
            span,
        },

        ExprKind::Within {
            entity,
            position,
            radius,
        } => Expr {
            kind: ExprKind::Within {
                entity,
                position: Box::new(desugar_expr(*position)),
                radius: Box::new(desugar_expr(*radius)),
            },
            span,
        },

        ExprKind::Filter { source, predicate } => Expr {
            kind: ExprKind::Filter {
                source: Box::new(desugar_expr(*source)),
                predicate: Box::new(desugar_expr(*predicate)),
            },
            span,
        },

        ExprKind::Struct { ty, fields } => Expr {
            kind: ExprKind::Struct {
                ty,
                fields: fields
                    .into_iter()
                    .map(|(name, expr)| (name, desugar_expr(expr)))
                    .collect(),
            },
            span,
        },

        ExprKind::FieldAccess { object, field } => Expr {
            kind: ExprKind::FieldAccess {
                object: Box::new(desugar_expr(*object)),
                field,
            },
            span,
        },

        // Leaf cases
        _ => expr,
    }
}

/// Desugar a list of attributes
pub fn desugar_attributes(attrs: Vec<Attribute>) -> Vec<Attribute> {
    attrs
        .into_iter()
        .map(|mut attr| {
            attr.args = attr.args.into_iter().map(desugar_expr).collect();
            attr
        })
        .collect()
}

/// Desugar a block body (expression or statements)
pub fn desugar_block_body(body: BlockBody) -> BlockBody {
    match body {
        BlockBody::Expression(expr) => BlockBody::Expression(desugar_expr(expr)),
        BlockBody::Statements(stmts) => {
            BlockBody::Statements(stmts.into_iter().map(desugar_stmt).collect())
        }
        BlockBody::TypedExpression(_) | BlockBody::TypedStatements(_) => {
            panic!(
                "Typed AST encountered during desugar pass: desugaring must happen before typing"
            )
        }
    }
}

/// Desugar a statement
pub fn desugar_stmt(stmt: Stmt<Expr>) -> Stmt<Expr> {
    match stmt {
        Stmt::Let { name, value, span } => Stmt::Let {
            name,
            value: desugar_expr(value),
            span,
        },
        Stmt::SignalAssign {
            target,
            value,
            span,
        } => Stmt::SignalAssign {
            target,
            value: desugar_expr(value),
            span,
        },
        Stmt::FieldAssign {
            target,
            position,
            value,
            span,
        } => Stmt::FieldAssign {
            target,
            position: desugar_expr(position),
            value: desugar_expr(value),
            span,
        },
        Stmt::Expr(expr) => Stmt::Expr(desugar_expr(expr)),
    }
}

/// Desugar a warmup block
pub fn desugar_warmup(warmup: WarmupBlock) -> WarmupBlock {
    WarmupBlock {
        attrs: desugar_attributes(warmup.attrs),
        iterate: desugar_expr(warmup.iterate),
        span: warmup.span,
    }
}

/// Desugar a when block
pub fn desugar_when(when: WhenBlock) -> WhenBlock {
    WhenBlock {
        conditions: when.conditions.into_iter().map(desugar_expr).collect(),
        span: when.span,
    }
}

/// Desugar an observe block
pub fn desugar_observe(observe: ObserveBlock) -> ObserveBlock {
    ObserveBlock {
        when_clauses: observe
            .when_clauses
            .into_iter()
            .map(|when| ObserveWhen {
                condition: desugar_expr(when.condition),
                emit_block: when.emit_block.into_iter().map(desugar_stmt).collect(),
                span: when.span,
            })
            .collect(),
        span: observe.span,
    }
}

/// Desugar a node (signal, field, operator, etc)
pub fn desugar_node<I: Index>(mut node: Node<I>) -> Node<I> {
    node.attributes = desugar_attributes(node.attributes);
    node.execution_blocks = node
        .execution_blocks
        .into_iter()
        .map(|(name, body)| (name, desugar_block_body(body)))
        .collect();

    node.warmup = node.warmup.map(desugar_warmup);
    node.when = node.when.map(desugar_when);
    node.observe = node.observe.map(desugar_observe);

    node
}

/// Desugar an entity declaration
pub fn desugar_entity(mut entity: Entity) -> Entity {
    entity.attributes = desugar_attributes(entity.attributes);
    entity
}

/// Desugar a stratum declaration
pub fn desugar_stratum(mut stratum: Stratum) -> Stratum {
    stratum.attributes = desugar_attributes(stratum.attributes);
    stratum
}

/// Desugar an era declaration
pub fn desugar_era(mut era: EraDecl) -> EraDecl {
    era.attributes = desugar_attributes(era.attributes);
    era.dt = era.dt.map(desugar_expr);
    era.transitions = era
        .transitions
        .into_iter()
        .map(|mut t| {
            t.conditions = t.conditions.into_iter().map(desugar_expr).collect();
            t
        })
        .collect();
    era
}

/// Desugar a world declaration
pub fn desugar_world(mut world: WorldDecl) -> WorldDecl {
    if let Some(mut warmup) = world.warmup {
        warmup.attributes = desugar_attributes(warmup.attributes);
        world.warmup = Some(warmup);
    }
    world.attributes = desugar_attributes(world.attributes);
    world
}

/// Main entry point for desugaring all declarations in a world
pub fn desugar_declarations(decls: Vec<Declaration>) -> Vec<Declaration> {
    decls
        .into_iter()
        .map(|decl| match decl {
            Declaration::Node(node) => Declaration::Node(desugar_node(node)),
            Declaration::Member(node) => Declaration::Member(desugar_node(node)),
            Declaration::Entity(entity) => Declaration::Entity(desugar_entity(entity)),
            Declaration::Stratum(stratum) => Declaration::Stratum(desugar_stratum(stratum)),
            Declaration::Era(era) => Declaration::Era(desugar_era(era)),
            Declaration::World(world) => Declaration::World(desugar_world(world)),
            Declaration::Const(mut entries) => {
                for entry in &mut entries {
                    entry.value = desugar_expr(entry.value.clone());
                }
                Declaration::Const(entries)
            }
            Declaration::Config(mut entries) => {
                for entry in &mut entries {
                    entry.default = entry.default.clone().map(desugar_expr);
                }
                Declaration::Config(entries)
            }
            Declaration::Type(_) => decl,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{BinaryOp, UnaryOp};
    use crate::foundation::Span;

    fn make_span() -> Span {
        Span::new(0, 0, 10, 1)
    }

    fn make_literal(value: f64) -> Expr {
        Expr {
            kind: ExprKind::Literal { value, unit: None },
            span: make_span(),
        }
    }

    #[test]
    fn test_desugar_binary_add() {
        let left = make_literal(1.0);
        let right = make_literal(2.0);
        let expr = Expr::binary(BinaryOp::Add, left, right, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::KernelCall { kernel, args } => {
                assert_eq!(kernel, KernelId::new("maths", "add"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected KernelCall, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_binary_mul() {
        let left = make_literal(3.0);
        let right = make_literal(4.0);
        let expr = Expr::binary(BinaryOp::Mul, left, right, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::KernelCall { kernel, args } => {
                assert_eq!(kernel, KernelId::new("maths", "mul"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected KernelCall, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_comparison() {
        let left = make_literal(5.0);
        let right = make_literal(10.0);
        let expr = Expr::binary(BinaryOp::Lt, left, right, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::KernelCall { kernel, args } => {
                assert_eq!(kernel, KernelId::new("compare", "lt"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected KernelCall, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_logical_and() {
        let left = make_literal(1.0);
        let right = make_literal(0.0);
        let expr = Expr::binary(BinaryOp::And, left, right, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::KernelCall { kernel, args } => {
                assert_eq!(kernel, KernelId::new("logic", "and"));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected KernelCall, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_unary_neg() {
        let operand = make_literal(5.0);
        let expr = Expr::unary(UnaryOp::Neg, operand, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::KernelCall { kernel, args } => {
                assert_eq!(kernel, KernelId::new("maths", "neg"));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected KernelCall, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_unary_not() {
        let operand = make_literal(1.0);
        let expr = Expr::unary(UnaryOp::Not, operand, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::KernelCall { kernel, args } => {
                assert_eq!(kernel, KernelId::new("logic", "not"));
                assert_eq!(args.len(), 1);
            }
            _ => panic!("Expected KernelCall, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_if_expression() {
        let condition = make_literal(1.0);
        let then_branch = make_literal(10.0);
        let else_branch = make_literal(20.0);
        let expr = Expr {
            kind: ExprKind::If {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            },
            span: make_span(),
        };

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::KernelCall { kernel, args } => {
                assert_eq!(kernel, KernelId::new("logic", "select"));
                assert_eq!(args.len(), 3);
            }
            _ => panic!("Expected KernelCall, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_nested_operators() {
        // (a + b) * c
        let a = make_literal(1.0);
        let b = make_literal(2.0);
        let c = make_literal(3.0);

        let add = Expr::binary(BinaryOp::Add, a, b, make_span());
        let mul = Expr::binary(BinaryOp::Mul, add, c, make_span());

        let desugared = desugar_expr(mul);

        // Outer should be mul
        match desugared.kind {
            ExprKind::KernelCall { kernel, args } => {
                assert_eq!(kernel, KernelId::new("maths", "mul"));
                assert_eq!(args.len(), 2);

                // First arg should be add
                match &args[0].kind {
                    ExprKind::KernelCall { kernel, .. } => {
                        assert_eq!(*kernel, KernelId::new("maths", "add"));
                    }
                    _ => panic!("Expected nested KernelCall"),
                }
            }
            _ => panic!("Expected KernelCall, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_preserves_let_bindings() {
        let value = Expr::binary(
            BinaryOp::Add,
            make_literal(1.0),
            make_literal(2.0),
            make_span(),
        );
        let body = make_literal(3.0);

        let expr = Expr {
            kind: ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(value),
                body: Box::new(body),
            },
            span: make_span(),
        };

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::Let { name, value, .. } => {
                assert_eq!(name, "x");
                // Value should be desugared to KernelCall
                match value.kind {
                    ExprKind::KernelCall { kernel, .. } => {
                        assert_eq!(kernel, KernelId::new("maths", "add"));
                    }
                    _ => panic!("Expected KernelCall in let value"),
                }
            }
            _ => panic!("Expected Let, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_arithmetic_ops() {
        let test_cases = vec![
            (BinaryOp::Sub, "maths", "sub"),
            (BinaryOp::Div, "maths", "div"),
            (BinaryOp::Mod, "maths", "mod"),
            (BinaryOp::Pow, "maths", "pow"),
        ];

        for (op, namespace, name) in test_cases {
            let left = make_literal(10.0);
            let right = make_literal(3.0);
            let expr = Expr::binary(op, left, right, make_span());
            let desugared = desugar_expr(expr);

            match desugared.kind {
                ExprKind::KernelCall { kernel, args } => {
                    assert_eq!(kernel, KernelId::new(namespace, name));
                    assert_eq!(args.len(), 2);
                    // Verify argument order preserved
                    assert!(
                        matches!(args[0].kind, ExprKind::Literal { value, .. } if (value - 10.0).abs() < 1e-10)
                    );
                    assert!(
                        matches!(args[1].kind, ExprKind::Literal { value, .. } if (value - 3.0).abs() < 1e-10)
                    );
                }
                _ => panic!("Expected KernelCall for {:?}, got {:?}", op, desugared.kind),
            }
        }
    }

    #[test]
    fn test_desugar_comparison_ops() {
        let test_cases = vec![
            (BinaryOp::Eq, "compare", "eq"),
            (BinaryOp::Ne, "compare", "ne"),
            (BinaryOp::Le, "compare", "le"),
            (BinaryOp::Gt, "compare", "gt"),
            (BinaryOp::Ge, "compare", "ge"),
        ];

        for (op, namespace, name) in test_cases {
            let left = make_literal(5.0);
            let right = make_literal(10.0);
            let expr = Expr::binary(op, left, right, make_span());
            let desugared = desugar_expr(expr);

            match desugared.kind {
                ExprKind::KernelCall { kernel, args } => {
                    assert_eq!(kernel, KernelId::new(namespace, name));
                    assert_eq!(args.len(), 2);
                    // Verify argument order preserved
                    assert!(
                        matches!(args[0].kind, ExprKind::Literal { value, .. } if (value - 5.0).abs() < 1e-10)
                    );
                    assert!(
                        matches!(args[1].kind, ExprKind::Literal { value, .. } if (value - 10.0).abs() < 1e-10)
                    );
                }
                _ => panic!("Expected KernelCall for {:?}, got {:?}", op, desugared.kind),
            }
        }
    }

    #[test]
    fn test_desugar_logical_or() {
        let left = make_literal(1.0);
        let right = make_literal(0.0);
        let expr = Expr::binary(BinaryOp::Or, left, right, make_span());

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::KernelCall { kernel, args } => {
                assert_eq!(kernel, KernelId::new("logic", "or"));
                assert_eq!(args.len(), 2);
                // Verify argument order
                assert!(
                    matches!(args[0].kind, ExprKind::Literal { value, .. } if (value - 1.0).abs() < 1e-10)
                );
                assert!(
                    matches!(args[1].kind, ExprKind::Literal { value, .. } if (value - 0.0).abs() < 1e-10)
                );
            }
            _ => panic!("Expected KernelCall, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_recurses_in_vector() {
        let elem1 = Expr::binary(
            BinaryOp::Add,
            make_literal(1.0),
            make_literal(2.0),
            make_span(),
        );
        let elem2 = Expr::binary(
            BinaryOp::Mul,
            make_literal(3.0),
            make_literal(4.0),
            make_span(),
        );
        let vector = Expr {
            kind: ExprKind::Vector(vec![elem1, elem2]),
            span: make_span(),
        };

        let desugared = desugar_expr(vector);

        match desugared.kind {
            ExprKind::Vector(elements) => {
                assert_eq!(elements.len(), 2);
                // First element should be desugared to add
                assert!(
                    matches!(elements[0].kind, ExprKind::KernelCall { ref kernel, .. } if *kernel == KernelId::new("maths", "add"))
                );
                // Second element should be desugared to mul
                assert!(
                    matches!(elements[1].kind, ExprKind::KernelCall { ref kernel, .. } if *kernel == KernelId::new("maths", "mul"))
                );
            }
            _ => panic!("Expected Vector, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_recurses_in_if_branches() {
        let condition = Expr::binary(
            BinaryOp::Lt,
            make_literal(1.0),
            make_literal(2.0),
            make_span(),
        );
        let then_branch = Expr::binary(
            BinaryOp::Add,
            make_literal(10.0),
            make_literal(20.0),
            make_span(),
        );
        let else_branch = Expr::binary(
            BinaryOp::Mul,
            make_literal(30.0),
            make_literal(40.0),
            make_span(),
        );

        let expr = Expr {
            kind: ExprKind::If {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            },
            span: make_span(),
        };

        let desugared = desugar_expr(expr);

        match desugared.kind {
            ExprKind::KernelCall { kernel, args } => {
                assert_eq!(kernel, KernelId::new("logic", "select"));
                assert_eq!(args.len(), 3);
                // Condition should be desugared to lt
                assert!(
                    matches!(args[0].kind, ExprKind::KernelCall { ref kernel, .. } if *kernel == KernelId::new("compare", "lt"))
                );
                // Then branch should be desugared to add
                assert!(
                    matches!(args[1].kind, ExprKind::KernelCall { ref kernel, .. } if *kernel == KernelId::new("maths", "add"))
                );
                // Else branch should be desugared to mul
                assert!(
                    matches!(args[2].kind, ExprKind::KernelCall { ref kernel, .. } if *kernel == KernelId::new("maths", "mul"))
                );
            }
            _ => panic!("Expected KernelCall, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_recurses_in_call_args() {
        use crate::foundation::Path;

        let arg1 = Expr::binary(
            BinaryOp::Add,
            make_literal(1.0),
            make_literal(2.0),
            make_span(),
        );
        let arg2 = make_literal(5.0);
        let call = Expr {
            kind: ExprKind::Call {
                func: Path::from_path_str("some.function"),
                args: vec![arg1, arg2],
            },
            span: make_span(),
        };

        let desugared = desugar_expr(call);

        match desugared.kind {
            ExprKind::Call { func, args } => {
                assert_eq!(func, Path::from_path_str("some.function"));
                assert_eq!(args.len(), 2);
                // First arg should be desugared to add
                assert!(
                    matches!(args[0].kind, ExprKind::KernelCall { ref kernel, .. } if *kernel == KernelId::new("maths", "add"))
                );
                // Second arg should be literal
                assert!(matches!(args[1].kind, ExprKind::Literal { .. }));
            }
            _ => panic!("Expected Call, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_recurses_in_struct_fields() {
        use crate::foundation::Path;

        let field1_value = Expr::binary(
            BinaryOp::Add,
            make_literal(1.0),
            make_literal(2.0),
            make_span(),
        );
        let field2_value = make_literal(10.0);
        let struct_expr = Expr {
            kind: ExprKind::Struct {
                ty: Path::from_path_str("MyType"),
                fields: vec![
                    ("x".to_string(), field1_value),
                    ("y".to_string(), field2_value),
                ],
            },
            span: make_span(),
        };

        let desugared = desugar_expr(struct_expr);

        match desugared.kind {
            ExprKind::Struct { ty, fields } => {
                assert_eq!(ty, Path::from_path_str("MyType"));
                assert_eq!(fields.len(), 2);
                // First field value should be desugared
                assert_eq!(fields[0].0, "x");
                assert!(
                    matches!(fields[0].1.kind, ExprKind::KernelCall { ref kernel, .. } if *kernel == KernelId::new("maths", "add"))
                );
                // Second field value should be literal
                assert_eq!(fields[1].0, "y");
                assert!(matches!(fields[1].1.kind, ExprKind::Literal { .. }));
            }
            _ => panic!("Expected Struct, got {:?}", desugared.kind),
        }
    }

    #[test]
    fn test_desugar_preserves_spans() {
        use crate::foundation::Span;

        // Create a custom span to verify preservation
        let custom_span = Span::new(0, 42, 100, 5);

        let left = make_literal(1.0);
        let right = make_literal(2.0);
        let expr = Expr::binary(BinaryOp::Add, left, right, custom_span);

        let desugared = desugar_expr(expr);

        // Outer span should be preserved
        assert_eq!(desugared.span.start, 42);
        assert_eq!(desugared.span.end, 100);
        assert_eq!(desugared.span.start_line, 5);
    }

    #[test]
    fn test_desugar_aggregate_and_fold() {
        use crate::foundation::{AggregateOp, EntityId};

        // Aggregate: sum(entity) { a + b }
        let body = Expr::binary(
            BinaryOp::Add,
            make_literal(1.0),
            make_literal(2.0),
            make_span(),
        );
        let span = make_span();
        let aggregate = Expr {
            kind: ExprKind::Aggregate {
                op: AggregateOp::Sum,
                source: Box::new(Expr::new(ExprKind::Entity(EntityId::new("plate")), span)),
                binding: "p".to_string(),
                body: Box::new(body),
            },
            span,
        };

        let desugared_agg = desugar_expr(aggregate);
        match desugared_agg.kind {
            ExprKind::Aggregate { body, .. } => {
                assert!(matches!(body.kind, ExprKind::KernelCall { .. }));
            }
            _ => panic!("Expected Aggregate"),
        }

        // Fold: fold(entity, 0.0, acc, elem) { acc + elem }
        let fold_body = Expr::binary(
            BinaryOp::Add,
            Expr::local("acc", make_span()),
            Expr::local("elem", make_span()),
            make_span(),
        );
        let fold = Expr {
            kind: ExprKind::Fold {
                source: Box::new(Expr::new(ExprKind::Entity(EntityId::new("plate")), span)),
                init: Box::new(make_literal(0.0)),
                acc: "acc".to_string(),
                elem: "elem".to_string(),
                body: Box::new(fold_body),
            },
            span,
        };

        let desugared_fold = desugar_expr(fold);
        match desugared_fold.kind {
            ExprKind::Fold { init, body, .. } => {
                assert!(matches!(init.kind, ExprKind::Literal { .. }));
                assert!(matches!(body.kind, ExprKind::KernelCall { .. }));
            }
            _ => panic!("Expected Fold"),
        }
    }

    #[test]
    fn test_desugar_statements() {
        use crate::ast::Stmt;
        use crate::foundation::Path;

        // Let statement: let x = a + b;
        let stmt = Stmt::Let {
            name: "x".to_string(),
            value: Expr::binary(
                BinaryOp::Add,
                make_literal(1.0),
                make_literal(2.0),
                make_span(),
            ),
            span: make_span(),
        };
        let desugared_let = desugar_stmt(stmt);
        match desugared_let {
            Stmt::Let { value, .. } => {
                assert!(matches!(value.kind, ExprKind::KernelCall { .. }));
            }
            _ => panic!("Expected Let stmt"),
        }

        // SignalAssign: signal = a + b;
        let assign = Stmt::SignalAssign {
            target: Path::from_path_str("temp"),
            value: Expr::binary(
                BinaryOp::Add,
                make_literal(1.0),
                make_literal(2.0),
                make_span(),
            ),
            span: make_span(),
        };
        let desugared_assign = desugar_stmt(assign);
        match desugared_assign {
            Stmt::SignalAssign { value, .. } => {
                assert!(matches!(value.kind, ExprKind::KernelCall { .. }));
            }
            _ => panic!("Expected SignalAssign"),
        }

        // FieldAssign: field[pos] = value;
        let field_assign = Stmt::FieldAssign {
            target: Path::from_path_str("pressure"),
            position: Expr::binary(
                BinaryOp::Add,
                make_literal(1.0),
                make_literal(2.0),
                make_span(),
            ),
            value: Expr::binary(
                BinaryOp::Mul,
                make_literal(3.0),
                make_literal(4.0),
                make_span(),
            ),
            span: make_span(),
        };
        let desugared_field = desugar_stmt(field_assign);
        match desugared_field {
            Stmt::FieldAssign {
                position, value, ..
            } => {
                assert!(matches!(position.kind, ExprKind::KernelCall { .. }));
                assert!(matches!(value.kind, ExprKind::KernelCall { .. }));
            }
            _ => panic!("Expected FieldAssign"),
        }
    }

    #[test]
    fn test_desugar_node_structure() {
        use crate::ast::{BlockBody, Node, RoleData};
        use crate::foundation::Path;

        let mut node = Node::new(
            Path::from_path_str("test"),
            make_span(),
            RoleData::Signal,
            (),
        );

        // Add an execution block: resolve { a + b }
        let expr = Expr::binary(
            BinaryOp::Add,
            make_literal(1.0),
            make_literal(2.0),
            make_span(),
        );
        node.execution_blocks
            .push(("resolve".to_string(), BlockBody::Expression(expr)));

        let desugared = desugar_node(node);
        assert_eq!(desugared.execution_blocks.len(), 1);
        match &desugared.execution_blocks[0].1 {
            BlockBody::Expression(e) => {
                assert!(matches!(e.kind, ExprKind::KernelCall { .. }));
            }
            _ => panic!("Expected Expression block"),
        }
    }

    #[test]
    fn test_desugar_era_and_world() {
        use crate::ast::{EraDecl, TransitionDecl, WorldDecl};
        use crate::foundation::Path;

        // Era: dt = a + b
        let era = EraDecl {
            path: Path::from_path_str("test"),
            dt: Some(Expr::binary(
                BinaryOp::Add,
                make_literal(1.0),
                make_literal(2.0),
                make_span(),
            )),
            strata_policy: vec![],
            transitions: vec![TransitionDecl {
                target: Path::from_path_str("next"),
                conditions: vec![Expr::binary(
                    BinaryOp::Lt,
                    make_literal(1.0),
                    make_literal(2.0),
                    make_span(),
                )],
                span: make_span(),
            }],
            attributes: vec![],
            span: make_span(),
            doc: None,
        };

        let desugared_era = desugar_era(era);
        assert!(matches!(
            desugared_era.dt.unwrap().kind,
            ExprKind::KernelCall { .. }
        ));
        assert!(matches!(
            desugared_era.transitions[0].conditions[0].kind,
            ExprKind::KernelCall { .. }
        ));

        // World with attributes
        let mut world = WorldDecl {
            path: Path::from_path_str("terra"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span: make_span(),
            doc: None,
            debug: false,
            policy: crate::ast::WorldPolicy::default(),
        };
        // Add attribute with expression arg
        use crate::ast::Attribute;
        world.attributes.push(Attribute {
            name: "test".to_string(),
            args: vec![Expr::binary(
                BinaryOp::Add,
                make_literal(1.0),
                make_literal(2.0),
                make_span(),
            )],
            span: make_span(),
        });

        let desugared_world = desugar_world(world);
        assert!(matches!(
            desugared_world.attributes[0].args[0].kind,
            ExprKind::KernelCall { .. }
        ));
    }

    #[test]
    fn test_desugar_declarations_mix() {
        use crate::ast::{ConstEntry, Declaration, Node, RoleData, TypeExpr};
        use crate::foundation::Path;

        let decls = vec![
            Declaration::Node(Node::new(
                Path::from_path_str("s1"),
                make_span(),
                RoleData::Signal,
                (),
            )),
            Declaration::Const(vec![ConstEntry {
                path: Path::from_path_str("C"),
                value: Expr::binary(
                    BinaryOp::Add,
                    make_literal(1.0),
                    make_literal(2.0),
                    make_span(),
                ),
                type_expr: TypeExpr::Scalar { unit: None },
                span: make_span(),
                doc: None,
            }]),
        ];

        let desugared = desugar_declarations(decls);
        assert_eq!(desugared.len(), 2);
        match &desugared[1] {
            Declaration::Const(entries) => {
                assert!(matches!(entries[0].value.kind, ExprKind::KernelCall { .. }));
            }
            _ => panic!("Expected Const declaration"),
        }
    }
}
