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

use continuum_cdsl_ast::foundation::Span;
use continuum_cdsl_ast::{
    Attribute, BlockBody, Declaration, Entity, EraDecl, Expr, Index, KernelId, Node, ObserveBlock,
    ObserveWhen, Stmt, Stratum, UntypedKind as ExprKind, WarmupBlock, WhenBlock, WorldDecl,
};
use continuum_kernel_registry::get_constant_mapping;

/// Construct a kernel call expression
fn kernel_call(kernel: KernelId, args: Vec<Expr>, span: Span) -> Expr {
    Expr {
        kind: ExprKind::KernelCall { kernel, args },
        span,
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
        // Keep Binary nodes for type-aware dispatch during type resolution
        ExprKind::Binary { op, left, right } => Expr {
            kind: ExprKind::Binary {
                op,
                left: Box::new(desugar_expr(*left)),
                right: Box::new(desugar_expr(*right)),
            },
            span,
        },

        ExprKind::Unary { op, operand } => Expr {
            kind: ExprKind::Unary {
                op,
                operand: Box::new(desugar_expr(*operand)),
            },
            span,
        },

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

        // Mathematical constants: desugar bare identifiers registered as constants
        // Example: PI, π → maths.PI(), TAU, τ → maths.TAU()
        ExprKind::Local(ref name) => {
            if let Some((namespace, kernel_name)) = get_constant_mapping(name) {
                kernel_call(KernelId::new(namespace, kernel_name), vec![], span)
            } else {
                expr
            }
        }

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
        Stmt::MemberSignalAssign {
            entity,
            instance,
            member,
            value,
            span,
        } => Stmt::MemberSignalAssign {
            entity,
            instance: desugar_expr(instance),
            member,
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
        Stmt::Assert {
            condition,
            severity,
            message,
            span,
        } => Stmt::Assert {
            condition: desugar_expr(condition),
            severity,
            message,
            span,
        },
        Stmt::EmitEvent { path, fields, span } => Stmt::EmitEvent {
            path,
            fields: fields
                .into_iter()
                .map(|(name, expr)| (name, desugar_expr(expr)))
                .collect(),
            span,
        },
        Stmt::Expr(expr) => Stmt::Expr(desugar_expr(expr)),
        Stmt::If {
            condition,
            then_branch,
            else_branch,
            span,
        } => Stmt::If {
            condition: desugar_expr(condition),
            then_branch: then_branch.into_iter().map(desugar_stmt).collect(),
            else_branch: else_branch.into_iter().map(desugar_stmt).collect(),
            span,
        },
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
            Declaration::Function(func) => Declaration::Function(func),
        })
        .collect()
}
