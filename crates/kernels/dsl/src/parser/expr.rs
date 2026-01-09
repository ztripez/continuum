//! Expression parser

use chumsky::prelude::*;

use crate::ast::{AggregateOp, BinaryOp, Expr, Literal, MathConst, Path, Spanned, UnaryOp};

use super::primitives::{ident, number, path, string_lit, unit, ws};
use super::ParseError;

/// Type alias for parser Extra to ensure consistency across helper functions
type Ex<'src> = extra::Err<ParseError<'src>>;

/// Type alias for boxed expression parser
type ExprBox<'src> = Boxed<'src, 'src, &'src str, Expr, Ex<'src>>;

/// Expression parser
///
/// Uses `.boxed()` at strategic points to reduce compile times by breaking the type chain.
/// Without boxing, chumsky's parser combinators create deeply nested generic types that
/// cause exponential compile time growth.
pub fn expr<'src>() -> impl Parser<'src, &'src str, Expr, Ex<'src>> + Clone {
    recursive(|expr| {
        // Box the recursive expr to prevent type explosion
        let expr_boxed = expr.clone().boxed();

        // Arguments list for function calls
        let args = expr_boxed
            .clone()
            .map_with(|e, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                Spanned::new(e, span.start..span.end)
            })
            .separated_by(just(',').padded_by(ws()))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just('(').padded_by(ws()), just(')').padded_by(ws()));

        // Entity expression atoms - boxed to reduce type complexity
        let entity_atoms = entity_expr_atoms(expr_boxed.clone()).boxed();

        // Core atoms (non-entity)
        let core_atoms = choice((
            text::keyword("prev").to(Expr::Prev),
            just("dt_raw").to(Expr::DtRaw),
            // Math constants (ASCII and Unicode)
            just("PI").or(just("π")).to(Expr::MathConst(MathConst::Pi)),
            just("TAU").or(just("τ")).to(Expr::MathConst(MathConst::Tau)),
            just("PHI").or(just("φ")).to(Expr::MathConst(MathConst::Phi)),
            just("E").or(just("ℯ")).to(Expr::MathConst(MathConst::E)),
            just("I").or(just("ⅈ")).to(Expr::MathConst(MathConst::I)),
            text::keyword("payload").to(Expr::Payload),
            text::keyword("sum")
                .ignore_then(
                    just('(')
                        .padded_by(ws())
                        .ignore_then(text::keyword("inputs"))
                        .ignore_then(just(')').padded_by(ws())),
                )
                .to(Expr::SumInputs),
            text::keyword("signal")
                .ignore_then(just('.'))
                .ignore_then(path())
                .map(Expr::SignalRef),
            text::keyword("const")
                .ignore_then(just('.'))
                .ignore_then(path())
                .map(Expr::ConstRef),
            text::keyword("config")
                .ignore_then(just('.'))
                .ignore_then(path())
                .map(Expr::ConfigRef),
            text::keyword("field")
                .ignore_then(just('.'))
                .ignore_then(path())
                .map(Expr::FieldRef),
        ))
        .boxed();

        // Additional atoms (function calls, literals, paths)
        let other_atoms = choice((
            // Function call: name(args) or path.to.func(args)
            path()
                .then(args.clone())
                .map(|(p, args)| Expr::Call {
                    function: Box::new(Spanned::new(Expr::Path(p), 0..0)),
                    args,
                }),
            number()
                .then(unit().padded_by(ws()).or_not())
                .map(|(lit, unit_opt)| match unit_opt {
                    Some(u) => Expr::LiteralWithUnit { value: lit, unit: u },
                    None => Expr::Literal(lit),
                }),
            string_lit().map(|s| Expr::Literal(Literal::String(s))),
            expr_boxed
                .clone()
                .padded_by(ws())
                .delimited_by(just('('), just(')')),
            // Plain path (must come after function call attempt)
            path().map(Expr::Path),
        ))
        .boxed();

        // Combine all atoms
        let atom = choice((entity_atoms, core_atoms, other_atoms))
            .padded_by(ws())
            .boxed();

        // Method calls: expr.method(args)
        let postfix = atom.foldl(
            just('.')
                .padded_by(ws())
                .ignore_then(ident())
                .then(args.or_not())
                .repeated(),
            |obj, (method, maybe_args)| match maybe_args {
                Some(args) => Expr::Call {
                    function: Box::new(Spanned::new(
                        Expr::Path(Path::new(vec![method])),
                        0..0,
                    )),
                    args: std::iter::once(Spanned::new(obj, 0..0))
                        .chain(args)
                        .collect(),
                },
                None => Expr::FieldAccess {
                    object: Box::new(Spanned::new(obj, 0..0)),
                    field: method,
                },
            },
        );

        let unary = just('-').repeated().foldr(postfix, |_, operand| Expr::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Spanned::new(operand, 0..0)),
        });

        let product = unary.clone().foldl(
            choice((just('*').to(BinaryOp::Mul), just('/').to(BinaryOp::Div)))
                .padded_by(ws())
                .then(unary)
                .repeated(),
            |left, (op, right)| Expr::Binary {
                op,
                left: Box::new(Spanned::new(left, 0..0)),
                right: Box::new(Spanned::new(right, 0..0)),
            },
        );

        let sum = product.clone().foldl(
            choice((just('+').to(BinaryOp::Add), just('-').to(BinaryOp::Sub)))
                .padded_by(ws())
                .then(product)
                .repeated(),
            |left, (op, right)| Expr::Binary {
                op,
                left: Box::new(Spanned::new(left, 0..0)),
                right: Box::new(Spanned::new(right, 0..0)),
            },
        );

        let comparison = sum.clone().foldl(
            choice((
                just("==").to(BinaryOp::Eq),
                just("!=").to(BinaryOp::Ne),
                just("<=").to(BinaryOp::Le),
                just(">=").to(BinaryOp::Ge),
                just('<').to(BinaryOp::Lt),
                just('>').to(BinaryOp::Gt),
            ))
            .padded_by(ws())
            .then(sum)
            .repeated(),
            |left, (op, right)| Expr::Binary {
                op,
                left: Box::new(Spanned::new(left, 0..0)),
                right: Box::new(Spanned::new(right, 0..0)),
            },
        );

        // Logical AND has lower precedence than comparison
        let logical_and = comparison.clone().foldl(
            just("&&")
                .to(BinaryOp::And)
                .padded_by(ws())
                .then(comparison)
                .repeated(),
            |left, (op, right)| Expr::Binary {
                op,
                left: Box::new(Spanned::new(left, 0..0)),
                right: Box::new(Spanned::new(right, 0..0)),
            },
        );

        // Logical OR has lower precedence than AND
        let logical_or = logical_and.clone().foldl(
            just("||")
                .to(BinaryOp::Or)
                .padded_by(ws())
                .then(logical_and)
                .repeated(),
            |left, (op, right)| Expr::Binary {
                op,
                left: Box::new(Spanned::new(left, 0..0)),
                right: Box::new(Spanned::new(right, 0..0)),
            },
        );

        // Box logical_or before using in if_expr to reduce type complexity
        let logical_or_boxed = logical_or.clone().boxed();

        // Let expression: let name = value \n body
        // Multiple lets chain together: let a = 1 \n let b = 2 \n a + b
        let let_expr = text::keyword("let")
            .padded_by(ws())
            .ignore_then(ident())
            .then_ignore(just('=').padded_by(ws()))
            .then(expr_boxed.clone().map_with(|e, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                Spanned::new(e, span.start..span.end)
            }))
            .then(expr_boxed.clone().map_with(|e, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                Spanned::new(e, span.start..span.end)
            }))
            .map(|((name, value), body)| Expr::Let {
                name,
                value: Box::new(value),
                body: Box::new(body),
            });

        // If expression: if condition { then } else { else }
        // The condition uses logical_or (not expr) to avoid infinite recursion
        // (conditions shouldn't be let or if expressions without braces)
        let if_expr = text::keyword("if")
            .padded_by(ws())
            .ignore_then(logical_or_boxed.map_with(|e, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                Spanned::new(e, span.start..span.end)
            }))
            .then(
                expr_boxed
                    .clone()
                    .map_with(|e, extra| {
                        let span: chumsky::span::SimpleSpan = extra.span();
                        Spanned::new(e, span.start..span.end)
                    })
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .then(
                text::keyword("else")
                    .padded_by(ws())
                    .ignore_then(
                        expr_boxed
                            .map_with(|e, extra| {
                                let span: chumsky::span::SimpleSpan = extra.span();
                                Spanned::new(e, span.start..span.end)
                            })
                            .padded_by(ws())
                            .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
                    )
                    .or_not(),
            )
            .map(|((condition, then_branch), else_branch)| Expr::If {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: else_branch.map(Box::new),
            });

        // Let and if expressions have lowest precedence - they consume the rest as body
        choice((let_expr, if_expr, logical_or))
    })
}

/// Entity expression atoms - separated to reduce type complexity
fn entity_expr_atoms<'src>(expr_boxed: ExprBox<'src>) -> impl Parser<'src, &'src str, Expr, Ex<'src>> + Clone {
    // Helper for spanned expressions
    let spanned = |p: ExprBox<'src>| {
        p.map_with(|e, extra| {
            let span: chumsky::span::SimpleSpan = extra.span();
            Spanned::new(e, span.start..span.end)
        })
    };

    choice((
        // self.field - current entity instance field access
        text::keyword("self")
            .ignore_then(just('.').padded_by(ws()))
            .ignore_then(ident())
            .map(Expr::SelfField),
        // entity.path["name"] - entity instance access
        text::keyword("entity")
            .ignore_then(just('.'))
            .ignore_then(path())
            .then(
                just('[')
                    .padded_by(ws())
                    .ignore_then(spanned(expr_boxed.clone()).padded_by(ws()))
                    .then_ignore(just(']').padded_by(ws()))
                    .or_not(),
            )
            .map(|(entity, instance)| match instance {
                Some(inst) => Expr::EntityAccess {
                    entity,
                    instance: Box::new(inst),
                },
                None => Expr::EntityRef(entity),
            }),
        // count(entity.path) - special case, no body expression needed
        text::keyword("count")
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(text::keyword("entity"))
                    .ignore_then(just('.'))
                    .ignore_then(path())
                    .then_ignore(just(')').padded_by(ws())),
            )
            .map(|entity| Expr::Aggregate {
                op: AggregateOp::Count,
                entity,
                body: Box::new(Spanned::new(Expr::Literal(Literal::Integer(1)), 0..0)),
            }),
        // other(entity.path) - self-exclusion for N-body
        text::keyword("other")
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(text::keyword("entity"))
                    .ignore_then(just('.'))
                    .ignore_then(path())
                    .then_ignore(just(')').padded_by(ws())),
            )
            .map(Expr::Other),
        // pairs(entity.path) - pairwise iteration
        text::keyword("pairs")
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(text::keyword("entity"))
                    .ignore_then(just('.'))
                    .ignore_then(path())
                    .then_ignore(just(')').padded_by(ws())),
            )
            .map(Expr::Pairs),
    ))
    .or(entity_aggregate_atoms(expr_boxed))
}

/// Entity aggregate operations - further split to reduce type complexity
fn entity_aggregate_atoms<'src>(expr_boxed: ExprBox<'src>) -> impl Parser<'src, &'src str, Expr, Ex<'src>> + Clone {
    // Helper for spanned expressions
    let spanned = move |p: ExprBox<'src>| {
        p.map_with(|e, extra| {
            let span: chumsky::span::SimpleSpan = extra.span();
            Spanned::new(e, span.start..span.end)
        })
    };

    // Aggregate with body: sum(entity.path, expr)
    let aggregate_with_body = aggregate_op_with_body()
        .then(
            just('(')
                .padded_by(ws())
                .ignore_then(text::keyword("entity"))
                .ignore_then(just('.'))
                .ignore_then(path())
                .then_ignore(just(',').padded_by(ws()))
                .then(spanned(expr_boxed.clone()))
                .then_ignore(just(')').padded_by(ws())),
        )
        .map(|(op, (entity, body))| Expr::Aggregate {
            op,
            entity,
            body: Box::new(body),
        });

    // filter(entity.path, predicate)
    let filter_expr = text::keyword("filter")
        .ignore_then(
            just('(')
                .padded_by(ws())
                .ignore_then(text::keyword("entity"))
                .ignore_then(just('.'))
                .ignore_then(path())
                .then_ignore(just(',').padded_by(ws()))
                .then(spanned(expr_boxed.clone()))
                .then_ignore(just(')').padded_by(ws())),
        )
        .map(|(entity, predicate)| Expr::Filter {
            entity,
            predicate: Box::new(predicate),
        });

    // first(entity.path, predicate)
    let first_expr = text::keyword("first")
        .ignore_then(
            just('(')
                .padded_by(ws())
                .ignore_then(text::keyword("entity"))
                .ignore_then(just('.'))
                .ignore_then(path())
                .then_ignore(just(',').padded_by(ws()))
                .then(spanned(expr_boxed.clone()))
                .then_ignore(just(')').padded_by(ws())),
        )
        .map(|(entity, predicate)| Expr::First {
            entity,
            predicate: Box::new(predicate),
        });

    // nearest(entity.path, position)
    let nearest_expr = text::keyword("nearest")
        .ignore_then(
            just('(')
                .padded_by(ws())
                .ignore_then(text::keyword("entity"))
                .ignore_then(just('.'))
                .ignore_then(path())
                .then_ignore(just(',').padded_by(ws()))
                .then(spanned(expr_boxed.clone()))
                .then_ignore(just(')').padded_by(ws())),
        )
        .map(|(entity, position)| Expr::Nearest {
            entity,
            position: Box::new(position),
        });

    // within(entity.path, position, radius)
    let within_expr = text::keyword("within")
        .ignore_then(
            just('(')
                .padded_by(ws())
                .ignore_then(text::keyword("entity"))
                .ignore_then(just('.'))
                .ignore_then(path())
                .then_ignore(just(',').padded_by(ws()))
                .then(spanned(expr_boxed.clone()))
                .then_ignore(just(',').padded_by(ws()))
                .then(spanned(expr_boxed))
                .then_ignore(just(')').padded_by(ws())),
        )
        .map(|((entity, position), radius)| Expr::Within {
            entity,
            position: Box::new(position),
            radius: Box::new(radius),
        });

    choice((
        aggregate_with_body,
        filter_expr,
        first_expr,
        nearest_expr,
        within_expr,
    ))
}

/// Spanned expression
pub fn spanned_expr<'src>() -> impl Parser<'src, &'src str, Spanned<Expr>, Ex<'src>> + Clone {
    expr().map_with(|e, extra| Spanned::new(e, extra.span().into()))
}

/// Parser for aggregate operations that take a body expression
fn aggregate_op_with_body<'src>() -> impl Parser<'src, &'src str, AggregateOp, Ex<'src>> + Clone {
    choice((
        text::keyword("sum").to(AggregateOp::Sum),
        text::keyword("product").to(AggregateOp::Product),
        text::keyword("min").to(AggregateOp::Min),
        text::keyword("max").to(AggregateOp::Max),
        text::keyword("mean").to(AggregateOp::Mean),
        text::keyword("any").to(AggregateOp::Any),
        text::keyword("all").to(AggregateOp::All),
        text::keyword("none").to(AggregateOp::None),
    ))
}
