//! Expression parser for the Continuum DSL.

use chumsky::input::MapExtra;
use chumsky::prelude::*;

use crate::ast::{AggregateOp, BinaryOp, CallArg, Expr, Literal, MathConst, Spanned, UnaryOp};

use super::lexer::Token;
use super::primitives::{ident, number, path, string_lit, unit};
use super::{ParseError, ParserInput};

/// Type alias for boxed spanned expression parser
type SpannedExprBox<'src> =
    Boxed<'src, 'src, ParserInput<'src>, Spanned<Expr>, extra::Err<ParseError<'src>>>;

/// Helper to create a span covering two spanned values
fn span_union<T, U>(left: &Spanned<T>, right: &Spanned<U>) -> SimpleSpan {
    SimpleSpan::from(left.span.start..right.span.end)
}

/// Expression parser - returns just the expression without span info
#[allow(dead_code)]
pub fn expr<'src>()
-> impl Parser<'src, ParserInput<'src>, Expr, extra::Err<ParseError<'src>>> + Clone {
    spanned_expr_inner().map(|spanned| spanned.node)
}

/// Internal spanned expression parser - produces `Spanned<Expr>` with proper spans throughout
fn spanned_expr_inner<'src>()
-> impl Parser<'src, ParserInput<'src>, Spanned<Expr>, extra::Err<ParseError<'src>>> + Clone {
    recursive(|expr| {
        let expr_boxed: SpannedExprBox<'src> = expr.clone().boxed();

        let call_arg = choice((
            ident()
                .then_ignore(just(Token::Colon))
                .then(expr_boxed.clone())
                .map(|(name, value)| CallArg::named(name, value)),
            expr_boxed.clone().map(CallArg::positional),
        ));

        let args = call_arg
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        let entity_atoms = entity_expr_atoms_spanned(expr_boxed.clone()).boxed();

        let core_atoms = choice((
            just(Token::Prev).to(Expr::Prev),
            just(Token::DtRaw).to(Expr::DtRaw),
            just(Token::Pi).to(Expr::MathConst(MathConst::Pi)),
            just(Token::Tau).to(Expr::MathConst(MathConst::Tau)),
            just(Token::Phi).to(Expr::MathConst(MathConst::Phi)),
            just(Token::E).to(Expr::MathConst(MathConst::E)),
            just(Token::I).to(Expr::MathConst(MathConst::I)),
            just(Token::Payload).to(Expr::Payload),
            just(Token::Collected).to(Expr::Collected),
            just(Token::Signal)
                .ignore_then(just(Token::Dot))
                .ignore_then(path())
                .map(Expr::SignalRef),
            just(Token::Const)
                .ignore_then(just(Token::Dot))
                .ignore_then(path())
                .map(Expr::ConstRef),
            just(Token::Config)
                .ignore_then(just(Token::Dot))
                .ignore_then(path())
                .map(Expr::ConfigRef),
            just(Token::Field)
                .ignore_then(just(Token::Dot))
                .ignore_then(path())
                .map(Expr::FieldRef),
        ))
        .map_with(|e, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
            Spanned::new(e, extra.span().into())
        })
        .boxed();

        let other_atoms = choice((
            path()
                .map_with(|p, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    Spanned::new(Expr::Path(p), extra.span().into())
                })
                .then(args.clone())
                .map_with(
                    |(func, args), extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                        Spanned::new(
                            Expr::Call {
                                function: Box::new(func),
                                args,
                            },
                            extra.span().into(),
                        )
                    },
                ),
            number().then(unit().or_not()).map_with(
                |(lit, unit_opt), extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    let e = match unit_opt {
                        Some(u) => Expr::LiteralWithUnit {
                            value: lit,
                            unit: u,
                        },
                        None => Expr::Literal(lit),
                    };
                    Spanned::new(e, extra.span().into())
                },
            ),
            string_lit().map_with(|s, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                Spanned::new(Expr::Literal(Literal::String(s)), extra.span().into())
            }),
            expr_boxed
                .clone()
                .delimited_by(just(Token::LParen), just(Token::RParen)),
            path().map_with(|p, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                Spanned::new(Expr::Path(p), extra.span().into())
            }),
        ))
        .boxed();

        let atom = choice((entity_atoms, core_atoms, other_atoms)).boxed();

        let args_with_span =
            args.clone()
                .map_with(|a, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    (a, extra.span().end)
                });

        let postfix = atom.foldl(
            just(Token::Dot)
                .ignore_then(ident().map_with(
                    |m, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| (m, extra.span()),
                ))
                .then(args_with_span.or_not())
                .repeated(),
            |obj: Spanned<Expr>, ((method, method_span), maybe_args)| {
                let actual_end = match maybe_args {
                    Some((_, end)) => end,
                    None => method_span.end,
                };
                let new_expr = match maybe_args {
                    Some((args, _paren_end)) => Expr::MethodCall {
                        object: Box::new(obj.clone()),
                        method,
                        args,
                    },
                    None => Expr::FieldAccess {
                        object: Box::new(obj.clone()),
                        field: method,
                    },
                };
                Spanned::new(new_expr, (obj.span.start..actual_end).into())
            },
        );

        let unary = choice((
            just(Token::Minus).to(UnaryOp::Neg),
            just(Token::Not).to(UnaryOp::Not),
            just(Token::NotKeyword).to(UnaryOp::Not),
        ))
        .map_with(|op, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
            (op, extra.span().start)
        })
        .repeated()
        .foldr(postfix, |(op, op_start), operand| {
            let span = (op_start..operand.span.end).into();
            Spanned::new(
                Expr::Unary {
                    op,
                    operand: Box::new(operand),
                },
                span,
            )
        });

        let product = unary.clone().foldl(
            choice((
                just(Token::Star).to(BinaryOp::Mul),
                just(Token::Slash).to(BinaryOp::Div),
            ))
            .then(unary)
            .repeated(),
            |left, (op, right)| {
                let span = span_union(&left, &right);
                Spanned::new(
                    Expr::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span.into(),
                )
            },
        );

        let sum = product.clone().foldl(
            choice((
                just(Token::Plus).to(BinaryOp::Add),
                just(Token::Minus).to(BinaryOp::Sub),
            ))
            .then(product)
            .repeated(),
            |left, (op, right)| {
                let span = span_union(&left, &right);
                Spanned::new(
                    Expr::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span.into(),
                )
            },
        );

        let comparison = sum
            .clone()
            .then(
                choice((
                    just(Token::Equals).to(BinaryOp::Eq),
                    just(Token::NotEquals).to(BinaryOp::Ne),
                    just(Token::LessEquals).to(BinaryOp::Le),
                    just(Token::GreaterEquals).to(BinaryOp::Ge),
                    just(Token::LAngle).to(BinaryOp::Lt),
                    just(Token::RAngle).to(BinaryOp::Gt),
                ))
                .then(sum)
                .or_not(),
            )
            .map(|(left, maybe_op_right)| match maybe_op_right {
                Some((op, right)) => {
                    let span = span_union(&left, &right);
                    Spanned::new(
                        Expr::Binary {
                            op,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        span.into(),
                    )
                }
                None => left,
            });

        let and_op = choice((
            just(Token::And).to(BinaryOp::And),
            just(Token::AndKeyword).to(BinaryOp::And),
        ));
        let logical_and =
            comparison
                .clone()
                .foldl(and_op.then(comparison).repeated(), |left, (op, right)| {
                    let span = span_union(&left, &right);
                    Spanned::new(
                        Expr::Binary {
                            op,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        span.into(),
                    )
                });

        let or_op = choice((
            just(Token::Or).to(BinaryOp::Or),
            just(Token::OrKeyword).to(BinaryOp::Or),
        ));
        let logical_or =
            logical_and
                .clone()
                .foldl(or_op.then(logical_and).repeated(), |left, (op, right)| {
                    let span = span_union(&left, &right);
                    Spanned::new(
                        Expr::Binary {
                            op,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        span.into(),
                    )
                });

        let logical_or_boxed = logical_or.clone().boxed();

        let let_expr = just(Token::Let)
            .map_with(|_, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| extra.span().start)
            .then(ident())
            .then_ignore(just(Token::Assign))
            .then(expr_boxed.clone())
            .then_ignore(just(Token::In))
            .then(expr_boxed.clone())
            .map(|(((let_start, name), value), body)| {
                let span = (let_start..body.span.end).into();
                Spanned::new(
                    Expr::Let {
                        name,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    span,
                )
            });

        let if_expr = just(Token::If)
            .map_with(|_, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| extra.span().start)
            .then(logical_or_boxed)
            .then(
                expr_boxed
                    .clone()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .then(
                just(Token::Else)
                    .ignore_then(expr_boxed.delimited_by(just(Token::LBrace), just(Token::RBrace)))
                    .or_not(),
            )
            .map(
                |(((if_start, condition), then_branch), else_branch): (
                    ((_, _), _),
                    Option<Spanned<Expr>>,
                )| {
                    let span_end = else_branch
                        .as_ref()
                        .map(|e| e.span.end)
                        .unwrap_or(then_branch.span.end);
                    Spanned::new(
                        Expr::If {
                            condition: Box::new(condition),
                            then_branch: Box::new(then_branch),
                            else_branch: else_branch.map(Box::new),
                        },
                        (if_start..span_end).into(),
                    )
                },
            );

        choice((let_expr, if_expr, logical_or))
    })
}

fn entity_expr_atoms_spanned<'src>(
    expr_boxed: SpannedExprBox<'src>,
) -> impl Parser<'src, ParserInput<'src>, Spanned<Expr>, extra::Err<ParseError<'src>>> + Clone {
    choice((
        just(Token::SelfToken)
            .ignore_then(just(Token::Dot))
            .ignore_then(ident())
            .map_with(
                |field, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    Spanned::new(Expr::SelfField(field), extra.span().into())
                },
            ),
        just(Token::Entity)
            .ignore_then(just(Token::Dot))
            .ignore_then(path())
            .then(
                just(Token::LBracket)
                    .ignore_then(expr_boxed.clone())
                    .then_ignore(just(Token::RBracket))
                    .or_not(),
            )
            .map_with(
                |(entity, instance), extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    let e = match instance {
                        Some(inst) => Expr::EntityAccess {
                            entity,
                            instance: Box::new(inst),
                        },
                        None => Expr::EntityRef(entity),
                    };
                    Spanned::new(e, extra.span().into())
                },
            ),
        just(Token::Count)
            .ignore_then(
                just(Token::LParen)
                    .ignore_then(just(Token::Entity))
                    .ignore_then(just(Token::Dot))
                    .ignore_then(path())
                    .then_ignore(just(Token::RParen)),
            )
            .map_with(
                |entity, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    let span = extra.span();
                    Spanned::new(
                        Expr::Aggregate {
                            op: AggregateOp::Count,
                            entity,
                            body: Box::new(Spanned::new(
                                Expr::Literal(Literal::Integer(1)),
                                span.into(),
                            )),
                        },
                        span.into(),
                    )
                },
            ),
        just(Token::Other)
            .ignore_then(
                just(Token::LParen)
                    .ignore_then(just(Token::Entity))
                    .ignore_then(just(Token::Dot))
                    .ignore_then(path())
                    .then_ignore(just(Token::RParen)),
            )
            .map_with(
                |entity, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    Spanned::new(Expr::Other(entity), extra.span().into())
                },
            ),
        just(Token::Pairs)
            .ignore_then(
                just(Token::LParen)
                    .ignore_then(just(Token::Entity))
                    .ignore_then(just(Token::Dot))
                    .ignore_then(path())
                    .then_ignore(just(Token::RParen)),
            )
            .map_with(
                |entity, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    Spanned::new(Expr::Pairs(entity), extra.span().into())
                },
            ),
    ))
    .or(entity_aggregate_atoms_spanned(expr_boxed))
}

fn entity_aggregate_atoms_spanned<'src>(
    expr_boxed: SpannedExprBox<'src>,
) -> impl Parser<'src, ParserInput<'src>, Spanned<Expr>, extra::Err<ParseError<'src>>> + Clone {
    let aggregate_with_body = aggregate_op_with_body()
        .then(
            just(Token::LParen)
                .ignore_then(just(Token::Entity))
                .ignore_then(just(Token::Dot))
                .ignore_then(path())
                .then_ignore(just(Token::Comma))
                .then(expr_boxed.clone())
                .then_ignore(just(Token::RParen)),
        )
        .map_with(
            |(op, (entity, body)), extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                Spanned::new(
                    Expr::Aggregate {
                        op,
                        entity,
                        body: Box::new(body),
                    },
                    extra.span().into(),
                )
            },
        );

    let filter_expr = just(Token::Filter)
        .ignore_then(
            just(Token::LParen)
                .ignore_then(just(Token::Entity))
                .ignore_then(just(Token::Dot))
                .ignore_then(path())
                .then_ignore(just(Token::Comma))
                .then(expr_boxed.clone())
                .then_ignore(just(Token::RParen)),
        )
        .map_with(
            |(entity, predicate), extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                Spanned::new(
                    Expr::Filter {
                        entity,
                        predicate: Box::new(predicate),
                    },
                    extra.span().into(),
                )
            },
        );

    let first_expr = just(Token::First)
        .ignore_then(
            just(Token::LParen)
                .ignore_then(just(Token::Entity))
                .ignore_then(just(Token::Dot))
                .ignore_then(path())
                .then_ignore(just(Token::Comma))
                .then(expr_boxed.clone())
                .then_ignore(just(Token::RParen)),
        )
        .map_with(
            |(entity, predicate), extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                Spanned::new(
                    Expr::First {
                        entity,
                        predicate: Box::new(predicate),
                    },
                    extra.span().into(),
                )
            },
        );

    let nearest_expr = just(Token::Nearest)
        .ignore_then(
            just(Token::LParen)
                .ignore_then(just(Token::Entity))
                .ignore_then(just(Token::Dot))
                .ignore_then(path())
                .then_ignore(just(Token::Comma))
                .then(expr_boxed.clone())
                .then_ignore(just(Token::RParen)),
        )
        .map_with(
            |(entity, position), extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                Spanned::new(
                    Expr::Nearest {
                        entity,
                        position: Box::new(position),
                    },
                    extra.span().into(),
                )
            },
        );

    let within_expr = just(Token::Within)
        .ignore_then(
            just(Token::LParen)
                .ignore_then(just(Token::Entity))
                .ignore_then(just(Token::Dot))
                .ignore_then(path())
                .then_ignore(just(Token::Comma))
                .then(expr_boxed.clone())
                .then_ignore(just(Token::Comma))
                .then(expr_boxed)
                .then_ignore(just(Token::RParen)),
        )
        .map_with(
            |((entity, position), radius), extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                Spanned::new(
                    Expr::Within {
                        entity,
                        position: Box::new(position),
                        radius: Box::new(radius),
                    },
                    extra.span().into(),
                )
            },
        );

    choice((
        aggregate_with_body,
        filter_expr,
        first_expr,
        nearest_expr,
        within_expr,
    ))
}

pub fn spanned_expr<'src>()
-> impl Parser<'src, ParserInput<'src>, Spanned<Expr>, extra::Err<ParseError<'src>>> + Clone {
    spanned_expr_inner()
}

fn aggregate_op_with_body<'src>()
-> impl Parser<'src, ParserInput<'src>, AggregateOp, extra::Err<ParseError<'src>>> + Clone {
    choice((
        just(Token::Sum).to(AggregateOp::Sum),
        just(Token::Product).to(AggregateOp::Product),
        just(Token::Min).to(AggregateOp::Min),
        just(Token::Max).to(AggregateOp::Max),
        just(Token::Mean).to(AggregateOp::Mean),
        just(Token::Any).to(AggregateOp::Any),
        just(Token::All).to(AggregateOp::All),
        just(Token::None).to(AggregateOp::None),
    ))
}
