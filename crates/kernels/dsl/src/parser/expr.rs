//! Expression parser for the Continuum DSL.

use chumsky::input::MapExtra;
use chumsky::prelude::*;

use crate::ast::{AggregateOp, BinaryOp, CallArg, Expr, Literal, MathConst, Spanned, UnaryOp};

use super::lexer::Token;
use super::primitives::{ident, number, path, string_lit, tok, unit};
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
    spanned_expr_inner(false).map(|spanned| spanned.node)
}

/// Internal spanned expression parser - produces `Spanned<Expr>` with proper spans throughout
fn spanned_expr_inner<'src>(
    allow_emit: bool,
) -> impl Parser<'src, ParserInput<'src>, Spanned<Expr>, extra::Err<ParseError<'src>>> + Clone {
    recursive(|expr| {
        let expr_boxed: SpannedExprBox<'src> = expr.clone().boxed();

        let call_arg = choice((
            ident()
                .then_ignore(tok(Token::Colon))
                .then(expr_boxed.clone())
                .map(|(name, value)| CallArg::named(name, value)),
            expr_boxed.clone().map(CallArg::positional),
        ));

        let args = call_arg
            .separated_by(tok(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(tok(Token::LParen), tok(Token::RParen));

        let entity_atoms = entity_expr_atoms_spanned(expr_boxed.clone()).boxed();

        let core_atoms = choice((
            tok(Token::Prev).to(Expr::Prev),
            tok(Token::DtRaw).to(Expr::DtRaw),
            tok(Token::SimTime).to(Expr::SimTime),
            tok(Token::Pi).to(Expr::MathConst(MathConst::Pi)),
            tok(Token::Tau).to(Expr::MathConst(MathConst::Tau)),
            tok(Token::Phi).to(Expr::MathConst(MathConst::Phi)),
            tok(Token::E).to(Expr::MathConst(MathConst::E)),
            tok(Token::I).to(Expr::MathConst(MathConst::I)),
            tok(Token::Payload).to(Expr::Payload),
            tok(Token::Collected).to(Expr::Collected),
            // Look up constants from the registry
            ident()
                .filter(|name| crate::math_consts::lookup(name).is_some())
                .map(|name| {
                    Expr::Literal(Literal::Float(crate::math_consts::lookup(&name).unwrap()))
                }),
            tok(Token::Signal)
                .ignore_then(tok(Token::Dot))
                .ignore_then(path())
                .map(Expr::SignalRef),
            tok(Token::Const)
                .ignore_then(tok(Token::Dot))
                .ignore_then(path())
                .map(Expr::ConstRef),
            tok(Token::Config)
                .ignore_then(tok(Token::Dot))
                .ignore_then(path())
                .map(Expr::ConfigRef),
            tok(Token::Field)
                .ignore_then(tok(Token::Dot))
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
            select! { Token::Bool(b) => Expr::Literal(Literal::Bool(b)) }.map_with(
                |e, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    Spanned::new(e, extra.span().into())
                },
            ),
            expr_boxed
                .clone()
                .delimited_by(tok(Token::LParen), tok(Token::RParen)),
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
            tok(Token::Dot)
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
            tok(Token::Minus).to(UnaryOp::Neg),
            tok(Token::Not).to(UnaryOp::Not),
            tok(Token::NotKeyword).to(UnaryOp::Not),
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
        })
        .boxed();

        let product = unary
            .clone()
            .foldl(
                choice((
                    tok(Token::Star).to(BinaryOp::Mul),
                    tok(Token::Slash).to(BinaryOp::Div),
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
            )
            .boxed();

        let sum = product
            .clone()
            .foldl(
                choice((
                    tok(Token::Plus).to(BinaryOp::Add),
                    tok(Token::Minus).to(BinaryOp::Sub),
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
            )
            .boxed();

        let comparison = sum
            .clone()
            .then(
                choice((
                    tok(Token::Equals).to(BinaryOp::Eq),
                    tok(Token::NotEquals).to(BinaryOp::Ne),
                    tok(Token::LessEquals).to(BinaryOp::Le),
                    tok(Token::GreaterEquals).to(BinaryOp::Ge),
                    tok(Token::LAngle).to(BinaryOp::Lt),
                    tok(Token::RAngle).to(BinaryOp::Gt),
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
            })
            .boxed();

        let and_op = choice((
            tok(Token::And).to(BinaryOp::And),
            tok(Token::AndKeyword).to(BinaryOp::And),
        ));
        let logical_and = comparison
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
            })
            .boxed();

        let or_op = choice((
            tok(Token::Or).to(BinaryOp::Or),
            tok(Token::OrKeyword).to(BinaryOp::Or),
        ));
        let logical_or = logical_and
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
            })
            .boxed();

        let logical_or_boxed = logical_or.clone();

        // Emit expression: signal.path <- value
        let emit_expr = tok(Token::Signal)
            .ignore_then(tok(Token::Dot))
            .ignore_then(path())
            .map_with(|p, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                (p, extra.span().start)
            })
            .then_ignore(tok(Token::EmitArrow))
            .then(expr_boxed.clone())
            .map(|((target, start), value)| {
                let span = (start..value.span.end).into();
                Spanned::new(
                    Expr::EmitSignal {
                        target,
                        value: Box::new(value),
                    },
                    span,
                )
            })
            .boxed();

        // If expression: if condition { then } else { else }
        let if_expr = {
            // Braced block: { expr ; expr ; ... } or { expr }
            let braced = tok(Token::LBrace)
                .ignore_then(
                    expr_boxed
                        .clone()
                        .separated_by(tok(Token::Semicolon).or_not())
                        .allow_trailing()
                        .at_least(1)
                        .collect::<Vec<_>>(),
                )
                .then_ignore(tok(Token::RBrace))
                .map_with(
                    |exprs, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                        let span = extra.span();
                        if exprs.len() == 1 {
                            exprs.into_iter().next().unwrap()
                        } else {
                            Spanned::new(Expr::Block(exprs), span.into())
                        }
                    },
                );

            let if_head = tok(Token::If)
                .map_with(|_, e: &mut MapExtra<'src, '_, ParserInput<'src>, _>| e.span().start)
                .then(logical_or_boxed.clone())
                .then(braced.clone());

            let else_if_clause = tok(Token::Else)
                .ignore_then(tok(Token::If))
                .ignore_then(logical_or_boxed.clone())
                .then(braced.clone());

            let else_final = tok(Token::Else).ignore_then(braced.clone());

            if_head
                .then(else_if_clause.repeated().collect::<Vec<_>>())
                .then(else_final.or_not())
                .map(|((((if_start, cond), then_block), else_ifs), else_final)| {
                    let mut else_branch = else_final;

                    for (ei_cond, ei_block) in else_ifs.into_iter().rev() {
                        let span_start = ei_cond.span.start;
                        let span_end = else_branch
                            .as_ref()
                            .map(|e| e.span.end)
                            .unwrap_or(ei_block.span.end);
                        else_branch = Some(Spanned::new(
                            Expr::If {
                                condition: Box::new(ei_cond),
                                then_branch: Box::new(ei_block),
                                else_branch: else_branch.map(Box::new),
                            },
                            (span_start..span_end).into(),
                        ));
                    }

                    let final_span_end = else_branch
                        .as_ref()
                        .map(|e| e.span.end)
                        .unwrap_or(then_block.span.end);

                    Spanned::new(
                        Expr::If {
                            condition: Box::new(cond),
                            then_branch: Box::new(then_block),
                            else_branch: else_branch.map(Box::new),
                        },
                        (if_start..final_span_end).into(),
                    )
                })
                .boxed()
        };

        let expr_without_let = {
            let mut without_let_parsers = vec![if_expr.clone(), logical_or.clone()];
            if allow_emit {
                without_let_parsers.insert(0, emit_expr.clone());
            }
            choice(without_let_parsers).boxed()
        };

        // Let expression: let name = value in body
        let let_expr = {
            let let_binding = tok(Token::Let)
                .map_with(|_, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    extra.span().start
                })
                .then(ident())
                .then_ignore(tok(Token::Assign))
                .then(expr_without_let)
                .then_ignore(tok(Token::In));

            let_binding
                .repeated()
                .at_least(1)
                .collect::<Vec<_>>()
                .then(expr_boxed.clone())
                .map(|(bindings, body)| {
                    let mut result = body;
                    for ((let_start, name), value) in bindings.into_iter().rev() {
                        let span = (let_start..result.span.end).into();
                        result = Spanned::new(
                            Expr::Let {
                                name,
                                value: Box::new(value),
                                body: Box::new(result),
                            },
                            span,
                        );
                    }
                    result
                })
                .boxed()
        };

        let mut final_parsers = vec![let_expr, if_expr, logical_or];
        if allow_emit {
            final_parsers.insert(0, emit_expr);
        }
        choice(final_parsers)
    })
}

fn entity_expr_atoms_spanned<'src>(
    expr_boxed: SpannedExprBox<'src>,
) -> impl Parser<'src, ParserInput<'src>, Spanned<Expr>, extra::Err<ParseError<'src>>> + Clone {
    choice((
        tok(Token::SelfToken)
            .ignore_then(tok(Token::Dot))
            .ignore_then(ident())
            .map_with(
                |field, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    Spanned::new(Expr::SelfField(field), extra.span().into())
                },
            ),
        tok(Token::Entity)
            .ignore_then(tok(Token::Dot))
            .ignore_then(path())
            .then(
                tok(Token::LBracket)
                    .ignore_then(expr_boxed.clone())
                    .then_ignore(tok(Token::RBracket))
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
        tok(Token::Count)
            .ignore_then(
                tok(Token::LParen)
                    .ignore_then(tok(Token::Entity))
                    .ignore_then(tok(Token::Dot))
                    .ignore_then(path())
                    .then_ignore(tok(Token::RParen)),
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
        tok(Token::Other)
            .ignore_then(
                tok(Token::LParen)
                    .ignore_then(tok(Token::Entity))
                    .ignore_then(tok(Token::Dot))
                    .ignore_then(path())
                    .then_ignore(tok(Token::RParen)),
            )
            .map_with(
                |entity, extra: &mut MapExtra<'src, '_, ParserInput<'src>, _>| {
                    Spanned::new(Expr::Other(entity), extra.span().into())
                },
            ),
        tok(Token::Pairs)
            .ignore_then(
                tok(Token::LParen)
                    .ignore_then(tok(Token::Entity))
                    .ignore_then(tok(Token::Dot))
                    .ignore_then(path())
                    .then_ignore(tok(Token::RParen)),
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
            tok(Token::LParen)
                .ignore_then(tok(Token::Entity))
                .ignore_then(tok(Token::Dot))
                .ignore_then(path())
                .then_ignore(tok(Token::Comma))
                .then(expr_boxed.clone())
                .then_ignore(tok(Token::RParen)),
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

    let filter_expr = tok(Token::Filter)
        .ignore_then(
            tok(Token::LParen)
                .ignore_then(tok(Token::Entity))
                .ignore_then(tok(Token::Dot))
                .ignore_then(path())
                .then_ignore(tok(Token::Comma))
                .then(expr_boxed.clone())
                .then_ignore(tok(Token::RParen)),
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

    let first_expr = tok(Token::First)
        .ignore_then(
            tok(Token::LParen)
                .ignore_then(tok(Token::Entity))
                .ignore_then(tok(Token::Dot))
                .ignore_then(path())
                .then_ignore(tok(Token::Comma))
                .then(expr_boxed.clone())
                .then_ignore(tok(Token::RParen)),
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

    let nearest_expr = tok(Token::Nearest)
        .ignore_then(
            tok(Token::LParen)
                .ignore_then(tok(Token::Entity))
                .ignore_then(tok(Token::Dot))
                .ignore_then(path())
                .then_ignore(tok(Token::Comma))
                .then(expr_boxed.clone())
                .then_ignore(tok(Token::RParen)),
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

    let within_expr = tok(Token::Within)
        .ignore_then(
            tok(Token::LParen)
                .ignore_then(tok(Token::Entity))
                .ignore_then(tok(Token::Dot))
                .ignore_then(path())
                .then_ignore(tok(Token::Comma))
                .then(expr_boxed.clone())
                .then_ignore(tok(Token::Comma))
                .then(expr_boxed)
                .then_ignore(tok(Token::RParen)),
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
    spanned_expr_inner(false)
}

pub fn spanned_effect_expr<'src>()
-> impl Parser<'src, ParserInput<'src>, Spanned<Expr>, extra::Err<ParseError<'src>>> + Clone {
    spanned_expr_inner(true)
}

fn aggregate_op_with_body<'src>()
-> impl Parser<'src, ParserInput<'src>, AggregateOp, extra::Err<ParseError<'src>>> + Clone {
    choice((
        tok(Token::Sum).to(AggregateOp::Sum),
        tok(Token::Product).to(AggregateOp::Product),
        tok(Token::Min).to(AggregateOp::Min),
        tok(Token::Max).to(AggregateOp::Max),
        tok(Token::Mean).to(AggregateOp::Mean),
        tok(Token::Any).to(AggregateOp::Any),
        tok(Token::All).to(AggregateOp::All),
        tok(Token::None).to(AggregateOp::None),
    ))
}
