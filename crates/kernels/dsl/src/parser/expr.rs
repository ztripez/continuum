//! Expression parser for the Continuum DSL.
//!
//! This module implements the expression grammar with proper operator precedence.
//! Expressions can appear in resolve blocks, measure blocks, assertions, and
//! other executable contexts.
//!
//! # Operator Precedence (lowest to highest)
//!
//! 1. **Ternary**: `if cond { then } else { else }`
//! 2. **Logical OR**: `or`
//! 3. **Logical AND**: `and`
//! 4. **Comparison**: `<`, `<=`, `>`, `>=`, `==`, `!=`
//! 5. **Addition/Subtraction**: `+`, `-`
//! 6. **Multiplication/Division**: `*`, `/`, `%`
//! 7. **Exponentiation**: `^`
//! 8. **Unary**: `-`, `!`
//! 9. **Primary**: Literals, identifiers, function calls, parentheses
//!
//! # Expression Types
//!
//! - **Literals**: `42`, `3.14`, `"string"`
//! - **References**: `signal.name`, `config.value`, `const.physics.g`
//! - **Keywords**: `prev`, `dt_raw`, `collected`, `payload`
//! - **Math constants**: `PI` / `π`, `TAU` / `τ`, `E` / `ℯ`, `PHI` / `φ`
//! - **Function calls**: `sin(x)`, `clamp(v, 0, 1)`, `integrate(prev, rate)`
//! - **Let bindings**: `let x = expr in body`
//! - **Conditionals**: `if cond { a } else { b }`
//!
//! # Compile Time Optimization
//!
//! The parser uses strategic `.boxed()` calls to prevent exponential compile
//! time growth from deeply nested generic types.

use chumsky::prelude::*;

use crate::ast::{AggregateOp, BinaryOp, CallArg, Expr, Literal, MathConst, Spanned, UnaryOp};

use super::primitives::{ident, number, path, string_lit, unit, ws};
use super::ParseError;

/// Type alias for parser Extra to ensure consistency across helper functions
type Ex<'src> = extra::Err<ParseError<'src>>;

/// Type alias for boxed spanned expression parser
type SpannedExprBox<'src> = Boxed<'src, 'src, &'src str, Spanned<Expr>, Ex<'src>>;

/// Helper to create a span covering two spanned values
fn span_union<T, U>(left: &Spanned<T>, right: &Spanned<U>) -> std::ops::Range<usize> {
    left.span.start..right.span.end
}

/// Expression parser - returns just the expression without span info
///
/// Uses `.boxed()` at strategic points to reduce compile times by breaking the type chain.
/// Without boxing, chumsky's parser combinators create deeply nested generic types that
/// cause exponential compile time growth.
#[allow(dead_code)]
pub fn expr<'src>() -> impl Parser<'src, &'src str, Expr, Ex<'src>> + Clone {
    spanned_expr_inner().map(|spanned| spanned.node)
}

/// Internal spanned expression parser - produces `Spanned<Expr>` with proper spans throughout
fn spanned_expr_inner<'src>() -> impl Parser<'src, &'src str, Spanned<Expr>, Ex<'src>> + Clone {
    recursive(|expr| {
        // Box the recursive expr (which returns Spanned<Expr>) to prevent type explosion
        let expr_boxed: SpannedExprBox<'src> = expr.clone().boxed();

        // Single argument parser - handles both named (name: expr) and positional (expr)
        // Named arguments use `name: value` syntax
        let call_arg = choice((
            // Named argument: name: value
            // Use look-ahead to distinguish from a path expression followed by comparison
            ident()
                .then_ignore(just(':').padded_by(ws()))
                .then(expr_boxed.clone())
                .map(|(name, value)| CallArg::named(name, value)),
            // Positional argument: just an expression
            expr_boxed.clone().map(CallArg::positional),
        ));

        // Arguments list for function calls - produces Vec<CallArg>
        let args = call_arg
            .separated_by(just(',').padded_by(ws()))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just('(').padded_by(ws()), just(')').padded_by(ws()));

        // Entity expression atoms - boxed to reduce type complexity
        // Returns Spanned<Expr>
        let entity_atoms = entity_expr_atoms_spanned(expr_boxed.clone()).boxed();

        // Core atoms (non-entity) - wrap with map_with to capture spans
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
            text::keyword("collected").to(Expr::Collected),
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
        .map_with(|e, extra| {
            let span: chumsky::span::SimpleSpan = extra.span();
            Spanned::new(e, span.start..span.end)
        })
        .boxed();

        // Additional atoms (function calls, literals, paths)
        let other_atoms = choice((
            // Function call: name(args) or path.to.func(args)
            path()
                .map_with(|p, extra| {
                    let span: chumsky::span::SimpleSpan = extra.span();
                    Spanned::new(Expr::Path(p), span.start..span.end)
                })
                .then(args.clone())
                .map_with(|(func, args), extra| {
                    let span: chumsky::span::SimpleSpan = extra.span();
                    Spanned::new(
                        Expr::Call {
                            function: Box::new(func),
                            args,
                        },
                        span.start..span.end,
                    )
                }),
            number()
                .then(unit().padded_by(ws()).or_not())
                .map_with(|(lit, unit_opt), extra| {
                    let span: chumsky::span::SimpleSpan = extra.span();
                    let e = match unit_opt {
                        Some(u) => Expr::LiteralWithUnit { value: lit, unit: u },
                        None => Expr::Literal(lit),
                    };
                    Spanned::new(e, span.start..span.end)
                }),
            string_lit().map_with(|s, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                Spanned::new(Expr::Literal(Literal::String(s)), span.start..span.end)
            }),
            expr_boxed
                .clone()
                .padded_by(ws())
                .delimited_by(just('('), just(')')),
            // Plain path (must come after function call attempt)
            path().map_with(|p, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                Spanned::new(Expr::Path(p), span.start..span.end)
            }),
        ))
        .boxed();

        // Combine all atoms - all now return Spanned<Expr>
        let atom = choice((entity_atoms, core_atoms, other_atoms))
            .padded_by(ws())
            .boxed();

        // Arguments with span end position for accurate MethodCall spans
        let args_with_span = args.clone().map_with(|a, extra| {
            let span: chumsky::span::SimpleSpan = extra.span();
            (a, span.end)
        });

        // Method calls: expr.method(args) - now works with Spanned<Expr>
        let postfix = atom.foldl(
            just('.')
                .padded_by(ws())
                .ignore_then(ident().map_with(|m, extra| {
                    let span: chumsky::span::SimpleSpan = extra.span();
                    (m, span.start..span.end)
                }))
                .then(args_with_span.or_not())
                .repeated(),
            |obj, ((method, method_span), maybe_args)| {
                let (new_expr, span_end) = match maybe_args {
                    Some((args, paren_end)) => (
                        Expr::MethodCall {
                            object: Box::new(obj.clone()),
                            method,
                            args,
                        },
                        paren_end,
                    ),
                    None => (
                        Expr::FieldAccess {
                            object: Box::new(obj.clone()),
                            field: method,
                        },
                        method_span.end,
                    ),
                };
                Spanned::new(new_expr, obj.span.start..span_end)
            },
        );

        // Unary operators: negation (-) and logical not (! or 'not')
        // The 'not' keyword uses text::keyword to ensure proper word boundary handling
        let unary = choice((
            just('-').to(UnaryOp::Neg),
            just('!').to(UnaryOp::Not),
            text::keyword("not").to(UnaryOp::Not),
        ))
            .map_with(|op, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                (op, span.start)
            })
            .repeated()
            .foldr(postfix, |(op, op_start), operand| {
                let span = op_start..operand.span.end;
                Spanned::new(
                    Expr::Unary {
                        op,
                        operand: Box::new(operand),
                    },
                    span,
                )
            });

        // Binary operators - helper macro to reduce repetition
        let product = unary.clone().foldl(
            choice((just('*').to(BinaryOp::Mul), just('/').to(BinaryOp::Div)))
                .padded_by(ws())
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
                    span,
                )
            },
        );

        let sum = product.clone().foldl(
            choice((just('+').to(BinaryOp::Add), just('-').to(BinaryOp::Sub)))
                .padded_by(ws())
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
                    span,
                )
            },
        );

        // Comparison operators do NOT chain: a < b < c is disallowed
        // Use .or_not() instead of .repeated() to prevent chaining
        let comparison = sum
            .clone()
            .then(
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
                        span,
                    )
                }
                None => left,
            });

        // Logical AND has lower precedence than comparison
        // Accept both '&&' and 'and' keyword
        let and_op = choice((
            just("&&").to(BinaryOp::And),
            text::keyword("and").to(BinaryOp::And),
        ));
        let logical_and = comparison.clone().foldl(
            and_op
                .padded_by(ws())
                .then(comparison)
                .repeated(),
            |left, (op, right)| {
                let span = span_union(&left, &right);
                Spanned::new(
                    Expr::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            },
        );

        // Logical OR has lower precedence than AND
        // Accept both '||' and 'or' keyword
        let or_op = choice((
            just("||").to(BinaryOp::Or),
            text::keyword("or").to(BinaryOp::Or),
        ));
        let logical_or = logical_and.clone().foldl(
            or_op
                .padded_by(ws())
                .then(logical_and)
                .repeated(),
            |left, (op, right)| {
                let span = span_union(&left, &right);
                Spanned::new(
                    Expr::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            },
        );

        // Box logical_or before using in if_expr to reduce type complexity
        let logical_or_boxed = logical_or.clone().boxed();

        // Emit expression: signal.path <- value
        // Emits a value to a signal target. Used in fracture emit blocks.
        let emit_expr = text::keyword("signal")
            .ignore_then(just('.'))
            .ignore_then(path())
            .map_with(|p, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                (p, span.start)
            })
            .then_ignore(just("<-").padded_by(ws()))
            .then(expr_boxed.clone())
            .map(|((target, start), value)| {
                let span = start..value.span.end;
                Spanned::new(
                    Expr::EmitSignal {
                        target,
                        value: Box::new(value),
                    },
                    span,
                )
            });

        // Let expression: let name = value in body
        // Multiple lets chain together: let a = 1 in let b = 2 in a + b
        let let_expr = text::keyword("let")
            .padded_by(ws())
            .map_with(|_, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                span.start
            })
            .then(ident())
            .then_ignore(just('=').padded_by(ws()))
            .then(expr_boxed.clone())
            .then_ignore(text::keyword("in").padded_by(ws()))
            .then(expr_boxed.clone())
            .map(|(((let_start, name), value), body)| {
                let span = let_start..body.span.end;
                Spanned::new(
                    Expr::Let {
                        name,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    span,
                )
            });

        // If expression: if condition { then } else { else }
        // Also supports: if cond { a } else if cond2 { b } else { c }
        //
        // Uses iterative parsing of else-if chains to avoid recursive parser issues.
        // Pattern: if COND { BLOCK } (else if COND { BLOCK })* (else { BLOCK })?
        let if_expr = {
            // Braced expression: { expr }
            let braced = just('{')
                .padded_by(ws())
                .ignore_then(expr_boxed.clone().padded_by(ws()))
                .then_ignore(just('}').padded_by(ws()));

            // Initial if clause: if COND { BLOCK }
            let if_head = text::keyword("if")
                .map_with(|_, e| {
                    let span: chumsky::span::SimpleSpan = e.span();
                    span.start
                })
                .then_ignore(ws())
                .then(logical_or_boxed.clone())
                .then_ignore(ws())
                .then(braced.clone());

            // Else-if clause: else if COND { BLOCK }
            // We match "else" + whitespace + "if" as a sequence, then condition + block
            let else_if_clause = text::keyword("else")
                .then_ignore(ws())
                .then(text::keyword("if"))
                .then_ignore(ws())
                .ignore_then(logical_or_boxed.clone())
                .then_ignore(ws())
                .then(braced.clone());

            // Final else clause: else { BLOCK }
            let else_final = text::keyword("else")
                .then_ignore(ws())
                .ignore_then(braced.clone());

            // Combine: if + (else-if)* + (else)?
            // The nested tuples come from chained .then() calls:
            // if_head produces ((usize, Spanned), Spanned)
            if_head
                .then(else_if_clause.repeated().collect::<Vec<_>>())
                .then(else_final.or_not())
                .map(|((((if_start, cond), then_block), else_ifs), else_final)| {
                    // Build nested If expressions from right to left
                    let mut else_branch = else_final;

                    // Fold else-if clauses from right to left
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
                            span_start..span_end,
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
                        if_start..final_span_end,
                    )
                })
        };

        // Let and if expressions have lowest precedence - they consume the rest as body
        choice((emit_expr, let_expr, if_expr, logical_or))
    })
}

/// Entity expression atoms (spanned) - separated to reduce type complexity
fn entity_expr_atoms_spanned<'src>(
    expr_boxed: SpannedExprBox<'src>,
) -> impl Parser<'src, &'src str, Spanned<Expr>, Ex<'src>> + Clone {
    choice((
        // self.field - current entity instance field access
        text::keyword("self")
            .ignore_then(just('.').padded_by(ws()))
            .ignore_then(ident())
            .map_with(|field, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                Spanned::new(Expr::SelfField(field), span.start..span.end)
            }),
        // entity.path["name"] - entity instance access
        text::keyword("entity")
            .ignore_then(just('.'))
            .ignore_then(path())
            .then(
                just('[')
                    .padded_by(ws())
                    .ignore_then(expr_boxed.clone().padded_by(ws()))
                    .then_ignore(just(']').padded_by(ws()))
                    .or_not(),
            )
            .map_with(|(entity, instance), extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                let e = match instance {
                    Some(inst) => Expr::EntityAccess {
                        entity,
                        instance: Box::new(inst),
                    },
                    None => Expr::EntityRef(entity),
                };
                Spanned::new(e, span.start..span.end)
            }),
        // count(entity.path) - special case, body is implicit "1" with span from count keyword
        text::keyword("count")
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(text::keyword("entity"))
                    .ignore_then(just('.'))
                    .ignore_then(path())
                    .then_ignore(just(')').padded_by(ws())),
            )
            .map_with(|entity, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                // Use the full span for the implicit body since it's synthetic
                Spanned::new(
                    Expr::Aggregate {
                        op: AggregateOp::Count,
                        entity,
                        body: Box::new(Spanned::new(
                            Expr::Literal(Literal::Integer(1)),
                            span.start..span.end,
                        )),
                    },
                    span.start..span.end,
                )
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
            .map_with(|entity, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                Spanned::new(Expr::Other(entity), span.start..span.end)
            }),
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
            .map_with(|entity, extra| {
                let span: chumsky::span::SimpleSpan = extra.span();
                Spanned::new(Expr::Pairs(entity), span.start..span.end)
            }),
    ))
    .or(entity_aggregate_atoms_spanned(expr_boxed))
}

/// Entity aggregate operations (spanned) - further split to reduce type complexity
fn entity_aggregate_atoms_spanned<'src>(
    expr_boxed: SpannedExprBox<'src>,
) -> impl Parser<'src, &'src str, Spanned<Expr>, Ex<'src>> + Clone {
    // Aggregate with body: sum(entity.path, expr)
    let aggregate_with_body = aggregate_op_with_body()
        .then(
            just('(')
                .padded_by(ws())
                .ignore_then(text::keyword("entity"))
                .ignore_then(just('.'))
                .ignore_then(path())
                .then_ignore(just(',').padded_by(ws()))
                .then(expr_boxed.clone())
                .then_ignore(just(')').padded_by(ws())),
        )
        .map_with(|(op, (entity, body)), extra| {
            let span: chumsky::span::SimpleSpan = extra.span();
            Spanned::new(
                Expr::Aggregate {
                    op,
                    entity,
                    body: Box::new(body),
                },
                span.start..span.end,
            )
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
                .then(expr_boxed.clone())
                .then_ignore(just(')').padded_by(ws())),
        )
        .map_with(|(entity, predicate), extra| {
            let span: chumsky::span::SimpleSpan = extra.span();
            Spanned::new(
                Expr::Filter {
                    entity,
                    predicate: Box::new(predicate),
                },
                span.start..span.end,
            )
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
                .then(expr_boxed.clone())
                .then_ignore(just(')').padded_by(ws())),
        )
        .map_with(|(entity, predicate), extra| {
            let span: chumsky::span::SimpleSpan = extra.span();
            Spanned::new(
                Expr::First {
                    entity,
                    predicate: Box::new(predicate),
                },
                span.start..span.end,
            )
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
                .then(expr_boxed.clone())
                .then_ignore(just(')').padded_by(ws())),
        )
        .map_with(|(entity, position), extra| {
            let span: chumsky::span::SimpleSpan = extra.span();
            Spanned::new(
                Expr::Nearest {
                    entity,
                    position: Box::new(position),
                },
                span.start..span.end,
            )
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
                .then(expr_boxed.clone())
                .then_ignore(just(',').padded_by(ws()))
                .then(expr_boxed)
                .then_ignore(just(')').padded_by(ws())),
        )
        .map_with(|((entity, position), radius), extra| {
            let span: chumsky::span::SimpleSpan = extra.span();
            Spanned::new(
                Expr::Within {
                    entity,
                    position: Box::new(position),
                    radius: Box::new(radius),
                },
                span.start..span.end,
            )
        });

    choice((
        aggregate_with_body,
        filter_expr,
        first_expr,
        nearest_expr,
        within_expr,
    ))
}

/// Spanned expression - public API that uses the internal spanned parser
pub fn spanned_expr<'src>() -> impl Parser<'src, &'src str, Spanned<Expr>, Ex<'src>> + Clone {
    spanned_expr_inner()
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
