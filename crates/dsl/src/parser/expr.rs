//! Expression parser

use chumsky::prelude::*;

use crate::ast::{BinaryOp, Expr, Literal, MathConst, Spanned, UnaryOp};

use super::primitives::{number, path, string_lit, unit, ws};
use super::ParseError;

/// Expression parser
pub fn expr<'src>() -> impl Parser<'src, &'src str, Expr, extra::Err<ParseError<'src>>> + Clone {
    recursive(|expr| {
        let atom = choice((
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
            number()
                .then(unit().padded_by(ws()).or_not())
                .map(|(lit, unit_opt)| match unit_opt {
                    Some(u) => Expr::LiteralWithUnit { value: lit, unit: u },
                    None => Expr::Literal(lit),
                }),
            string_lit().map(|s| Expr::Literal(Literal::String(s))),
            expr.clone()
                .padded_by(ws())
                .delimited_by(just('('), just(')')),
            path().map(Expr::Path),
        ))
        .padded_by(ws());

        let unary = just('-').repeated().foldr(atom, |_, operand| Expr::Unary {
            op: UnaryOp::Neg,
            operand: Box::new(Spanned::new(operand, 0..0)),
        });

        let product = unary.clone().foldl(
            choice((just('*').to(BinaryOp::Mul), just('/').to(BinaryOp::Div)))
                .padded_by(ws())
                .then(unary.clone())
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
                .then(product.clone())
                .repeated(),
            |left, (op, right)| Expr::Binary {
                op,
                left: Box::new(Spanned::new(left, 0..0)),
                right: Box::new(Spanned::new(right, 0..0)),
            },
        );

        sum.clone().foldl(
            choice((
                just("==").to(BinaryOp::Eq),
                just("!=").to(BinaryOp::Ne),
                just("<=").to(BinaryOp::Le),
                just(">=").to(BinaryOp::Ge),
                just('<').to(BinaryOp::Lt),
                just('>').to(BinaryOp::Gt),
            ))
            .padded_by(ws())
            .then(sum.clone())
            .repeated(),
            |left, (op, right)| Expr::Binary {
                op,
                left: Box::new(Spanned::new(left, 0..0)),
                right: Box::new(Spanned::new(right, 0..0)),
            },
        )
    })
}

/// Spanned expression
pub fn spanned_expr<'src>(
) -> impl Parser<'src, &'src str, Spanned<Expr>, extra::Err<ParseError<'src>>> + Clone {
    expr().map_with(|e, extra| Spanned::new(e, extra.span().into()))
}
