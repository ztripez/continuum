//! Type and function definition parsers.

use chumsky::prelude::*;

use crate::ast::{FnDef, FnParam, TypeDef, TypeExpr, TypeField};

use super::super::expr::spanned_expr;
use super::super::lexer::Token;
use super::super::primitives::{ident, spanned, spanned_path};
use super::super::{ParseError, ParserInput};

mod registry;
use registry::primitive_type_parser;

// === Type Definitions ===

pub fn type_def<'src>()
-> impl Parser<'src, ParserInput<'src>, TypeDef, extra::Err<ParseError<'src>>> {
    just(Token::Type)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned(ident()))
        .then(
            type_field()
                .repeated()
                .collect()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|(name, fields)| TypeDef {
            doc: None,
            name,
            fields,
        })
}

fn type_field<'src>()
-> impl Parser<'src, ParserInput<'src>, TypeField, extra::Err<ParseError<'src>>> {
    spanned(ident())
        .then_ignore(just(Token::Colon))
        .then(spanned(type_expr()))
        .map(|(name, ty)| TypeField { name, ty })
}

pub fn type_expr<'src>()
-> impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
    recursive(|type_expr_recurse| {
        primitive_type_parser(type_expr_recurse.clone()).or(ident().map(TypeExpr::Named))
    })
}

// === Function Definitions ===

pub fn fn_def<'src>() -> impl Parser<'src, ParserInput<'src>, FnDef, extra::Err<ParseError<'src>>> {
    just(Token::Fn)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned_path())
        .then(
            spanned(ident())
                .separated_by(just(Token::Comma))
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LAngle), just(Token::RAngle))
                .or_not()
                .map(|o: Option<Vec<_>>| o.unwrap_or_default()),
        )
        .then(
            fn_param()
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect()
                .delimited_by(just(Token::LParen), just(Token::RParen)),
        )
        .then(
            just(Token::Arrow)
                .ignore_then(spanned(type_expr()))
                .or_not(),
        )
        .then(spanned_expr().delimited_by(just(Token::LBrace), just(Token::RBrace)))
        .map(|((((path, generics), params), return_type), body)| FnDef {
            doc: None,
            path,
            generics,
            params,
            return_type,
            body,
        })
}

fn fn_param<'src>()
-> impl Parser<'src, ParserInput<'src>, FnParam, extra::Err<ParseError<'src>>> + Clone {
    spanned(ident())
        .then(
            just(Token::Colon)
                .ignore_then(spanned(type_expr()))
                .or_not(),
        )
        .map(|(name, ty)| FnParam { name, ty })
}

#[cfg(test)]
mod tests;
