//! Time structure parsers (strata and eras).

use chumsky::prelude::*;

use crate::ast::{
    EraDef, Expr, Spanned, StrataDef, StrataState, StrataStateKind, Transition, ValueWithUnit,
};

use super::super::expr::spanned_expr;
use super::super::lexer::Token;
use super::super::primitives::{
    attr_flag, attr_int, attr_string, ident, literal, spanned, spanned_path, unit,
};
use super::super::{ParseError, ParserInput};

// === Strata ===

pub fn strata_def<'src>()
-> impl Parser<'src, ParserInput<'src>, StrataDef, extra::Err<ParseError<'src>>> {
    just(Token::Strata)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned_path())
        .then(
            strata_attr()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|(path, attrs)| {
            let mut def = StrataDef {
                doc: None,
                path,
                title: None,
                symbol: None,
                stride: None,
            };
            for attr in attrs {
                match attr {
                    StrataAttr::Title(t) => def.title = Some(t),
                    StrataAttr::Symbol(s) => def.symbol = Some(s),
                    StrataAttr::Stride(s) => def.stride = Some(s),
                }
            }
            def
        })
}

#[derive(Clone)]
enum StrataAttr {
    Title(Spanned<String>),
    Symbol(Spanned<String>),
    Stride(Spanned<u32>),
}

fn strata_attr<'src>()
-> impl Parser<'src, ParserInput<'src>, StrataAttr, extra::Err<ParseError<'src>>> + Clone {
    choice((
        attr_string(Token::Title).map(StrataAttr::Title),
        attr_string(Token::Symbol).map(StrataAttr::Symbol),
        attr_int(Token::Stride).map(StrataAttr::Stride),
    ))
}

// === Era ===

pub fn era_def<'src>() -> impl Parser<'src, ParserInput<'src>, EraDef, extra::Err<ParseError<'src>>>
{
    just(Token::Era)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned(ident()))
        .then(
            era_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|(name, contents)| {
            let mut def = EraDef {
                doc: None,
                name,
                is_initial: false,
                is_terminal: false,
                title: None,
                dt: None,
                config_overrides: vec![],
                strata_states: vec![],
                transitions: vec![],
            };
            for content in contents {
                match content {
                    EraContent::Initial => def.is_initial = true,
                    EraContent::Terminal => def.is_terminal = true,
                    EraContent::Title(t) => def.title = Some(t),
                    EraContent::Dt(d) => def.dt = Some(d),
                    EraContent::Strata(s) => def.strata_states = s,
                    EraContent::Transition(t) => def.transitions.push(t),
                }
            }
            def
        })
}

#[derive(Clone)]
enum EraContent {
    Initial,
    Terminal,
    Title(Spanned<String>),
    Dt(Spanned<ValueWithUnit>),
    Strata(Vec<StrataState>),
    Transition(Transition),
}

fn era_content<'src>()
-> impl Parser<'src, ParserInput<'src>, EraContent, extra::Err<ParseError<'src>>> + Clone {
    choice((
        attr_flag(Token::Initial).to(EraContent::Initial),
        attr_flag(Token::Terminal).to(EraContent::Terminal),
        attr_string(Token::Title).map(EraContent::Title),
        just(Token::Colon)
            .ignore_then(just(Token::Dt))
            .ignore_then(
                spanned(value_with_unit()).delimited_by(just(Token::LParen), just(Token::RParen)),
            )
            .map(EraContent::Dt),
        just(Token::Strata)
            .ignore_then(
                strata_state()
                    .repeated()
                    .collect()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(EraContent::Strata),
        just(Token::Transition)
            .ignore_then(transition().delimited_by(just(Token::LBrace), just(Token::RBrace)))
            .map(EraContent::Transition),
    ))
}

fn value_with_unit<'src>()
-> impl Parser<'src, ParserInput<'src>, ValueWithUnit, extra::Err<ParseError<'src>>> + Clone {
    literal()
        .then(unit())
        .map(|(value, unit)| ValueWithUnit { value, unit })
}

fn strata_state<'src>()
-> impl Parser<'src, ParserInput<'src>, StrataState, extra::Err<ParseError<'src>>> + Clone {
    spanned_path()
        .then_ignore(just(Token::Colon))
        .then(strata_state_kind())
        .map(|(strata, state)| StrataState { strata, state })
}

fn strata_state_kind<'src>()
-> impl Parser<'src, ParserInput<'src>, StrataStateKind, extra::Err<ParseError<'src>>> + Clone {
    choice((
        just(Token::Active)
            .ignore_then(
                just(Token::LParen)
                    .ignore_then(just(Token::Stride))
                    .ignore_then(just(Token::Colon))
                    .ignore_then(select! { Token::Integer(i) => i as u32 })
                    .then_ignore(just(Token::RParen))
                    .or_not(),
            )
            .map(|stride| match stride {
                Some(s) => StrataStateKind::ActiveWithStride(s),
                None => StrataStateKind::Active,
            }),
        just(Token::Gated).to(StrataStateKind::Gated),
    ))
}

fn transition<'src>()
-> impl Parser<'src, ParserInput<'src>, Transition, extra::Err<ParseError<'src>>> + Clone {
    just(Token::To)
        .ignore_then(just(Token::Colon))
        .ignore_then(just(Token::Era))
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned_path())
        .then(
            just(Token::When)
                .ignore_then(
                    spanned_expr()
                        .repeated()
                        .collect()
                        .delimited_by(just(Token::LBrace), just(Token::RBrace)),
                )
                .or_not()
                .map(|c: Option<Vec<Spanned<Expr>>>| c.unwrap_or_default()),
        )
        .map(|(target, conditions)| Transition { target, conditions })
}
