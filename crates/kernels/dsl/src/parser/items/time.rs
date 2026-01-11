//! Time structure parsers (strata and eras).
//!
//! This module handles:
//! - `strata.name { ... }` time strata definitions
//! - `era.name { ... }` simulation era definitions

use chumsky::prelude::*;

use crate::ast::{
    EraDef, Spanned, StrataDef, StrataState, StrataStateKind, Transition, ValueWithUnit,
};

use super::super::ParseError;
use super::super::expr::spanned_expr;
use super::super::primitives::{
    attr_flag, attr_int, attr_string, ident, literal, spanned, spanned_path, unit, ws,
};

// === Strata ===

pub fn strata_def<'src>() -> impl Parser<'src, &'src str, StrataDef, extra::Err<ParseError<'src>>> {
    text::keyword("strata")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            strata_attr()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
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
-> impl Parser<'src, &'src str, StrataAttr, extra::Err<ParseError<'src>>> + Clone {
    choice((
        attr_string("title").map(StrataAttr::Title),
        attr_string("symbol").map(StrataAttr::Symbol),
        attr_int("stride").map(StrataAttr::Stride),
    ))
}

// === Era ===

pub fn era_def<'src>() -> impl Parser<'src, &'src str, EraDef, extra::Err<ParseError<'src>>> {
    text::keyword("era")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned(ident()))
        .padded_by(ws())
        .then(
            era_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
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
-> impl Parser<'src, &'src str, EraContent, extra::Err<ParseError<'src>>> + Clone {
    choice((
        attr_flag("initial").to(EraContent::Initial),
        attr_flag("terminal").to(EraContent::Terminal),
        attr_string("title").map(EraContent::Title),
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("dt"))
            .ignore_then(
                spanned(value_with_unit())
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(EraContent::Dt),
        text::keyword("strata")
            .padded_by(ws())
            .ignore_then(
                strata_state()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(EraContent::Strata),
        text::keyword("transition")
            .padded_by(ws())
            .ignore_then(
                transition().delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(EraContent::Transition),
    ))
}

fn value_with_unit<'src>()
-> impl Parser<'src, &'src str, ValueWithUnit, extra::Err<ParseError<'src>>> + Clone {
    literal()
        .padded_by(ws())
        .then(unit())
        .map(|(value, unit)| ValueWithUnit { value, unit })
}

fn strata_state<'src>()
-> impl Parser<'src, &'src str, StrataState, extra::Err<ParseError<'src>>> + Clone {
    spanned_path()
        .then_ignore(just(':').padded_by(ws()))
        .then(strata_state_kind())
        .map(|(strata, state)| StrataState { strata, state })
}

fn strata_state_kind<'src>()
-> impl Parser<'src, &'src str, StrataStateKind, extra::Err<ParseError<'src>>> + Clone {
    choice((
        text::keyword("active")
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(text::keyword("stride"))
                    .ignore_then(just(':').padded_by(ws()))
                    .ignore_then(text::int(10).map(|s: &str| s.parse::<u32>().unwrap_or(1)))
                    .then_ignore(just(')').padded_by(ws()))
                    .or_not(),
            )
            .map(|stride| match stride {
                Some(s) => StrataStateKind::ActiveWithStride(s),
                None => StrataStateKind::Active,
            }),
        text::keyword("gated").to(StrataStateKind::Gated),
    ))
}

fn transition<'src>()
-> impl Parser<'src, &'src str, Transition, extra::Err<ParseError<'src>>> + Clone {
    text::keyword("to")
        .padded_by(ws())
        .ignore_then(just(':').padded_by(ws()))
        .ignore_then(text::keyword("era").padded_by(ws()))
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .then(
            text::keyword("when")
                .padded_by(ws())
                .ignore_then(
                    spanned_expr()
                        .padded_by(ws())
                        .repeated()
                        .collect()
                        .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
                )
                .or_not()
                .map(|c| c.unwrap_or_default()),
        )
        .map(|(target, conditions)| Transition { target, conditions })
}
