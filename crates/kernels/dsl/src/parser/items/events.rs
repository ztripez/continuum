//! Event-related parsers (impulses, fractures, chronicles).

use chumsky::prelude::*;

use crate::ast::{
    ApplyBlock, ChronicleDef, ConfigEntry, Expr, FractureDef, ImpulseDef, ObserveBlock,
    ObserveHandler, Path, Spanned, TypeExpr,
};

use super::super::expr::{spanned_effect_expr, spanned_expr};
use super::super::lexer::Token;
use super::super::primitives::{attr_path, attr_string, ident, spanned, spanned_path};
use super::super::{ParseError, ParserInput};
use super::config::config_entry;
use super::types::type_expr;

// === Impulse ===

pub fn impulse_def<'src>()
-> impl Parser<'src, ParserInput<'src>, ImpulseDef, extra::Err<ParseError<'src>>> {
    just(Token::Impulse)
        .ignore_then(spanned_path())
        .then(
            impulse_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|(path, contents)| {
            let mut def = ImpulseDef {
                doc: None,
                path,
                payload_type: None,
                title: None,
                symbol: None,
                local_config: vec![],
                apply: None,
                uses: vec![],
            };
            for content in contents {
                match content {
                    ImpulseContent::Type(t) => def.payload_type = Some(t),
                    ImpulseContent::Title(t) => def.title = Some(t),
                    ImpulseContent::Symbol(s) => def.symbol = Some(s),
                    ImpulseContent::Uses(u) => def.uses.push(u),
                    ImpulseContent::Config(c) => def.local_config = c,
                    ImpulseContent::Apply(a) => def.apply = Some(a),
                }
            }
            def
        })
}

#[derive(Clone)]
enum ImpulseContent {
    Type(Spanned<TypeExpr>),
    Title(Spanned<String>),
    Symbol(Spanned<String>),
    Uses(String),
    Config(Vec<ConfigEntry>),
    Apply(ApplyBlock),
}

fn impulse_content<'src>()
-> impl Parser<'src, ParserInput<'src>, ImpulseContent, extra::Err<ParseError<'src>>> {
    choice((
        attr_string(Token::Title).map(ImpulseContent::Title),
        attr_string(Token::Symbol).map(ImpulseContent::Symbol),
        // : uses(namespace.key) - generic uses declaration
        just(Token::Colon)
            .ignore_then(just(Token::Uses))
            .ignore_then(spanned_path().delimited_by(just(Token::LParen), just(Token::RParen)))
            .map(|path| ImpulseContent::Uses(path.node.join("."))),
        // Type expression: `: TypeExpr`
        just(Token::Colon)
            .ignore_then(spanned(type_expr()))
            .map(ImpulseContent::Type),
        // Config block: `config { ... }`
        just(Token::Config)
            .ignore_then(
                config_entry()
                    .repeated()
                    .collect()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(ImpulseContent::Config),
        // Apply block: `apply { expr; expr; ... }` - supports semicolon-separated expressions
        just(Token::Apply)
            .ignore_then(
                spanned_effect_expr()
                    .separated_by(just(Token::Semicolon).or_not())
                    .allow_trailing()
                    .at_least(1)
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map_with(|exprs, extra| {
                let span = extra.span();
                let body = if exprs.len() == 1 {
                    exprs.into_iter().next().unwrap()
                } else {
                    Spanned::new(Expr::Block(exprs), span.into())
                };
                ImpulseContent::Apply(ApplyBlock { body })
            }),
    ))
}

// === Fracture ===

pub fn fracture_def<'src>()
-> impl Parser<'src, ParserInput<'src>, FractureDef, extra::Err<ParseError<'src>>> {
    just(Token::Fracture)
        .ignore_then(spanned_path())
        .then(
            fracture_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|(path, contents)| {
            let mut strata = None;
            let mut local_config = vec![];
            let mut conditions = vec![];
            let mut emit = None;
            let mut uses = vec![];
            for content in contents {
                match content {
                    FractureContent::Strata(s) => strata = Some(s),
                    FractureContent::Uses(u) => uses.push(u),
                    FractureContent::Config(c) => local_config = c,
                    FractureContent::When(w) => conditions = w,
                    FractureContent::Emit(e) => emit = Some(e),
                }
            }
            FractureDef {
                doc: None,
                path,
                strata,
                local_config,
                conditions,
                emit,
                uses,
            }
        })
}

#[derive(Clone)]
enum FractureContent {
    Strata(Spanned<Path>),
    Uses(String),
    Config(Vec<ConfigEntry>),
    When(Vec<Spanned<Expr>>),
    Emit(Spanned<Expr>),
}

fn fracture_content<'src>()
-> impl Parser<'src, ParserInput<'src>, FractureContent, extra::Err<ParseError<'src>>> {
    choice((
        attr_path(Token::Strata).map(FractureContent::Strata),
        // : uses(namespace.key) - generic uses declaration
        just(Token::Colon)
            .ignore_then(just(Token::Uses))
            .ignore_then(spanned_path().delimited_by(just(Token::LParen), just(Token::RParen)))
            .map(|path| FractureContent::Uses(path.node.join("."))),
        // config { ... } - local config block
        just(Token::Config)
            .ignore_then(
                config_entry()
                    .repeated()
                    .collect()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(FractureContent::Config),
        // when { ... } - trigger conditions
        just(Token::When)
            .ignore_then(
                spanned_expr()
                    .repeated()
                    .collect()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(FractureContent::When),
        // emit { expr... } - emit expression(s), supports let bindings
        // Multiple expressions are wrapped in a Block
        just(Token::Emit)
            .ignore_then(
                spanned_effect_expr()
                    .separated_by(just(Token::Semicolon).or_not())
                    .allow_trailing()
                    .at_least(1)
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map_with(|exprs, extra| {
                let span = extra.span();
                if exprs.len() == 1 {
                    FractureContent::Emit(exprs.into_iter().next().unwrap())
                } else {
                    // Multiple expressions -> wrap in a Block
                    FractureContent::Emit(Spanned::new(Expr::Block(exprs), span.into()))
                }
            }),
    ))
}

// === Chronicle ===

pub fn chronicle_def<'src>()
-> impl Parser<'src, ParserInput<'src>, ChronicleDef, extra::Err<ParseError<'src>>> {
    just(Token::Chronicle)
        .ignore_then(spanned_path())
        .then(
            observe_block()
                .or_not()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|(path, observe)| ChronicleDef {
            doc: None,
            path,
            observe,
        })
}

fn observe_block<'src>()
-> impl Parser<'src, ParserInput<'src>, ObserveBlock, extra::Err<ParseError<'src>>> {
    just(Token::Observe)
        .ignore_then(
            observe_handler()
                .repeated()
                .collect()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|handlers| ObserveBlock { handlers })
}

fn observe_handler<'src>()
-> impl Parser<'src, ParserInput<'src>, ObserveHandler, extra::Err<ParseError<'src>>> {
    just(Token::When)
        .ignore_then(spanned_expr())
        .then(
            just(Token::Emit)
                .ignore_then(just(Token::Event))
                .ignore_then(just(Token::Dot))
                .ignore_then(spanned_path())
                .then(
                    event_field()
                        .repeated()
                        .collect()
                        .delimited_by(just(Token::LBrace), just(Token::RBrace)),
                )
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|(condition, (event_name, event_fields))| ObserveHandler {
            condition,
            event_name,
            event_fields,
        })
}

fn event_field<'src>()
-> impl Parser<'src, ParserInput<'src>, (Spanned<String>, Spanned<Expr>), extra::Err<ParseError<'src>>>
{
    spanned(ident())
        .then_ignore(just(Token::Colon))
        .then(spanned_expr())
}
