//! Event-related parsers (impulses, fractures, chronicles).

use chumsky::prelude::*;

use crate::ast::{
    ApplyBlock, ChronicleDef, EmitStatement, Expr, FractureDef, ImpulseDef, ObserveBlock,
    ObserveHandler, Spanned, TypeExpr,
};

use super::super::expr::spanned_expr;
use super::super::lexer::Token;
use super::super::primitives::{ident, spanned, spanned_path};
use super::super::{ParseError, ParserInput};
use super::types::type_expr;

// === Impulse ===

pub fn impulse_def<'src>()
-> impl Parser<'src, ParserInput<'src>, ImpulseDef, extra::Err<ParseError<'src>>> {
    just(Token::Impulse)
        .ignore_then(just(Token::Dot))
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
                local_config: vec![],
                apply: None,
            };
            for content in contents {
                match content {
                    ImpulseContent::Type(t) => def.payload_type = Some(t),
                    ImpulseContent::Apply(a) => def.apply = Some(a),
                }
            }
            def
        })
}

#[derive(Clone)]
enum ImpulseContent {
    Type(Spanned<TypeExpr>),
    Apply(ApplyBlock),
}

fn impulse_content<'src>()
-> impl Parser<'src, ParserInput<'src>, ImpulseContent, extra::Err<ParseError<'src>>> {
    choice((
        just(Token::Colon)
            .ignore_then(spanned(type_expr()))
            .map(ImpulseContent::Type),
        just(Token::Apply)
            .ignore_then(spanned_expr().delimited_by(just(Token::LBrace), just(Token::RBrace)))
            .map(|body| ImpulseContent::Apply(ApplyBlock { body })),
    ))
}

// === Fracture ===

pub fn fracture_def<'src>()
-> impl Parser<'src, ParserInput<'src>, FractureDef, extra::Err<ParseError<'src>>> {
    just(Token::Fracture)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned_path())
        .then(
            fracture_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(|(path, contents)| {
            let mut conditions = vec![];
            let mut emit = vec![];
            for content in contents {
                match content {
                    FractureContent::When(w) => conditions = w,
                    FractureContent::Emit(e) => emit = e,
                }
            }
            FractureDef {
                doc: None,
                path,
                conditions,
                emit,
            }
        })
}

#[derive(Clone)]
enum FractureContent {
    When(Vec<Spanned<Expr>>),
    Emit(Vec<EmitStatement>),
}

fn fracture_content<'src>()
-> impl Parser<'src, ParserInput<'src>, FractureContent, extra::Err<ParseError<'src>>> {
    choice((
        just(Token::When)
            .ignore_then(
                spanned_expr()
                    .repeated()
                    .collect()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(FractureContent::When),
        just(Token::Emit)
            .ignore_then(
                emit_statement()
                    .repeated()
                    .collect()
                    .delimited_by(just(Token::LBrace), just(Token::RBrace)),
            )
            .map(FractureContent::Emit),
    ))
}

fn emit_statement<'src>()
-> impl Parser<'src, ParserInput<'src>, EmitStatement, extra::Err<ParseError<'src>>> + Clone {
    just(Token::Signal)
        .ignore_then(just(Token::Dot))
        .ignore_then(spanned_path())
        .then_ignore(just(Token::EmitArrow))
        .then(spanned_expr())
        .map(|(target, value)| EmitStatement { target, value })
}

// === Chronicle ===

pub fn chronicle_def<'src>()
-> impl Parser<'src, ParserInput<'src>, ChronicleDef, extra::Err<ParseError<'src>>> {
    just(Token::Chronicle)
        .ignore_then(just(Token::Dot))
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
