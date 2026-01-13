//! Member signal definition parsers.

use chumsky::prelude::*;

use crate::ast::{ConfigEntry, MemberDef, ResolveBlock, Spanned};

use super::super::expr::spanned_expr;
use super::super::lexer::Token;
use super::super::primitives::{attr_path, attr_string, spanned, spanned_path, tok};
use super::super::{ParseError, ParserInput};
use super::common::assert_block;
use super::config::config_entry;
use super::types::type_expr;

// === Member Signal ===

pub fn member_def<'src>()
-> impl Parser<'src, ParserInput<'src>, MemberDef, extra::Err<ParseError<'src>>> {
    tok(Token::Member)
        .ignore_then(tok(Token::Dot))
        .ignore_then(spanned_path())
        .then(
            member_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(tok(Token::LBrace), tok(Token::RBrace)),
        )
        .map(|(path, contents)| {
            let mut def = MemberDef {
                doc: None,
                path,
                ty: None,
                strata: None,
                title: None,
                symbol: None,
                local_config: vec![],
                initial: None,
                resolve: None,
                assertions: None,
            };
            for content in contents {
                match content {
                    MemberContent::Type(t) => def.ty = Some(t),
                    MemberContent::Strata(s) => def.strata = Some(s),
                    MemberContent::Title(t) => def.title = Some(t),
                    MemberContent::Symbol(s) => def.symbol = Some(s),
                    MemberContent::Config(c) => def.local_config = c,
                    MemberContent::Initial(i) => def.initial = Some(i),
                    MemberContent::Resolve(r) => def.resolve = Some(r),
                    MemberContent::Assert(a) => def.assertions = Some(a),
                }
            }
            def
        })
}

#[derive(Clone)]
enum MemberContent {
    Type(Spanned<crate::ast::TypeExpr>),
    Strata(Spanned<crate::ast::Path>),
    Title(Spanned<String>),
    Symbol(Spanned<String>),
    Config(Vec<ConfigEntry>),
    Initial(ResolveBlock),
    Resolve(ResolveBlock),
    Assert(crate::ast::AssertBlock),
}

fn member_content<'src>()
-> impl Parser<'src, ParserInput<'src>, MemberContent, extra::Err<ParseError<'src>>> {
    choice((
        // : strata(path) - must come before Type to avoid matching "strata" as Named type
        attr_path(Token::Strata).map(MemberContent::Strata),
        // : title("...")
        attr_string(Token::Title).map(MemberContent::Title),
        // : symbol("...")
        attr_string(Token::Symbol).map(MemberContent::Symbol),
        // : Type - comes after specific attributes
        tok(Token::Colon)
            .ignore_then(spanned(type_expr()))
            .map(MemberContent::Type),
        // config { entries }
        tok(Token::Config)
            .ignore_then(
                config_entry()
                    .repeated()
                    .collect()
                    .delimited_by(tok(Token::LBrace), tok(Token::RBrace)),
            )
            .map(MemberContent::Config),
        // initial { expr }
        tok(Token::Initial)
            .ignore_then(spanned_expr().delimited_by(tok(Token::LBrace), tok(Token::RBrace)))
            .map(|body| MemberContent::Initial(ResolveBlock { body })),
        // resolve { expr }
        tok(Token::Resolve)
            .ignore_then(spanned_expr().delimited_by(tok(Token::LBrace), tok(Token::RBrace)))
            .map(|body| MemberContent::Resolve(ResolveBlock { body })),
        // assert { assertions }
        assert_block().map(MemberContent::Assert),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use chumsky::input::{Input, Stream};
    use logos::Logos;

    fn lex_map(
        tok_span: (
            Result<Token, <Token as logos::Logos>::Error>,
            std::ops::Range<usize>,
        ),
    ) -> (Token, std::ops::Range<usize>) {
        let (token, range) = tok_span;
        (token.unwrap_or(Token::Error), range)
    }

    fn attach_span(token_span: (Token, std::ops::Range<usize>)) -> (Token, SimpleSpan) {
        let (token, range) = token_span;
        (token, SimpleSpan::from(range))
    }

    #[test]
    fn test_parse_simple_member() {
        let src = r#"member.human.person.age {
            : Scalar
            : strata(human.physiology)
            resolve { prev + 1 }
        }"#;

        let lexer = Token::lexer(src).spanned().map(lex_map as fn(_) -> _);
        let stream = Stream::from_iter(lexer).map(
            SimpleSpan::from(src.len()..src.len()),
            attach_span as fn(_) -> _,
        );

        let result = member_def().parse(stream);
        assert!(result.has_output());
        let member = result.into_output().unwrap();
        assert_eq!(member.path.node.to_string(), "human.person.age");
        assert!(member.ty.is_some());
        assert!(member.strata.is_some());
        assert!(member.resolve.is_some());
    }

    #[test]
    fn test_parse_member_with_config() {
        let src = r#"member.stellar.moon.mass {
            : Scalar<kg>
            : strata(stellar.orbital)
            : title("Moon Mass")

            config {
                initial: 1e22
            }

            resolve { prev }
        }"#;

        let lexer = Token::lexer(src).spanned().map(lex_map as fn(_) -> _);
        let stream = Stream::from_iter(lexer).map(
            SimpleSpan::from(src.len()..src.len()),
            attach_span as fn(_) -> _,
        );

        let result = member_def().parse(stream);
        assert!(result.has_output());
        let member = result.into_output().unwrap();
        assert_eq!(member.path.node.to_string(), "stellar.moon.mass");
        assert!(member.title.is_some());
        assert!(!member.local_config.is_empty());
    }

    #[test]
    fn test_parse_member_with_initial() {
        let src = r#"member.stellar.star.rotation_period {
            : Scalar<day, 0.1..100>
            : strata(stellar.activity)
            initial { config.stellar.default_rotation_period_days }
            resolve { prev }
        }"#;

        let lexer = Token::lexer(src).spanned().map(lex_map as fn(_) -> _);
        let stream = Stream::from_iter(lexer).map(
            SimpleSpan::from(src.len()..src.len()),
            attach_span as fn(_) -> _,
        );

        let result = member_def().parse(stream);
        assert!(result.has_output());
        let member = result.into_output().unwrap();
        assert_eq!(member.path.node.to_string(), "stellar.star.rotation_period");
        assert!(member.initial.is_some(), "initial block should be parsed");
        assert!(member.resolve.is_some());
    }
}
