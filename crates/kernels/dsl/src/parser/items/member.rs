//! Member signal definition parsers.
//!
//! This module handles `member.entity.field { ... }` per-entity state declarations.
//! Member signals are top-level primitives with their own resolve blocks.

use chumsky::prelude::*;

use crate::ast::{ConfigEntry, MemberDef, ResolveBlock, Spanned};

use super::super::ParseError;
use super::super::expr::spanned_expr;
use super::super::primitives::{attr_path, attr_string, spanned, spanned_path, ws};
use super::common::assert_block;
use super::config::config_entry;
use super::types::type_expr;

// === Member Signal ===

/// Parses a member signal definition.
///
/// # DSL Syntax
///
/// ```cdsl
/// member.human.person.age {
///     : Scalar
///     : strata(human.physiology)
///     : title("Age")
///     : symbol("🎂")
///
///     config {
///         initial: 0.0
///     }
///
///     resolve { integrate(prev, 1) }
/// }
/// ```
pub fn member_def<'src>() -> impl Parser<'src, &'src str, MemberDef, extra::Err<ParseError<'src>>> {
    text::keyword("member")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            member_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
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
-> impl Parser<'src, &'src str, MemberContent, extra::Err<ParseError<'src>>> {
    choice((
        // : strata(path) - must come before Type to avoid matching "strata" as Named type
        attr_path("strata").map(MemberContent::Strata),
        // : title("...")
        attr_string("title").map(MemberContent::Title),
        // : symbol("...")
        attr_string("symbol").map(MemberContent::Symbol),
        // : Type - comes after specific attributes
        just(':')
            .padded_by(ws())
            .ignore_then(spanned(type_expr()))
            .map(MemberContent::Type),
        // config { entries }
        text::keyword("config")
            .padded_by(ws())
            .ignore_then(
                config_entry()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(MemberContent::Config),
        // initial { expr }
        text::keyword("initial")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| MemberContent::Initial(ResolveBlock { body })),
        // resolve { expr }
        text::keyword("resolve")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| MemberContent::Resolve(ResolveBlock { body })),
        // assert { assertions }
        assert_block().map(MemberContent::Assert),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_member() {
        let src = r#"member.human.person.age {
            : Scalar
            : strata(human.physiology)
            resolve { prev + 1 }
        }"#;

        let result = member_def().parse(src);
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

        let result = member_def().parse(src);
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

        let result = member_def().parse(src);
        assert!(result.has_output());
        let member = result.into_output().unwrap();
        assert_eq!(member.path.node.to_string(), "stellar.star.rotation_period");
        assert!(member.initial.is_some(), "initial block should be parsed");
        assert!(member.resolve.is_some());
    }
}
