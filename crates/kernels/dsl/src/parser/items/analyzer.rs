//! Parser for analyzer definitions.
//!
//! Analyzers are post-hoc analysis queries on field snapshots.

use chumsky::prelude::*;

use crate::ast::{AnalyzerDef, Severity, Spanned, ValidationCheck};
use crate::parser::expr::spanned_expr;
use crate::parser::lexer::Token;
use crate::parser::primitives::{spanned_path, string_lit, tok};
use crate::parser::{ParseError, ParserInput};

/// Parse an analyzer definition.
///
/// # Grammar
///
/// ```text
/// analyzer terra.name {
///     : doc "description"
///     : requires(fields: [field1, field2])
///     : compute { <expr> }
///     : validate { check <expr> : severity(...) : message("...") }
/// }
/// ```
pub fn analyzer_def<'src>()
-> impl Parser<'src, ParserInput<'src>, AnalyzerDef, extra::Err<ParseError<'src>>> {
    tok(Token::Analyzer)
        .ignore_then(spanned_path())
        .then(
            analyzer_content()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(tok(Token::LBrace), tok(Token::RBrace)),
        )
        .map(|(path, contents)| {
            let mut def = AnalyzerDef {
                doc: None,
                path,
                required_fields: Vec::new(),
                compute: None,
                validations: Vec::new(),
            };
            for content in contents {
                match content {
                    AnalyzerContent::Doc(d) => def.doc = Some(d),
                    AnalyzerContent::Requires(fields) => def.required_fields = fields,
                    AnalyzerContent::Compute(expr) => def.compute = Some(expr),
                    AnalyzerContent::Validate(checks) => def.validations = checks,
                }
            }
            def
        })
}

/// Content variants for analyzer body.
#[derive(Clone)]
enum AnalyzerContent {
    Doc(String),
    Requires(Vec<Spanned<crate::ast::Path>>),
    Compute(Spanned<crate::ast::Expr>),
    Validate(Vec<ValidationCheck>),
}

/// Parse analyzer content items.
fn analyzer_content<'src>()
-> impl Parser<'src, ParserInput<'src>, AnalyzerContent, extra::Err<ParseError<'src>>> {
    choice((
        // : doc "description"
        tok(Token::Colon)
            .ignore_then(tok(Token::Doc))
            .ignore_then(string_lit())
            .map(AnalyzerContent::Doc),
        // : requires(fields: [path, path, ...])
        tok(Token::Colon)
            .ignore_then(tok(Token::Requires))
            .ignore_then(
                tok(Token::LParen)
                    .ignore_then(tok(Token::Fields))
                    .ignore_then(tok(Token::Colon))
                    .ignore_then(
                        spanned_path()
                            .separated_by(tok(Token::Comma))
                            .allow_trailing()
                            .collect::<Vec<_>>()
                            .delimited_by(tok(Token::LBracket), tok(Token::RBracket)),
                    )
                    .then_ignore(tok(Token::RParen)),
            )
            .map(AnalyzerContent::Requires),
        // : compute { expr }
        tok(Token::Colon)
            .ignore_then(tok(Token::Compute))
            .ignore_then(spanned_expr().delimited_by(tok(Token::LBrace), tok(Token::RBrace)))
            .map(AnalyzerContent::Compute),
        // : validate { check ... }
        tok(Token::Colon)
            .ignore_then(tok(Token::Validate))
            .ignore_then(
                validation_check()
                    .repeated()
                    .at_least(1)
                    .collect::<Vec<_>>()
                    .delimited_by(tok(Token::LBrace), tok(Token::RBrace)),
            )
            .map(AnalyzerContent::Validate),
    ))
}

/// Parse a single validation check.
///
/// ```text
/// check <condition> : severity(warning) : message("...")
/// ```
fn validation_check<'src>()
-> impl Parser<'src, ParserInput<'src>, ValidationCheck, extra::Err<ParseError<'src>>> {
    tok(Token::Check)
        .ignore_then(spanned_expr())
        .then(validation_attrs().repeated().collect::<Vec<_>>())
        .map(|(condition, attrs)| {
            let mut severity = Severity::Error; // Default
            let mut message = None;
            for attr in attrs {
                match attr {
                    ValidationAttr::Severity(s) => severity = s,
                    ValidationAttr::Message(m) => message = Some(m),
                }
            }
            ValidationCheck {
                condition,
                severity,
                message,
            }
        })
}

/// Validation check attributes.
#[derive(Clone)]
enum ValidationAttr {
    Severity(Severity),
    Message(Spanned<String>),
}

/// Parse validation attributes: `: level(...)` or `: message("...")`.
fn validation_attrs<'src>()
-> impl Parser<'src, ParserInput<'src>, ValidationAttr, extra::Err<ParseError<'src>>> {
    choice((
        // : level(error|warn)
        tok(Token::Colon)
            .ignore_then(tok(Token::Level))
            .ignore_then(
                choice((
                    tok(Token::Error).to(Severity::Error),
                    tok(Token::Warn).to(Severity::Warning),
                    // "info" could be added if needed
                ))
                .delimited_by(tok(Token::LParen), tok(Token::RParen)),
            )
            .map(ValidationAttr::Severity),
        // : message("...")
        tok(Token::Colon)
            .ignore_then(tok(Token::Message))
            .ignore_then(
                string_lit()
                    .map_with(|s, e| Spanned::new(s, e.span().into()))
                    .delimited_by(tok(Token::LParen), tok(Token::RParen)),
            )
            .map(ValidationAttr::Message),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::lexer::Token;
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

    fn parse_analyzer(src: &str) -> chumsky::prelude::ParseResult<AnalyzerDef, ParseError<'_>> {
        let lexer = Token::lexer(src).spanned().map(lex_map as fn(_) -> _);
        let stream = Stream::from_iter(lexer).map(
            SimpleSpan::from(src.len()..src.len()),
            attach_span as fn(_) -> _,
        );
        analyzer_def().parse(stream)
    }

    #[test]
    fn test_parse_simple_analyzer() {
        let src = r#"analyzer terra.test {
            : doc "Test analyzer"
        }"#;
        let result = parse_analyzer(src);
        assert!(
            result.has_output(),
            "Failed to parse: {:?}",
            result.errors().collect::<Vec<_>>()
        );
        let def = result.into_output().unwrap();
        assert_eq!(def.path.node.to_string(), "terra.test");
        assert_eq!(def.doc, Some("Test analyzer".to_string()));
    }

    #[test]
    fn test_parse_analyzer_with_requires() {
        let src = r#"analyzer terra.test {
            : doc "Test analyzer"
            : requires(fields: [elevation.map, thickness.map])
        }"#;
        let result = parse_analyzer(src);
        assert!(
            result.has_output(),
            "Failed to parse: {:?}",
            result.errors().collect::<Vec<_>>()
        );
        let def = result.into_output().unwrap();
        assert_eq!(def.required_fields.len(), 2);
    }

    #[test]
    fn test_parse_analyzer_with_compute() {
        let src = r#"analyzer terra.test {
            : doc "Test"
            : compute {
                42.0
            }
        }"#;
        let result = parse_analyzer(src);
        assert!(
            result.has_output(),
            "Failed to parse: {:?}",
            result.errors().collect::<Vec<_>>()
        );
        let def = result.into_output().unwrap();
        assert!(def.compute.is_some());
    }

    #[test]
    fn test_parse_analyzer_with_validate() {
        let src = r#"analyzer terra.test {
            : doc "Test"
            : validate {
                check 1.0 > 0.5 : level(warn) : message("Value too low")
            }
        }"#;
        let result = parse_analyzer(src);
        assert!(
            result.has_output(),
            "Failed to parse: {:?}",
            result.errors().collect::<Vec<_>>()
        );
        let def = result.into_output().unwrap();
        assert_eq!(def.validations.len(), 1);
        assert_eq!(def.validations[0].severity, Severity::Warning);
        assert!(def.validations[0].message.is_some());
    }
}
