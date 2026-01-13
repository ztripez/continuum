use chumsky::Parser;
use chumsky::input::{Input, Stream};
use chumsky::span::SimpleSpan;
use logos::Logos;

use super::super::super::ParserInput;
use super::super::super::lexer::Token;
use super::type_expr;
use crate::ast::TypeExpr;

fn lex_map<'src>(
    tok_span: (
        Result<Token, <Token as Logos<'src>>::Error>,
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

fn parser_input(source: &str) -> ParserInput<'_> {
    let lexer = Token::lexer(source).spanned().map(lex_map as fn(_) -> _);
    Stream::from_iter(lexer).map(
        SimpleSpan::from(source.len()..source.len()),
        attach_span as fn(_) -> _,
    )
}

fn parse_type_expr(source: &str) -> TypeExpr {
    let input = parser_input(source);
    type_expr().parse(input).into_output_errors().0.unwrap()
}

#[test]
fn parse_core_type_primitives() {
    assert!(matches!(parse_type_expr("Scalar"), TypeExpr::Scalar { .. }));
    assert!(matches!(
        parse_type_expr("Vec3"),
        TypeExpr::Vector { dim: 3, .. }
    ));
    assert!(matches!(
        parse_type_expr("Tensor<2, 3, kg>"),
        TypeExpr::Tensor {
            rows: 2,
            cols: 3,
            ..
        }
    ));
    assert!(matches!(
        parse_type_expr("Grid<4, 4, Scalar>"),
        TypeExpr::Grid {
            width: 4,
            height: 4,
            ..
        }
    ));
    assert!(matches!(
        parse_type_expr("Seq<Scalar>"),
        TypeExpr::Seq { .. }
    ));
}

#[test]
fn named_types_still_parse() {
    let parsed = parse_type_expr("MyCustomType");
    assert_eq!(parsed, TypeExpr::Named("MyCustomType".to_string()));
}
