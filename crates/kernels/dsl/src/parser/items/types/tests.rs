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

fn assert_primitive_name(expr: &TypeExpr, expected: &str) {
    match expr {
        TypeExpr::Primitive(primitive) => {
            assert_eq!(primitive.id.name(), expected);
        }
        TypeExpr::Named(name) => panic!("Expected primitive type, got named '{name}'"),
    }
}

#[test]
fn parse_core_type_primitives() {
    assert_primitive_name(&parse_type_expr("Scalar"), "Scalar");
    assert_primitive_name(&parse_type_expr("Vec3"), "Vec3");
    assert_primitive_name(&parse_type_expr("Quat"), "Quat");
    assert_primitive_name(&parse_type_expr("Tensor<2, 3, kg>"), "Tensor");
    assert_primitive_name(&parse_type_expr("Grid<4, 4, Scalar>"), "Grid");
    assert_primitive_name(&parse_type_expr("Seq<Scalar>"), "Seq");
}

#[test]
fn named_types_still_parse() {
    let parsed = parse_type_expr("MyCustomType");
    assert_eq!(parsed, TypeExpr::Named("MyCustomType".to_string()));
}
