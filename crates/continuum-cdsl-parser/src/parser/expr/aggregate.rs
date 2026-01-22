//! Aggregate expression parsing - agg.sum/product/min/max/etc.

use super::super::{ParseError, TokenStream};
use continuum_cdsl_ast::foundation::AggregateOp;
use continuum_cdsl_ast::{Expr, UntypedKind};
use continuum_cdsl_lexer::Token;

/// Parse aggregate: agg.op(source) or agg.op(source, body).
pub(super) fn parse_aggregate(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let start = stream.current_pos();
    stream.expect(Token::Agg)?;
    stream.expect(Token::Dot)?;

    let op = {
        let span = stream.current_span();
        match stream.advance() {
            Some(Token::Ident(s)) => match s.as_str() {
                "sum" => AggregateOp::Sum,
                "product" => AggregateOp::Product,
                "min" => AggregateOp::Min,
                "max" => AggregateOp::Max,
                "mean" => AggregateOp::Mean,
                "count" => AggregateOp::Count,
                "any" => AggregateOp::Any,
                "all" => AggregateOp::All,
                "none" => AggregateOp::None,
                "map" => AggregateOp::Map,
                "first" => AggregateOp::First,
                _ => {
                    return Err(ParseError::invalid_syntax(
                        format!("unknown aggregate operation: {}", s),
                        span,
                    ));
                }
            },
            other => {
                return Err(ParseError::unexpected_token(
                    other,
                    "aggregate operation",
                    span,
                ));
            }
        }
    };

    stream.expect(Token::LParen)?;
    let source = super::parse_expr(stream)?;

    // count() takes only source, others take source + body
    let body = if op == AggregateOp::Count {
        if matches!(stream.peek(), Some(Token::RParen)) {
            // count(source) - use default body
            Expr::new(UntypedKind::BoolLiteral(true), stream.current_span())
        } else {
            stream.expect(Token::Comma)?;
            super::parse_expr(stream)?
        }
    } else {
        stream.expect(Token::Comma)?;
        super::parse_expr(stream)?
    };

    stream.expect(Token::RParen)?;

    Ok(Expr::new(
        UntypedKind::Aggregate {
            op,
            source: Box::new(source),
            binding: "self".to_string(),
            body: Box::new(body),
        },
        stream.span_from(start),
    ))
}
