use chumsky::prelude::*;

use chumsky::prelude::{Parser, Rich, choice, extra, select};
use chumsky::span::SimpleSpan;

use continuum_foundation::{
    PrimitiveParamKind, PrimitiveParamSpec, PrimitiveTypeDef, primitive_type_by_name,
};

use crate::ast::{PrimitiveParamValue, PrimitiveTypeExpr, Range, TypeExpr};

use super::super::super::lexer::Token;
use super::super::super::primitives::{float, ident, tok, unit_string};
use super::super::super::{ParseError, ParserInput};

/// Parser for all primitive types registered in the DSL.
pub fn primitive_type_parser<'src>(
    type_expr_recurse: impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>>
    + Clone,
) -> impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>> + Clone {
    let params_parser = type_params_parser(type_expr_recurse);

    ident()
        .then(params_parser.or_not())
        .try_map(|(name, params), span| {
            let Some(def) = primitive_type_by_name(&name) else {
                if params.is_some() {
                    return Err(Rich::custom(
                        span.into(),
                        format!("unknown primitive type '{name}'"),
                    ));
                }
                return Ok(TypeExpr::Named(name));
            };

            let params = params.unwrap_or_default();
            let params = validate_params(def, params, span)?;

            Ok(TypeExpr::Primitive(PrimitiveTypeExpr {
                id: def.id,
                params,
                constraints: Vec::new(),
                seq_constraints: Vec::new(),
            }))
        })
}

#[derive(Debug, Clone)]
enum RawParamValue {
    Unit(String),
    Range(Range),
    Integer(i64),
    Number(f64),
    TypeExpr(TypeExpr),
}

#[derive(Debug, Clone)]
enum TypeParamInput {
    Named(String, RawParamValue),
    Positional(RawParamValue),
}

fn type_params_parser<'src>(
    type_expr_recurse: impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>>
    + Clone,
) -> impl Parser<'src, ParserInput<'src>, Vec<TypeParamInput>, extra::Err<ParseError<'src>>> + Clone
{
    let raw_value = raw_param_value_parser(type_expr_recurse);

    let named = ident()
        .then_ignore(tok(Token::Colon))
        .then(raw_value.clone())
        .map(|(name, value)| TypeParamInput::Named(name, value));

    let positional = raw_value.map(TypeParamInput::Positional);

    choice((named, positional))
        .separated_by(tok(Token::Comma))
        .allow_trailing()
        .collect()
        .delimited_by(tok(Token::LAngle), tok(Token::RAngle))
}

fn raw_param_value_parser<'src>(
    type_expr_recurse: impl Parser<'src, ParserInput<'src>, TypeExpr, extra::Err<ParseError<'src>>>
    + Clone,
) -> impl Parser<'src, ParserInput<'src>, RawParamValue, extra::Err<ParseError<'src>>> + Clone {
    choice((
        range().map(RawParamValue::Range),
        integer_value().map(RawParamValue::Integer),
        numeric_value().map(RawParamValue::Number),
        unit_string()
            .then_ignore(tok(Token::LAngle).not())
            .map(RawParamValue::Unit),
        type_expr_recurse.map(RawParamValue::TypeExpr),
    ))
}

fn range<'src>() -> impl Parser<'src, ParserInput<'src>, Range, extra::Err<ParseError<'src>>> + Clone
{
    numeric_value()
        .then_ignore(tok(Token::DotDot))
        .then(numeric_value())
        .map(|(min, max)| Range { min, max })
}

fn numeric_value<'src>()
-> impl Parser<'src, ParserInput<'src>, f64, extra::Err<ParseError<'src>>> + Clone {
    let value = choice((
        float(),
        ident().try_map(|name: String, span: SimpleSpan| {
            crate::math_consts::lookup(&name)
                .ok_or_else(|| Rich::custom(span.into(), format!("unknown math constant '{name}'")))
        }),
    ));

    tok(Token::Minus)
        .or_not()
        .then(value)
        .map(|(minus, val): (Option<Token>, f64)| if minus.is_some() { -val } else { val })
}

fn integer_value<'src>()
-> impl Parser<'src, ParserInput<'src>, i64, extra::Err<ParseError<'src>>> + Clone {
    tok(Token::Minus)
        .or_not()
        .then(select! { Token::Integer(i) => i })
        .map(|(minus, val): (Option<Token>, i64)| if minus.is_some() { -val } else { val })
}

fn validate_params<'src>(
    def: &'static PrimitiveTypeDef,
    params: Vec<TypeParamInput>,
    span: SimpleSpan,
) -> Result<Vec<PrimitiveParamValue>, Rich<'src, Token>> {
    let mut positional_specs: Vec<&PrimitiveParamSpec> = def
        .params
        .iter()
        .filter(|spec| spec.position.is_some())
        .collect();
    positional_specs.sort_by_key(|spec| spec.position);

    let mut values = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let mut positional_iter = positional_specs.iter();

    for input in params {
        match input {
            TypeParamInput::Positional(raw) => {
                let Some(spec) = positional_iter.next() else {
                    return Err(Rich::custom(
                        span.into(),
                        format!("too many positional parameters for {}", def.name),
                    ));
                };
                let value = convert_param_value(spec.kind, raw, span)?;
                insert_param_value(&mut values, &mut seen, value, span)?;
            }
            TypeParamInput::Named(name, raw) => {
                let Some(spec) = def.params.iter().find(|spec| spec.name == name) else {
                    return Err(Rich::custom(
                        span.into(),
                        format!("unknown parameter '{name}' for {}", def.name),
                    ));
                };
                let value = convert_param_value(spec.kind, raw, span)?;
                insert_param_value(&mut values, &mut seen, value, span)?;
            }
        }
    }

    for spec in positional_iter {
        if !spec.optional {
            return Err(Rich::custom(
                span.into(),
                format!(
                    "missing required parameter '{}' for {}",
                    spec.name, def.name
                ),
            ));
        }
    }

    Ok(values)
}

fn insert_param_value<'src>(
    values: &mut Vec<PrimitiveParamValue>,
    seen: &mut std::collections::HashSet<PrimitiveParamKind>,
    value: PrimitiveParamValue,
    span: SimpleSpan,
) -> Result<(), Rich<'src, Token>> {
    let kind = value.kind();
    if !seen.insert(kind) {
        return Err(Rich::custom(
            span.into(),
            format!("duplicate parameter '{}'", kind_name(kind)),
        ));
    }
    values.push(value);
    Ok(())
}

fn kind_name(kind: PrimitiveParamKind) -> &'static str {
    match kind {
        PrimitiveParamKind::Unit => "unit",
        PrimitiveParamKind::Range => "range",
        PrimitiveParamKind::Magnitude => "magnitude",
        PrimitiveParamKind::Rows => "rows",
        PrimitiveParamKind::Cols => "cols",
        PrimitiveParamKind::Width => "width",
        PrimitiveParamKind::Height => "height",
        PrimitiveParamKind::ElementType => "element_type",
    }
}

fn range_from_value(value: f64) -> Range {
    Range {
        min: value,
        max: value,
    }
}

fn convert_param_value<'src>(
    kind: PrimitiveParamKind,
    raw: RawParamValue,
    span: SimpleSpan,
) -> Result<PrimitiveParamValue, Rich<'src, Token>> {
    match (kind, raw) {
        (PrimitiveParamKind::Unit, RawParamValue::Unit(unit)) => {
            Ok(PrimitiveParamValue::Unit(unit))
        }
        (PrimitiveParamKind::Unit, RawParamValue::TypeExpr(TypeExpr::Named(name))) => {
            Ok(PrimitiveParamValue::Unit(name))
        }
        (PrimitiveParamKind::Unit, RawParamValue::Integer(value)) => {
            Ok(PrimitiveParamValue::Unit(value.to_string()))
        }
        (PrimitiveParamKind::Range, RawParamValue::Range(range)) => {
            Ok(PrimitiveParamValue::Range(range))
        }
        (PrimitiveParamKind::Range, RawParamValue::Integer(value)) => {
            Ok(PrimitiveParamValue::Range(range_from_value(value as f64)))
        }
        (PrimitiveParamKind::Range, RawParamValue::Number(value)) => {
            Ok(PrimitiveParamValue::Range(range_from_value(value)))
        }
        (PrimitiveParamKind::Magnitude, RawParamValue::Range(range)) => {
            Ok(PrimitiveParamValue::Magnitude(range))
        }
        (PrimitiveParamKind::Magnitude, RawParamValue::Integer(value)) => Ok(
            PrimitiveParamValue::Magnitude(range_from_value(value as f64)),
        ),
        (PrimitiveParamKind::Magnitude, RawParamValue::Number(value)) => {
            Ok(PrimitiveParamValue::Magnitude(range_from_value(value)))
        }
        (PrimitiveParamKind::Rows, RawParamValue::Integer(value)) => {
            Ok(PrimitiveParamValue::Rows(to_u8(value, span)?))
        }
        (PrimitiveParamKind::Cols, RawParamValue::Integer(value)) => {
            Ok(PrimitiveParamValue::Cols(to_u8(value, span)?))
        }
        (PrimitiveParamKind::Width, RawParamValue::Integer(value)) => {
            Ok(PrimitiveParamValue::Width(to_u32(value, span)?))
        }
        (PrimitiveParamKind::Height, RawParamValue::Integer(value)) => {
            Ok(PrimitiveParamValue::Height(to_u32(value, span)?))
        }
        (PrimitiveParamKind::ElementType, RawParamValue::TypeExpr(expr)) => {
            Ok(PrimitiveParamValue::ElementType(Box::new(expr)))
        }
        (PrimitiveParamKind::ElementType, RawParamValue::Unit(name)) => Ok(
            PrimitiveParamValue::ElementType(Box::new(TypeExpr::Named(name))),
        ),
        (expected, _) => Err(Rich::custom(
            span.into(),
            format!("invalid value for parameter '{}'", kind_name(expected)),
        )),
    }
}

fn to_u8<'src>(value: i64, span: SimpleSpan) -> Result<u8, Rich<'src, Token>> {
    if value < 0 || value > u8::MAX as i64 {
        return Err(Rich::custom(
            span.into(),
            format!("integer value {value} out of range"),
        ));
    }
    Ok(value as u8)
}

fn to_u32<'src>(value: i64, span: SimpleSpan) -> Result<u32, Rich<'src, Token>> {
    if value < 0 || value > u32::MAX as i64 {
        return Err(Rich::custom(
            span.into(),
            format!("integer value {value} out of range"),
        ));
    }
    Ok(value as u32)
}
