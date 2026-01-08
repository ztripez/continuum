//! Parser for Continuum DSL
//!
//! Uses Chumsky for direct string parsing with good error recovery.

use chumsky::prelude::*;

use crate::ast::{
    BinaryOp, CompilationUnit, ConfigBlock, ConfigEntry, ConstBlock, ConstEntry, EraDef,
    Expr, FieldDef, FractureDef, ImpulseDef, Item, Literal, OperatorDef, Path, Range,
    SignalDef, Spanned, StrataDef, StrataState, StrataStateKind, Topology, Transition,
    TypeDef, TypeExpr, TypeField, UnaryOp, ValueWithUnit, ResolveBlock,
    MeasureBlock, OperatorPhase, OperatorBody, ApplyBlock, EmitStatement, ChronicleDef,
    ObserveBlock, ObserveHandler,
};

/// Parse error type
pub type ParseError<'src> = Rich<'src, char>;

/// Parse source code into a compilation unit
pub fn parse(source: &str) -> (Option<CompilationUnit>, Vec<ParseError<'_>>) {
    compilation_unit()
        .parse(source)
        .into_output_errors()
}

// =============================================================================
// Helper Combinators
// =============================================================================

/// Parse whitespace and comments
fn ws<'src>() -> impl Parser<'src, &'src str, (), extra::Err<ParseError<'src>>> + Clone {
    let line_comment = just("//")
        .then(any().and_is(just('\n').not()).repeated())
        .padded();
    let hash_comment = just("#")
        .then(any().and_is(just('\n').not()).repeated())
        .padded();
    let block_comment = just("/*")
        .then(any().and_is(just("*/").not()).repeated())
        .then(just("*/"))
        .padded();

    choice((
        line_comment.ignored(),
        hash_comment.ignored(),
        block_comment.ignored(),
        text::whitespace().at_least(1).ignored(),
    ))
    .repeated()
    .ignored()
}

/// Parse an identifier (lowercase with underscores)
fn ident<'src>() -> impl Parser<'src, &'src str, String, extra::Err<ParseError<'src>>> + Clone {
    text::ascii::ident()
        .map(|s: &str| s.to_string())
}

/// Parse a path (dot-separated identifiers)
fn path<'src>() -> impl Parser<'src, &'src str, Path, extra::Err<ParseError<'src>>> + Clone {
    ident()
        .separated_by(just('.'))
        .at_least(1)
        .collect::<Vec<_>>()
        .map(Path::new)
}

/// Parse a spanned path
fn spanned_path<'src>(
) -> impl Parser<'src, &'src str, Spanned<Path>, extra::Err<ParseError<'src>>> + Clone {
    path().map_with(|p, e| Spanned::new(p, e.span().into()))
}

/// Parse a string literal
fn string_lit<'src>() -> impl Parser<'src, &'src str, String, extra::Err<ParseError<'src>>> + Clone
{
    none_of("\"\\")
        .or(just('\\').ignore_then(any()))
        .repeated()
        .collect::<String>()
        .delimited_by(just('"'), just('"'))
}

/// Parse a float
fn float<'src>() -> impl Parser<'src, &'src str, f64, extra::Err<ParseError<'src>>> + Clone {
    just('-')
        .or_not()
        .then(text::int(10))
        .then(just('.').then(text::digits(10)).or_not())
        .then(
            one_of("eE")
                .then(one_of("+-").or_not())
                .then(text::digits(10))
                .or_not(),
        )
        .to_slice()
        .map(|s: &str| s.parse().unwrap_or(0.0))
}

/// Parse a number (float or integer)
fn number<'src>() -> impl Parser<'src, &'src str, Literal, extra::Err<ParseError<'src>>> + Clone {
    float().map(Literal::Float)
}

/// Parse a literal
fn literal<'src>() -> impl Parser<'src, &'src str, Literal, extra::Err<ParseError<'src>>> + Clone {
    choice((
        number(),
        string_lit().map(Literal::String),
    ))
}

/// Parse a unit like <K> or <W/m²>
fn unit<'src>() -> impl Parser<'src, &'src str, String, extra::Err<ParseError<'src>>> + Clone {
    none_of(">")
        .repeated()
        .at_least(1)
        .collect::<String>()
        .delimited_by(just('<'), just('>'))
}

/// Parse an optional unit
fn optional_unit<'src>(
) -> impl Parser<'src, &'src str, Option<Spanned<String>>, extra::Err<ParseError<'src>>> + Clone {
    unit()
        .map_with(|u, e| Spanned::new(u, e.span().into()))
        .or_not()
}

// =============================================================================
// Top-level Parser
// =============================================================================

/// Parse a complete compilation unit
fn compilation_unit<'src>(
) -> impl Parser<'src, &'src str, CompilationUnit, extra::Err<ParseError<'src>>> {
    ws().ignore_then(
        item()
            .map_with(|i, e| Spanned::new(i, e.span().into()))
            .padded_by(ws())
            .repeated()
            .collect()
            .map(|items| CompilationUnit { items }),
    )
}

/// Parse a top-level item
fn item<'src>() -> impl Parser<'src, &'src str, Item, extra::Err<ParseError<'src>>> {
    choice((
        const_block().map(Item::ConstBlock),
        config_block().map(Item::ConfigBlock),
        type_def().map(Item::TypeDef),
        strata_def().map(Item::StrataDef),
        era_def().map(Item::EraDef),
        signal_def().map(Item::SignalDef),
        field_def().map(Item::FieldDef),
        operator_def().map(Item::OperatorDef),
        impulse_def().map(Item::ImpulseDef),
        fracture_def().map(Item::FractureDef),
        chronicle_def().map(Item::ChronicleDef),
    ))
}

// =============================================================================
// Const and Config Blocks
// =============================================================================

/// Parse: const { entries... }
fn const_block<'src>(
) -> impl Parser<'src, &'src str, ConstBlock, extra::Err<ParseError<'src>>> {
    text::keyword("const")
        .padded_by(ws())
        .ignore_then(
            const_entry()
                .padded_by(ws())
                .repeated()
                .collect()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|entries| ConstBlock { entries })
}

/// Parse: path: value <unit>
fn const_entry<'src>() -> impl Parser<'src, &'src str, ConstEntry, extra::Err<ParseError<'src>>> {
    spanned_path()
        .then_ignore(just(':').padded_by(ws()))
        .then(literal().map_with(|l, e| Spanned::new(l, e.span().into())))
        .then(optional_unit().padded_by(ws()))
        .map(|((path, value), unit)| ConstEntry { path, value, unit })
}

/// Parse: config { entries... }
fn config_block<'src>(
) -> impl Parser<'src, &'src str, ConfigBlock, extra::Err<ParseError<'src>>> {
    text::keyword("config")
        .padded_by(ws())
        .ignore_then(
            config_entry()
                .padded_by(ws())
                .repeated()
                .collect()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|entries| ConfigBlock { entries })
}

/// Parse: path: value <unit>
fn config_entry<'src>() -> impl Parser<'src, &'src str, ConfigEntry, extra::Err<ParseError<'src>>>
{
    spanned_path()
        .then_ignore(just(':').padded_by(ws()))
        .then(literal().map_with(|l, e| Spanned::new(l, e.span().into())))
        .then(optional_unit().padded_by(ws()))
        .map(|((path, value), unit)| ConfigEntry { path, value, unit })
}

// =============================================================================
// Type Definitions
// =============================================================================

/// Parse: type.Name { fields... }
fn type_def<'src>() -> impl Parser<'src, &'src str, TypeDef, extra::Err<ParseError<'src>>> {
    text::keyword("type")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(ident().map_with(|n, e| Spanned::new(n, e.span().into())))
        .padded_by(ws())
        .then(
            type_field()
                .padded_by(ws())
                .repeated()
                .collect()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(name, fields)| TypeDef { name, fields })
}

/// Parse: name: Type
fn type_field<'src>() -> impl Parser<'src, &'src str, TypeField, extra::Err<ParseError<'src>>> {
    ident()
        .map_with(|n, e| Spanned::new(n, e.span().into()))
        .then_ignore(just(':').padded_by(ws()))
        .then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
        .map(|(name, ty)| TypeField { name, ty })
}

/// Parse a type expression
fn type_expr<'src>() -> impl Parser<'src, &'src str, TypeExpr, extra::Err<ParseError<'src>>> {
    choice((
        // Scalar<unit> or Scalar<unit, range>
        text::keyword("Scalar")
            .ignore_then(
                just('<')
                    .padded_by(ws())
                    .ignore_then(ident())
                    .then(
                        just(',')
                            .padded_by(ws())
                            .ignore_then(range())
                            .or_not(),
                    )
                    .then_ignore(just('>').padded_by(ws())),
            )
            .map(|(unit, range)| TypeExpr::Scalar { unit, range }),
        // Vec2<unit>, Vec3<unit>, Vec4<unit>
        choice((
            text::keyword("Vec2").to(2u8),
            text::keyword("Vec3").to(3u8),
            text::keyword("Vec4").to(4u8),
        ))
        .then(
            just('<')
                .padded_by(ws())
                .ignore_then(ident())
                .then_ignore(just('>').padded_by(ws())),
        )
        .map(|(dim, unit)| TypeExpr::Vector {
            dim,
            unit,
            magnitude: None,
        }),
        // Named type reference
        ident().map(TypeExpr::Named),
    ))
}

/// Parse a range: min..max
fn range<'src>() -> impl Parser<'src, &'src str, Range, extra::Err<ParseError<'src>>> {
    float()
        .then_ignore(just("..").padded_by(ws()))
        .then(float())
        .map(|(min, max)| Range { min, max })
}

// =============================================================================
// Strata Definition
// =============================================================================

/// Parse: strata.path { attributes... }
fn strata_def<'src>() -> impl Parser<'src, &'src str, StrataDef, extra::Err<ParseError<'src>>> {
    text::keyword("strata")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            strata_attribute()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, attrs)| {
            let mut def = StrataDef {
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

fn strata_attribute<'src>(
) -> impl Parser<'src, &'src str, StrataAttr, extra::Err<ParseError<'src>>> {
    just(':')
        .padded_by(ws())
        .ignore_then(choice((
            text::keyword("title")
                .ignore_then(
                    string_lit()
                        .map_with(|s, e| Spanned::new(s, e.span().into()))
                        .padded_by(ws())
                        .delimited_by(just('('), just(')')),
                )
                .map(StrataAttr::Title),
            text::keyword("symbol")
                .ignore_then(
                    string_lit()
                        .map_with(|s, e| Spanned::new(s, e.span().into()))
                        .padded_by(ws())
                        .delimited_by(just('('), just(')')),
                )
                .map(StrataAttr::Symbol),
            text::keyword("stride")
                .ignore_then(
                    text::int(10)
                        .map(|s: &str| s.parse::<u32>().unwrap_or(1))
                        .map_with(|n: u32, e| {
                            let span: chumsky::span::SimpleSpan = e.span();
                            Spanned::new(n, span.start..span.end)
                        })
                        .padded_by(ws())
                        .delimited_by(just('('), just(')')),
                )
                .map(StrataAttr::Stride),
        )))
}

// =============================================================================
// Era Definition
// =============================================================================

/// Parse: era.name { attributes... }
fn era_def<'src>() -> impl Parser<'src, &'src str, EraDef, extra::Err<ParseError<'src>>> {
    text::keyword("era")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(ident().map_with(|n, e| Spanned::new(n, e.span().into())))
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

fn era_content<'src>() -> impl Parser<'src, &'src str, EraContent, extra::Err<ParseError<'src>>> {
    choice((
        // : initial
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("initial"))
            .to(EraContent::Initial),
        // : terminal
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("terminal"))
            .to(EraContent::Terminal),
        // : title("...")
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("title"))
            .ignore_then(
                string_lit()
                    .map_with(|s, e| Spanned::new(s, e.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(EraContent::Title),
        // : dt(value <unit>)
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("dt"))
            .ignore_then(
                value_with_unit()
                    .map_with(|v, e| Spanned::new(v, e.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(EraContent::Dt),
        // strata { ... }
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
        // transition { ... }
        text::keyword("transition")
            .padded_by(ws())
            .ignore_then(
                transition().delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(EraContent::Transition),
    ))
}

fn value_with_unit<'src>(
) -> impl Parser<'src, &'src str, ValueWithUnit, extra::Err<ParseError<'src>>> {
    literal()
        .padded_by(ws())
        .then(unit())
        .map(|(value, unit)| ValueWithUnit { value, unit })
}

fn strata_state<'src>() -> impl Parser<'src, &'src str, StrataState, extra::Err<ParseError<'src>>>
{
    spanned_path()
        .then_ignore(just(':').padded_by(ws()))
        .then(strata_state_kind())
        .map(|(strata, state)| StrataState { strata, state })
}

fn strata_state_kind<'src>(
) -> impl Parser<'src, &'src str, StrataStateKind, extra::Err<ParseError<'src>>> {
    choice((
        text::keyword("active")
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(text::keyword("stride"))
                    .ignore_then(just(':').padded_by(ws()))
                    .ignore_then(
                        text::int(10)
                            .map(|s: &str| s.parse::<u32>().unwrap_or(1)),
                    )
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

fn transition<'src>() -> impl Parser<'src, &'src str, Transition, extra::Err<ParseError<'src>>> {
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
                    simple_expr()
                        .map_with(|e, extra| Spanned::new(e, extra.span().into()))
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

// =============================================================================
// Signal Definition
// =============================================================================

/// Parse: signal.path { ... }
fn signal_def<'src>() -> impl Parser<'src, &'src str, SignalDef, extra::Err<ParseError<'src>>> {
    text::keyword("signal")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            signal_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = SignalDef {
                path,
                ty: None,
                strata: None,
                title: None,
                symbol: None,
                dt_raw: false,
                local_consts: vec![],
                local_config: vec![],
                warmup: None,
                resolve: None,
            };
            for content in contents {
                match content {
                    SignalContent::Type(t) => def.ty = Some(t),
                    SignalContent::Strata(s) => def.strata = Some(s),
                    SignalContent::Title(t) => def.title = Some(t),
                    SignalContent::Symbol(s) => def.symbol = Some(s),
                    SignalContent::DtRaw => def.dt_raw = true,
                    SignalContent::Resolve(r) => def.resolve = Some(r),
                }
            }
            def
        })
}

#[derive(Clone)]
enum SignalContent {
    Type(Spanned<TypeExpr>),
    Strata(Spanned<Path>),
    Title(Spanned<String>),
    Symbol(Spanned<String>),
    DtRaw,
    Resolve(ResolveBlock),
}

fn signal_content<'src>(
) -> impl Parser<'src, &'src str, SignalContent, extra::Err<ParseError<'src>>> {
    choice((
        // : strata(path) - must come before type_expr since "strata" looks like an identifier
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("strata"))
            .ignore_then(spanned_path().padded_by(ws()).delimited_by(just('('), just(')')))
            .map(SignalContent::Strata),
        // : title("...")
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("title"))
            .ignore_then(
                string_lit()
                    .map_with(|s, e| Spanned::new(s, e.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(SignalContent::Title),
        // : symbol("...")
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("symbol"))
            .ignore_then(
                string_lit()
                    .map_with(|s, e| Spanned::new(s, e.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(SignalContent::Symbol),
        // : dt_raw
        just(':')
            .padded_by(ws())
            .ignore_then(just("dt_raw"))
            .to(SignalContent::DtRaw),
        // : Type<...> - must come after keyword-specific parsers
        just(':')
            .padded_by(ws())
            .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
            .map(SignalContent::Type),
        // resolve { expr }
        resolve_block().map(SignalContent::Resolve),
    ))
}

fn resolve_block<'src>(
) -> impl Parser<'src, &'src str, ResolveBlock, extra::Err<ParseError<'src>>> {
    text::keyword("resolve")
        .padded_by(ws())
        .ignore_then(
            simple_expr()
                .map_with(|e, extra| Spanned::new(e, extra.span().into()))
                .padded_by(ws())
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|body| ResolveBlock { body })
}

// =============================================================================
// Field Definition
// =============================================================================

fn field_def<'src>() -> impl Parser<'src, &'src str, FieldDef, extra::Err<ParseError<'src>>> {
    text::keyword("field")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            field_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = FieldDef {
                path,
                ty: None,
                strata: None,
                topology: None,
                title: None,
                symbol: None,
                measure: None,
            };
            for content in contents {
                match content {
                    FieldContent::Type(t) => def.ty = Some(t),
                    FieldContent::Strata(s) => def.strata = Some(s),
                    FieldContent::Topology(t) => def.topology = Some(t),
                    FieldContent::Title(t) => def.title = Some(t),
                    FieldContent::Symbol(s) => def.symbol = Some(s),
                    FieldContent::Measure(m) => def.measure = Some(m),
                }
            }
            def
        })
}

#[derive(Clone)]
enum FieldContent {
    Type(Spanned<TypeExpr>),
    Strata(Spanned<Path>),
    Topology(Spanned<Topology>),
    Title(Spanned<String>),
    Symbol(Spanned<String>),
    Measure(MeasureBlock),
}

fn field_content<'src>(
) -> impl Parser<'src, &'src str, FieldContent, extra::Err<ParseError<'src>>> {
    choice((
        // : strata(path) - must come before type_expr since "strata" looks like an identifier
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("strata"))
            .ignore_then(spanned_path().padded_by(ws()).delimited_by(just('('), just(')')))
            .map(FieldContent::Strata),
        // : topology(kind)
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("topology"))
            .ignore_then(
                topology()
                    .map_with(|t, e| Spanned::new(t, e.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(FieldContent::Topology),
        // : title("...")
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("title"))
            .ignore_then(
                string_lit()
                    .map_with(|s, e| Spanned::new(s, e.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(FieldContent::Title),
        // : symbol("...")
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("symbol"))
            .ignore_then(
                string_lit()
                    .map_with(|s, e| Spanned::new(s, e.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(FieldContent::Symbol),
        // : Type<...> - must come after keyword-specific parsers
        just(':')
            .padded_by(ws())
            .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
            .map(FieldContent::Type),
        // measure { expr }
        measure_block().map(FieldContent::Measure),
    ))
}

fn topology<'src>() -> impl Parser<'src, &'src str, Topology, extra::Err<ParseError<'src>>> {
    choice((
        text::keyword("sphere_surface").to(Topology::SphereSurface),
        text::keyword("point_cloud").to(Topology::PointCloud),
        text::keyword("volume").to(Topology::Volume),
    ))
}

fn measure_block<'src>(
) -> impl Parser<'src, &'src str, MeasureBlock, extra::Err<ParseError<'src>>> {
    text::keyword("measure")
        .padded_by(ws())
        .ignore_then(
            simple_expr()
                .map_with(|e, extra| Spanned::new(e, extra.span().into()))
                .padded_by(ws())
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|body| MeasureBlock { body })
}

// =============================================================================
// Operator Definition
// =============================================================================

fn operator_def<'src>() -> impl Parser<'src, &'src str, OperatorDef, extra::Err<ParseError<'src>>>
{
    text::keyword("operator")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            operator_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = OperatorDef {
                path,
                strata: None,
                phase: None,
                body: None,
            };
            for content in contents {
                match content {
                    OperatorContent::Strata(s) => def.strata = Some(s),
                    OperatorContent::Phase(p) => def.phase = Some(p),
                    OperatorContent::Body(b) => def.body = Some(b),
                }
            }
            def
        })
}

#[derive(Clone)]
enum OperatorContent {
    Strata(Spanned<Path>),
    Phase(Spanned<OperatorPhase>),
    Body(OperatorBody),
}

fn operator_content<'src>(
) -> impl Parser<'src, &'src str, OperatorContent, extra::Err<ParseError<'src>>> {
    choice((
        // : strata(path)
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("strata"))
            .ignore_then(spanned_path().padded_by(ws()).delimited_by(just('('), just(')')))
            .map(OperatorContent::Strata),
        // : phase(kind)
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("phase"))
            .ignore_then(
                operator_phase()
                    .map_with(|p, e| Spanned::new(p, e.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(OperatorContent::Phase),
        // collect { ... }
        text::keyword("collect")
            .padded_by(ws())
            .ignore_then(
                simple_expr()
                    .map_with(|e, extra| Spanned::new(e, extra.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|e| OperatorContent::Body(OperatorBody::Collect(e))),
        // measure { ... }
        text::keyword("measure")
            .padded_by(ws())
            .ignore_then(
                simple_expr()
                    .map_with(|e, extra| Spanned::new(e, extra.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|e| OperatorContent::Body(OperatorBody::Measure(e))),
    ))
}

fn operator_phase<'src>(
) -> impl Parser<'src, &'src str, OperatorPhase, extra::Err<ParseError<'src>>> {
    choice((
        text::keyword("warmup").to(OperatorPhase::Warmup),
        text::keyword("collect").to(OperatorPhase::Collect),
        text::keyword("measure").to(OperatorPhase::Measure),
    ))
}

// =============================================================================
// Impulse Definition
// =============================================================================

fn impulse_def<'src>() -> impl Parser<'src, &'src str, ImpulseDef, extra::Err<ParseError<'src>>> {
    text::keyword("impulse")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            impulse_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = ImpulseDef {
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

fn impulse_content<'src>(
) -> impl Parser<'src, &'src str, ImpulseContent, extra::Err<ParseError<'src>>> {
    choice((
        // : Type
        just(':')
            .padded_by(ws())
            .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
            .map(ImpulseContent::Type),
        // apply { ... }
        text::keyword("apply")
            .padded_by(ws())
            .ignore_then(
                simple_expr()
                    .map_with(|e, extra| Spanned::new(e, extra.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| ImpulseContent::Apply(ApplyBlock { body })),
    ))
}

// =============================================================================
// Fracture Definition
// =============================================================================

fn fracture_def<'src>() -> impl Parser<'src, &'src str, FractureDef, extra::Err<ParseError<'src>>>
{
    text::keyword("fracture")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            fracture_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
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

fn fracture_content<'src>(
) -> impl Parser<'src, &'src str, FractureContent, extra::Err<ParseError<'src>>> {
    choice((
        // when { conditions... }
        text::keyword("when")
            .padded_by(ws())
            .ignore_then(
                simple_expr()
                    .map_with(|e, extra| Spanned::new(e, extra.span().into()))
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(FractureContent::When),
        // emit { statements... }
        text::keyword("emit")
            .padded_by(ws())
            .ignore_then(
                emit_statement()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(FractureContent::Emit),
    ))
}

fn emit_statement<'src>(
) -> impl Parser<'src, &'src str, EmitStatement, extra::Err<ParseError<'src>>> {
    text::keyword("signal")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .then_ignore(just("<-").padded_by(ws()))
        .then(simple_expr().map_with(|e, extra| Spanned::new(e, extra.span().into())))
        .map(|(target, value)| EmitStatement { target, value })
}

// =============================================================================
// Chronicle Definition
// =============================================================================

fn chronicle_def<'src>(
) -> impl Parser<'src, &'src str, ChronicleDef, extra::Err<ParseError<'src>>> {
    text::keyword("chronicle")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            observe_block()
                .padded_by(ws())
                .or_not()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, observe)| ChronicleDef { path, observe })
}

fn observe_block<'src>(
) -> impl Parser<'src, &'src str, ObserveBlock, extra::Err<ParseError<'src>>> {
    text::keyword("observe")
        .padded_by(ws())
        .ignore_then(
            observe_handler()
                .padded_by(ws())
                .repeated()
                .collect()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|handlers| ObserveBlock { handlers })
}

fn observe_handler<'src>(
) -> impl Parser<'src, &'src str, ObserveHandler, extra::Err<ParseError<'src>>> {
    text::keyword("when")
        .padded_by(ws())
        .ignore_then(simple_expr().map_with(|e, extra| Spanned::new(e, extra.span().into())))
        .then(
            text::keyword("emit")
                .padded_by(ws())
                .ignore_then(text::keyword("event"))
                .padded_by(ws())
                .ignore_then(just('.'))
                .ignore_then(spanned_path())
                .then(
                    event_field()
                        .padded_by(ws())
                        .repeated()
                        .collect()
                        .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
                )
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(condition, (event_name, event_fields))| ObserveHandler {
            condition,
            event_name,
            event_fields,
        })
}

fn event_field<'src>(
) -> impl Parser<'src, &'src str, (Spanned<String>, Spanned<Expr>), extra::Err<ParseError<'src>>>
{
    ident()
        .map_with(|n, e| Spanned::new(n, e.span().into()))
        .then_ignore(just(':').padded_by(ws()))
        .then(simple_expr().map_with(|e, extra| Spanned::new(e, extra.span().into())))
}

// =============================================================================
// Simple Expression Parser
// =============================================================================

/// A simplified expression parser for now
fn simple_expr<'src>() -> impl Parser<'src, &'src str, Expr, extra::Err<ParseError<'src>>> + Clone
{
    recursive(|expr| {
        let atom = choice((
            // prev
            text::keyword("prev").to(Expr::Prev),
            // dt
            text::keyword("dt").to(Expr::Dt),
            // payload
            text::keyword("payload").to(Expr::Payload),
            // sum(inputs)
            text::keyword("sum")
                .ignore_then(
                    just('(')
                        .padded_by(ws())
                        .ignore_then(text::keyword("inputs"))
                        .ignore_then(just(')').padded_by(ws())),
                )
                .to(Expr::SumInputs),
            // signal.path
            text::keyword("signal")
                .ignore_then(just('.'))
                .ignore_then(path())
                .map(Expr::SignalRef),
            // const.path
            text::keyword("const")
                .ignore_then(just('.'))
                .ignore_then(path())
                .map(Expr::ConstRef),
            // config.path
            text::keyword("config")
                .ignore_then(just('.'))
                .ignore_then(path())
                .map(Expr::ConfigRef),
            // field.path
            text::keyword("field")
                .ignore_then(just('.'))
                .ignore_then(path())
                .map(Expr::FieldRef),
            // Number literal
            number().map(Expr::Literal),
            // String literal
            string_lit().map(|s| Expr::Literal(Literal::String(s))),
            // Parenthesized expression
            expr.clone()
                .padded_by(ws())
                .delimited_by(just('('), just(')')),
            // Path/identifier
            path().map(Expr::Path),
        ))
        .padded_by(ws());

        // Unary negation
        let unary = just('-')
            .repeated()
            .foldr(atom, |_, operand| Expr::Unary {
                op: UnaryOp::Neg,
                operand: Box::new(Spanned::new(operand, 0..0)),
            });

        // Binary operators: * /
        let product = unary.clone().foldl(
            choice((just('*').to(BinaryOp::Mul), just('/').to(BinaryOp::Div)))
                .padded_by(ws())
                .then(unary.clone())
                .repeated(),
            |left, (op, right)| Expr::Binary {
                op,
                left: Box::new(Spanned::new(left, 0..0)),
                right: Box::new(Spanned::new(right, 0..0)),
            },
        );

        // Binary operators: + -
        let sum = product.clone().foldl(
            choice((just('+').to(BinaryOp::Add), just('-').to(BinaryOp::Sub)))
                .padded_by(ws())
                .then(product.clone())
                .repeated(),
            |left, (op, right)| Expr::Binary {
                op,
                left: Box::new(Spanned::new(left, 0..0)),
                right: Box::new(Spanned::new(right, 0..0)),
            },
        );

        // Comparison operators
        let comparison = sum.clone().foldl(
            choice((
                just("==").to(BinaryOp::Eq),
                just("!=").to(BinaryOp::Ne),
                just("<=").to(BinaryOp::Le),
                just(">=").to(BinaryOp::Ge),
                just('<').to(BinaryOp::Lt),
                just('>').to(BinaryOp::Gt),
            ))
            .padded_by(ws())
            .then(sum.clone())
            .repeated(),
            |left, (op, right)| Expr::Binary {
                op,
                left: Box::new(Spanned::new(left, 0..0)),
                right: Box::new(Spanned::new(right, 0..0)),
            },
        );

        comparison
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_const_block() {
        let source = r#"
            const {
                physics.stefan_boltzmann: 5.67e-8 <W>
            }
        "#;
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0].node {
            Item::ConstBlock(block) => {
                assert_eq!(block.entries.len(), 1);
                assert_eq!(
                    block.entries[0].path.node.join("."),
                    "physics.stefan_boltzmann"
                );
            }
            _ => panic!("expected ConstBlock"),
        }
    }

    #[test]
    fn test_parse_strata_def() {
        let source = r#"
            strata.terra.thermal {
                : title("Thermal")
                : symbol("Q")
                : stride(5)
            }
        "#;
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0].node {
            Item::StrataDef(def) => {
                assert_eq!(def.path.node.join("."), "terra.thermal");
                assert_eq!(def.title.as_ref().unwrap().node, "Thermal");
                assert_eq!(def.symbol.as_ref().unwrap().node, "Q");
                assert_eq!(def.stride.as_ref().unwrap().node, 5);
            }
            _ => panic!("expected StrataDef"),
        }
    }

    #[test]
    fn test_parse_signal_def() {
        let source = r#"
            signal.terra.core.temp {
                : Scalar<K, 100..10000>
                : strata(terra.thermal)

                resolve {
                    prev
                }
            }
        "#;
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0].node {
            Item::SignalDef(def) => {
                assert_eq!(def.path.node.join("."), "terra.core.temp");
                assert!(def.ty.is_some());
                assert!(def.strata.is_some());
                assert!(def.resolve.is_some());
            }
            _ => panic!("expected SignalDef"),
        }
    }

    #[test]
    fn test_parse_expression() {
        let source = "signal.terra.temp { resolve { prev + 1.0 } }";
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
    }

    #[test]
    fn test_parse_era_def() {
        let source = r#"
            era.hadean {
                : initial
                : title("Hadean")
                : dt(1 <Myr>)

                strata {
                    terra.thermal: active
                    terra.tectonics: gated
                }

                transition {
                    to: era.archean
                    when {
                        signal.time.planet_age > 500
                    }
                }
            }
        "#;
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0].node {
            Item::EraDef(def) => {
                assert_eq!(def.name.node, "hadean");
                assert!(def.is_initial);
                assert!(!def.is_terminal);
                assert!(def.title.is_some());
                assert!(def.dt.is_some());
                assert_eq!(def.strata_states.len(), 2);
                assert_eq!(def.transitions.len(), 1);
            }
            _ => panic!("expected EraDef"),
        }
    }

    #[test]
    fn test_parse_field_def() {
        let source = r#"
            field.terra.surface.temperature_map {
                : Scalar<K>
                : strata(terra.atmosphere)
                : topology(sphere_surface)
                : title("Surface Temperature")

                measure {
                    signal.terra.atmosphere.temp_profile
                }
            }
        "#;
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0].node {
            Item::FieldDef(def) => {
                assert_eq!(def.path.node.join("."), "terra.surface.temperature_map");
                assert!(def.ty.is_some());
                assert!(def.strata.is_some());
                assert!(def.topology.is_some());
                assert!(def.measure.is_some());
            }
            _ => panic!("expected FieldDef"),
        }
    }

    #[test]
    fn test_parse_fracture_def() {
        let source = r#"
            fracture.terra.climate.runaway_greenhouse {
                when {
                    signal.terra.atmosphere.co2 > 1000
                    signal.terra.surface.avg_temp > 350
                }

                emit {
                    signal.terra.atmosphere.feedback <- 1.5
                }
            }
        "#;
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0].node {
            Item::FractureDef(def) => {
                assert_eq!(def.path.node.join("."), "terra.climate.runaway_greenhouse");
                assert_eq!(def.conditions.len(), 2);
                assert_eq!(def.emit.len(), 1);
            }
            _ => panic!("expected FractureDef"),
        }
    }

    #[test]
    fn test_parse_complex_expression() {
        let source = r#"
            signal.terra.thermal.loss {
                : Scalar<W>
                : strata(terra.thermal)

                resolve {
                    const.physics.stefan_boltzmann * prev * prev * prev * prev
                }
            }
        "#;
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0].node {
            Item::SignalDef(def) => {
                assert!(def.resolve.is_some());
            }
            _ => panic!("expected SignalDef"),
        }
    }

    #[test]
    fn test_parse_type_def() {
        let source = r#"
            type.ThermalState {
                temperature: Scalar<K>
                flux: Scalar<W>
            }
        "#;
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0].node {
            Item::TypeDef(def) => {
                assert_eq!(def.name.node, "ThermalState");
                assert_eq!(def.fields.len(), 2);
            }
            _ => panic!("expected TypeDef"),
        }
    }

    #[test]
    fn test_parse_operator_def() {
        let source = r#"
            operator.terra.thermal.budget {
                : strata(terra.thermal)
                : phase(collect)

                collect {
                    signal.terra.geophysics.mantle.heat_j
                }
            }
        "#;
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0].node {
            Item::OperatorDef(def) => {
                assert_eq!(def.path.node.join("."), "terra.thermal.budget");
                assert!(def.strata.is_some());
                assert!(def.phase.is_some());
                assert!(def.body.is_some());
            }
            _ => panic!("expected OperatorDef"),
        }
    }

    #[test]
    fn test_parse_impulse_def() {
        let source = r#"
            impulse.terra.impact.asteroid {
                : ImpactEvent

                apply {
                    payload
                }
            }
        "#;
        let (result, errors) = parse(source);
        assert!(errors.is_empty(), "errors: {:?}", errors);
        let unit = result.unwrap();
        assert_eq!(unit.items.len(), 1);
        match &unit.items[0].node {
            Item::ImpulseDef(def) => {
                assert_eq!(def.path.node.join("."), "terra.impact.asteroid");
                assert!(def.payload_type.is_some());
                assert!(def.apply.is_some());
            }
            _ => panic!("expected ImpulseDef"),
        }
    }
}
