//! Item parsers for top-level DSL constructs

use chumsky::prelude::*;

use crate::ast::{
    ApplyBlock, AssertBlock, AssertSeverity, Assertion, ConfigBlock, ConfigEntry, ConstBlock,
    ConstEntry, CountBounds, ChronicleDef, EmitStatement, EntityDef, EntityFieldDef,
    EntitySchemaField, EraDef, Expr, FieldDef, FnDef, FnParam, FractureDef, ImpulseDef, Item,
    MeasureBlock, ObserveBlock, ObserveHandler, OperatorBody, OperatorDef, OperatorPhase, Path,
    Range, ResolveBlock, SignalDef, Spanned, StrataDef, StrataState, StrataStateKind, Topology,
    Transition, TypeDef, TypeExpr, TypeField, ValueWithUnit,
};

use super::expr::spanned_expr;
use super::primitives::{
    float, ident, literal, optional_unit, spanned_path, string_lit, unit, unit_string, ws,
};
use super::ParseError;

pub fn item<'src>() -> impl Parser<'src, &'src str, Item, extra::Err<ParseError<'src>>> {
    choice((
        const_block().map(Item::ConstBlock),
        config_block().map(Item::ConfigBlock),
        type_def().map(Item::TypeDef),
        fn_def().map(Item::FnDef),
        strata_def().map(Item::StrataDef),
        era_def().map(Item::EraDef),
        signal_def().map(Item::SignalDef),
        field_def().map(Item::FieldDef),
        operator_def().map(Item::OperatorDef),
        impulse_def().map(Item::ImpulseDef),
        fracture_def().map(Item::FractureDef),
        chronicle_def().map(Item::ChronicleDef),
        entity_def().map(Item::EntityDef),
    ))
}

// === Const/Config ===

fn const_block<'src>() -> impl Parser<'src, &'src str, ConstBlock, extra::Err<ParseError<'src>>> {
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

fn const_entry<'src>() -> impl Parser<'src, &'src str, ConstEntry, extra::Err<ParseError<'src>>> {
    spanned_path()
        .then_ignore(just(':').padded_by(ws()))
        .then(literal().map_with(|l, e| Spanned::new(l, e.span().into())))
        .then(optional_unit().padded_by(ws()))
        .map(|((path, value), unit)| ConstEntry { path, value, unit })
}

fn config_block<'src>() -> impl Parser<'src, &'src str, ConfigBlock, extra::Err<ParseError<'src>>> {
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

fn config_entry<'src>() -> impl Parser<'src, &'src str, ConfigEntry, extra::Err<ParseError<'src>>> {
    spanned_path()
        .then_ignore(just(':').padded_by(ws()))
        .then(literal().map_with(|l, e| Spanned::new(l, e.span().into())))
        .then(optional_unit().padded_by(ws()))
        .map(|((path, value), unit)| ConfigEntry { path, value, unit })
}

// === Types ===

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

fn type_field<'src>() -> impl Parser<'src, &'src str, TypeField, extra::Err<ParseError<'src>>> {
    ident()
        .map_with(|n, e| Spanned::new(n, e.span().into()))
        .then_ignore(just(':').padded_by(ws()))
        .then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
        .map(|(name, ty)| TypeField { name, ty })
}

fn type_expr<'src>() -> impl Parser<'src, &'src str, TypeExpr, extra::Err<ParseError<'src>>> {
    choice((
        text::keyword("Scalar")
            .ignore_then(
                just('<')
                    .padded_by(ws())
                    .ignore_then(unit_string())
                    .then(just(',').padded_by(ws()).ignore_then(range()).or_not())
                    .then_ignore(just('>').padded_by(ws())),
            )
            .map(|(unit, range)| TypeExpr::Scalar { unit, range }),
        choice((
            text::keyword("Vec2").to(2u8),
            text::keyword("Vec3").to(3u8),
            text::keyword("Vec4").to(4u8),
        ))
        .then(
            just('<')
                .padded_by(ws())
                .ignore_then(unit_string())
                .then_ignore(just('>').padded_by(ws())),
        )
        .map(|(dim, unit)| TypeExpr::Vector {
            dim,
            unit,
            magnitude: None,
        }),
        ident().map(TypeExpr::Named),
    ))
}

fn range<'src>() -> impl Parser<'src, &'src str, Range, extra::Err<ParseError<'src>>> {
    float()
        .then_ignore(just("..").padded_by(ws()))
        .then(float())
        .map(|(min, max)| Range { min, max })
}

// === Functions ===

/// Parse a user-defined function
/// Syntax: fn.path.to.func(param1: Type, param2: Type) -> ReturnType { body_expr }
fn fn_def<'src>() -> impl Parser<'src, &'src str, FnDef, extra::Err<ParseError<'src>>> {
    text::keyword("fn")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .then(
            fn_param()
                .padded_by(ws())
                .separated_by(just(',').padded_by(ws()))
                .allow_trailing()
                .collect()
                .delimited_by(just('(').padded_by(ws()), just(')').padded_by(ws())),
        )
        .then(
            just("->")
                .padded_by(ws())
                .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
                .or_not(),
        )
        .then(
            spanned_expr()
                .padded_by(ws())
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(((path, params), return_type), body)| FnDef {
            path,
            params,
            return_type,
            body,
        })
}

/// Parse a function parameter: name or name: Type
fn fn_param<'src>() -> impl Parser<'src, &'src str, FnParam, extra::Err<ParseError<'src>>> {
    ident()
        .map_with(|n, e| Spanned::new(n, e.span().into()))
        .then(
            just(':')
                .padded_by(ws())
                .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
                .or_not(),
        )
        .map(|(name, ty)| FnParam { name, ty })
}

// === Strata ===

fn strata_def<'src>() -> impl Parser<'src, &'src str, StrataDef, extra::Err<ParseError<'src>>> {
    text::keyword("strata")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            strata_attr()
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

fn strata_attr<'src>() -> impl Parser<'src, &'src str, StrataAttr, extra::Err<ParseError<'src>>> {
    just(':').padded_by(ws()).ignore_then(choice((
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

// === Era ===

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
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("initial"))
            .to(EraContent::Initial),
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("terminal"))
            .to(EraContent::Terminal),
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

fn strata_state<'src>() -> impl Parser<'src, &'src str, StrataState, extra::Err<ParseError<'src>>> {
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
                    .ignore_then(text::int(10).map(|s: &str| s.parse::<u32>().unwrap_or(1)))
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
                    spanned_expr()
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

// === Signal ===

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
                assertions: None,
            };
            for content in contents {
                match content {
                    SignalContent::Type(t) => def.ty = Some(t),
                    SignalContent::Strata(s) => def.strata = Some(s),
                    SignalContent::Title(t) => def.title = Some(t),
                    SignalContent::Symbol(s) => def.symbol = Some(s),
                    SignalContent::DtRaw => def.dt_raw = true,
                    SignalContent::LocalConst(c) => def.local_consts.extend(c),
                    SignalContent::LocalConfig(c) => def.local_config.extend(c),
                    SignalContent::Resolve(r) => def.resolve = Some(r),
                    SignalContent::Assert(a) => def.assertions = Some(a),
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
    LocalConst(Vec<ConstEntry>),
    LocalConfig(Vec<ConfigEntry>),
    Resolve(ResolveBlock),
    Assert(AssertBlock),
}

fn signal_content<'src>(
) -> impl Parser<'src, &'src str, SignalContent, extra::Err<ParseError<'src>>> {
    choice((
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("strata"))
            .ignore_then(spanned_path().padded_by(ws()).delimited_by(just('('), just(')')))
            .map(SignalContent::Strata),
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
        just(':')
            .padded_by(ws())
            .ignore_then(just("dt_raw"))
            .to(SignalContent::DtRaw),
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("uses"))
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(just("dt_raw"))
                    .then_ignore(just(')').padded_by(ws())),
            )
            .to(SignalContent::DtRaw),
        just(':')
            .padded_by(ws())
            .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
            .map(SignalContent::Type),
        text::keyword("const")
            .padded_by(ws())
            .ignore_then(
                const_entry()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(SignalContent::LocalConst),
        text::keyword("config")
            .padded_by(ws())
            .ignore_then(
                config_entry()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(SignalContent::LocalConfig),
        text::keyword("resolve")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| SignalContent::Resolve(ResolveBlock { body })),
        assert_block().map(SignalContent::Assert),
    ))
}

// === Field ===

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

fn field_content<'src>() -> impl Parser<'src, &'src str, FieldContent, extra::Err<ParseError<'src>>>
{
    choice((
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("strata"))
            .ignore_then(spanned_path().padded_by(ws()).delimited_by(just('('), just(')')))
            .map(FieldContent::Strata),
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
        just(':')
            .padded_by(ws())
            .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
            .map(FieldContent::Type),
        text::keyword("measure")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| FieldContent::Measure(MeasureBlock { body })),
    ))
}

fn topology<'src>() -> impl Parser<'src, &'src str, Topology, extra::Err<ParseError<'src>>> {
    choice((
        text::keyword("sphere_surface").to(Topology::SphereSurface),
        text::keyword("point_cloud").to(Topology::PointCloud),
        text::keyword("volume").to(Topology::Volume),
    ))
}

// === Operator ===

fn operator_def<'src>() -> impl Parser<'src, &'src str, OperatorDef, extra::Err<ParseError<'src>>> {
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
                assertions: None,
            };
            for content in contents {
                match content {
                    OperatorContent::Strata(s) => def.strata = Some(s),
                    OperatorContent::Phase(p) => def.phase = Some(p),
                    OperatorContent::Body(b) => def.body = Some(b),
                    OperatorContent::Assert(a) => def.assertions = Some(a),
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
    Assert(AssertBlock),
}

fn operator_content<'src>(
) -> impl Parser<'src, &'src str, OperatorContent, extra::Err<ParseError<'src>>> {
    choice((
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("strata"))
            .ignore_then(spanned_path().padded_by(ws()).delimited_by(just('('), just(')')))
            .map(OperatorContent::Strata),
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("phase"))
            .ignore_then(
                choice((
                    text::keyword("warmup").to(OperatorPhase::Warmup),
                    text::keyword("collect").to(OperatorPhase::Collect),
                    text::keyword("measure").to(OperatorPhase::Measure),
                ))
                .map_with(|p, e| {
                    let span: chumsky::span::SimpleSpan = e.span();
                    Spanned::new(p, span.start..span.end)
                })
                .padded_by(ws())
                .delimited_by(just('('), just(')')),
            )
            .map(OperatorContent::Phase),
        text::keyword("collect")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|e| OperatorContent::Body(OperatorBody::Collect(e))),
        text::keyword("measure")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|e| OperatorContent::Body(OperatorBody::Measure(e))),
        assert_block().map(OperatorContent::Assert),
    ))
}

// === Impulse ===

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
        just(':')
            .padded_by(ws())
            .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
            .map(ImpulseContent::Type),
        text::keyword("apply")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| ImpulseContent::Apply(ApplyBlock { body })),
    ))
}

// === Fracture ===

fn fracture_def<'src>() -> impl Parser<'src, &'src str, FractureDef, extra::Err<ParseError<'src>>> {
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
        text::keyword("when")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(FractureContent::When),
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
        .then(spanned_expr())
        .map(|(target, value)| EmitStatement { target, value })
}

// === Chronicle ===

fn chronicle_def<'src>() -> impl Parser<'src, &'src str, ChronicleDef, extra::Err<ParseError<'src>>>
{
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

fn observe_block<'src>() -> impl Parser<'src, &'src str, ObserveBlock, extra::Err<ParseError<'src>>>
{
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
        .ignore_then(spanned_expr())
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
) -> impl Parser<'src, &'src str, (Spanned<String>, Spanned<Expr>), extra::Err<ParseError<'src>>> {
    ident()
        .map_with(|n, e| Spanned::new(n, e.span().into()))
        .then_ignore(just(':').padded_by(ws()))
        .then(spanned_expr())
}

// === Assertions ===

fn assert_block<'src>() -> impl Parser<'src, &'src str, AssertBlock, extra::Err<ParseError<'src>>> {
    text::keyword("assert")
        .padded_by(ws())
        .ignore_then(
            assertion()
                .padded_by(ws())
                .repeated()
                .at_least(1)
                .collect()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|assertions| AssertBlock { assertions })
}

fn assertion<'src>() -> impl Parser<'src, &'src str, Assertion, extra::Err<ParseError<'src>>> {
    // Parse: condition [: severity] [, "message"]
    spanned_expr()
        .then(
            just(':')
                .padded_by(ws())
                .ignore_then(assert_severity())
                .or_not(),
        )
        .then(
            just(',')
                .padded_by(ws())
                .ignore_then(string_lit().map_with(|s, e| Spanned::new(s, e.span().into())))
                .or_not(),
        )
        .map(|((condition, severity), message)| Assertion {
            condition,
            severity: severity.unwrap_or_default(),
            message,
        })
}

fn assert_severity<'src>() -> impl Parser<'src, &'src str, AssertSeverity, extra::Err<ParseError<'src>>>
{
    choice((
        text::keyword("warn").to(AssertSeverity::Warn),
        text::keyword("error").to(AssertSeverity::Error),
        text::keyword("fatal").to(AssertSeverity::Fatal),
    ))
}

// === Entity ===

fn entity_def<'src>() -> impl Parser<'src, &'src str, EntityDef, extra::Err<ParseError<'src>>> {
    text::keyword("entity")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(spanned_path())
        .padded_by(ws())
        .then(
            entity_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(path, contents)| {
            let mut def = EntityDef {
                path,
                strata: None,
                count_source: None,
                count_bounds: None,
                schema: vec![],
                config_defaults: vec![],
                resolve: None,
                assertions: None,
                fields: vec![],
            };
            for content in contents {
                match content {
                    EntityContent::Strata(s) => def.strata = Some(s),
                    EntityContent::CountSource(c) => def.count_source = Some(c),
                    EntityContent::CountBounds(b) => def.count_bounds = Some(b),
                    EntityContent::Schema(s) => def.schema = s,
                    EntityContent::Config(c) => def.config_defaults = c,
                    EntityContent::Resolve(r) => def.resolve = Some(r),
                    EntityContent::Assert(a) => def.assertions = Some(a),
                    EntityContent::Field(f) => def.fields.push(f),
                }
            }
            def
        })
}

#[derive(Clone)]
enum EntityContent {
    Strata(Spanned<Path>),
    CountSource(Spanned<Path>),
    CountBounds(CountBounds),
    Schema(Vec<EntitySchemaField>),
    Config(Vec<ConfigEntry>),
    Resolve(ResolveBlock),
    Assert(AssertBlock),
    Field(EntityFieldDef),
}

fn entity_content<'src>(
) -> impl Parser<'src, &'src str, EntityContent, extra::Err<ParseError<'src>>> {
    choice((
        // : strata(path)
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("strata"))
            .ignore_then(spanned_path().padded_by(ws()).delimited_by(just('('), just(')')))
            .map(EntityContent::Strata),
        // : count(config.path) - count from config
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("count"))
            .ignore_then(
                just('(')
                    .padded_by(ws())
                    .ignore_then(text::keyword("config"))
                    .ignore_then(just('.'))
                    .ignore_then(spanned_path())
                    .then_ignore(just(')').padded_by(ws())),
            )
            .map(EntityContent::CountSource),
        // : count(min..max) - count bounds
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("count"))
            .ignore_then(
                count_bounds()
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(EntityContent::CountBounds),
        // schema { fields }
        text::keyword("schema")
            .padded_by(ws())
            .ignore_then(
                entity_schema_field()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(EntityContent::Schema),
        // config { defaults }
        text::keyword("config")
            .padded_by(ws())
            .ignore_then(
                config_entry()
                    .padded_by(ws())
                    .repeated()
                    .collect()
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(EntityContent::Config),
        // resolve { expr }
        text::keyword("resolve")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| EntityContent::Resolve(ResolveBlock { body })),
        // assert { assertions }
        assert_block().map(EntityContent::Assert),
        // field.name { ... } - nested field definition
        entity_field_def().map(EntityContent::Field),
    ))
}

fn count_bounds<'src>() -> impl Parser<'src, &'src str, CountBounds, extra::Err<ParseError<'src>>> {
    text::int(10)
        .map(|s: &str| s.parse::<u32>().unwrap_or(0))
        .then_ignore(just("..").padded_by(ws()))
        .then(text::int(10).map(|s: &str| s.parse::<u32>().unwrap_or(u32::MAX)))
        .map(|(min, max)| CountBounds { min, max })
}

fn entity_schema_field<'src>(
) -> impl Parser<'src, &'src str, EntitySchemaField, extra::Err<ParseError<'src>>> {
    ident()
        .map_with(|n, e| Spanned::new(n, e.span().into()))
        .then_ignore(just(':').padded_by(ws()))
        .then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
        .map(|(name, ty)| EntitySchemaField { name, ty })
}

fn entity_field_def<'src>(
) -> impl Parser<'src, &'src str, EntityFieldDef, extra::Err<ParseError<'src>>> {
    text::keyword("field")
        .padded_by(ws())
        .ignore_then(just('.'))
        .ignore_then(ident().map_with(|n, e| Spanned::new(n, e.span().into())))
        .padded_by(ws())
        .then(
            entity_field_content()
                .padded_by(ws())
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
        )
        .map(|(name, contents)| {
            let mut def = EntityFieldDef {
                name,
                ty: None,
                topology: None,
                measure: None,
            };
            for content in contents {
                match content {
                    EntityFieldContent::Type(t) => def.ty = Some(t),
                    EntityFieldContent::Topology(t) => def.topology = Some(t),
                    EntityFieldContent::Measure(m) => def.measure = Some(m),
                }
            }
            def
        })
}

#[derive(Clone)]
enum EntityFieldContent {
    Type(Spanned<TypeExpr>),
    Topology(Spanned<Topology>),
    Measure(MeasureBlock),
}

fn entity_field_content<'src>(
) -> impl Parser<'src, &'src str, EntityFieldContent, extra::Err<ParseError<'src>>> {
    choice((
        just(':')
            .padded_by(ws())
            .ignore_then(text::keyword("topology"))
            .ignore_then(
                topology()
                    .map_with(|t, e| Spanned::new(t, e.span().into()))
                    .padded_by(ws())
                    .delimited_by(just('('), just(')')),
            )
            .map(EntityFieldContent::Topology),
        just(':')
            .padded_by(ws())
            .ignore_then(type_expr().map_with(|t, e| Spanned::new(t, e.span().into())))
            .map(EntityFieldContent::Type),
        text::keyword("measure")
            .padded_by(ws())
            .ignore_then(
                spanned_expr()
                    .padded_by(ws())
                    .delimited_by(just('{').padded_by(ws()), just('}').padded_by(ws())),
            )
            .map(|body| EntityFieldContent::Measure(MeasureBlock { body })),
    ))
}
