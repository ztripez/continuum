//! Logos-based lexer for Continuum DSL.

use logos::Logos;
use std::fmt;

#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\n\f]+")]
#[logos(skip r"//[^/!][^\n]*")]
#[logos(skip r"#[^\n]*")]
#[logos(skip r"/\*([^*]|\*[^/])*\*/")]
pub enum Token {
    // --- Keywords ---
    #[token("world")]
    World,
    #[token("strata")]
    Strata,
    #[token("era")]
    Era,
    #[token("signal")]
    Signal,
    #[token("field")]
    Field,
    #[token("operator")]
    Operator,
    #[token("fn")]
    Fn,
    #[token("type")]
    Type,
    #[token("impulse")]
    Impulse,
    #[token("fracture")]
    Fracture,
    #[token("chronicle")]
    Chronicle,
    #[token("entity")]
    Entity,
    #[token("count")]
    Count,
    #[token("member")]
    Member,
    #[token("const")]
    Const,
    #[token("config")]
    Config,
    #[token("policy")]
    Policy,
    #[token("version")]
    Version,

    #[token("resolve")]
    Resolve,
    #[token("measure")]
    Measure,
    #[token("when")]
    When,
    #[token("emit")]
    Emit,
    #[token("assert")]
    Assert,
    #[token("transition")]
    Transition,
    #[token("dt")]
    Dt,
    #[token("to")]
    To,
    #[token("gated")]
    Gated,

    #[token("apply")]
    Apply,
    #[token("observe")]
    Observe,
    #[token("event")]
    Event,

    #[token("warn")]
    Warn,
    #[token("error")]
    Error,
    #[token("fatal")]
    Fatal,
    #[token("sphere_surface")]
    SphereSurface,
    #[token("point_cloud")]
    PointCloud,
    #[token("volume")]
    Volume,

    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("let")]
    Let,
    #[token("in")]
    In,
    #[token("for")]
    For,

    #[token("prev")]
    Prev,
    #[token("dt_raw")]
    DtRaw,
    #[token("collected")]
    Collected,
    #[token("payload")]
    Payload,

    // --- Attribute Keywords (often used after :) ---
    #[token("initial")]
    Initial,
    #[token("terminal")]
    Terminal,
    #[token("stride")]
    Stride,
    #[token("title")]
    Title,
    #[token("symbol")]
    Symbol,
    #[token("uses")]
    Uses,
    #[token("active")]
    Active,
    #[token("converge")]
    Converge,
    #[token("warmup")]
    Warmup,
    #[token("iterate")]
    Iterate,

    // --- Built-in Types ---
    #[token("Scalar")]
    Scalar,
    #[token("Vec2")]
    Vec2,
    #[token("Vec3")]
    Vec3,
    #[token("Vec4")]
    Vec4,
    #[token("Vector")]
    Vector,
    #[token("Tensor")]
    Tensor,
    #[token("Grid")]
    Grid,
    #[token("Seq")]
    Seq,
    #[token("magnitude")]
    Magnitude,
    #[token("symmetric")]
    Symmetric,
    #[token("positive_definite")]
    PositiveDefinite,
    #[token("each")]
    Each,
    #[token("sum")]
    Sum,
    #[token("product")]
    Product,
    #[token("min")]
    Min,
    #[token("max")]
    Max,
    #[token("mean")]
    Mean,
    #[token("any")]
    Any,
    #[token("all")]
    All,
    #[token("none")]
    None,
    #[token("phase")]
    Phase,
    #[token("collect")]
    Collect,
    #[token("topology")]
    Topology,
    #[token("self")]
    SelfToken,
    #[token("filter")]
    Filter,
    #[token("first")]
    First,
    #[token("nearest")]
    Nearest,
    #[token("within")]
    Within,
    #[token("other")]
    Other,
    #[token("pairs")]
    Pairs,
    #[token("not")]
    NotKeyword,
    #[token("and")]
    AndKeyword,
    #[token("or")]
    OrKeyword,

    // --- Math Constants ---
    #[token("PI")]
    #[token("π")]
    Pi,
    #[token("TAU")]
    #[token("τ")]
    Tau,
    #[token("PHI")]
    #[token("φ")]
    Phi,
    #[token("E")]
    #[token("ℯ")]
    E,
    #[token("I")]
    #[token("ⅈ")]
    I,

    // --- Operators & Punctuation ---
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("=")]
    Assign,
    #[token("==")]
    Equals,
    #[token("!=")]
    NotEquals,
    #[token("<")]
    LAngle,
    #[token(">")]
    RAngle,
    #[token("<=")]
    LessEquals,
    #[token(">=")]
    GreaterEquals,
    #[token("&&")]
    And,
    #[token("||")]
    Or,
    #[token("!")]
    Not,
    #[token("<-")]
    EmitArrow,
    #[token("->")]
    Arrow,
    #[token("..")]
    DotDot,
    #[token(".")]
    Dot,
    #[token(",")]
    Comma,
    #[token(":")]
    Colon,
    #[token("::")]
    DoubleColon,

    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,

    // --- Literals ---
    #[token("true", |_| true)]
    #[token("false", |_| false)]
    Bool(bool),

    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string(), priority = 1)]
    Ident(String),

    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        s[1..s.len()-1].to_string()
    })]
    String(String),

    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok(), priority = 1)]
    Integer(i64),

    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", |lex| lex.slice().parse::<f64>().ok(), priority = 1)]
    #[regex(r"\.[0-9]+([eE][+-]?[0-9]+)?", |lex| lex.slice().parse::<f64>().ok(), priority = 1)]
    #[regex(r"[0-9]+[eE][+-]?[0-9]+", |lex| lex.slice().parse::<f64>().ok(), priority = 1)]
    Float(f64),

    // --- Trivia ---
    #[regex(r"///[^\n]*", |lex| lex.slice()[3..].trim().to_string(), priority = 3, allow_greedy = true)]
    DocComment(String),

    #[regex(r"//![^\n]*", |lex| lex.slice()[3..].trim().to_string(), priority = 3, allow_greedy = true)]
    ModuleDoc(String),

    // Special parts for units that don't match standard Ident/Float (superscripts etc)
    #[regex(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺₀₁₂₃₄₅₆₇₈₉°·]+", |lex| lex.slice().to_string())]
    UnitPart(String),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Ident(s) => write!(f, "{}", s),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Integer(i) => write!(f, "{}", i),
            Token::Float(fl) => write!(f, "{}", fl),
            Token::Bool(b) => write!(f, "{}", b),
            Token::Plus => write!(f, "+"),
            Token::Minus => write!(f, "-"),
            Token::Star => write!(f, "*"),
            Token::Slash => write!(f, "/"),
            Token::Assign => write!(f, "="),
            Token::Equals => write!(f, "=="),
            Token::NotEquals => write!(f, "!="),
            Token::LAngle => write!(f, "<"),
            Token::RAngle => write!(f, ">"),
            Token::LessEquals => write!(f, "<="),
            Token::GreaterEquals => write!(f, ">="),
            Token::And => write!(f, "&&"),
            Token::Or => write!(f, "||"),
            Token::Not => write!(f, "!"),
            Token::EmitArrow => write!(f, "<-"),
            Token::Arrow => write!(f, "->"),
            Token::DotDot => write!(f, ".."),
            Token::Dot => write!(f, "."),
            Token::Comma => write!(f, ","),
            Token::Colon => write!(f, ":"),
            Token::DoubleColon => write!(f, "::"),
            Token::LParen => write!(f, "("),
            Token::RParen => write!(f, ")"),
            Token::LBracket => write!(f, "["),
            Token::RBracket => write!(f, "]"),
            Token::LBrace => write!(f, "{{"),
            Token::RBrace => write!(f, "}}"),
            Token::UnitPart(s) => write!(f, "{}", s),
            _ => write!(f, "{:?}", self),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use logos::Logos;

    #[test]
    fn test_lex_keywords() {
        let mut lex = Token::lexer(
            "world strata era signal field operator fn type impulse fracture chronicle entity member const config",
        );
        assert_eq!(lex.next(), Some(Ok(Token::World)));
        assert_eq!(lex.next(), Some(Ok(Token::Strata)));
        assert_eq!(lex.next(), Some(Ok(Token::Era)));
        assert_eq!(lex.next(), Some(Ok(Token::Signal)));
        assert_eq!(lex.next(), Some(Ok(Token::Field)));
        assert_eq!(lex.next(), Some(Ok(Token::Operator)));
        assert_eq!(lex.next(), Some(Ok(Token::Fn)));
        assert_eq!(lex.next(), Some(Ok(Token::Type)));
        assert_eq!(lex.next(), Some(Ok(Token::Impulse)));
        assert_eq!(lex.next(), Some(Ok(Token::Fracture)));
        assert_eq!(lex.next(), Some(Ok(Token::Chronicle)));
        assert_eq!(lex.next(), Some(Ok(Token::Entity)));
        assert_eq!(lex.next(), Some(Ok(Token::Member)));
        assert_eq!(lex.next(), Some(Ok(Token::Const)));
        assert_eq!(lex.next(), Some(Ok(Token::Config)));
    }

    #[test]
    fn test_lex_literals() {
        let mut lex = Token::lexer("true false 123 45.6 \"hello world\"");
        assert_eq!(lex.next(), Some(Ok(Token::Bool(true))));
        assert_eq!(lex.next(), Some(Ok(Token::Bool(false))));
        assert_eq!(lex.next(), Some(Ok(Token::Integer(123))));
        assert_eq!(lex.next(), Some(Ok(Token::Float(45.6))));
        assert_eq!(
            lex.next(),
            Some(Ok(Token::String("hello world".to_string())))
        );
    }

    #[test]
    fn test_lex_operators() {
        let mut lex =
            Token::lexer("+ - * / = == != < > <= >= && || ! <- -> .. . , : :: ( ) [ ] { }");
        assert_eq!(lex.next(), Some(Ok(Token::Plus)));
        assert_eq!(lex.next(), Some(Ok(Token::Minus)));
        assert_eq!(lex.next(), Some(Ok(Token::Star)));
        assert_eq!(lex.next(), Some(Ok(Token::Slash)));
        assert_eq!(lex.next(), Some(Ok(Token::Assign)));
        assert_eq!(lex.next(), Some(Ok(Token::Equals)));
        assert_eq!(lex.next(), Some(Ok(Token::NotEquals)));
        assert_eq!(lex.next(), Some(Ok(Token::LAngle)));
        assert_eq!(lex.next(), Some(Ok(Token::RAngle)));
        assert_eq!(lex.next(), Some(Ok(Token::LessEquals)));
        assert_eq!(lex.next(), Some(Ok(Token::GreaterEquals)));
        assert_eq!(lex.next(), Some(Ok(Token::And)));
        assert_eq!(lex.next(), Some(Ok(Token::Or)));
        assert_eq!(lex.next(), Some(Ok(Token::Not)));
        assert_eq!(lex.next(), Some(Ok(Token::EmitArrow)));
        assert_eq!(lex.next(), Some(Ok(Token::Arrow)));
        assert_eq!(lex.next(), Some(Ok(Token::DotDot)));
        assert_eq!(lex.next(), Some(Ok(Token::Dot)));
        assert_eq!(lex.next(), Some(Ok(Token::Comma)));
        assert_eq!(lex.next(), Some(Ok(Token::Colon)));
        assert_eq!(lex.next(), Some(Ok(Token::DoubleColon)));
        assert_eq!(lex.next(), Some(Ok(Token::LParen)));
        assert_eq!(lex.next(), Some(Ok(Token::RParen)));
        assert_eq!(lex.next(), Some(Ok(Token::LBracket)));
        assert_eq!(lex.next(), Some(Ok(Token::RBracket)));
        assert_eq!(lex.next(), Some(Ok(Token::LBrace)));
        assert_eq!(lex.next(), Some(Ok(Token::RBrace)));
    }

    #[test]
    fn test_lex_comments() {
        let mut lex = Token::lexer(
            "// regular comment\n# hash comment\n/// doc comment\n//! module doc\n/* block comment */",
        );
        // Regular comments are skipped now
        assert_eq!(
            lex.next(),
            Some(Ok(Token::DocComment("doc comment".to_string())))
        );
        assert_eq!(
            lex.next(),
            Some(Ok(Token::ModuleDoc("module doc".to_string())))
        );
    }
}
