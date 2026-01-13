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
    #[token("world", priority = 2)]
    World,
    #[token("strata", priority = 2)]
    Strata,
    #[token("era", priority = 2)]
    Era,
    #[token("signal", priority = 2)]
    Signal,
    #[token("field", priority = 2)]
    Field,
    #[token("operator", priority = 2)]
    Operator,
    #[token("fn", priority = 2)]
    Fn,
    #[token("type", priority = 2)]
    Type,
    #[token("impulse", priority = 2)]
    Impulse,
    #[token("fracture", priority = 2)]
    Fracture,
    #[token("chronicle", priority = 2)]
    Chronicle,
    #[token("entity", priority = 2)]
    Entity,
    #[token("count", priority = 2)]
    Count,
    #[token("member", priority = 2)]
    Member,
    #[token("const", priority = 2)]
    Const,
    #[token("config", priority = 2)]
    Config,
    #[token("policy", priority = 2)]
    Policy,
    #[token("version", priority = 2)]
    Version,

    #[token("resolve", priority = 2)]
    Resolve,
    #[token("measure", priority = 2)]
    Measure,
    #[token("when", priority = 2)]
    When,
    #[token("emit", priority = 2)]
    Emit,
    #[token("assert", priority = 2)]
    Assert,
    #[token("transition", priority = 2)]
    Transition,
    #[token("dt", priority = 2)]
    Dt,
    #[token("to", priority = 2)]
    To,
    #[token("gated", priority = 2)]
    Gated,

    #[token("apply", priority = 2)]
    Apply,
    #[token("observe", priority = 2)]
    Observe,
    #[token("event", priority = 2)]
    Event,

    #[token("warn", priority = 2)]
    Warn,
    #[token("error", priority = 2)]
    Error,
    #[token("fatal", priority = 2)]
    Fatal,
    #[token("sphere_surface", priority = 2)]
    SphereSurface,
    #[token("point_cloud", priority = 2)]
    PointCloud,
    #[token("volume", priority = 2)]
    Volume,

    #[token("if", priority = 2)]
    If,
    #[token("else", priority = 2)]
    Else,
    #[token("let", priority = 2)]
    Let,
    #[token("in", priority = 2)]
    In,
    #[token("for", priority = 2)]
    For,

    #[token("prev", priority = 2)]
    Prev,
    #[token("dt_raw", priority = 2)]
    DtRaw,
    #[token("sim_time", priority = 2)]
    SimTime,
    #[token("collected", priority = 2)]
    Collected,
    #[token("payload", priority = 2)]
    Payload,

    // --- Attribute Keywords (often used after :) ---
    #[token("initial", priority = 2)]
    Initial,
    #[token("terminal", priority = 2)]
    Terminal,
    #[token("stride", priority = 2)]
    Stride,
    #[token("title", priority = 2)]
    Title,
    #[token("symbol", priority = 2)]
    Symbol,
    #[token("uses", priority = 2)]
    Uses,
    #[token("active", priority = 2)]
    Active,
    #[token("converge", priority = 2)]
    Converge,
    #[token("warmup", priority = 2)]
    Warmup,
    #[token("iterate", priority = 2)]
    Iterate,

    // --- Built-in Types ---
    #[token("symmetric", priority = 2)]
    Symmetric,
    #[token("positive_definite", priority = 2)]
    PositiveDefinite,
    #[token("each", priority = 2)]
    Each,
    #[token("sum", priority = 2)]
    Sum,
    #[token("product", priority = 2)]
    Product,
    #[token("min", priority = 2)]
    Min,
    #[token("max", priority = 2)]
    Max,
    #[token("mean", priority = 2)]
    Mean,
    #[token("any", priority = 2)]
    Any,
    #[token("all", priority = 2)]
    All,
    #[token("none", priority = 2)]
    None,
    #[token("phase", priority = 2)]
    Phase,
    #[token("collect", priority = 2)]
    Collect,
    #[token("topology", priority = 2)]
    Topology,
    #[token("self", priority = 2)]
    SelfToken,
    #[token("filter", priority = 2)]
    Filter,
    #[token("first", priority = 2)]
    First,
    #[token("nearest", priority = 2)]
    Nearest,
    #[token("within", priority = 2)]
    Within,
    #[token("other", priority = 2)]
    Other,
    #[token("pairs", priority = 2)]
    Pairs,
    #[token("not", priority = 2)]
    NotKeyword,
    #[token("and", priority = 2)]
    AndKeyword,
    #[token("or", priority = 2)]
    OrKeyword,

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
    #[token(";")]
    Semicolon,
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
    #[token("true", |_| true, priority = 2)]
    #[token("false", |_| false, priority = 2)]
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
