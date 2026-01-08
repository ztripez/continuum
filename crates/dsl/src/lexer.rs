//! Lexer for Continuum DSL (.cdsl files)
//!
//! Uses Logos for fast, compile-time optimized tokenization.

use logos::{Logos, Span};

/// Token type for the Continuum DSL
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n\f]+")]
pub enum Token<'src> {
    // === Comments ===
    #[regex(r"//[^\n]*", logos::skip, allow_greedy = true)]
    #[regex(r"#[^\n]*", logos::skip, allow_greedy = true)]
    #[regex(r"/\*([^*]|\*[^/])*\*/", logos::skip)]
    Comment,

    // === Keywords ===
    #[token("const")]
    Const,
    #[token("config")]
    Config,
    #[token("type")]
    Type,
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
    #[token("impulse")]
    Impulse,
    #[token("fracture")]
    Fracture,
    #[token("chronicle")]
    Chronicle,

    // === Block keywords ===
    #[token("resolve")]
    Resolve,
    #[token("warmup")]
    Warmup,
    #[token("iterate")]
    Iterate,
    #[token("measure")]
    Measure,
    #[token("collect")]
    Collect,
    #[token("apply")]
    Apply,
    #[token("when")]
    When,
    #[token("emit")]
    Emit,
    #[token("observe")]
    Observe,
    #[token("transition")]
    Transition,

    // === Control flow ===
    #[token("let")]
    Let,
    #[token("if")]
    If,
    #[token("else")]
    Else,
    #[token("for")]
    For,
    #[token("in")]
    In,

    // === Logical operators ===
    #[token("and")]
    And,
    #[token("or")]
    Or,
    #[token("not")]
    Not,

    // === Built-in references ===
    #[token("prev")]
    Prev,
    #[token("dt")]
    Dt,
    #[token("payload")]
    Payload,
    #[token("sum")]
    Sum,
    #[token("map")]
    Map,
    #[token("fold")]
    Fold,

    // === Era attributes ===
    #[token("initial")]
    Initial,
    #[token("terminal")]
    Terminal,
    #[token("to")]
    To,
    #[token("active")]
    Active,
    #[token("gated")]
    Gated,

    // === Type keywords ===
    #[token("Scalar")]
    ScalarType,
    #[token("Vec2")]
    Vec2Type,
    #[token("Vec3")]
    Vec3Type,
    #[token("Vec4")]
    Vec4Type,
    #[token("Tensor")]
    TensorType,
    #[token("Seq")]
    SeqType,
    #[token("Grid")]
    GridType,

    // === Constraint keywords ===
    #[token("magnitude")]
    Magnitude,
    #[token("symmetric")]
    Symmetric,
    #[token("positive_definite")]
    PositiveDefinite,
    #[token("each")]
    Each,

    // === Topology keywords ===
    #[token("topology")]
    Topology,
    #[token("sphere_surface")]
    SphereSurface,
    #[token("point_cloud")]
    PointCloud,
    #[token("volume")]
    Volume,

    // === Warmup keywords ===
    #[token("iterations")]
    Iterations,
    #[token("convergence")]
    Convergence,

    // === Other keywords ===
    #[token("title")]
    Title,
    #[token("symbol")]
    Symbol,
    #[token("stride")]
    Stride,
    #[token("phase")]
    Phase,
    #[token("kernel")]
    Kernel,
    #[token("event")]
    Event,

    // === Literals ===
    /// Integer literal (may have sign)
    #[regex(r"-?[0-9]+", |lex| lex.slice())]
    Integer(&'src str),

    /// Float literal (scientific notation supported)
    #[regex(r"-?[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", |lex| lex.slice())]
    #[regex(r"-?[0-9]+[eE][+-]?[0-9]+", |lex| lex.slice())]
    Float(&'src str),

    /// String literal
    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        &s[1..s.len()-1]
    })]
    String(&'src str),

    // === Identifiers ===
    /// Simple identifier (lowercase, may contain underscores)
    #[regex(r"[a-z_][a-z0-9_]*", |lex| lex.slice())]
    Ident(&'src str),

    // === Punctuation ===
    #[token("{")]
    BraceOpen,
    #[token("}")]
    BraceClose,
    #[token("(")]
    ParenOpen,
    #[token(")")]
    ParenClose,
    #[token("[")]
    BracketOpen,
    #[token("]")]
    BracketClose,
    #[token("<")]
    AngleOpen,
    #[token(">")]
    AngleClose,

    #[token(":")]
    Colon,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("..")]
    DotDot,

    // === Operators ===
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("^")]
    Caret,
    #[token("=")]
    Equals,
    #[token("<-")]
    Arrow,

    // === Comparison ===
    #[token(">=")]
    GreaterEq,
    #[token("<=")]
    LessEq,
    #[token("==")]
    DoubleEq,
    #[token("!=")]
    NotEq,

    // === Unit content ===
    /// Unit content (letters, digits, slashes, Unicode superscripts, etc.)
    /// This matches content inside angle brackets for units like K, W/m²/K⁴, m/s
    #[regex(r"[A-Za-z][A-Za-z0-9/²³⁴⁵⁶⁷⁸⁹⁰]*", |lex| lex.slice(), priority = 1)]
    UnitIdent(&'src str),
}

/// A token with its source span
#[derive(Debug, Clone, PartialEq)]
pub struct Spanned<T> {
    pub token: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    pub fn new(token: T, span: Span) -> Self {
        Self { token, span }
    }
}

/// Tokenize source code into a vector of spanned tokens
pub fn lex(source: &str) -> Result<Vec<Spanned<Token<'_>>>, LexError> {
    let mut lexer = Token::lexer(source);
    let mut tokens = Vec::new();

    while let Some(result) = lexer.next() {
        match result {
            Ok(token) => {
                // Skip the Comment variant (it's handled by logos::skip but
                // the variant still exists)
                if !matches!(token, Token::Comment) {
                    tokens.push(Spanned::new(token, lexer.span()));
                }
            }
            Err(()) => {
                return Err(LexError {
                    span: lexer.span(),
                    slice: lexer.slice().to_string(),
                });
            }
        }
    }

    Ok(tokens)
}

/// Error during lexing
#[derive(Debug, Clone)]
pub struct LexError {
    pub span: Span,
    pub slice: String,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "unexpected character(s) '{}' at {:?}",
            self.slice, self.span
        )
    }
}

impl std::error::Error for LexError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keywords() {
        let tokens = lex("const config signal field").unwrap();
        assert_eq!(tokens.len(), 4);
        assert_eq!(tokens[0].token, Token::Const);
        assert_eq!(tokens[1].token, Token::Config);
        assert_eq!(tokens[2].token, Token::Signal);
        assert_eq!(tokens[3].token, Token::Field);
    }

    #[test]
    fn test_numbers() {
        let tokens = lex("42 -17 3.14 1e10 5.67e-8").unwrap();
        assert_eq!(tokens.len(), 5);
        assert_eq!(tokens[0].token, Token::Integer("42"));
        assert_eq!(tokens[1].token, Token::Integer("-17"));
        assert_eq!(tokens[2].token, Token::Float("3.14"));
        assert_eq!(tokens[3].token, Token::Float("1e10"));
        assert_eq!(tokens[4].token, Token::Float("5.67e-8"));
    }

    #[test]
    fn test_string() {
        let tokens = lex(r#""hello world""#).unwrap();
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].token, Token::String("hello world"));
    }

    #[test]
    fn test_identifiers() {
        let tokens = lex("terra thermal core_temp").unwrap();
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0].token, Token::Ident("terra"));
        assert_eq!(tokens[1].token, Token::Ident("thermal"));
        assert_eq!(tokens[2].token, Token::Ident("core_temp"));
    }

    #[test]
    fn test_punctuation() {
        let tokens = lex("{ } ( ) [ ] : , . ..").unwrap();
        assert_eq!(tokens.len(), 10);
        assert_eq!(tokens[0].token, Token::BraceOpen);
        assert_eq!(tokens[1].token, Token::BraceClose);
        assert_eq!(tokens[2].token, Token::ParenOpen);
        assert_eq!(tokens[3].token, Token::ParenClose);
        assert_eq!(tokens[4].token, Token::BracketOpen);
        assert_eq!(tokens[5].token, Token::BracketClose);
        assert_eq!(tokens[6].token, Token::Colon);
        assert_eq!(tokens[7].token, Token::Comma);
        assert_eq!(tokens[8].token, Token::Dot);
        assert_eq!(tokens[9].token, Token::DotDot);
    }

    #[test]
    fn test_operators() {
        let tokens = lex("+ - * / ^ = <- >= <= == !=").unwrap();
        assert_eq!(tokens.len(), 11);
        assert_eq!(tokens[0].token, Token::Plus);
        assert_eq!(tokens[1].token, Token::Minus);
        assert_eq!(tokens[2].token, Token::Star);
        assert_eq!(tokens[3].token, Token::Slash);
        assert_eq!(tokens[4].token, Token::Caret);
        assert_eq!(tokens[5].token, Token::Equals);
        assert_eq!(tokens[6].token, Token::Arrow);
        assert_eq!(tokens[7].token, Token::GreaterEq);
        assert_eq!(tokens[8].token, Token::LessEq);
        assert_eq!(tokens[9].token, Token::DoubleEq);
        assert_eq!(tokens[10].token, Token::NotEq);
    }

    #[test]
    fn test_units() {
        // Units are now parsed as AngleOpen + UnitIdent + AngleClose
        let tokens = lex("<K> <W/m²/K⁴> <m/s>").unwrap();
        assert_eq!(tokens.len(), 9);
        assert_eq!(tokens[0].token, Token::AngleOpen);
        assert_eq!(tokens[1].token, Token::UnitIdent("K"));
        assert_eq!(tokens[2].token, Token::AngleClose);
        assert_eq!(tokens[3].token, Token::AngleOpen);
        assert_eq!(tokens[4].token, Token::UnitIdent("W/m²/K⁴"));
        assert_eq!(tokens[5].token, Token::AngleClose);
        assert_eq!(tokens[6].token, Token::AngleOpen);
        assert_eq!(tokens[7].token, Token::UnitIdent("m/s"));
        assert_eq!(tokens[8].token, Token::AngleClose);
    }

    #[test]
    fn test_comments() {
        let tokens = lex("signal // this is a comment\nfield").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].token, Token::Signal);
        assert_eq!(tokens[1].token, Token::Field);
    }

    #[test]
    fn test_hash_comments() {
        let tokens = lex("signal # this is a comment\nfield").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].token, Token::Signal);
        assert_eq!(tokens[1].token, Token::Field);
    }

    #[test]
    fn test_multiline_comments() {
        let tokens = lex("signal /* multi\nline\ncomment */ field").unwrap();
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].token, Token::Signal);
        assert_eq!(tokens[1].token, Token::Field);
    }

    #[test]
    fn test_signal_declaration() {
        let source = r#"
            signal.terra.core.temp {
                : Scalar<K, 100..10000>
                : strata(terra.thermal)
            }
        "#;
        let tokens = lex(source).unwrap();

        // signal . terra . core . temp { : Scalar < K , 100 .. 10000 > ...
        assert!(tokens.len() > 10);
        assert_eq!(tokens[0].token, Token::Signal);
        assert_eq!(tokens[1].token, Token::Dot);
        assert_eq!(tokens[2].token, Token::Ident("terra"));
    }

    #[test]
    fn test_const_block() {
        let source = r#"
            const {
                physics.stefan_boltzmann: 5.67e-8 <W/m²/K⁴>
            }
        "#;
        let tokens = lex(source).unwrap();
        assert_eq!(tokens[0].token, Token::Const);
        assert_eq!(tokens[1].token, Token::BraceOpen);
    }

    #[test]
    fn test_type_keywords() {
        let tokens = lex("Scalar Vec2 Vec3 Vec4 Tensor Seq Grid").unwrap();
        assert_eq!(tokens.len(), 7);
        assert_eq!(tokens[0].token, Token::ScalarType);
        assert_eq!(tokens[1].token, Token::Vec2Type);
        assert_eq!(tokens[2].token, Token::Vec3Type);
        assert_eq!(tokens[3].token, Token::Vec4Type);
        assert_eq!(tokens[4].token, Token::TensorType);
        assert_eq!(tokens[5].token, Token::SeqType);
        assert_eq!(tokens[6].token, Token::GridType);
    }
}
