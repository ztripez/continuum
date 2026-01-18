//! Lexical analysis for Continuum DSL.
//!
//! This module provides tokenization of CDSL source code using logos.
//!
//! # Design
//!
//! - `Token` — all CDSL token types (keywords, operators, literals, identifiers)
//! - Comments are stripped during lexing (not tokens)
//! - Token strings defined once in `TOKEN_STRINGS` table (single source of truth for Display)
//!
//! # Examples
//!
//! ```
//! # use continuum_cdsl::lexer::*;
//! # use logos::Logos;
//! let source = "signal temp : Scalar<K> { resolve { prev + 1 } }";
//! let tokens: Vec<Result<Token, ()>> = Token::lexer(source).collect();
//! ```

use logos::Logos;

/// CDSL token.
///
/// Represents all lexical elements of the CDSL language including keywords,
/// operators, literals, and identifiers.
///
/// Token strings for keywords, operators, and delimiters are defined once
/// in the `TOKEN_STRINGS` table and indexed by discriminant for Display.
///
/// # Layout
///
/// Uses `#[repr(u16)]` to guarantee discriminant values are stable and
/// can be safely used to index into `TOKEN_STRINGS`.
#[derive(Logos, Debug, Clone, PartialEq)]
#[repr(u16)]
#[logos(skip r"[ \t\r\n]+")] // Skip whitespace
#[logos(skip r"//[^\n]*")] // Skip // comments (non-doc)
#[logos(skip r"#[^\n]*")] // Skip # comments
#[logos(skip r"/\*([^*]|\*[^/])*\*/")] // Skip /* */ comments
pub enum Token {
    // === Keywords ===

    // Primitives
    /// Keyword `signal`
    #[token("signal")]
    Signal,
    /// Keyword `field`
    #[token("field")]
    Field,
    /// Keyword `operator`
    #[token("operator")]
    Operator,
    /// Keyword `impulse`
    #[token("impulse")]
    Impulse,
    /// Keyword `fracture`
    #[token("fracture")]
    Fracture,
    /// Keyword `chronicle`
    #[token("chronicle")]
    Chronicle,

    // Structure
    /// Keyword `entity`
    #[token("entity")]
    Entity,
    /// Keyword `member`
    #[token("member")]
    Member,
    /// Keyword `strata`
    #[token("strata")]
    Strata,
    /// Keyword `era`
    #[token("era")]
    Era,
    /// Keyword `type`
    #[token("type")]
    Type,
    /// Keyword `const`
    #[token("const")]
    Const,
    /// Keyword `config`
    #[token("config")]
    Config,

    // Phases & Blocks
    /// Keyword `resolve`
    #[token("resolve")]
    Resolve,
    /// Keyword `warmup`
    #[token("warmup")]
    WarmUp,
    /// Keyword `iterate`
    #[token("iterate")]
    Iterate,
    /// Keyword `collect`
    #[token("collect")]
    Collect,
    /// Keyword `apply`
    #[token("apply")]
    Apply,
    /// Keyword `measure`
    #[token("measure")]
    Measure,
    /// Keyword `assert`
    #[token("assert")]
    Assert,
    /// Keyword `transition`
    #[token("transition")]
    Transition,
    /// Keyword `when`
    #[token("when")]
    When,
    /// Keyword `to`
    #[token("to")]
    To,
    /// Keyword `for`
    #[token("for")]
    For,
    /// Keyword `emit`
    #[token("emit")]
    Emit,
    /// Keyword `observe`
    #[token("observe")]
    Observe,
    /// Keyword `world`
    #[token("world")]
    World,

    // Expression keywords
    /// Keyword `let`
    #[token("let")]
    Let,
    /// Keyword `in`
    #[token("in")]
    In,
    /// Keyword `if`
    #[token("if")]
    If,
    /// Keyword `else`
    #[token("else")]
    Else,

    // Context keywords
    /// Keyword `prev`
    #[token("prev")]
    Prev,
    /// Keyword `current`
    #[token("current")]
    Current,
    /// Keyword `inputs`
    #[token("inputs")]
    Inputs,
    /// Keyword `dt`
    #[token("dt")]
    Dt,
    /// Keyword `payload`
    #[token("payload")]
    Payload,
    /// Keyword `self`
    #[token("self")]
    Self_,
    /// Keyword `other`
    #[token("other")]
    Other,
    /// Keyword `pairs`
    #[token("pairs")]
    Pairs,

    // Boolean literals
    /// Boolean literal `true`
    #[token("true")]
    True,
    /// Boolean literal `false`
    #[token("false")]
    False,

    // === Operators ===

    // Arithmetic
    /// Operator `+`
    #[token("+")]
    Plus,
    /// Operator `-`
    #[token("-")]
    Minus,
    /// Operator `*`
    #[token("*")]
    Star,
    /// Operator `/`
    #[token("/")]
    Slash,
    /// Operator `%`
    #[token("%")]
    Percent,
    /// Operator `^`
    #[token("^")]
    Caret,

    // Comparison
    /// Operator `==`
    #[token("==")]
    EqEq,
    /// Operator `!=`
    #[token("!=")]
    BangEq,
    /// Operator `<` (high priority to avoid matching as start of Unit)
    #[token("<", priority = 10)]
    Lt,
    /// Operator `<=`
    #[token("<=")]
    LtEq,
    /// Operator `>` (high priority to avoid matching as end of Unit)
    #[token(">", priority = 10)]
    Gt,
    /// Operator `>=`
    #[token(">=")]
    GtEq,

    // Logic
    /// Keyword `and` (logical and)
    #[token("and")]
    And,
    /// Keyword `or` (logical or)
    #[token("or")]
    Or,
    /// Keyword `not` (logical not)
    #[token("not")]
    Not,

    // Assignment & Type
    /// Operator `=`
    #[token("=")]
    Eq,
    /// Operator `:`
    #[token(":")]
    Colon,
    /// Operator `->`
    #[token("->")]
    Arrow,
    /// Operator `<-` (signal/field assignment)
    #[token("<-")]
    LeftArrow,

    // Range
    /// Operator `..`
    #[token("..")]
    DotDot,
    /// Operator `..=`
    #[token("..=")]
    DotDotEq,

    // Other
    /// Operator `.`
    #[token(".")]
    Dot,
    /// Operator `,`
    #[token(",")]
    Comma,
    /// Operator `;`
    #[token(";")]
    Semicolon,
    /// Operator `|`
    #[token("|")]
    Pipe,

    // === Delimiters ===
    /// Delimiter `(`
    #[token("(")]
    LParen,
    /// Delimiter `)`
    #[token(")")]
    RParen,
    /// Delimiter `{`
    #[token("{")]
    LBrace,
    /// Delimiter `}`
    #[token("}")]
    RBrace,
    /// Delimiter `[`
    #[token("[")]
    LBracket,
    /// Delimiter `]`
    #[token("]")]
    RBracket,

    // === Literals ===
    /// Integer literal (e.g., 42, 0, 1000)
    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    Integer(i64),

    /// Float literal (e.g., 3.14, 1.0, 5.67e-8)
    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", |lex| lex.slice().parse::<f64>().ok())]
    #[regex(r"[0-9]+[eE][+-]?[0-9]+", |lex| lex.slice().parse::<f64>().ok())]
    Float(f64),

    /// String literal (e.g., "hello", "world")
    #[regex(r#""([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        // Strip quotes and unescape
        let content = &s[1..s.len()-1];
        Some(content.to_string())
    })]
    String(String),

    /// Identifier (e.g., velocity, temperature, Scalar, Vec3, m, kg)
    ///
    /// Simple identifier without dots. Dotted paths are parsed as
    /// sequences of Ident separated by Dot tokens.
    /// Allows both lowercase and uppercase (for type names).
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Ident(String),

    /// Doc comment `/// ...`
    ///
    /// Captures documentation comments that start with `///`.
    /// The captured string excludes the `///` prefix and leading whitespace.
    /// High priority ensures it's matched before `//` skip rule.
    #[regex(r"///[^\n]*", |lex| {
        let s = lex.slice();
        // Strip /// prefix and trim leading/trailing whitespace
        s.strip_prefix("///").unwrap_or(s).trim().to_string()
    }, priority = 10)]
    DocComment(String),
}

/// Token string lookup table.
///
/// Maps discriminant indices to their string representation.
/// This is the single source of truth for token display strings,
/// indexed by the enum discriminant order.
///
/// NOTE: The `#[token("...")]` attributes above must match these strings.
/// This duplication is unavoidable due to logos requiring literal strings,
/// but this table at least consolidates Display logic to avoid a large match.
const TOKEN_STRINGS: &[&str] = &[
    "signal",
    "field",
    "operator",
    "impulse",
    "fracture",
    "chronicle", // primitives
    "entity",
    "member",
    "strata",
    "era",
    "type",
    "const",
    "config", // structure
    "resolve",
    "warmup",
    "iterate",
    "collect",
    "apply",
    "measure",
    "assert",
    "transition",
    "when",
    "to",
    "for",
    "emit",
    "observe",
    "world", // phases & blocks
    "let",
    "in",
    "if",
    "else", // expressions
    "prev",
    "current",
    "inputs",
    "dt",
    "payload",
    "self",
    "other",
    "pairs", // context
    "true",
    "false", // booleans
    "+",
    "-",
    "*",
    "/",
    "%",
    "^", // arithmetic
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=", // comparison
    "and",
    "or",
    "not", // logic
    "=",
    ":",
    "->",
    "<-", // assignment
    "..",
    "..=", // range
    ".",
    ",",
    ";",
    "|", // other
    "(",
    ")",
    "{",
    "}",
    "[",
    "]", // delimiters
];

impl Token {
    /// Get the index into TOKEN_STRINGS for simple tokens.
    ///
    /// # Returns
    ///
    /// Index for simple tokens (keywords, operators, delimiters), or panics for data tokens.
    ///
    /// # Safety
    ///
    /// Safe due to `#[repr(u16)]` on Token enum ensuring stable discriminants.
    fn token_string_index(&self) -> usize {
        // Safe: Token has #[repr(u16)] so discriminant values are stable
        let discriminant = unsafe { *(self as *const Token as *const u16) };
        discriminant as usize
    }
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // Literals with data (not in TOKEN_STRINGS table)
            Token::Integer(n) => write!(f, "{}", n),
            Token::Float(x) => write!(f, "{}", x),
            Token::String(s) => write!(f, "\"{}\"", s),
            Token::Ident(id) => write!(f, "{}", id),
            Token::DocComment(text) => write!(f, "/// {}", text),

            // Simple tokens (keywords, operators, delimiters)
            // Index into TOKEN_STRINGS using discriminant
            _ => {
                let idx = self.token_string_index();
                let s = TOKEN_STRINGS
                    .get(idx)
                    .expect("BUG: token discriminant out of bounds for TOKEN_STRINGS");
                write!(f, "{}", s)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test helper: lex source and filter out errors.
    ///
    /// This is lenient for testing valid token sequences. For tests that need
    /// to verify error handling, use `Token::lexer()` directly and check the
    /// `Result` stream (see `test_lexer_error_detection`).
    fn lex(source: &str) -> Vec<Token> {
        Token::lexer(source)
            .filter_map(|result| result.ok())
            .collect()
    }

    /// Test helper: lex source and panic on any error.
    ///
    /// Use this when testing syntax that must be valid.
    #[allow(dead_code)]
    fn lex_strict(source: &str) -> Vec<Token> {
        Token::lexer(source)
            .collect::<Result<Vec<_>, _>>()
            .expect("Lexing failed - invalid token encountered")
    }

    #[test]
    fn test_keywords() {
        let tokens = lex("signal field operator impulse");
        assert_eq!(
            tokens,
            vec![Token::Signal, Token::Field, Token::Operator, Token::Impulse,]
        );
    }

    #[test]
    fn test_identifiers() {
        let tokens = lex("velocity temperature my_var x");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("velocity".to_string()),
                Token::Ident("temperature".to_string()),
                Token::Ident("my_var".to_string()),
                Token::Ident("x".to_string()),
            ]
        );
    }

    #[test]
    fn test_numbers() {
        let tokens = lex("42 3.14 5.67e-8 1e10");
        assert_eq!(
            tokens,
            vec![
                Token::Integer(42),
                Token::Float(3.14),
                Token::Float(5.67e-8),
                Token::Float(1e10),
            ]
        );
    }

    #[test]
    fn test_strings() {
        let tokens = lex(r#""hello" "world""#);
        assert_eq!(
            tokens,
            vec![
                Token::String("hello".to_string()),
                Token::String("world".to_string()),
            ]
        );
    }

    #[test]
    fn test_units() {
        // Units are lexed as angle brackets with identifiers/operators
        // Parser will recognize these as unit literals based on context
        // Note: Superscripts (², ⁴) are skipped by lexer as they're not valid tokens
        let tokens = lex("<m> <kg/s> <W/m²/K⁴>");
        assert_eq!(
            tokens,
            vec![
                // <m>
                Token::Lt,
                Token::Ident("m".to_string()),
                Token::Gt,
                // <kg/s>
                Token::Lt,
                Token::Ident("kg".to_string()),
                Token::Slash,
                Token::Ident("s".to_string()),
                Token::Gt,
                // <W/m²/K⁴> - superscripts are skipped
                Token::Lt,
                Token::Ident("W".to_string()),
                Token::Slash,
                Token::Ident("m".to_string()),
                Token::Slash,
                Token::Ident("K".to_string()),
                Token::Gt,
            ]
        );
    }

    #[test]
    fn test_operators() {
        let tokens = lex("+ - * / == != < <= > >=");
        assert_eq!(
            tokens,
            vec![
                Token::Plus,
                Token::Minus,
                Token::Star,
                Token::Slash,
                Token::EqEq,
                Token::BangEq,
                Token::Lt,
                Token::LtEq,
                Token::Gt,
                Token::GtEq,
            ]
        );
    }

    #[test]
    fn test_delimiters() {
        let tokens = lex("( ) { } [ ] , ; : .");
        assert_eq!(
            tokens,
            vec![
                Token::LParen,
                Token::RParen,
                Token::LBrace,
                Token::RBrace,
                Token::LBracket,
                Token::RBracket,
                Token::Comma,
                Token::Semicolon,
                Token::Colon,
                Token::Dot,
            ]
        );
    }

    #[test]
    fn test_dotted_path() {
        let tokens = lex("terra.core.temperature");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("terra".to_string()),
                Token::Dot,
                Token::Ident("core".to_string()),
                Token::Dot,
                Token::Ident("temperature".to_string()),
            ]
        );
    }

    #[test]
    fn test_signal_declaration() {
        let source = "signal temp : Scalar<K> { resolve { prev + 1.0 } }";
        let tokens = lex(source);
        assert_eq!(
            tokens,
            vec![
                Token::Signal,
                Token::Ident("temp".to_string()),
                Token::Colon,
                Token::Ident("Scalar".to_string()),
                Token::Lt,
                Token::Ident("K".to_string()),
                Token::Gt,
                Token::LBrace,
                Token::Resolve,
                Token::LBrace,
                Token::Prev,
                Token::Plus,
                Token::Float(1.0),
                Token::RBrace,
                Token::RBrace,
            ]
        );
    }

    #[test]
    fn test_line_comments() {
        let source = "signal // comment\ntemp";
        let tokens = lex(source);
        assert_eq!(
            tokens,
            vec![Token::Signal, Token::Ident("temp".to_string()),]
        );
    }

    #[test]
    fn test_hash_comments() {
        let source = "signal # comment\ntemp";
        let tokens = lex(source);
        assert_eq!(
            tokens,
            vec![Token::Signal, Token::Ident("temp".to_string()),]
        );
    }

    #[test]
    fn test_block_comments() {
        let source = "signal /* multi\nline\ncomment */ temp";
        let tokens = lex(source);
        assert_eq!(
            tokens,
            vec![Token::Signal, Token::Ident("temp".to_string()),]
        );
    }

    #[test]
    fn test_booleans() {
        let tokens = lex("true false");
        assert_eq!(tokens, vec![Token::True, Token::False,]);
    }

    #[test]
    fn test_range_operators() {
        let tokens = lex("0..100 0..=100");
        assert_eq!(
            tokens,
            vec![
                Token::Integer(0),
                Token::DotDot,
                Token::Integer(100),
                Token::Integer(0),
                Token::DotDotEq,
                Token::Integer(100),
            ]
        );
    }

    #[test]
    fn test_lambda() {
        let source = "|p| p.mass";
        let tokens = lex(source);
        assert_eq!(
            tokens,
            vec![
                Token::Pipe,
                Token::Ident("p".to_string()),
                Token::Pipe,
                Token::Ident("p".to_string()),
                Token::Dot,
                Token::Ident("mass".to_string()),
            ]
        );
    }

    #[test]
    fn test_let_expression() {
        let source = "let x = 42 in x + 1";
        let tokens = lex(source);
        assert_eq!(
            tokens,
            vec![
                Token::Let,
                Token::Ident("x".to_string()),
                Token::Eq,
                Token::Integer(42),
                Token::In,
                Token::Ident("x".to_string()),
                Token::Plus,
                Token::Integer(1),
            ]
        );
    }

    #[test]
    fn test_if_expression() {
        let source = "if x > 0 { 1 } else { 0 }";
        let tokens = lex(source);
        assert_eq!(
            tokens,
            vec![
                Token::If,
                Token::Ident("x".to_string()),
                Token::Gt,
                Token::Integer(0),
                Token::LBrace,
                Token::Integer(1),
                Token::RBrace,
                Token::Else,
                Token::LBrace,
                Token::Integer(0),
                Token::RBrace,
            ]
        );
    }

    #[test]
    fn test_whitespace_handling() {
        let source = "  signal\t\ntemp\r\n";
        let tokens = lex(source);
        assert_eq!(
            tokens,
            vec![Token::Signal, Token::Ident("temp".to_string()),]
        );
    }

    #[test]
    fn test_invalid_token() {
        // The lex() helper filters out errors for convenience.
        // This tests that valid tokens are still recognized when invalid chars are present.
        // For proper error detection, see test_lexer_error_detection below.
        let source = "signal @ temp";
        let tokens = lex(source);
        assert_eq!(
            tokens,
            vec![
                Token::Signal,
                // @ is skipped (error filtered by helper)
                Token::Ident("temp".to_string()),
            ]
        );
    }

    #[test]
    fn test_lexer_error_detection() {
        // Test that we can detect errors if we don't filter
        let source = "signal @ temp";
        let results: Vec<_> = Token::lexer(source).collect();
        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok()); // signal
        assert!(results[1].is_err()); // @
        assert!(results[2].is_ok()); // temp
    }

    /// Verify that TOKEN_STRINGS matches token definitions.
    ///
    /// This test ensures display strings are consistent with lexer tokens.
    #[test]
    fn test_token_string_consistency() {
        // Test a sample of tokens to ensure Display matches expected strings
        assert_eq!(Token::Signal.to_string(), "signal");
        assert_eq!(Token::Field.to_string(), "field");
        assert_eq!(Token::Plus.to_string(), "+");
        assert_eq!(Token::EqEq.to_string(), "==");
        assert_eq!(Token::Arrow.to_string(), "->");
        assert_eq!(Token::LBrace.to_string(), "{");
        assert_eq!(Token::RBracket.to_string(), "]");
        assert_eq!(Token::True.to_string(), "true");
        assert_eq!(Token::DotDotEq.to_string(), "..=");
    }

    #[test]
    fn test_type_parameter_angle_brackets() {
        // Debug test to see what <K> produces
        let source = "<K>";
        let results: Vec<_> = Token::lexer(source).collect();
        eprintln!("Lexing '<K>' produces: {:?}", results);

        let tokens: Vec<_> = results.into_iter().filter_map(|r| r.ok()).collect();
        assert_eq!(
            tokens,
            vec![Token::Lt, Token::Ident("K".to_string()), Token::Gt]
        );
    }

    #[test]
    fn test_new_keywords() {
        let tokens = lex("for emit observe world and or not");
        assert_eq!(
            tokens,
            vec![
                Token::For,
                Token::Emit,
                Token::Observe,
                Token::World,
                Token::And,
                Token::Or,
                Token::Not,
            ]
        );
    }

    #[test]
    fn test_left_arrow() {
        let tokens = lex("temp.x <- value");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("temp".to_string()),
                Token::Dot,
                Token::Ident("x".to_string()),
                Token::LeftArrow,
                Token::Ident("value".to_string()),
            ]
        );
    }

    #[test]
    fn test_doc_comments() {
        let source = "/// This is a doc comment\n///   Another line\nsignal";
        let tokens = lex(source);
        assert_eq!(
            tokens,
            vec![
                Token::DocComment("This is a doc comment".to_string()),
                Token::DocComment("Another line".to_string()),
                Token::Signal,
            ]
        );
    }

    #[test]
    fn test_doc_vs_regular_comments() {
        let source = "// Regular comment\n/// Doc comment\n# Hash comment\nsignal";
        let tokens = lex(source);
        // Regular and hash comments are skipped, only doc comment captured
        assert_eq!(
            tokens,
            vec![Token::DocComment("Doc comment".to_string()), Token::Signal,]
        );
    }

    #[test]
    fn test_and_or_not_keywords() {
        // Test that word forms work (&&, ||, ! are not valid - only and, or, not)
        let tokens = lex("a and b or c not d");
        assert_eq!(
            tokens,
            vec![
                Token::Ident("a".to_string()),
                Token::And,
                Token::Ident("b".to_string()),
                Token::Or,
                Token::Ident("c".to_string()),
                Token::Not,
                Token::Ident("d".to_string()),
            ]
        );
    }
}
