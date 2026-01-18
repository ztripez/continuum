//! Parser for Continuum DSL using chumsky combinators.
//!
//! This module implements a recursive descent parser that transforms a stream
//! of tokens from the lexer into an untyped AST ([`Expr`]).
//!
//! # Design Principles
//!
//! ## Token-Based Parsing
//!
//! The parser operates on [`Token`](crate::lexer::Token) streams, not source text.
//! This provides:
//! - Clean separation between lexing and parsing
//! - Structured error recovery
//! - Consistent span tracking
//!
//! ## Unit Expression Parsing
//!
//! Units are **NOT** lexed as single tokens. The lexer produces:
//! - `Lt` (`<`) - angle bracket start
//! - Identifier/operators (`m`, `/`, `^`, etc.)
//! - `Gt` (`>`) - angle bracket end
//!
//! The parser reconstructs unit expressions from these token sequences:
//! ```text
//! <m/s>  →  [Lt, Ident("m"), Slash, Ident("s"), Gt]  →  UnitExpr::Divide(Base("m"), Base("s"))
//! ```
//!
//! ## Operator Precedence
//!
//! Binary operators are parsed with precedence climbing:
//! 1. Logical OR (`||`) - lowest
//! 2. Logical AND (`&&`)
//! 3. Comparison (`<`, `<=`, `>`, `>=`, `==`, `!=`)
//! 4. Addition/Subtraction (`+`, `-`)
//! 5. Multiplication/Division/Modulo (`*`, `/`, `%`)
//! 6. Power (`^`) - highest precedence (right-associative: `2 ^ 3 ^ 4` → `2 ^ (3 ^ 4)`)
//!
//! ## Error Recovery
//!
//! The parser does not perform error recovery.
//! Parse failures produce Rich<Token> errors rather than
//! [`ExprKind::ParseError`] placeholders.
//!
//! # Examples
//!
//! ```rust,ignore
//! use continuum_cdsl::parser::parse_expr;
//! use continuum_cdsl::lexer::Token;
//!
//! let source = "10.0<m> + velocity.x";
//! let tokens: Vec<_> = Token::lexer(source)
//!     .collect::<Result<Vec<_>, _>>()
//!     .expect("lexer should not fail on valid input");
//! let expr = parse_expr(&tokens)?;
//! ```

use chumsky::prelude::*;

use crate::ast::{
    Attribute, BinaryOp, BlockBody, ConfigEntry, ConstEntry, Declaration, Entity, EraDecl, Expr,
    Node, ObserveBlock, ObserveWhen, RoleData, Stmt, Stratum, StratumPolicyEntry, StratumState,
    TransitionDecl, TypeDecl, TypeExpr, TypeField, UnaryOp, UnitExpr, UntypedKind as ExprKind,
    WarmupBlock, WarmupPolicy, WarmupTimeout, WhenBlock, WorldDecl,
};
use crate::foundation::{EntityId, Path, Span, StratumId};
use crate::lexer::Token;

// Need continuum_foundation::Path for EntityId/StratumId
use continuum_foundation::Path as FoundationPath;

/// Parse an expression from a token stream.
///
/// # Parameters
///
/// - `tokens`: Slice of tokens to parse
///
/// # Returns
///
/// - `Ok(Expr)`: Successfully parsed expression
///
/// # Errors
///
/// Returns `Err(Vec<Rich<Token>>)` when parsing fails. Each [`Rich<Token>`](chumsky::error::Rich)
/// error contains:
/// - **Span**: Token index range into the `tokens` slice (not byte offsets in source).
///   For example, `span.start` and `span.end` are indices into the token array.
/// - **Expected tokens**: The `Token` variants the parser was expecting at this position
/// - **Found token**: The actual `Token` encountered, or `None` at end-of-input
/// - **Reason/Label**: Human-readable error context from chumsky
///
/// Multiple errors may be returned if the parser encounters several issues.
///
/// To check for errors, use `.into_result().is_err()` on the returned [`ParseResult`].
///
/// # Examples
///
/// ```rust,ignore
/// let tokens = vec![
///     Token::Integer(10),
///     Token::Plus,
///     Token::Integer(20),
/// ];
/// let expr = parse_expr(&tokens)?;
/// ```
pub fn parse_expr(tokens: &[Token]) -> ParseResult<Expr, Rich<'_, Token>> {
    expr_parser().parse(tokens)
}

/// Parse declarations from a token stream.
///
/// # Parameters
///
/// - `tokens`: Slice of tokens to parse
///
/// # Returns
///
/// - `Ok(Vec<Declaration>)`: Successfully parsed declarations
///
/// # Errors
///
/// Returns parse errors if the token stream doesn't match the declaration grammar.
pub fn parse_declarations(tokens: &[Token]) -> ParseResult<Vec<Declaration>, Rich<'_, Token>> {
    declarations_parser().parse(tokens)
}

/// Main declarations parser - parses a complete CDSL file.
fn declarations_parser<'src>()
-> impl Parser<'src, &'src [Token], Vec<Declaration>, extra::Err<Rich<'src, Token>>> + Clone {
    choice((
        world_parser(),
        type_decl_parser(),
        const_block_parser(),
        config_block_parser(),
        entity_parser(),
        member_parser(),
        stratum_parser(),
        era_parser(),
        signal_parser(),
        field_parser(),
        operator_parser(),
        impulse_parser(),
        fracture_parser(),
        chronicle_parser(),
    ))
    .repeated()
    .collect::<Vec<_>>()
    .then_ignore(end())
}

/// Main expression parser (recursive).
///
/// Parses the full expression grammar with explicit precedence levels
/// (lowest to highest):
///
/// 1. **Let bindings**: `let name = value in body`
/// 2. **If expressions**: `if condition { then_expr } else { else_expr }`
/// 3. **Logical OR**: `a || b`
/// 4. **Logical AND**: `a && b`
/// 5. **Comparison**: `<`, `<=`, `>`, `>=`, `==`, `!=`
/// 6. **Addition/Subtraction**: `a + b`, `a - b`
/// 7. **Multiplication/Division/Modulo**: `a * b`, `a / b`, `a % b`
/// 8. **Power** (right-associative): `a ^ b ^ c` = `a ^ (b ^ c)`
/// 9. **Unary operators**: `-expr`, `!expr`
/// 10. **Postfix** (highest): field access `expr.field`, function calls `f(args)`
/// 11. **Atoms**: literals, identifiers, parentheses, vectors
///
/// # Grammar Summary
///
/// ```text
/// expr       := let_expr | if_expr | or_expr
/// let_expr   := 'let' IDENT '=' expr 'in' expr
/// if_expr    := 'if' expr '{' expr '}' 'else' '{' expr '}'
/// or_expr    := and_expr ('||' and_expr)*
/// and_expr   := cmp_expr ('&&' cmp_expr)*
/// cmp_expr   := add_expr (CMP_OP add_expr)*
/// add_expr   := mul_expr (('+' | '-') mul_expr)*
/// mul_expr   := power_expr (('*' | '/' | '%') power_expr)*
/// power_expr := unary_expr ('^' unary_expr)*  // right-associative
/// unary_expr := ('-' | '!')* postfix_expr
/// postfix    := atom ('.' IDENT | '(' args ')')*
/// atom       := NUMBER | BOOL | IDENT | '(' expr ')' | '[' exprs ']'
/// ```
///
/// # Parsing Features
///
/// - Literals: numeric (with optional units `<m>`), boolean, vector `[1, 2, 3]`
/// - Operators: binary (14 ops), unary (neg, not)
/// - Let bindings: introduce local variables
/// - If expressions: eager evaluation (both branches evaluated)
/// - Function calls: `f(arg1, arg2)`
/// - Field access: `expr.field`
/// - Context values: `prev`, `current`, `inputs`, `dt`, `self`, `other`, `payload`
///
/// # Notes
///
/// - Power operator is right-associative: `2 ^ 3 ^ 4` parses as `2 ^ (3 ^ 4)`
/// - If expressions require braces: `if x { a } else { b }`
/// - Let bindings require `in` keyword: `let x = 1 in x + 1`
/// - Parentheses override precedence: `(a + b) * c`
fn expr_parser<'src>()
-> impl Parser<'src, &'src [Token], Expr, extra::Err<Rich<'src, Token>>> + Clone {
    recursive(|expr| {
        // === Atoms ===

        // Boolean literals
        let bool_literal = select! {
            Token::True => ExprKind::BoolLiteral(true),
            Token::False => ExprKind::BoolLiteral(false),
        }
        .map_with(|kind, e| Expr::new(kind, token_span(e.span())));

        // Numeric literals (integers and floats)
        let numeric_literal = select! {
            Token::Integer(n) => n as f64,
            Token::Float(f) => f,
        }
        .then(
            // Optional unit: < unit_expr >
            just(Token::Lt)
                .ignore_then(unit_expr_parser())
                .then_ignore(just(Token::Gt))
                .or_not(),
        )
        .map_with(|(value, unit), e| {
            Expr::new(ExprKind::Literal { value, unit }, token_span(e.span()))
        });

        // Context values (prev, current, inputs, dt, self, other, payload)
        let context_value = select! {
            Token::Prev => ExprKind::Prev,
            Token::Current => ExprKind::Current,
            Token::Inputs => ExprKind::Inputs,
            Token::Dt => ExprKind::Dt,
            Token::Self_ => ExprKind::Self_,
            Token::Other => ExprKind::Other,
            Token::Payload => ExprKind::Payload,
        }
        .map_with(|kind, e| Expr::new(kind, token_span(e.span())));

        // Identifiers - can be called
        let identifier = select! {
            Token::Ident(name) => name,
        }
        .map_with(|name, e| Expr::new(ExprKind::Local(name), token_span(e.span())))
        .foldl(
            expr.clone()
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .repeated(),
            |func_expr, args| {
                let span = func_expr.span;
                match &func_expr.kind {
                    ExprKind::Local(name) => {
                        let path = Path::from_str(name);
                        Expr::new(ExprKind::Call { func: path, args }, span)
                    }
                    ExprKind::Call { .. } => {
                        // Calling a call result - not supported yet
                        Expr::new(
                            ExprKind::ParseError("nested function calls not supported".to_string()),
                            span,
                        )
                    }
                    _ => unreachable!("identifier foldl should only produce Local or Call"),
                }
            },
        );

        // Parenthesized expressions
        let parens = expr
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // Vector literals: [expr, expr, ...]
        let vector_literal = expr
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map_with(|elements, e| Expr::new(ExprKind::Vector(elements), token_span(e.span())));

        // Atom: any of the above
        let atom = choice((
            bool_literal,
            numeric_literal,
            context_value,
            vector_literal,
            parens,
            identifier,
        ));

        // === Postfix (field access only - calls handled above) ===

        // Field access: atom.field
        let postfix = atom.foldl(
            just(Token::Dot)
                .ignore_then(select! { Token::Ident(name) => name })
                .repeated(),
            |object, field| {
                let span = object.span;
                Expr::new(
                    ExprKind::FieldAccess {
                        object: Box::new(object),
                        field,
                    },
                    span,
                )
            },
        );

        // === Unary operators ===

        let unary = choice((
            just(Token::Minus).to(UnaryOp::Neg),
            just(Token::Not).to(UnaryOp::Not),
        ))
        .repeated()
        .foldr(postfix, |op, operand| {
            let span = operand.span;
            Expr::new(
                ExprKind::Unary {
                    op,
                    operand: Box::new(operand),
                },
                span,
            )
        });

        // === Binary operators with precedence ===

        // Power (highest precedence, right-associative)
        // For `a ^ b ^ c`, parse as `a ^ (b ^ c)`
        let power_op = just(Token::Caret).to(BinaryOp::Pow);
        let power = unary
            .clone()
            .then(power_op.then(unary.clone()).repeated().collect::<Vec<_>>())
            .map(|(first, rest)| {
                if rest.is_empty() {
                    return first;
                }

                // Helper to build power binary node (eliminates duplication)
                let make_pow = |left: Expr, right: Expr| {
                    let span = left.span;
                    Expr::new(
                        ExprKind::Binary {
                            op: BinaryOp::Pow,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        span,
                    )
                };

                // Build right-to-left using iterator fold
                // For `2 ^ 3 ^ 4`, rest = [(Pow, 3), (Pow, 4)]
                // Reverse to [(Pow, 4), (Pow, 3)], start with 4, fold left with 3
                let mut iter = rest.into_iter().rev();
                let (_, rightmost) = iter.next().unwrap();

                let right_tree = iter.fold(rightmost, |acc, (_, operand)| make_pow(operand, acc));

                make_pow(first, right_tree)
            });

        // Multiplication, division, modulo
        let mul_op = choice((
            just(Token::Star).to(BinaryOp::Mul),
            just(Token::Slash).to(BinaryOp::Div),
            just(Token::Percent).to(BinaryOp::Mod),
        ));
        let mul =
            power
                .clone()
                .foldl_with(mul_op.then(power).repeated(), |left, (op, right), _e| {
                    let span = left.span;
                    Expr::new(
                        ExprKind::Binary {
                            op,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        span,
                    )
                });

        // Addition, subtraction
        let add_op = choice((
            just(Token::Plus).to(BinaryOp::Add),
            just(Token::Minus).to(BinaryOp::Sub),
        ));
        let add = mul
            .clone()
            .foldl_with(add_op.then(mul).repeated(), |left, (op, right), _e| {
                let span = left.span;
                Expr::new(
                    ExprKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            });

        // Comparison operators
        let cmp_op = choice((
            just(Token::Lt).to(BinaryOp::Lt),
            just(Token::LtEq).to(BinaryOp::Le),
            just(Token::Gt).to(BinaryOp::Gt),
            just(Token::GtEq).to(BinaryOp::Ge),
            just(Token::EqEq).to(BinaryOp::Eq),
            just(Token::BangEq).to(BinaryOp::Ne),
        ));
        let comparison =
            add.clone()
                .foldl_with(cmp_op.then(add).repeated(), |left, (op, right), _e| {
                    let span = left.span;
                    Expr::new(
                        ExprKind::Binary {
                            op,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                        span,
                    )
                });

        // Logical AND
        let and = comparison.clone().foldl_with(
            just(Token::And)
                .to(BinaryOp::And)
                .then(comparison)
                .repeated(),
            |left, (op, right), _e| {
                let span = left.span;
                Expr::new(
                    ExprKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            },
        );

        // Logical OR
        let or = and.clone().foldl_with(
            just(Token::Or).to(BinaryOp::Or).then(and).repeated(),
            |left, (op, right), _e| {
                let span = left.span;
                Expr::new(
                    ExprKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            },
        );

        // If expression: if cond { then_expr } else { else_expr }
        let braced_expr = expr
            .clone()
            .delimited_by(just(Token::LBrace), just(Token::RBrace));
        let if_expr = just(Token::If)
            .ignore_then(expr.clone())
            .then(braced_expr.clone())
            .then_ignore(just(Token::Else))
            .then(braced_expr)
            .map_with(|((condition, then_branch), else_branch), e| {
                Expr::new(
                    ExprKind::If {
                        condition: Box::new(condition),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                    token_span(e.span()),
                )
            });

        // Let binding: let x = value in body
        let let_expr = just(Token::Let)
            .ignore_then(select! { Token::Ident(name) => name })
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .then_ignore(just(Token::In))
            .then(expr.clone())
            .map_with(|((name, value), body), e| {
                Expr::new(
                    ExprKind::Let {
                        name,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    token_span(e.span()),
                )
            });

        // Choice between let, if, or normal expression
        // Let and if are lower precedence than all operators
        let base_expr = choice((let_expr, if_expr, or));

        // Note: Struct construction and aggregates (sum, map, fold) are extension points.
        // The parser will produce errors for these constructs until parsers are added.

        // Return the lowest precedence parser (let/if or logical OR)
        base_expr
    })
}

/// Parse a unit expression from angle bracket sequence.
///
/// Units are lexed as token sequences: `<`, content, `>`
///
/// # Grammar
///
/// ```text
/// unit_expr := '<' unit_term '>'
/// unit_term := base_unit
///            | unit_term '*' unit_term
///            | unit_term '/' unit_term
///            | unit_term '^' integer
///            | '(' unit_term ')'
/// base_unit := identifier
/// ```
///
/// # Examples
///
/// ```text
/// <m>       → Base("m")
/// <m/s>     → Divide(Base("m"), Base("s"))
/// <kg*m^2>  → Multiply(Base("kg"), Power(Base("m"), 2))
/// <>        → Dimensionless
/// ```
///
/// # Parameters
///
/// - `tokens`: Token stream positioned at `<`
///
/// # Returns
///
/// Parsed unit expression or error.
fn unit_expr_parser<'src>()
-> impl Parser<'src, &'src [Token], UnitExpr, extra::Err<Rich<'src, Token>>> + Clone {
    recursive(|unit_term| {
        // Base unit: identifier
        let base = select! {
            Token::Ident(name) => UnitExpr::Base(name),
        };

        // Parenthesized unit
        let parens = unit_term
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        // Primary: base or parens
        let primary = choice((base, parens));

        // Power: primary ^ integer
        let power = primary
            .then(
                just(Token::Caret)
                    .ignore_then(select! {
                        Token::Integer(n) => n as i8,
                    })
                    .or_not(),
            )
            .map(|(base_unit, exp)| {
                if let Some(exp) = exp {
                    UnitExpr::Power(Box::new(base_unit), exp)
                } else {
                    base_unit
                }
            });

        // Multiply/Divide (left-associative)
        let mul_div = power.clone().foldl(
            choice((
                just(Token::Star).to(true),   // true = multiply
                just(Token::Slash).to(false), // false = divide
            ))
            .then(power)
            .repeated(),
            |left, (is_mul, right)| {
                if is_mul {
                    UnitExpr::Multiply(Box::new(left), Box::new(right))
                } else {
                    UnitExpr::Divide(Box::new(left), Box::new(right))
                }
            },
        );

        mul_div
    })
}

/// Parse a type expression.
///
/// # Grammar
///
/// ```text
/// type_expr := 'Scalar' [ '<' unit_expr '>' ]
///            | 'Vector' '<' integer ',' unit_expr '>'
///            | 'Matrix' '<' integer ',' integer ',' unit_expr '>'
///            | 'Bool'
///            | identifier
/// ```
///
/// # Examples
///
/// ```text
/// Scalar<m>           → Scalar { unit: Some(Base("m")) }
/// Vector<3, m/s>      → Vector { dim: 3, unit: Some(Divide(...)) }
/// Matrix<3, 3, kg>    → Matrix { rows: 3, cols: 3, unit: Some(Base("kg")) }
/// Bool                → Bool
/// OrbitalElements     → User(Path(...))
/// ```
fn type_expr_parser<'src>()
-> impl Parser<'src, &'src [Token], TypeExpr, extra::Err<Rich<'src, Token>>> + Clone {
    // Bool type
    let bool_type = just(Token::Ident("Bool".to_string())).to(TypeExpr::Bool);

    // Scalar<unit>
    let scalar_type = just(Token::Ident("Scalar".to_string()))
        .then(
            just(Token::Lt)
                .ignore_then(unit_expr_parser())
                .then_ignore(just(Token::Gt))
                .or_not(),
        )
        .map(|(_, unit)| TypeExpr::Scalar { unit });

    // Vector<dim, unit>
    let vector_type = just(Token::Ident("Vector".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(select! { Token::Integer(n) => n as u8 })
        .then_ignore(just(Token::Comma))
        .then(unit_expr_parser())
        .then_ignore(just(Token::Gt))
        .map(|(dim, unit)| TypeExpr::Vector {
            dim,
            unit: Some(unit),
        });

    // Matrix<rows, cols, unit>
    let matrix_type = just(Token::Ident("Matrix".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(select! { Token::Integer(n) => n as u8 })
        .then_ignore(just(Token::Comma))
        .then(select! { Token::Integer(n) => n as u8 })
        .then_ignore(just(Token::Comma))
        .then(unit_expr_parser())
        .then_ignore(just(Token::Gt))
        .map(|((rows, cols), unit)| TypeExpr::Matrix {
            rows,
            cols,
            unit: Some(unit),
        });

    // Vec2<unit>, Vec3<unit>, Vec4<unit> - sugar for Vector<2/3/4, unit>
    let vec2_type = just(Token::Ident("Vec2".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(unit_expr_parser())
        .then_ignore(just(Token::Gt))
        .map(|unit| TypeExpr::Vector {
            dim: 2,
            unit: Some(unit),
        });

    let vec3_type = just(Token::Ident("Vec3".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(unit_expr_parser())
        .then_ignore(just(Token::Gt))
        .map(|unit| TypeExpr::Vector {
            dim: 3,
            unit: Some(unit),
        });

    let vec4_type = just(Token::Ident("Vec4".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(unit_expr_parser())
        .then_ignore(just(Token::Gt))
        .map(|unit| TypeExpr::Vector {
            dim: 4,
            unit: Some(unit),
        });

    // Mat2<unit>, Mat3<unit>, Mat4<unit> - sugar for Matrix<2/3/4, 2/3/4, unit>
    let mat2_type = just(Token::Ident("Mat2".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(unit_expr_parser())
        .then_ignore(just(Token::Gt))
        .map(|unit| TypeExpr::Matrix {
            rows: 2,
            cols: 2,
            unit: Some(unit),
        });

    let mat3_type = just(Token::Ident("Mat3".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(unit_expr_parser())
        .then_ignore(just(Token::Gt))
        .map(|unit| TypeExpr::Matrix {
            rows: 3,
            cols: 3,
            unit: Some(unit),
        });

    let mat4_type = just(Token::Ident("Mat4".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(unit_expr_parser())
        .then_ignore(just(Token::Gt))
        .map(|unit| TypeExpr::Matrix {
            rows: 4,
            cols: 4,
            unit: Some(unit),
        });

    // Quat or Quaternion - sugar for Vector<4, 1> (unit quaternion)
    let quat_type = choice((
        just(Token::Ident("Quat".to_string())),
        just(Token::Ident("Quaternion".to_string())),
    ))
    .map(|_| TypeExpr::Vector {
        dim: 4,
        unit: None, // Quaternions are dimensionless
    });

    // User types (identifiers)
    let user_type = select! {
        Token::Ident(name) => TypeExpr::User(Path::from_str(&name)),
    };

    choice((
        bool_type,
        scalar_type,
        vector_type,
        matrix_type,
        vec2_type,
        vec3_type,
        vec4_type,
        mat2_type,
        mat3_type,
        mat4_type,
        quat_type,
        user_type,
    ))
}

/// Helper to convert chumsky span to Continuum Span.
///
/// # Parameters
///
/// - `span`: Chumsky simple span (start, end offsets)
///
/// # Returns
///
/// Continuum [`Span`] with file ID and line information.
///
/// # Implementation Note
///
/// Currently returns a placeholder span. Full implementation needs:
/// - File ID mapping
/// - Line number calculation from byte offsets
/// - Source map integration
///
/// # Current Implementation
///
/// This function currently returns placeholder spans with file_id=0 and line=0.
/// Proper span conversion requires a SourceMap context which is not yet implemented.
/// When SourceMap integration is added, this function will be updated to:
/// - Accept a SourceMap/FileId parameter
/// - Look up line numbers from byte offsets
/// - Return properly mapped spans
///
/// The current placeholder implementation is sufficient for parser development
/// but will need to be replaced before production use.
fn token_span(span: SimpleSpan) -> Span {
    // Placeholder span conversion until SourceMap integration is complete.
    // Returns byte offsets with file_id=0 and line=0.
    Span::new(0, span.start as u32, span.end as u32, 0)
}

// =============================================================================
// Declaration Parsers
// =============================================================================

/// Parse a dotted path: `terra.temperature` or `plate.area`
///
/// Paths are sequences of identifiers separated by dots.
/// They identify signals, fields, entities, strata, eras, etc.
fn path_parser<'src>()
-> impl Parser<'src, &'src [Token], Path, extra::Err<Rich<'src, Token>>> + Clone {
    select! { Token::Ident(name) => name }
        .separated_by(just(Token::Dot))
        .at_least(1)
        .collect::<Vec<_>>()
        .map(|segments| Path::from_str(&segments.join(".")))
}

/// Parse type annotation: `: type TypeExpr`
///
/// Per architecture feedback, type annotations use explicit `type` keyword
/// to disambiguate from attributes: `: type Scalar<K>` vs `: title("Temp")`
fn type_annotation_parser<'src>()
-> impl Parser<'src, &'src [Token], TypeExpr, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Colon)
        .ignore_then(just(Token::Type))
        .ignore_then(type_expr_parser())
}

/// Parse attribute: `: name(args)` or `: name`
///
/// Attributes are parsed generically. Validation of attribute names and
/// argument types happens in the analyzer, not the parser.
fn attribute_parser<'src>()
-> impl Parser<'src, &'src [Token], Attribute, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Colon)
        .ignore_then(select! { Token::Ident(name) => name })
        .then(
            expr_parser()
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .or_not(),
        )
        .map_with(|(name, args), e| Attribute {
            name,
            args: args.unwrap_or_default(),
            span: token_span(e.span()),
        })
}

/// Parse a statement
///
/// Statements appear in effect blocks (collect, apply, emit).
/// Supported statements:
/// - `let x = expr` - local binding
/// - `path <- expr` - signal assignment
/// - `path <- position, value` - field assignment
/// - `expr` - expression statement
fn stmt_parser<'src>(
    expr: impl Parser<'src, &'src [Token], Expr, extra::Err<Rich<'src, Token>>> + Clone,
) -> impl Parser<'src, &'src [Token], Stmt, extra::Err<Rich<'src, Token>>> + Clone {
    // Let statement: `let x = expr`
    let let_stmt = just(Token::Let)
        .ignore_then(select! { Token::Ident(name) => name })
        .then_ignore(just(Token::Eq))
        .then(expr.clone())
        .map_with(|(name, value), e| Stmt::Let {
            name,
            value,
            span: token_span(e.span()),
        });

    // Assignment statement: `path <- expr` or `path <- position, value`
    let assign_stmt = path_parser()
        .then_ignore(just(Token::LeftArrow))
        .then(expr.clone())
        .then(just(Token::Comma).ignore_then(expr.clone()).or_not())
        .map_with(|((target, first), second), e| {
            let span = token_span(e.span());
            match second {
                Some(value) => Stmt::FieldAssign {
                    target,
                    position: first,
                    value,
                    span,
                },
                None => Stmt::SignalAssign {
                    target,
                    value: first,
                    span,
                },
            }
        });

    // Expression statement
    let expr_stmt = expr.map(Stmt::Expr);

    choice((let_stmt, assign_stmt, expr_stmt))
}

/// Parse block body: expression or statement list
///
/// Per architecture feedback:
/// - Pure phases (resolve, measure, assert) use Expression bodies
/// - Effect phases (collect, apply, emit) use Statement bodies
///
/// Statements must be separated by semicolons.
fn block_body_parser<'src>()
-> impl Parser<'src, &'src [Token], BlockBody, extra::Err<Rich<'src, Token>>> + Clone {
    let expr = expr_parser();

    // Try to parse as statement list first (more specific)
    let stmt_list = stmt_parser(expr.clone())
        .separated_by(just(Token::Semicolon))
        .at_least(1)
        .collect::<Vec<_>>()
        .map(BlockBody::Statements);

    // Fall back to single expression
    let single_expr = expr.map(BlockBody::Expression);

    // Try statement list first, then expression
    choice((stmt_list, single_expr))
}

/// Parse warmup block: `warmup { :iterations(N) iterate { expr } }`
///
/// Warmup blocks contain attributes (iterations, convergence) and an
/// iterate sub-block with the warmup expression.
fn warmup_parser<'src>()
-> impl Parser<'src, &'src [Token], WarmupBlock, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::WarmUp)
        .ignore_then(
            attribute_parser()
                .repeated()
                .collect::<Vec<_>>()
                .then(just(Token::Iterate).ignore_then(
                    expr_parser().delimited_by(just(Token::LBrace), just(Token::RBrace)),
                ))
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(|(attrs, iterate), e| WarmupBlock {
            attrs,
            iterate,
            span: token_span(e.span()),
        })
}

/// Parse when block: `when { condition1; condition2; ... }`
///
/// When blocks contain conditions that must all be true.
/// Conditions are expressions separated by semicolons or newlines.
fn when_parser<'src>()
-> impl Parser<'src, &'src [Token], WhenBlock, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::When)
        .ignore_then(
            expr_parser()
                .separated_by(just(Token::Semicolon))
                .at_least(1)
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(|conditions, e| WhenBlock {
            conditions,
            span: token_span(e.span()),
        })
}

/// Parse observe block for chronicles: `observe { when condition { emit ... } }`
///
/// Observe blocks contain when clauses with associated emit blocks.
/// Example:
/// ```cdsl
/// observe {
///     when signal.diversity < -0.5 {
///         emit event.extinction { severity: 0.8 }
///     }
/// }
/// ```
fn observe_parser<'src>()
-> impl Parser<'src, &'src [Token], ObserveBlock, extra::Err<Rich<'src, Token>>> + Clone {
    // when clause: `when condition { emit_statements }`
    let when_clause = just(Token::When)
        .ignore_then(expr_parser())
        .then(
            stmt_parser(expr_parser())
                .separated_by(just(Token::Semicolon))
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(|(condition, emit_block), e| ObserveWhen {
            condition,
            emit_block,
            span: token_span(e.span()),
        });

    just(Token::Observe)
        .ignore_then(
            when_clause
                .repeated()
                .at_least(1)
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(|when_clauses, e| ObserveBlock {
            when_clauses,
            span: token_span(e.span()),
        })
}

/// Parse transition block: `transition target_era when { conditions }`
///
/// Transitions define when to switch from one era to another.
/// Example:
/// ```cdsl
/// transition stable when {
///     signal.temp < 1000<K>;
///     signal.time > 1e9<s>
/// }
/// ```
fn transition_parser<'src>()
-> impl Parser<'src, &'src [Token], TransitionDecl, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Transition)
        .ignore_then(path_parser())
        .then(when_parser())
        .map_with(|(target, when_block), e| TransitionDecl {
            target,
            conditions: when_block.conditions,
            span: token_span(e.span()),
        })
}

// =============================================================================
// Block Parsers (for role declarations)
// =============================================================================

/// Parse a simple execution block: `block_name { body }`
///
/// Execution blocks contain the logic for different phases.
/// Examples: `resolve { expr }`, `collect { stmts }`, `measure { expr }`
fn execution_block_parser<'src>()
-> impl Parser<'src, &'src [Token], (String, BlockBody), extra::Err<Rich<'src, Token>>> + Clone {
    let block_keyword = select! {
        Token::Resolve => "resolve",
        Token::Collect => "collect",
        Token::Apply => "apply",
        Token::Measure => "measure",
        Token::Assert => "assert",
        Token::Emit => "emit",
    };

    block_keyword
        .then(block_body_parser().delimited_by(just(Token::LBrace), just(Token::RBrace)))
        .map(|(name, body)| (name.to_string(), body))
}

// =============================================================================
// Role Declaration Parsers
// =============================================================================

/// Parse a signal declaration.
///
/// Example:
/// ```cdsl
/// signal temp {
///     : type Scalar<K>
///     : strata(simulation)
///     warmup { :iterations(100) iterate { prev * 0.9 } }
///     resolve { prev + 1 }
/// }
/// ```
fn signal_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Signal)
        .ignore_then(path_parser())
        .then(type_annotation_parser().or_not())
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(warmup_parser().or_not())
        .then(execution_block_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(|((((path, type_expr), _attrs), _warmup), blocks), e| {
            let span = token_span(e.span());
            let mut node = Node::new(path, span, RoleData::Signal, ());
            node.type_expr = type_expr;

            // Convert BlockBody to execution expressions (placeholder - needs proper conversion)
            for (name, body) in blocks {
                // For now, only handle Expression bodies
                // TODO: Handle Statement bodies properly
                match body {
                    BlockBody::Expression(expr) => {
                        node.execution_exprs.push((name, expr));
                    }
                    BlockBody::Statements(_stmts) => {
                        // TODO: Convert statements to expression or handle differently
                        // For now, skip statement blocks
                    }
                }
            }

            // TODO: Apply attributes (extract title, symbol, stratum, etc.)
            // TODO: Handle warmup block

            Declaration::Node(node)
        })
}

/// Parse a field declaration.
///
/// Example:
/// ```cdsl
/// field temp_map {
///     : type Scalar<K>
///     measure { signal.temp }
/// }
/// ```
fn field_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Field)
        .ignore_then(path_parser())
        .then(type_annotation_parser().or_not())
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(execution_block_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(|(((path, type_expr), _attrs), blocks), e| {
            let span = token_span(e.span());
            let mut node = Node::new(
                path,
                span,
                RoleData::Field {
                    reconstruction: None,
                },
                (),
            );
            node.type_expr = type_expr;

            for (name, body) in blocks {
                match body {
                    BlockBody::Expression(expr) => {
                        node.execution_exprs.push((name, expr));
                    }
                    BlockBody::Statements(_) => {}
                }
            }

            Declaration::Node(node)
        })
}

/// Parse an operator declaration.
///
/// Example:
/// ```cdsl
/// operator budget {
///     collect { signal.heat <- radiation - loss; }
/// }
/// ```
fn operator_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Operator)
        .ignore_then(path_parser())
        .then(type_annotation_parser().or_not())
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(execution_block_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(|(((path, type_expr), _attrs), blocks), e| {
            let span = token_span(e.span());
            let mut node = Node::new(path, span, RoleData::Operator, ());
            node.type_expr = type_expr;

            for (name, body) in blocks {
                match body {
                    BlockBody::Expression(expr) => {
                        node.execution_exprs.push((name, expr));
                    }
                    BlockBody::Statements(_) => {}
                }
            }

            Declaration::Node(node)
        })
}

/// Parse an impulse declaration.
///
/// Example:
/// ```cdsl
/// impulse asteroid {
///     : type ImpactEvent
///     apply { signal.energy <- payload.energy; }
/// }
/// ```
fn impulse_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Impulse)
        .ignore_then(path_parser())
        .then(type_annotation_parser().or_not())
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(execution_block_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(|(((path, type_expr), _attrs), blocks), e| {
            let span = token_span(e.span());
            let mut node = Node::new(path, span, RoleData::Impulse { payload: None }, ());
            node.type_expr = type_expr;

            for (name, body) in blocks {
                match body {
                    BlockBody::Expression(expr) => {
                        node.execution_exprs.push((name, expr));
                    }
                    BlockBody::Statements(_) => {}
                }
            }

            Declaration::Node(node)
        })
}

/// Parse a fracture declaration.
///
/// Example:
/// ```cdsl
/// fracture runaway {
///     when { signal.temp > 350<K> }
///     emit { signal.feedback <- 1.5; }
/// }
/// ```
fn fracture_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Fracture)
        .ignore_then(path_parser())
        .then(type_annotation_parser().or_not())
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(when_parser().or_not())
        .then(execution_block_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(|((((path, type_expr), _attrs), _when), blocks), e| {
            let span = token_span(e.span());
            let mut node = Node::new(path, span, RoleData::Fracture, ());
            node.type_expr = type_expr;

            // TODO: Handle when block

            for (name, body) in blocks {
                match body {
                    BlockBody::Expression(expr) => {
                        node.execution_exprs.push((name, expr));
                    }
                    BlockBody::Statements(_) => {}
                }
            }

            Declaration::Node(node)
        })
}

/// Parse a chronicle declaration.
///
/// Example:
/// ```cdsl
/// chronicle extinction {
///     observe {
///         when signal.diversity < -0.5 {
///             emit event.extinction { severity: 0.8 };
///         }
///     }
/// }
/// ```
fn chronicle_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Chronicle)
        .ignore_then(path_parser())
        .then(type_annotation_parser().or_not())
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(observe_parser().or_not())
        .then_ignore(just(Token::RBrace))
        .map_with(|(((path, type_expr), _attrs), _observe), e| {
            let span = token_span(e.span());
            let mut node = Node::new(path, span, RoleData::Chronicle, ());
            node.type_expr = type_expr;

            // TODO: Handle observe block

            Declaration::Node(node)
        })
}

// =============================================================================
// Structural Declaration Parsers
// =============================================================================

/// Parse an entity declaration.
///
/// Example:
/// ```cdsl
/// entity plate {
///     : count(5..50)
/// }
/// ```
fn entity_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Entity)
        .ignore_then(path_parser())
        .then_ignore(just(Token::LBrace))
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(|(path, _attrs), e| {
            let span = token_span(e.span());
            // Convert Path to FoundationPath for EntityId
            let foundation_path = FoundationPath::from_str(&path.to_string());
            let entity = Entity::new(EntityId(foundation_path), path, span);
            // TODO: Extract count from attributes
            Declaration::Entity(entity)
        })
}

/// Parse a member declaration.
///
/// Example:
/// ```cdsl
/// member plate.area {
///     : type Scalar<m2>
///     resolve { prev }
/// }
/// ```
fn member_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Member)
        .ignore_then(path_parser())
        .then(type_annotation_parser().or_not())
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(execution_block_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(|(((full_path, type_expr), _attrs), blocks), e| {
            let span = token_span(e.span());

            // Extract entity ID from path: plate.area -> entity=plate
            // For simplicity, assume entity is all segments except last
            let path_str = full_path.to_string();
            let parts: Vec<&str> = path_str.split('.').collect();
            let entity_path_str = if parts.len() > 1 {
                parts[..parts.len() - 1].join(".")
            } else {
                // No entity prefix, use empty path (will fail validation)
                String::new()
            };

            // Convert to FoundationPath for EntityId
            let foundation_path = FoundationPath::from_str(&entity_path_str);
            let entity_id = EntityId(foundation_path);
            let mut node = Node::new(full_path, span, RoleData::Signal, entity_id);
            node.type_expr = type_expr;

            for (name, body) in blocks {
                match body {
                    BlockBody::Expression(expr) => {
                        node.execution_exprs.push((name, expr));
                    }
                    BlockBody::Statements(_) => {}
                }
            }

            Declaration::Member(node)
        })
}

/// Parse a stratum declaration.
///
/// Example:
/// ```cdsl
/// strata simulation {
///     : stride(1)
///     : title("Simulation")
/// }
/// ```
fn stratum_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Strata)
        .ignore_then(path_parser())
        .then_ignore(just(Token::LBrace))
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(|(path, _attrs), e| {
            let span = token_span(e.span());
            // TODO: Extract stride from attributes, default to 1
            let cadence = 1;
            // Convert Path to FoundationPath for StratumId
            let foundation_path = FoundationPath::from_str(&path.to_string());
            let stratum = Stratum::new(StratumId(foundation_path), path, cadence, span);
            Declaration::Stratum(stratum)
        })
}

/// Parse an era declaration.
///
/// Example:
/// ```cdsl
/// era main {
///     : initial
///     : dt(1<s>)
///     strata { simulation: active }
///     transition stable when { signal.temp < 1000<K> }
/// }
/// ```
fn era_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    // Strata policy entry: `path : active` or `path : gated`
    let strata_entry = path_parser()
        .then_ignore(just(Token::Colon))
        .then(select! {
            Token::Ident(s) if s == "active" => StratumState::Active,
            Token::Ident(s) if s == "gated" => StratumState::Gated,
        })
        .map_with(|(path, state), e| StratumPolicyEntry {
            stratum: path,
            state,
            stride: None,
            span: token_span(e.span()),
        });

    // Strata block: `strata { entries }`
    let strata_block = just(Token::Strata).ignore_then(
        strata_entry
            .separated_by(just(Token::Semicolon).or_not())
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace)),
    );

    just(Token::Era)
        .ignore_then(path_parser())
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(strata_block.or_not())
        .then(transition_parser().repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(|(((path, attrs), strata_policy), transitions), e| {
            let span = token_span(e.span());

            // TODO: Extract dt, is_initial, is_terminal from attributes
            let era = EraDecl {
                path,
                span,
                doc: None,
                dt: None,
                is_initial: attrs.iter().any(|a| a.name == "initial"),
                is_terminal: attrs.iter().any(|a| a.name == "terminal"),
                strata_policy: strata_policy.unwrap_or_default(),
                transitions,
            };

            Declaration::Era(era)
        })
}

/// Parse a type declaration.
///
/// Example:
/// ```cdsl
/// type ImpactEvent {
///     mass: Scalar<kg>
///     velocity: Vec3<m/s>
/// }
/// ```
fn type_decl_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    // Field: `name : TypeExpr`
    let field = select! { Token::Ident(name) => name }
        .then_ignore(just(Token::Colon))
        .then(just(Token::Type).ignore_then(type_expr_parser()))
        .map_with(|(name, type_expr), e| TypeField {
            name,
            type_expr,
            span: token_span(e.span()),
        });

    just(Token::Type)
        .ignore_then(select! { Token::Ident(name) => name })
        .then_ignore(just(Token::LBrace))
        .then(
            field
                .separated_by(just(Token::Semicolon).or_not())
                .collect::<Vec<_>>(),
        )
        .then_ignore(just(Token::RBrace))
        .map_with(|(name, fields), e| {
            Declaration::Type(TypeDecl {
                name,
                fields,
                span: token_span(e.span()),
                doc: None,
            })
        })
}

/// Parse a world block.
///
/// Example:
/// ```cdsl
/// world terra {
///     : title("Terra")
///     : version("1.0.0")
///     warmup {
///         :converged(maths.max_delta(signals) < 1e-6)
///         :max_iterations(1000)
///         :on_timeout(fail)
///     }
/// }
/// ```
fn world_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    // Warmup policy block
    let warmup_policy = just(Token::WarmUp)
        .ignore_then(
            attribute_parser()
                .repeated()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(|attrs, e| {
            // TODO: Extract converged expression, max_iterations, on_timeout from attributes
            // For now, create placeholder
            WarmupPolicy {
                converged: Expr::new(ExprKind::BoolLiteral(true), token_span(e.span())),
                max_iterations: 1000,
                on_timeout: WarmupTimeout::Fail,
                span: token_span(e.span()),
            }
        });

    just(Token::World)
        .ignore_then(path_parser())
        .then_ignore(just(Token::LBrace))
        .then(attribute_parser().repeated().collect::<Vec<_>>())
        .then(warmup_policy.or_not())
        .then_ignore(just(Token::RBrace))
        .map_with(|((path, attrs), warmup), e| {
            // TODO: Extract title, version from attributes
            Declaration::World(WorldDecl {
                path,
                span: token_span(e.span()),
                title: None,
                version: None,
                warmup,
                doc: None,
            })
        })
}

/// Parse a const block.
///
/// Example:
/// ```cdsl
/// const {
///     physics.gravity: 9.81
/// }
/// ```
fn const_block_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    // Entry: `path : expr`
    let entry = path_parser()
        .then_ignore(just(Token::Colon))
        .then(just(Token::Type).ignore_then(type_expr_parser()))
        .then_ignore(just(Token::Eq))
        .then(expr_parser())
        .map_with(|((path, type_expr), value), e| ConstEntry {
            path,
            value,
            type_expr,
            span: token_span(e.span()),
            doc: None,
        });

    just(Token::Const)
        .ignore_then(
            entry
                .separated_by(just(Token::Semicolon).or_not())
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(Declaration::Const)
}

/// Parse a config block.
///
/// Example:
/// ```cdsl
/// config {
///     initial_temp: 5500<K>
/// }
/// ```
fn config_block_parser<'src>()
-> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    // Entry: `path : type TypeExpr = expr` or `path : type TypeExpr`
    let entry = path_parser()
        .then_ignore(just(Token::Colon))
        .then(just(Token::Type).ignore_then(type_expr_parser()))
        .then(just(Token::Eq).ignore_then(expr_parser()).or_not())
        .map_with(|((path, type_expr), default), e| ConfigEntry {
            path,
            default,
            type_expr,
            span: token_span(e.span()),
            doc: None,
        });

    just(Token::Config)
        .ignore_then(
            entry
                .separated_by(just(Token::Semicolon).or_not())
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map(Declaration::Config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Token;
    use logos::Logos;

    /// Helper to lex and parse a source string.
    ///
    /// # Parameters
    ///
    /// - `source`: CDSL source code
    ///
    /// # Returns
    ///
    /// Parsed expression.
    ///
    /// # Panics
    ///
    /// Panics if lexing fails or if parsing produces errors. Test inputs
    /// should always be valid.
    fn lex_and_parse(source: &str) -> Expr {
        let tokens: Vec<_> = Token::lexer(source)
            .collect::<Result<Vec<_>, _>>()
            .expect("lexer should not produce errors for test inputs");
        parse_expr(&tokens).unwrap()
    }

    /// Helper to lex and parse a source string, returning Result.
    ///
    /// Used in tests to verify error paths. This helper intentionally panics
    /// on lexer errors (which should not occur for valid test inputs) and
    /// returns parse errors as a boolean (true = has errors).
    ///
    /// # Parameters
    ///
    /// - `source`: CDSL source code
    ///
    /// # Returns
    ///
    /// - `Ok(Expr)` if parsing succeeded with no errors
    /// - `Err(true)` if parsing failed (has errors)
    ///
    /// # Errors
    ///
    /// Returns `Err(true)` if the parser produces any errors. The boolean
    /// value is always `true` when present, indicating that parse errors
    /// occurred. Test helpers don't need full Rich<Token> error details,
    /// just success/failure indication.
    ///
    /// # Panics
    ///
    /// Panics if lexer produces any errors. This is intentional for test
    /// clarity - if a test wants to verify parse errors, the input must
    /// at least tokenize successfully.
    ///
    /// # Examples
    ///
    /// ```rust,ignore
    /// // Valid parse
    /// let expr = lex_and_parse_result("let x = 1 in x").unwrap();
    ///
    /// // Parse error (but lexes fine)
    /// assert!(lex_and_parse_result("let = 1").is_err());
    /// ```
    fn lex_and_parse_result(source: &str) -> Result<Expr, bool> {
        // Collect lexer results, panic on lexer errors (test inputs should always tokenize)
        let tokens: Vec<_> = Token::lexer(source)
            .collect::<Result<Vec<_>, _>>()
            .expect("lexer should not produce errors for test inputs");

        let result = parse_expr(&tokens);

        // Return parse result
        if result.output().is_some() && result.errors().len() == 0 {
            Ok(result.output().unwrap().clone())
        } else {
            // Return Err(true) to indicate parse errors exist
            // (test helpers don't need full Rich<Token> error details)
            Err(true)
        }
    }

    #[test]
    fn test_parse_literal() {
        let expr = lex_and_parse("42.0");
        match expr.kind {
            ExprKind::Literal { value, unit } => {
                assert_eq!(value, 42.0);
                assert_eq!(unit, None);
            }
            _ => panic!("expected literal, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_parse_bool_literal() {
        let expr = lex_and_parse("true");
        assert!(matches!(expr.kind, ExprKind::BoolLiteral(true)));

        let expr = lex_and_parse("false");
        assert!(matches!(expr.kind, ExprKind::BoolLiteral(false)));
    }

    #[test]
    fn test_parse_binary_add() {
        let expr = lex_and_parse("10 + 20");
        match expr.kind {
            ExprKind::Binary { op, left, right } => {
                assert_eq!(op, BinaryOp::Add);
                assert!(matches!(left.kind, ExprKind::Literal { value: 10.0, .. }));
                assert!(matches!(right.kind, ExprKind::Literal { value: 20.0, .. }));
            }
            _ => panic!("expected binary, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_parse_unit_meters() {
        // Units are token sequences: Lt, Ident("m"), Gt
        let expr = lex_and_parse("100.0<m>");
        match expr.kind {
            ExprKind::Literal { value, unit } => {
                assert_eq!(value, 100.0);
                assert_eq!(unit, Some(UnitExpr::Base("m".to_string())));
            }
            _ => panic!("expected literal with unit, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_parse_unit_velocity() {
        // <m/s> → Lt, Ident("m"), Slash, Ident("s"), Gt
        let expr = lex_and_parse("10.0<m/s>");
        match expr.kind {
            ExprKind::Literal { value, unit } => {
                assert_eq!(value, 10.0);
                match unit {
                    Some(UnitExpr::Divide(num, denom)) => {
                        assert_eq!(*num, UnitExpr::Base("m".to_string()));
                        assert_eq!(*denom, UnitExpr::Base("s".to_string()));
                    }
                    _ => panic!("expected divide unit, got {:?}", unit),
                }
            }
            _ => panic!("expected literal with unit, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_parse_unary_neg() {
        let expr = lex_and_parse("-42.0");
        match expr.kind {
            ExprKind::Unary { op, operand } => {
                assert_eq!(op, UnaryOp::Neg);
                assert!(matches!(
                    operand.kind,
                    ExprKind::Literal { value: 42.0, .. }
                ));
            }
            _ => panic!("expected unary, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_precedence_mul_over_add() {
        let expr = lex_and_parse("1 + 2 * 3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 1.0, .. }));
                assert!(matches!(
                    right.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
            }
            _ => panic!("expected add with mul on right, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_right_associative() {
        let expr = lex_and_parse("2 ^ 3 ^ 4");
        // Should parse as 2 ^ (3 ^ 4) for right-associativity
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                assert!(matches!(
                    right.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("expected pow chain, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_single() {
        // Single power should just work
        let expr = lex_and_parse("2 ^ 3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                assert!(matches!(right.kind, ExprKind::Literal { value: 3.0, .. }));
            }
            _ => panic!("expected single pow, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_vs_mul_precedence() {
        // Power should bind tighter than multiplication
        // 2 * 3 ^ 4 should parse as 2 * (3 ^ 4)
        let expr = lex_and_parse("2 * 3 ^ 4");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                assert!(matches!(
                    right.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("expected mul with pow on right, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_vs_mul_precedence_left() {
        // Power on left: 2 ^ 3 * 4 should parse as (2 ^ 3) * 4
        let expr = lex_and_parse("2 ^ 3 * 4");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    left.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
                assert!(matches!(right.kind, ExprKind::Literal { value: 4.0, .. }));
            }
            _ => panic!("expected mul with pow on left, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_parentheses_override() {
        // Parentheses should override right-associativity
        // (2 ^ 3) ^ 4 should parse as left-associative Binary(Binary(2, 3), 4)
        let expr = lex_and_parse("(2 ^ 3) ^ 4");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(
                    left.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
                assert!(matches!(right.kind, ExprKind::Literal { value: 4.0, .. }));
            }
            _ => panic!("expected pow with pow on left, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_with_unary_right() {
        // Power with unary on right: 2 ^ -3
        let expr = lex_and_parse("2 ^ -3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                assert!(matches!(
                    right.kind,
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
            }
            _ => panic!("expected pow with unary right, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_with_unary_left() {
        // Power with unary on left: -2 ^ 3
        // Unary binds to its operand first, so this is (-2) ^ 3
        let expr = lex_and_parse("-2 ^ 3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(
                    left.kind,
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
                assert!(matches!(right.kind, ExprKind::Literal { value: 3.0, .. }));
            }
            _ => panic!("expected pow with unary left, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_longer_chain() {
        // Longer chain: 2 ^ 3 ^ 4 ^ 5 should be 2 ^ (3 ^ (4 ^ 5))
        let expr = lex_and_parse("2 ^ 3 ^ 4 ^ 5");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                // right should be 3 ^ (4 ^ 5)
                match right.kind {
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        left: inner_left,
                        right: inner_right,
                    } => {
                        assert!(matches!(
                            inner_left.kind,
                            ExprKind::Literal { value: 3.0, .. }
                        ));
                        // inner_right should be 4 ^ 5 with full leaves verified
                        match inner_right.kind {
                            ExprKind::Binary {
                                op: BinaryOp::Pow,
                                left: leaf_left,
                                right: leaf_right,
                            } => {
                                assert!(matches!(
                                    leaf_left.kind,
                                    ExprKind::Literal { value: 4.0, .. }
                                ));
                                assert!(matches!(
                                    leaf_right.kind,
                                    ExprKind::Literal { value: 5.0, .. }
                                ));
                            }
                            _ => panic!("expected innermost pow, got {:?}", inner_right.kind),
                        }
                    }
                    _ => panic!("expected nested pow, got {:?}", right.kind),
                }
            }
            _ => panic!("expected pow chain, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_parenthesized_right() {
        // Explicit parentheses on right: 2 ^ (3 ^ 4)
        let expr = lex_and_parse("2 ^ (3 ^ 4)");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Pow,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 2.0, .. }));
                match right.kind {
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        left: rleft,
                        right: rright,
                    } => {
                        assert!(matches!(rleft.kind, ExprKind::Literal { value: 3.0, .. }));
                        assert!(matches!(rright.kind, ExprKind::Literal { value: 4.0, .. }));
                    }
                    _ => panic!("expected parenthesized pow on right, got {:?}", right.kind),
                }
            }
            _ => panic!("expected pow with parenthesized right, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_vs_comparison() {
        // Power should bind tighter than comparison: 1 < 2 ^ 3 is 1 < (2 ^ 3)
        let expr = lex_and_parse("1 < 2 ^ 3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Lt,
                left,
                right,
            } => {
                assert!(matches!(left.kind, ExprKind::Literal { value: 1.0, .. }));
                assert!(matches!(
                    right.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Pow,
                        ..
                    }
                ));
            }
            _ => panic!("expected comparison with pow on right, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_power_missing_rhs() {
        // Missing right operand should fail
        let tokens: Vec<_> = Token::lexer("2 ^").collect::<Result<Vec<_>, _>>().unwrap();
        assert!(parse_expr(&tokens).into_result().is_err());
    }

    #[test]
    fn test_power_missing_lhs() {
        // Missing left operand should fail
        let tokens: Vec<_> = Token::lexer("^ 2").collect::<Result<Vec<_>, _>>().unwrap();
        assert!(parse_expr(&tokens).into_result().is_err());
    }

    #[test]
    fn test_power_double_operator() {
        // Double operator should fail
        let tokens: Vec<_> = Token::lexer("2 ^ ^ 3")
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert!(parse_expr(&tokens).into_result().is_err());
    }

    #[test]
    fn test_rich_error_contains_span() {
        // Verify Rich errors contain span information
        let tokens: Vec<_> = Token::lexer("2 ^").collect::<Result<Vec<_>, _>>().unwrap();
        let result = parse_expr(&tokens);
        match result.into_result() {
            Err(errors) => {
                assert!(!errors.is_empty());
                // Rich errors should have spans
                let first_error = &errors[0];
                let span = first_error.span();
                // Span should be non-empty (covers at least one token)
                assert!(span.end >= span.start);
            }
            Ok(_) => panic!("expected parse error for incomplete expression"),
        }
    }

    #[test]
    fn test_rich_error_has_context() {
        // Verify Rich errors provide context about what was expected
        let tokens: Vec<_> = Token::lexer("^ 2").collect::<Result<Vec<_>, _>>().unwrap();
        let result = parse_expr(&tokens);
        match result.into_result() {
            Err(errors) => {
                assert!(!errors.is_empty());
                // Rich errors should provide some context (found/expected)
                let first_error = &errors[0];
                // Just verify the error exists and has a reason
                let _reason = first_error.reason();
                // We got here, so the error has diagnostic info
            }
            Ok(_) => panic!("expected parse error for malformed expression"),
        }
    }

    #[test]
    fn test_postfix_call_then_field() {
        let expr = lex_and_parse("f(1).x");
        match expr.kind {
            ExprKind::FieldAccess { object, field } => {
                assert_eq!(field, "x");
                assert!(matches!(object.kind, ExprKind::Call { .. }));
            }
            _ => panic!("expected field access on call, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_vector_empty() {
        let expr = lex_and_parse("[]");
        match expr.kind {
            ExprKind::Vector(elements) => {
                assert_eq!(elements.len(), 0);
            }
            _ => panic!("expected empty vector, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_vector_trailing_comma() {
        let expr = lex_and_parse("[1.0, 2.0, 3.0,]");
        match expr.kind {
            ExprKind::Vector(elements) => {
                assert_eq!(elements.len(), 3);
            }
            _ => panic!("expected vector, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_unary_chaining() {
        let expr = lex_and_parse("not -42.0");
        match expr.kind {
            ExprKind::Unary {
                op: UnaryOp::Not,
                operand,
            } => {
                assert!(matches!(
                    operand.kind,
                    ExprKind::Unary {
                        op: UnaryOp::Neg,
                        ..
                    }
                ));
            }
            _ => panic!("expected unary chain, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_parenthesis_override() {
        let expr = lex_and_parse("(1 + 2) * 3");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(
                    left.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                assert!(matches!(right.kind, ExprKind::Literal { value: 3.0, .. }));
            }
            _ => panic!("expected mul with add on left, got {:?}", expr.kind),
        }
    }

    // === Let Expression Tests ===

    #[test]
    fn test_let_basic() {
        // let x = 10 in x + 1
        let expr = lex_and_parse("let x = 10 in x + 1");
        match expr.kind {
            ExprKind::Let { name, value, body } => {
                assert_eq!(name, "x");
                assert!(matches!(value.kind, ExprKind::Literal { value: 10.0, .. }));
                assert!(matches!(
                    body.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("expected let expression, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_let_with_expression_value() {
        // let x = 2 * 3 in x + 1
        let expr = lex_and_parse("let x = 2 * 3 in x + 1");
        match expr.kind {
            ExprKind::Let { name, value, body } => {
                assert_eq!(name, "x");
                assert!(matches!(
                    value.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert!(matches!(
                    body.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("expected let expression, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_let_nested() {
        // let x = 10 in let y = 20 in x + y
        let expr = lex_and_parse("let x = 10 in let y = 20 in x + y");
        match expr.kind {
            ExprKind::Let { name, value, body } => {
                assert_eq!(name, "x");
                assert!(matches!(value.kind, ExprKind::Literal { value: 10.0, .. }));
                // Body should be another let
                match body.kind {
                    ExprKind::Let {
                        name: inner_name,
                        value: inner_value,
                        body: inner_body,
                    } => {
                        assert_eq!(inner_name, "y");
                        assert!(matches!(
                            inner_value.kind,
                            ExprKind::Literal { value: 20.0, .. }
                        ));
                        assert!(matches!(
                            inner_body.kind,
                            ExprKind::Binary {
                                op: BinaryOp::Add,
                                ..
                            }
                        ));
                    }
                    _ => panic!("expected nested let in body, got {:?}", body.kind),
                }
            }
            _ => panic!("expected let expression, got {:?}", expr.kind),
        }
    }

    // === If Expression Tests ===

    #[test]
    fn test_if_basic() {
        // if true { 1 } else { 2 }
        let expr = lex_and_parse("if true { 1 } else { 2 }");
        match expr.kind {
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                assert!(matches!(condition.kind, ExprKind::BoolLiteral(true)));
                assert!(matches!(
                    then_branch.kind,
                    ExprKind::Literal { value: 1.0, .. }
                ));
                assert!(matches!(
                    else_branch.kind,
                    ExprKind::Literal { value: 2.0, .. }
                ));
            }
            _ => panic!("expected if expression, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_if_with_comparison() {
        // if x < 10 { x } else { 10 }
        let expr = lex_and_parse("if x < 10 { x } else { 10 }");
        match expr.kind {
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                assert!(matches!(
                    condition.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Lt,
                        ..
                    }
                ));
                assert!(matches!(then_branch.kind, ExprKind::Local(_)));
                assert!(matches!(
                    else_branch.kind,
                    ExprKind::Literal { value: 10.0, .. }
                ));
            }
            _ => panic!("expected if expression, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_if_nested() {
        // if x { if y { 1 } else { 2 } } else { 3 }
        let expr = lex_and_parse("if x { if y { 1 } else { 2 } } else { 3 }");
        match expr.kind {
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                assert!(matches!(condition.kind, ExprKind::Local(_)));
                // Then branch should be another if
                match then_branch.kind {
                    ExprKind::If {
                        condition: inner_cond,
                        then_branch: inner_then,
                        else_branch: inner_else,
                    } => {
                        assert!(matches!(inner_cond.kind, ExprKind::Local(_)));
                        assert!(matches!(
                            inner_then.kind,
                            ExprKind::Literal { value: 1.0, .. }
                        ));
                        assert!(matches!(
                            inner_else.kind,
                            ExprKind::Literal { value: 2.0, .. }
                        ));
                    }
                    _ => panic!(
                        "expected nested if in then branch, got {:?}",
                        then_branch.kind
                    ),
                }
                assert!(matches!(
                    else_branch.kind,
                    ExprKind::Literal { value: 3.0, .. }
                ));
            }
            _ => panic!("expected if expression, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_if_with_expressions() {
        // if x + 1 < 10 { x * 2 } else { x / 2 }
        let expr = lex_and_parse("if x + 1 < 10 { x * 2 } else { x / 2 }");
        match expr.kind {
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                assert!(matches!(
                    condition.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Lt,
                        ..
                    }
                ));
                assert!(matches!(
                    then_branch.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
                assert!(matches!(
                    else_branch.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Div,
                        ..
                    }
                ));
            }
            _ => panic!("expected if expression, got {:?}", expr.kind),
        }
    }

    // === Let/If Interaction Tests ===

    #[test]
    fn test_let_inside_if() {
        // if x { let y = 10 in y } else { 0 }
        let expr = lex_and_parse("if x { let y = 10 in y } else { 0 }");
        match expr.kind {
            ExprKind::If {
                condition,
                then_branch,
                else_branch,
            } => {
                assert!(matches!(condition.kind, ExprKind::Local(_)));
                assert!(matches!(then_branch.kind, ExprKind::Let { .. }));
                assert!(matches!(
                    else_branch.kind,
                    ExprKind::Literal { value: 0.0, .. }
                ));
            }
            _ => panic!("expected if expression, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_if_inside_let() {
        // let x = if true { 1 } else { 2 } in x + 1
        let expr = lex_and_parse("let x = if true { 1 } else { 2 } in x + 1");
        match expr.kind {
            ExprKind::Let { name, value, body } => {
                assert_eq!(name, "x");
                assert!(matches!(value.kind, ExprKind::If { .. }));
                assert!(matches!(
                    body.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("expected let expression, got {:?}", expr.kind),
        }
    }

    #[test]
    fn test_let_if_precedence() {
        // let and if should be lower precedence than operators
        // let x = 1 + 2 in x * 3  (not let x = 1 + (2 in x) * 3)
        let expr = lex_and_parse("let x = 1 + 2 in x * 3");
        match expr.kind {
            ExprKind::Let { name, value, body } => {
                assert_eq!(name, "x");
                // Value should be 1 + 2, not just 1
                assert!(matches!(
                    value.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
                // Body should be x * 3, not just x
                assert!(matches!(
                    body.kind,
                    ExprKind::Binary {
                        op: BinaryOp::Mul,
                        ..
                    }
                ));
            }
            _ => panic!("expected let expression, got {:?}", expr.kind),
        }
    }

    // === Error Path Tests ===

    #[test]
    fn test_let_missing_eq() {
        // let x 10 in x  (missing =)
        let result = lex_and_parse_result("let x 10 in x");
        assert!(result.is_err(), "expected error for missing =");
    }

    #[test]
    fn test_let_missing_in() {
        // let x = 10 x  (missing 'in')
        let result = lex_and_parse_result("let x = 10 x");
        assert!(result.is_err(), "expected error for missing 'in'");
    }

    #[test]
    fn test_let_missing_identifier() {
        // let = 1 in 2  (missing identifier)
        let result = lex_and_parse_result("let = 1 in 2");
        assert!(result.is_err(), "expected error for missing identifier");
    }

    #[test]
    fn test_let_missing_value() {
        // let x = in x  (missing value expression)
        let result = lex_and_parse_result("let x = in x");
        assert!(result.is_err(), "expected error for missing value");
    }

    #[test]
    fn test_let_missing_body() {
        // let x = 1 in  (missing body expression)
        let result = lex_and_parse_result("let x = 1 in");
        assert!(result.is_err(), "expected error for missing body");
    }

    #[test]
    fn test_if_missing_else() {
        // if true { 1 }  (missing else)
        let result = lex_and_parse_result("if true { 1 }");
        assert!(result.is_err(), "expected error for missing else");
    }

    #[test]
    fn test_if_missing_braces() {
        // if true 1 else 2  (missing braces)
        let result = lex_and_parse_result("if true 1 else 2");
        assert!(result.is_err(), "expected error for missing braces");
    }

    #[test]
    fn test_if_missing_condition() {
        // if { 1 } else { 2 }  (missing condition)
        let result = lex_and_parse_result("if { 1 } else { 2 }");
        assert!(result.is_err(), "expected error for missing condition");
    }

    #[test]
    fn test_if_empty_then_branch() {
        // if true { } else { 2 }  (empty then branch)
        let result = lex_and_parse_result("if true { } else { 2 }");
        assert!(result.is_err(), "expected error for empty then branch");
    }

    #[test]
    fn test_if_empty_else_branch() {
        // if true { 1 } else { }  (empty else branch)
        let result = lex_and_parse_result("if true { 1 } else { }");
        assert!(result.is_err(), "expected error for empty else branch");
    }

    #[test]
    fn test_if_missing_else_body() {
        // if true { 1 } else  (missing else body)
        let result = lex_and_parse_result("if true { 1 } else");
        assert!(result.is_err(), "expected error for missing else body");
    }

    #[test]
    fn test_if_as_operand_requires_parentheses() {
        // if expression cannot be used directly as operand without parentheses
        // 1 + if true { 2 } else { 3 } should fail
        let result = lex_and_parse_result("1 + if true { 2 } else { 3 }");
        assert!(
            result.is_err(),
            "expected error: if cannot be used as operand without parentheses"
        );
    }

    #[test]
    fn test_if_with_parentheses_as_operand() {
        // if expression CAN be used as operand when parenthesized
        // 1 + (if true { 2 } else { 3 }) should parse
        let expr = lex_and_parse("1 + (if true { 2 } else { 3 })");
        match expr.kind {
            ExprKind::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                // Left should be 1
                assert!(matches!(left.kind, ExprKind::Literal { value: 1.0, .. }));
                // Right should be if expression
                assert!(matches!(right.kind, ExprKind::If { .. }));
            }
            _ => panic!(
                "expected Add with parenthesized if on right, got {:?}",
                expr.kind
            ),
        }
    }

    // ========================================================================
    // Declaration Parser Tests
    // ========================================================================

    /// Helper to lex and parse declarations
    fn lex_and_parse_decl(source: &str) -> Vec<Declaration> {
        let tokens: Vec<_> = Token::lexer(source)
            .collect::<Result<Vec<_>, _>>()
            .expect("lexer should not produce errors for test inputs");
        parse_declarations(&tokens)
            .into_result()
            .expect("parser failed")
    }

    /// Helper that returns Result for error testing
    fn lex_and_parse_decl_result(source: &str) -> Result<Vec<Declaration>, String> {
        let tokens: Vec<_> = Token::lexer(source)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "lexer error".to_string())?;
        parse_declarations(&tokens)
            .into_result()
            .map_err(|e| format!("parser error: {:?}", e))
    }

    // ------------------------------------------------------------------------
    // Core Parser Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_path_simple() {
        let source = "entity Plate {}";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Entity(entity) => {
                assert_eq!(entity.path.to_string(), "Plate");
            }
            _ => panic!("expected entity declaration"),
        }
    }

    #[test]
    fn test_path_dotted() {
        let source = "signal terra.temperature : type Scalar<K> { resolve { 273.15 } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                assert_eq!(node.path.to_string(), "terra.temperature");
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_type_annotation_with_type_keyword() {
        let source = "signal temp : type Scalar<K> { resolve { 0 } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                assert!(node.type_expr.is_some());
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_attribute_simple() {
        let source = "signal temp : initial { resolve { 0 } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(_node) => {
                // Attributes parsed but not yet attached to node
                // Just verify it parses successfully
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_attribute_with_args() {
        let source = "signal temp : stratum(physics) { resolve { 0 } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(_node) => {
                // Attributes parsed but not yet attached to node
                // Just verify it parses successfully
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_stmt_let_binding() {
        let source = "operator op { apply { let x = 10; x } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        // Successfully parsed
    }

    #[test]
    fn test_stmt_signal_assign() {
        let source = "operator op { apply { temperature <- 273.15 } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        // Successfully parsed
    }

    #[test]
    fn test_stmt_field_assign() {
        let source = "field vis { measure { temperature <- 273.15 } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        // Successfully parsed
    }

    #[test]
    fn test_block_body_expression() {
        let source = "signal temp : type Scalar<K> { resolve { 273.15 } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => match &node.role {
                RoleData::Signal => {
                    // Signal role confirmed
                    // resolve block is in execution_exprs
                }
                _ => panic!("expected signal role"),
            },
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_block_body_statements() {
        let source = "operator op { apply { let x = 10; let y = 20; x + y } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        // Successfully parsed
    }

    // ------------------------------------------------------------------------
    // Signal Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_signal_basic() {
        let source = "signal temperature : type Scalar<K> { resolve { 273.15 } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                assert_eq!(node.path.to_string(), "temperature");
                assert!(node.type_expr.is_some());
                match &node.role {
                    RoleData::Signal => {
                        // Signal role confirmed
                        // resolve block is in execution_exprs
                    }
                    _ => panic!("expected signal role"),
                }
            }
            _ => panic!("expected signal declaration"),
        }
    }

    // TODO: Add test for signal with doc comment once doc comment parsing is implemented
    // #[test]
    // fn test_signal_with_doc_comment() {
    //     let source = r#"
    // /// Surface temperature in Kelvin
    // signal temperature : type Scalar<K> { resolve { 273.15 } }
    // "#;
    //     let decls = lex_and_parse_decl(source);
    //     assert_eq!(decls.len(), 1);
    //     match &decls[0] {
    //         Declaration::Node(node) => {
    //             assert!(node.doc.is_some());
    //         }
    //         _ => panic!("expected signal declaration"),
    //     }
    // }

    #[test]
    fn test_signal_with_warmup() {
        let source = r#"
signal temperature : type Scalar<K> {
    warmup {
        :iterations(100)
        iterate { prev + 0.1 }
    }
    resolve { prev + 1.0 }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        // TODO: Verify warmup block is attached to signal
    }

    #[test]
    fn test_signal_with_multiple_attributes() {
        let source = r#"
signal temp 
    : type Scalar<K> 
    : initial 
    : stratum(physics) 
{ 
    resolve { 273.15 } 
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(_node) => {
                // Attributes parsed but not yet attached to node
                // Just verify it parses successfully
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_signal_with_collect_and_resolve() {
        let source = r#"
signal force : type Vector<3, N> {
    collect { gravity + wind }
    resolve { sum(inputs) }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => match &node.role {
                RoleData::Signal => {
                    // Signal role confirmed
                    // collect and resolve blocks are in execution_exprs
                }
                _ => panic!("expected signal role"),
            },
            _ => panic!("expected signal declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Field Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_field_basic() {
        let source = "field temperature : type Scalar<K> { measure { 273.15 } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                assert_eq!(node.path.to_string(), "temperature");
                match &node.role {
                    RoleData::Field { .. } => {
                        // Field role confirmed
                        // measure block is in execution_exprs, not in RoleData
                    }
                    _ => panic!("expected field role"),
                }
            }
            _ => panic!("expected field declaration"),
        }
    }

    #[test]
    fn test_field_with_reconstruction() {
        let source = r#"
field elevation 
    : type Scalar<m> 
    : reconstruction(idw)
    : samples(1000)
{
    measure { sample_elevation(location) }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(_node) => {
                // Attributes parsed but not yet attached to node
                // Just verify it parses successfully
            }
            _ => panic!("expected field declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Operator Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_operator_basic() {
        let source = r#"
operator apply_gravity {
    apply { velocity <- velocity + gravity * dt }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                match &node.role {
                    RoleData::Operator => {
                        // Operator role confirmed
                        // apply block is in execution_exprs
                    }
                    _ => panic!("expected operator role"),
                }
            }
            _ => panic!("expected operator declaration"),
        }
    }

    #[test]
    fn test_operator_with_collect_and_apply() {
        let source = r#"
operator accumulate_forces {
    collect { total_force + external_forces }
    apply { acceleration <- total_force / mass }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                match &node.role {
                    RoleData::Operator => {
                        // Operator role confirmed
                        // collect and apply blocks in execution_exprs
                    }
                    _ => panic!("expected operator role"),
                }
            }
            _ => panic!("expected operator declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Impulse Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_impulse_basic() {
        let source = r#"
impulse external_force : type Vector<3, N> {
    apply { force <- force + value }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                match &node.role {
                    RoleData::Impulse { .. } => {
                        // Impulse role confirmed
                        // apply block in execution_exprs
                    }
                    _ => panic!("expected impulse role"),
                }
            }
            _ => panic!("expected impulse declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Fracture Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_fracture_basic() {
        let source = r#"
fracture stress_exceeded {
    when { stress > yield_strength }
    emit { split_entity(self) }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                match &node.role {
                    RoleData::Fracture => {
                        // Fracture role confirmed
                        // when/emit blocks in execution_exprs
                    }
                    _ => panic!("expected fracture role"),
                }
            }
            _ => panic!("expected fracture declaration"),
        }
    }

    #[test]
    fn test_fracture_with_multiple_conditions() {
        let source = r#"
fracture boundary_collision {
    when { temperature > 1000 and pressure < 0.1 }
    emit { destroy(self) }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        // Successfully parsed
    }

    // ------------------------------------------------------------------------
    // Chronicle Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_chronicle_basic() {
        let source = r#"
chronicle plate_lifecycle {
    observe {
        when created { emit_event() }
    }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                match &node.role {
                    RoleData::Chronicle => {
                        // Chronicle role confirmed
                        // observe block parsed
                    }
                    _ => panic!("expected chronicle role"),
                }
            }
            _ => panic!("expected chronicle declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Entity Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_entity_basic() {
        let source = "entity Plate {}";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Entity(entity) => {
                assert_eq!(entity.path.to_string(), "Plate");
            }
            _ => panic!("expected entity declaration"),
        }
    }

    // TODO: Add test for entity with doc comment once doc comment parsing is implemented
    // #[test]
    // fn test_entity_with_doc_comment() {
    //     let source = r#"
    // /// Tectonic plate entity
    // entity Plate {}
    // "#;
    //     let decls = lex_and_parse_decl(source);
    //     assert_eq!(decls.len(), 1);
    //     match &decls[0] {
    //         Declaration::Entity(entity) => {
    //             assert!(entity.doc.is_some());
    //         }
    //         _ => panic!("expected entity declaration"),
    //     }
    // }

    // ------------------------------------------------------------------------
    // Member Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_member_basic() {
        let source = r#"
member Plate.velocity : type Vector<3, m/s> {
    resolve { prev + acceleration * dt }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Member(node) => {
                // Full path is Plate.velocity
                assert_eq!(node.path.to_string(), "Plate.velocity");
                // Entity ID should be Plate
                assert_eq!(node.index.0.to_string(), "Plate");
            }
            _ => panic!("expected member declaration"),
        }
    }

    #[test]
    fn test_member_with_stratum() {
        let source = r#"
member Plate.stress 
    : type Scalar<Pa> 
    : stratum(physics) 
{
    resolve { calculate_stress() }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Member(node) => {
                // Attributes are not yet extracted in parser
                // Just verify it parsed successfully
                assert_eq!(node.path.to_string(), "Plate.stress");
            }
            _ => panic!("expected member declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Stratum Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_stratum_basic() {
        let source = "strata physics {}";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Stratum(stratum) => {
                assert_eq!(stratum.path.to_string(), "physics");
            }
            _ => panic!("expected stratum declaration"),
        }
    }

    // TODO: Add test for stratum with doc comment once doc comment parsing is implemented
    // #[test]
    // fn test_stratum_with_doc_comment() {
    //     let source = r#"
    // /// Physics simulation stratum
    // stratum physics {}
    // "#;
    //     let decls = lex_and_parse_decl(source);
    //     assert_eq!(decls.len(), 1);
    //     match &decls[0] {
    //         Declaration::Stratum(stratum) => {
    //             assert!(stratum.doc.is_some());
    //         }
    //         _ => panic!("expected stratum declaration"),
    //     }
    // }

    // ------------------------------------------------------------------------
    // Era Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_era_basic() {
        let source = r#"
era simulation 
    : timestep(1.0)
    : is_initial
{
    strata {
        physics: active
    }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Era(era) => {
                assert_eq!(era.path.to_string(), "simulation");
                assert!(!era.strata_policy.is_empty());
            }
            _ => panic!("expected era declaration"),
        }
    }

    #[test]
    fn test_era_with_transitions() {
        let source = r#"
era initialization 
    : timestep(0.1)
    : is_initial
{
    strata {
        physics: active
    }
    transition simulation when { tick > 1000 }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Era(era) => {
                assert!(!era.transitions.is_empty());
                assert_eq!(era.transitions[0].target.to_string(), "simulation");
            }
            _ => panic!("expected era declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Type Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_type_decl_basic() {
        let source = r#"
type PlateState {
    position: type Vector<3, m>
    velocity: type Vector<3, m/s>
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Type(ty) => {
                assert_eq!(ty.name, "PlateState");
                assert_eq!(ty.fields.len(), 2);
                assert_eq!(ty.fields[0].name, "position");
                assert_eq!(ty.fields[1].name, "velocity");
            }
            _ => panic!("expected type declaration"),
        }
    }

    // TODO: Enable once doc comment parsing is implemented
    // #[test]
    // fn test_type_decl_with_doc_comments() {
    //     let source = r#"
    // /// Plate state container
    // type PlateState {
    //     position: type Vector<3, m>
    // }
    // "#;
    //     let decls = lex_and_parse_decl(source);
    //     assert_eq!(decls.len(), 1);
    //     match &decls[0] {
    //         Declaration::Type(ty) => {
    //             assert!(ty.doc.is_some());
    //         }
    //         _ => panic!("expected type declaration"),
    //     }
    // }

    // ------------------------------------------------------------------------
    // World Declaration Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_world_basic() {
        let source = r#"
world terra {
    warmup {
        :max_iterations(100)
        :converged(true)
    }
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::World(world) => {
                assert_eq!(world.path.to_string(), "terra");
                assert!(world.warmup.is_some());
            }
            _ => panic!("expected world declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Const Block Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_const_block_basic() {
        let source = r#"
const {
    PI: type Scalar = 3.14159
    GRAVITY: type Vector<3, m/s^2> = vec3(0, 0, -9.81)
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Const(entries) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].path.to_string(), "PI");
                assert_eq!(entries[1].path.to_string(), "GRAVITY");
            }
            _ => panic!("expected const declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Config Block Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_config_block_basic() {
        let source = r#"
config {
    timestep: type Scalar<s> = 1.0
    max_iterations: type Int = 1000
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Config(entries) => {
                assert_eq!(entries.len(), 2);
                assert_eq!(entries[0].path.to_string(), "timestep");
                assert!(entries[0].default.is_some());
            }
            _ => panic!("expected config declaration"),
        }
    }

    #[test]
    fn test_config_without_defaults() {
        let source = r#"
config {
    world_seed: type Int
}
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Config(entries) => {
                assert_eq!(entries.len(), 1);
                assert!(entries[0].default.is_none());
            }
            _ => panic!("expected config declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Type Expression Sugar Tests
    // ------------------------------------------------------------------------

    #[test]
    fn test_vec2_sugar() {
        let source = "signal pos : type Vec2<m> { resolve { vec2(0, 0) } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                if let Some(TypeExpr::Vector { dim, .. }) = &node.type_expr {
                    assert_eq!(*dim, 2);
                } else {
                    panic!("expected Vec2 to desugar to Vector<2, unit>");
                }
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_vec3_sugar() {
        let source = "signal pos : type Vec3<m> { resolve { vec3(0, 0, 0) } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                if let Some(TypeExpr::Vector { dim, .. }) = &node.type_expr {
                    assert_eq!(*dim, 3);
                } else {
                    panic!("expected Vec3 to desugar to Vector<3, unit>");
                }
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_vec4_sugar() {
        let source = "signal color : type Vec4<None> { resolve { vec4(1, 1, 1, 1) } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                if let Some(TypeExpr::Vector { dim, .. }) = &node.type_expr {
                    assert_eq!(*dim, 4);
                } else {
                    panic!("expected Vec4 to desugar to Vector<4, None>");
                }
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_quat_sugar() {
        let source = "signal rotation : type Quat { resolve { quat_identity() } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                if let Some(TypeExpr::Vector { dim, unit, .. }) = &node.type_expr {
                    assert_eq!(*dim, 4);
                    assert!(unit.is_none());
                } else {
                    panic!("expected Quat to desugar to Vector<4, None>");
                }
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_mat2_sugar() {
        let source = "signal transform : type Mat2<None> { resolve { mat2_identity() } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                if let Some(TypeExpr::Matrix { rows, cols, .. }) = &node.type_expr {
                    assert_eq!(*rows, 2);
                    assert_eq!(*cols, 2);
                } else {
                    panic!("expected Mat2 to desugar to Matrix<2, 2, unit>");
                }
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_mat3_sugar() {
        let source = "signal transform : type Mat3<None> { resolve { mat3_identity() } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                if let Some(TypeExpr::Matrix { rows, cols, .. }) = &node.type_expr {
                    assert_eq!(*rows, 3);
                    assert_eq!(*cols, 3);
                } else {
                    panic!("expected Mat3 to desugar to Matrix<3, 3, unit>");
                }
            }
            _ => panic!("expected signal declaration"),
        }
    }

    #[test]
    fn test_mat4_sugar() {
        let source = "signal transform : type Mat4<None> { resolve { mat4_identity() } }";
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 1);
        match &decls[0] {
            Declaration::Node(node) => {
                if let Some(TypeExpr::Matrix { rows, cols, .. }) = &node.type_expr {
                    assert_eq!(*rows, 4);
                    assert_eq!(*cols, 4);
                } else {
                    panic!("expected Mat4 to desugar to Matrix<4, 4, unit>");
                }
            }
            _ => panic!("expected signal declaration"),
        }
    }

    // ------------------------------------------------------------------------
    // Error Case Tests
    // ------------------------------------------------------------------------

    // TODO: Move to semantic analysis tests - parser accepts syntactically valid but semantically invalid input
    // #[test]
    // fn test_error_missing_resolve_block() {
    //     let source = "signal temp : type Scalar<K> {}";
    //     let result = lex_and_parse_decl_result(source);
    //     assert!(
    //         result.is_err(),
    //         "expected error for signal without resolve block"
    //     );
    // }

    #[test]
    fn test_error_missing_type_in_const() {
        let source = "const { PI = 3.14159 }";
        let result = lex_and_parse_decl_result(source);
        assert!(
            result.is_err(),
            "expected error for const entry without type"
        );
    }

    // TODO: Move to semantic analysis tests - parser accepts syntactically valid but semantically invalid input
    // #[test]
    // fn test_error_invalid_member_path() {
    //     let source = "member velocity { resolve { 0 } }";
    //     let result = lex_and_parse_decl_result(source);
    //     assert!(result.is_err(), "expected error for member without entity");
    // }

    #[test]
    fn test_error_empty_entity_body() {
        // Empty entity body should be valid (just {})
        let source = "entity Plate";
        let result = lex_and_parse_decl_result(source);
        assert!(result.is_err(), "expected error for entity without braces");
    }

    #[test]
    fn test_multiple_declarations() {
        let source = r#"
entity Plate {}

signal temperature : type Scalar<K> { resolve { 273.15 } }

field elevation : type Scalar<m> { measure { 0 } }
"#;
        let decls = lex_and_parse_decl(source);
        assert_eq!(decls.len(), 3);
        assert!(matches!(decls[0], Declaration::Entity(_)));
        assert!(matches!(decls[1], Declaration::Node(_)));
        assert!(matches!(decls[2], Declaration::Node(_)));
    }
}
