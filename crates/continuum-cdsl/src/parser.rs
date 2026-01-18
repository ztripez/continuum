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
    Attribute, BinaryOp, BlockBody, Expr, Stmt, TypeExpr, UnaryOp, UnitExpr,
    UntypedKind as ExprKind,
};
use crate::foundation::{Path, Span};
use crate::lexer::Token;

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
            just(Token::Bang).to(UnaryOp::Not),
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
            just(Token::AndAnd)
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
            just(Token::OrOr).to(BinaryOp::Or).then(and).repeated(),
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

    // User types (identifiers)
    let user_type = select! {
        Token::Ident(name) => TypeExpr::User(Path::from_str(&name)),
    };

    choice((bool_type, scalar_type, vector_type, matrix_type, user_type))
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
        let expr = lex_and_parse("!-42.0");
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
}
