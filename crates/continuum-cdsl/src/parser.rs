//! Parser for Continuum DSL using chumsky combinators.
//!
//! This module implements a recursive descent parser that transforms a stream
//! of tokens from the lexer into an untyped AST ([`Expr`]).

use chumsky::prelude::*;

use crate::ast::{
    Attribute, BinaryOp, BlockBody, ConfigEntry, ConstEntry, Declaration, Entity, EraDecl, Expr,
    Node, ObserveBlock, ObserveWhen, RawWarmupPolicy, RoleData, Stmt, Stratum, StratumPolicyEntry,
    TransitionDecl, TypeDecl, TypeExpr, TypeField, UnaryOp, UnitExpr, UntypedKind, WarmupBlock,
    WhenBlock, WorldDecl,
};

use crate::foundation::{EntityId, Path, Span, StratumId};
use crate::lexer::Token;

/// Parse an expression from a token stream.
///
/// # Parameters
/// - `tokens`: Token stream to parse.
/// - `file_id`: SourceMap file identifier for spans.
///
/// # Returns
/// Parsed [`Expr`] with spans referencing `file_id`.
///
/// # Errors
/// Returns a [`Rich`] parse error if the tokens are not a valid expression.
///
/// # Examples
/// ```rust
/// use continuum_cdsl::lexer::Token;
/// use continuum_cdsl::parser::parse_expr;
/// use logos::Logos;
///
/// let tokens: Vec<_> = Token::lexer("1 + 2")
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap();
/// let expr = parse_expr(&tokens, 0).unwrap();
/// let _ = expr;
/// ```
pub fn parse_expr(tokens: &[Token], file_id: u16) -> ParseResult<Expr, Rich<'_, Token>> {
    expr_parser(file_id).parse(tokens)
}

/// Parse declarations from a token stream.
///
/// # Parameters
/// - `tokens`: Token stream for a full CDSL file.
/// - `file_id`: SourceMap file identifier for spans.
///
/// # Returns
/// Parsed declarations in source order.
///
/// # Errors
/// Returns a [`Rich`] parse error if the token stream is invalid.
///
/// # Examples
/// ```rust
/// use continuum_cdsl::lexer::Token;
/// use continuum_cdsl::parser::parse_declarations;
/// use logos::Logos;
///
/// let tokens: Vec<_> = Token::lexer("world demo {}")
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap();
/// let decls = parse_declarations(&tokens, 0).unwrap();
/// assert_eq!(decls.len(), 1);
/// ```
pub fn parse_declarations(
    tokens: &[Token],
    file_id: u16,
) -> ParseResult<Vec<Declaration>, Rich<'_, Token>> {
    declarations_parser(file_id).parse(tokens)
}

/// Main declarations parser - parses a complete CDSL file.
fn declarations_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Vec<Declaration>, extra::Err<Rich<'src, Token>>> + Clone {
    choice((
        world_parser(file_id),
        type_decl_parser(file_id),
        const_block_parser(file_id),
        config_block_parser(file_id),
        entity_parser(file_id),
        member_parser(file_id),
        stratum_parser(file_id),
        era_parser(file_id),
        signal_parser(file_id),
        field_parser(file_id),
        operator_parser(file_id),
        impulse_parser(file_id),
        fracture_parser(file_id),
        chronicle_parser(file_id),
    ))
    .repeated()
    .collect::<Vec<_>>()
    .then_ignore(end())
}

fn expr_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Expr, extra::Err<Rich<'src, Token>>> + Clone {
    recursive(|expr| {
        // Boolean literals
        let bool_literal = select! {
            Token::True => UntypedKind::BoolLiteral(true),
            Token::False => UntypedKind::BoolLiteral(false),
        }
        .map_with(move |kind, e| Expr::new(kind, token_span(e.span(), file_id)));

        // String literals
        let string_literal = select! {
            Token::String(s) => UntypedKind::StringLiteral(s),
        }
        .map_with(move |kind, e| Expr::new(kind, token_span(e.span(), file_id)));

        // Numeric literals
        let numeric_literal = select! {
            Token::Integer(n) => n as f64,
            Token::Float(f) => f,
        }
        .then(
            just(Token::Lt)
                .ignore_then(unit_expr_parser(file_id))
                .then_ignore(just(Token::Gt))
                .or_not(),
        )
        .map_with(move |(value, unit), e| {
            Expr::new(
                UntypedKind::Literal { value, unit },
                token_span(e.span(), file_id),
            )
        });

        // Context values
        let context_value = select! {
            Token::Prev => UntypedKind::Prev,
            Token::Current => UntypedKind::Current,
            Token::Inputs => UntypedKind::Inputs,
            Token::Dt => UntypedKind::Dt,
            Token::Self_ => UntypedKind::Self_,
            Token::Other => UntypedKind::Other,
            Token::Payload => UntypedKind::Payload,
        }
        .map_with(move |kind, e| Expr::new(kind, token_span(e.span(), file_id)));

        // Identifiers
        let identifier = select! {
            Token::Ident(name) => name,
        }
        .map_with(move |name, e| {
            Expr::new(UntypedKind::Local(name), token_span(e.span(), file_id))
        });

        let parens = expr
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));

        let vector_literal = expr
            .clone()
            .separated_by(just(Token::Comma))
            .allow_trailing()
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBracket), just(Token::RBracket))
            .map_with(move |elements, e| {
                Expr::new(UntypedKind::Vector(elements), token_span(e.span(), file_id))
            });

        let atom = choice((
            bool_literal,
            numeric_literal,
            string_literal,
            context_value,
            vector_literal,
            parens,
            identifier,
        ));

        #[derive(Clone)]
        enum Postfix {
            Field(String),
            Call(Vec<Expr>),
        }

        let postfix = atom.foldl(
            choice((
                just(Token::Dot)
                    .ignore_then(select! { Token::Ident(name) => name })
                    .map(Postfix::Field),
                expr.clone()
                    .separated_by(just(Token::Comma))
                    .allow_trailing()
                    .collect::<Vec<_>>()
                    .delimited_by(just(Token::LParen), just(Token::RParen))
                    .map(Postfix::Call),
            ))
            .repeated(),
            |target, post| {
                let span = target.span;
                match post {
                    Postfix::Call(args) => {
                        // It's a call
                        if let Some(path) = target.as_path() {
                            Expr::new(UntypedKind::Call { func: path, args }, span)
                        } else {
                            Expr::new(
                                UntypedKind::ParseError(
                                    "Dynamic dispatch not supported".to_string(),
                                ),
                                span,
                            )
                        }
                    }
                    Postfix::Field(field) => {
                        // It's a field access
                        Expr::new(
                            UntypedKind::FieldAccess {
                                object: Box::new(target),
                                field,
                            },
                            span,
                        )
                    }
                }
            },
        );

        let unary = choice((
            just(Token::Minus).to(UnaryOp::Neg),
            just(Token::Not).to(UnaryOp::Not),
        ))
        .repeated()
        .foldr(postfix, |op, operand| {
            let span = operand.span;
            Expr::new(
                UntypedKind::Unary {
                    op,
                    operand: Box::new(operand),
                },
                span,
            )
        });

        let power_op = just(Token::Caret).to(BinaryOp::Pow);
        let power = unary
            .clone()
            .then(power_op.then(unary.clone()).repeated().collect::<Vec<_>>())
            .map(|(first, rest)| {
                if rest.is_empty() {
                    return first;
                }
                let mut iter = rest.into_iter().rev();
                let (_, rightmost) = iter.next().unwrap();
                let right_tree = iter.fold(rightmost, |acc, (_, operand)| {
                    let span = operand.span;
                    Expr::new(
                        UntypedKind::Binary {
                            op: BinaryOp::Pow,
                            left: Box::new(operand),
                            right: Box::new(acc),
                        },
                        span,
                    )
                });
                let span = first.span;
                Expr::new(
                    UntypedKind::Binary {
                        op: BinaryOp::Pow,
                        left: Box::new(first),
                        right: Box::new(right_tree),
                    },
                    span,
                )
            });

        let mul_op = choice((
            just(Token::Star).to(BinaryOp::Mul),
            just(Token::Slash).to(BinaryOp::Div),
            just(Token::Percent).to(BinaryOp::Mod),
        ));
        let mul = power
            .clone()
            .foldl(mul_op.then(power).repeated(), |left, (op, right)| {
                let span = left.span;
                Expr::new(
                    UntypedKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            });

        let add_op = choice((
            just(Token::Plus).to(BinaryOp::Add),
            just(Token::Minus).to(BinaryOp::Sub),
        ));
        let add = mul
            .clone()
            .foldl(add_op.then(mul).repeated(), |left, (op, right)| {
                let span = left.span;
                Expr::new(
                    UntypedKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            });

        let cmp_op = choice((
            just(Token::Lt).to(BinaryOp::Lt),
            just(Token::LtEq).to(BinaryOp::Le),
            just(Token::Gt).to(BinaryOp::Gt),
            just(Token::GtEq).to(BinaryOp::Ge),
            just(Token::EqEq).to(BinaryOp::Eq),
            just(Token::BangEq).to(BinaryOp::Ne),
        ));
        let comparison = add
            .clone()
            .foldl(cmp_op.then(add).repeated(), |left, (op, right)| {
                let span = left.span;
                Expr::new(
                    UntypedKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            });

        let and = comparison.clone().foldl(
            just(Token::And)
                .to(BinaryOp::And)
                .then(comparison)
                .repeated(),
            |left, (op, right)| {
                let span = left.span;
                Expr::new(
                    UntypedKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            },
        );

        let or = and.clone().foldl(
            just(Token::Or).to(BinaryOp::Or).then(and).repeated(),
            |left, (op, right)| {
                let span = left.span;
                Expr::new(
                    UntypedKind::Binary {
                        op,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    span,
                )
            },
        );

        let braced_expr = expr
            .clone()
            .delimited_by(just(Token::LBrace), just(Token::RBrace));
        let if_expr = just(Token::If)
            .ignore_then(expr.clone())
            .then(braced_expr.clone())
            .then_ignore(just(Token::Else))
            .then(braced_expr)
            .map_with(move |((condition, then_branch), else_branch), e| {
                Expr::new(
                    UntypedKind::If {
                        condition: Box::new(condition),
                        then_branch: Box::new(then_branch),
                        else_branch: Box::new(else_branch),
                    },
                    token_span(e.span(), file_id),
                )
            });

        let let_expr = just(Token::Let)
            .ignore_then(select! { Token::Ident(name) => name })
            .then_ignore(just(Token::Eq))
            .then(expr.clone())
            .then_ignore(just(Token::In))
            .then(expr.clone())
            .map_with(move |((name, value), body), e| {
                Expr::new(
                    UntypedKind::Let {
                        name,
                        value: Box::new(value),
                        body: Box::new(body),
                    },
                    token_span(e.span(), file_id),
                )
            });

        choice((let_expr, if_expr, or))
    })
}

fn unit_expr_parser<'src>(
    _file_id: u16,
) -> impl Parser<'src, &'src [Token], UnitExpr, extra::Err<Rich<'src, Token>>> + Clone {
    recursive(|unit_term| {
        let base = select! { Token::Ident(name) => UnitExpr::Base(name) };
        let parens = unit_term
            .clone()
            .delimited_by(just(Token::LParen), just(Token::RParen));
        let primary = choice((base, parens));
        let power = primary
            .then(
                just(Token::Caret)
                    .ignore_then(select! { Token::Integer(n) => n as i8 })
                    .or_not(),
            )
            .map(|(base_unit, exp)| {
                if let Some(exp) = exp {
                    UnitExpr::Power(Box::new(base_unit), exp)
                } else {
                    base_unit
                }
            });
        power.clone().foldl(
            choice((just(Token::Star).to(true), just(Token::Slash).to(false)))
                .then(power)
                .repeated(),
            |left, (is_mul, right)| {
                if is_mul {
                    UnitExpr::Multiply(Box::new(left), Box::new(right))
                } else {
                    UnitExpr::Divide(Box::new(left), Box::new(right))
                }
            },
        )
    })
}

fn type_expr_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], TypeExpr, extra::Err<Rich<'src, Token>>> + Clone {
    let bool_type = just(Token::Ident("Bool".to_string())).to(TypeExpr::Bool);
    let scalar_type = just(Token::Ident("Scalar".to_string()))
        .then(
            just(Token::Lt)
                .ignore_then(unit_expr_parser(file_id))
                .then_ignore(just(Token::Gt))
                .or_not(),
        )
        .map(|(_, unit)| TypeExpr::Scalar { unit });
    let vector_type = just(Token::Ident("Vector".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(select! { Token::Integer(n) => n as u8 })
        .then_ignore(just(Token::Comma))
        .then(unit_expr_parser(file_id))
        .then_ignore(just(Token::Gt))
        .map(|(dim, unit)| TypeExpr::Vector {
            dim,
            unit: Some(unit),
        });
    let matrix_type = just(Token::Ident("Matrix".to_string()))
        .ignore_then(just(Token::Lt))
        .ignore_then(select! { Token::Integer(n) => n as u8 })
        .then_ignore(just(Token::Comma))
        .then(select! { Token::Integer(n) => n as u8 })
        .then_ignore(just(Token::Comma))
        .then(unit_expr_parser(file_id))
        .then_ignore(just(Token::Gt))
        .map(|((rows, cols), unit)| TypeExpr::Matrix {
            rows,
            cols,
            unit: Some(unit),
        });
    let user_type = select! { Token::Ident(name) => TypeExpr::User(Path::from(name.as_str())) };
    choice((bool_type, scalar_type, vector_type, matrix_type, user_type))
}

fn token_span(span: SimpleSpan, file_id: u16) -> Span {
    Span::new(file_id, span.start as u32, span.end as u32, 0)
}

fn path_parser<'src>(
) -> impl Parser<'src, &'src [Token], Path, extra::Err<Rich<'src, Token>>> + Clone {
    select! { Token::Ident(name) => name }
        .separated_by(just(Token::Dot))
        .at_least(1)
        .collect::<Vec<_>>()
        .map(|segments| Path::from(segments.join(".").as_str()))
}

fn type_annotation_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], TypeExpr, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Colon)
        .ignore_then(just(Token::Type))
        .ignore_then(type_expr_parser(file_id))
}

fn attribute_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Attribute, extra::Err<Rich<'src, Token>>> + Clone {
    let attribute_name = select! {
        Token::Ident(name) => name,
        Token::Dt => "dt".to_string(),
    };
    just(Token::Colon)
        .ignore_then(attribute_name)
        .then(
            expr_parser(file_id)
                .separated_by(just(Token::Comma))
                .allow_trailing()
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LParen), just(Token::RParen))
                .or_not(),
        )
        .map_with(move |(name, args), e| {
            let args = match args {
                Some(args) => args,
                None => Vec::new(),
            };
            Attribute {
                name,
                args,
                span: token_span(e.span(), file_id),
            }
        })
}

fn stmt_parser<'src>(
    expr: impl Parser<'src, &'src [Token], Expr, extra::Err<Rich<'src, Token>>> + Clone,
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Stmt, extra::Err<Rich<'src, Token>>> + Clone {
    let let_stmt = just(Token::Let)
        .ignore_then(select! { Token::Ident(name) => name })
        .then_ignore(just(Token::Eq))
        .then(expr.clone())
        .map_with(move |(name, value), e| Stmt::Let {
            name,
            value,
            span: token_span(e.span(), file_id),
        });
    let assign_stmt = path_parser()
        .then_ignore(just(Token::LeftArrow))
        .then(expr.clone())
        .then(just(Token::Comma).ignore_then(expr.clone()).or_not())
        .map_with(move |((target, first), second), e| {
            let span = token_span(e.span(), file_id);
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
    choice((let_stmt, assign_stmt, expr.map(Stmt::Expr)))
}

fn block_body_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], BlockBody, extra::Err<Rich<'src, Token>>> + Clone {
    let expr = expr_parser(file_id);
    let stmt_list = stmt_parser(expr.clone(), file_id)
        .separated_by(just(Token::Semicolon))
        .at_least(1)
        .collect::<Vec<_>>()
        .map(BlockBody::Statements);
    choice((expr.map(BlockBody::Expression), stmt_list))
}

fn warmup_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], WarmupBlock, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::WarmUp)
        .ignore_then(
            attribute_parser(file_id)
                .repeated()
                .collect::<Vec<_>>()
                .then(just(Token::Iterate).ignore_then(
                    expr_parser(file_id).delimited_by(just(Token::LBrace), just(Token::RBrace)),
                ))
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(move |(attrs, iterate), e| WarmupBlock {
            attrs,
            iterate,
            span: token_span(e.span(), file_id),
        })
}

fn when_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], WhenBlock, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::When)
        .ignore_then(
            expr_parser(file_id)
                .separated_by(just(Token::Semicolon))
                .at_least(1)
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(move |conditions, e| WhenBlock {
            conditions,
            span: token_span(e.span(), file_id),
        })
}

fn observe_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], ObserveBlock, extra::Err<Rich<'src, Token>>> + Clone {
    let when_clause = just(Token::When)
        .ignore_then(expr_parser(file_id))
        .then(
            stmt_parser(expr_parser(file_id), file_id)
                .separated_by(just(Token::Semicolon))
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(move |(condition, emit_block), e| ObserveWhen {
            condition,
            emit_block,
            span: token_span(e.span(), file_id),
        });
    just(Token::Observe)
        .ignore_then(
            when_clause
                .repeated()
                .at_least(1)
                .collect::<Vec<_>>()
                .delimited_by(just(Token::LBrace), just(Token::RBrace)),
        )
        .map_with(move |when_clauses, e| ObserveBlock {
            when_clauses,
            span: token_span(e.span(), file_id),
        })
}

fn transition_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], TransitionDecl, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Transition)
        .ignore_then(path_parser())
        .then(when_parser(file_id))
        .map_with(move |(target, when_block), e| TransitionDecl {
            target,
            conditions: when_block.conditions,
            span: token_span(e.span(), file_id),
        })
}

fn execution_block_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], (String, BlockBody), extra::Err<Rich<'src, Token>>> + Clone {
    choice((
        just(Token::Resolve).to("resolve"),
        just(Token::Collect).to("collect"),
        just(Token::Apply).to("apply"),
        just(Token::Measure).to("measure"),
        just(Token::Assert).to("assert"),
        just(Token::Emit).to("emit"),
    ))
    .then(block_body_parser(file_id).delimited_by(just(Token::LBrace), just(Token::RBrace)))
    .map(|(name, body)| (name.to_string(), body))
}

fn signal_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Signal)
        .ignore_then(path_parser())
        .then(type_annotation_parser(file_id).or_not())
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(warmup_parser(file_id).or_not())
        .then(
            execution_block_parser(file_id)
                .repeated()
                .collect::<Vec<_>>(),
        )
        .then_ignore(just(Token::RBrace))
        .map_with(move |((((path, type_expr), attrs), warmup), blocks), e| {
            let mut node = Node::new(path, token_span(e.span(), file_id), RoleData::Signal, ());
            node.type_expr = type_expr;
            node.attributes = attrs;
            node.warmup = warmup;
            for (name, body) in blocks {
                node.execution_blocks.push((name, body));
            }
            Declaration::Node(node)
        })
}

fn field_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Field)
        .ignore_then(path_parser())
        .then(type_annotation_parser(file_id).or_not())
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(
            execution_block_parser(file_id)
                .repeated()
                .collect::<Vec<_>>(),
        )
        .then_ignore(just(Token::RBrace))
        .map_with(move |(((path, type_expr), attrs), blocks), e| {
            let mut node = Node::new(
                path,
                token_span(e.span(), file_id),
                RoleData::Field {
                    reconstruction: None,
                },
                (),
            );
            node.type_expr = type_expr;
            node.attributes = attrs;
            for (name, body) in blocks {
                node.execution_blocks.push((name, body));
            }
            Declaration::Node(node)
        })
}

fn operator_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Operator)
        .ignore_then(path_parser())
        .then(type_annotation_parser(file_id).or_not())
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(
            execution_block_parser(file_id)
                .repeated()
                .collect::<Vec<_>>(),
        )
        .then_ignore(just(Token::RBrace))
        .map_with(move |(((path, type_expr), attrs), blocks), e| {
            let mut node = Node::new(path, token_span(e.span(), file_id), RoleData::Operator, ());
            node.type_expr = type_expr;
            node.attributes = attrs;
            for (name, body) in blocks {
                node.execution_blocks.push((name, body));
            }
            Declaration::Node(node)
        })
}

fn impulse_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Impulse)
        .ignore_then(path_parser())
        .then(type_annotation_parser(file_id).or_not())
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(
            execution_block_parser(file_id)
                .repeated()
                .collect::<Vec<_>>(),
        )
        .then_ignore(just(Token::RBrace))
        .map_with(move |(((path, type_expr), attrs), blocks), e| {
            let mut node = Node::new(
                path,
                token_span(e.span(), file_id),
                RoleData::Impulse { payload: None },
                (),
            );
            node.type_expr = type_expr;
            node.attributes = attrs;
            for (name, body) in blocks {
                node.execution_blocks.push((name, body));
            }
            Declaration::Node(node)
        })
}

fn fracture_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Fracture)
        .ignore_then(path_parser())
        .then(type_annotation_parser(file_id).or_not())
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(when_parser(file_id).or_not())
        .then(
            execution_block_parser(file_id)
                .repeated()
                .collect::<Vec<_>>(),
        )
        .then_ignore(just(Token::RBrace))
        .map_with(
            move |((((path, type_expr), attrs), when_block), blocks), e| {
                let mut node =
                    Node::new(path, token_span(e.span(), file_id), RoleData::Fracture, ());
                node.type_expr = type_expr;
                node.attributes = attrs;
                node.when = when_block;
                for (name, body) in blocks {
                    node.execution_blocks.push((name, body));
                }
                Declaration::Node(node)
            },
        )
}

fn chronicle_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Chronicle)
        .ignore_then(path_parser())
        .then(type_annotation_parser(file_id).or_not())
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(observe_parser(file_id).or_not())
        .then_ignore(just(Token::RBrace))
        .map_with(move |(((path, type_expr), attrs), observe_block), e| {
            let mut node = Node::new(path, token_span(e.span(), file_id), RoleData::Chronicle, ());
            node.type_expr = type_expr;
            node.attributes = attrs;
            node.observe = observe_block;
            Declaration::Node(node)
        })
}

fn entity_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Entity)
        .ignore_then(path_parser())
        .then_ignore(just(Token::LBrace))
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(move |(path, attrs), e| {
            let mut entity = Entity::new(
                EntityId::new(path.to_string()),
                path,
                token_span(e.span(), file_id),
            );
            entity.attributes = attrs;
            Declaration::Entity(entity)
        })
}

fn member_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Member)
        .ignore_then(path_parser())
        .validate(|path, e, emitter| {
            if path.len() < 2 {
                emitter.emit(Rich::custom(
                    e.span(),
                    format!(
                        "Member declaration requires entity.member path structure, got '{}'",
                        path
                    ),
                ));
            }
            path
        })
        .then(type_annotation_parser(file_id).or_not())
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(
            execution_block_parser(file_id)
                .repeated()
                .collect::<Vec<_>>(),
        )
        .then_ignore(just(Token::RBrace))
        .map_with(move |(((full_path, type_expr), attrs), blocks), e| {
            let span = token_span(e.span(), file_id);
            let entity_path = full_path.parent().unwrap_or_else(|| full_path.clone());
            let entity_id = EntityId::new(entity_path.to_string());
            let mut node = Node::new(full_path, span, RoleData::Signal, entity_id);
            node.type_expr = type_expr;
            node.attributes = attrs;
            for (name, body) in blocks {
                node.execution_blocks.push((name, body));
            }
            Declaration::Member(node)
        })
}

fn stratum_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::Strata)
        .ignore_then(path_parser())
        .then_ignore(just(Token::LBrace))
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(move |(path, attrs), e| {
            let mut stratum = Stratum::new(
                StratumId::new(path.to_string()),
                path,
                token_span(e.span(), file_id),
            );
            stratum.attributes = attrs;
            Declaration::Stratum(stratum)
        })
}

fn era_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    let strata_entry = path_parser()
        .then_ignore(just(Token::Colon))
        .then(select! { Token::Ident(s) => s.to_string() })
        .map_with(move |(path, state_name), e| StratumPolicyEntry {
            stratum: path,
            state_name,
            stride: None,
            span: token_span(e.span(), file_id),
        });
    let strata_block = just(Token::Strata).ignore_then(
        strata_entry
            .separated_by(just(Token::Semicolon).or_not())
            .collect::<Vec<_>>()
            .delimited_by(just(Token::LBrace), just(Token::RBrace)),
    );
    just(Token::Era)
        .ignore_then(path_parser())
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::LBrace))
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then(strata_block.or_not())
        .then(transition_parser(file_id).repeated().collect::<Vec<_>>())
        .then_ignore(just(Token::RBrace))
        .map_with(
            move |((((path, header_attrs), body_attrs), strata_policy), transitions), e| {
                let mut dt_attrs = Vec::new();
                let mut attributes = Vec::new();
                for attr in header_attrs.into_iter().chain(body_attrs) {
                    if attr.name == "dt" {
                        dt_attrs.push(attr);
                    } else {
                        attributes.push(attr);
                    }
                }

                let dt = match dt_attrs.as_slice() {
                    [] => None,
                    [attr] => {
                        if attr.args.len() == 1 {
                            Some(attr.args[0].clone())
                        } else {
                            Some(Expr::new(
                                UntypedKind::ParseError(format!(
                                    "dt attribute expects exactly one argument, got {}",
                                    attr.args.len()
                                )),
                                attr.span,
                            ))
                        }
                    }
                    _ => Some(Expr::new(
                        UntypedKind::ParseError("multiple :dt attributes".to_string()),
                        dt_attrs[0].span,
                    )),
                };

                Declaration::Era(EraDecl {
                    path,
                    span: token_span(e.span(), file_id),
                    doc: None,
                    dt,
                    attributes,
                    strata_policy: match strata_policy {
                        Some(policy) => policy,
                        None => Vec::new(),
                    },
                    transitions,
                })
            },
        )
}

fn type_decl_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    let field = select! { Token::Ident(name) => name }
        .then_ignore(just(Token::Colon))
        .then(just(Token::Type).ignore_then(type_expr_parser(file_id)))
        .map_with(move |(name, type_expr), e| TypeField {
            name,
            type_expr,
            span: token_span(e.span(), file_id),
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
        .map_with(move |(name, fields), e| {
            Declaration::Type(TypeDecl {
                name,
                fields,
                span: token_span(e.span(), file_id),
                doc: None,
            })
        })
}

fn const_block_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    let entry = path_parser()
        .then_ignore(just(Token::Colon))
        .then(type_annotation_parser(file_id))
        .then_ignore(just(Token::Eq))
        .then(expr_parser(file_id))
        .map_with(move |((path, type_expr), value), e| ConstEntry {
            path,
            value,
            type_expr,
            span: token_span(e.span(), file_id),
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

fn config_block_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    let entry = path_parser()
        .then_ignore(just(Token::Colon))
        .then(type_annotation_parser(file_id))
        .then(just(Token::Eq).ignore_then(expr_parser(file_id)).or_not())
        .map_with(move |((path, type_expr), default), e| ConfigEntry {
            path,
            default,
            type_expr,
            span: token_span(e.span(), file_id),
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

fn world_parser<'src>(
    file_id: u16,
) -> impl Parser<'src, &'src [Token], Declaration, extra::Err<Rich<'src, Token>>> + Clone {
    just(Token::World)
        .ignore_then(path_parser())
        .then_ignore(just(Token::LBrace))
        .then(attribute_parser(file_id).repeated().collect::<Vec<_>>())
        .then(warmup_parser(file_id).or_not())
        .then_ignore(just(Token::RBrace))
        .map_with(move |((path, attrs), warmup), e| {
            Declaration::World(WorldDecl {
                path,
                title: None,
                version: None,
                warmup: warmup.map(|w| RawWarmupPolicy {
                    attributes: w.attrs,
                    span: w.span,
                }),
                attributes: attrs,
                span: token_span(e.span(), file_id),
                doc: None,
                debug: false,
            })
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Token;
    use logos::Logos;

    fn lex_and_parse(source: &str) -> Expr {
        let tokens: Vec<_> = Token::lexer(source).collect::<Result<Vec<_>, _>>().unwrap();
        parse_expr(&tokens, 0).unwrap()
    }

    #[test]
    fn test_parse_literal() {
        let expr = lex_and_parse("42.0");
        match expr.kind {
            UntypedKind::Literal { value, .. } => assert_eq!(value, 42.0),
            _ => panic!("expected literal"),
        }
    }

    #[test]
    fn test_parse_bool_literal() {
        let expr = lex_and_parse("true");
        assert!(matches!(expr.kind, UntypedKind::BoolLiteral(true)));
    }

    #[test]
    fn test_parse_binary_add() {
        let expr = lex_and_parse("10 + 20");
        assert!(matches!(
            expr.kind,
            UntypedKind::Binary {
                op: BinaryOp::Add,
                ..
            }
        ));
    }

    fn lex_and_parse_decl(source: &str) -> Vec<Declaration> {
        let tokens: Vec<_> = Token::lexer(source).collect::<Result<Vec<_>, _>>().unwrap();
        parse_declarations(&tokens, 0).into_result().unwrap()
    }

    #[test]
    fn test_path_simple() {
        let decls = lex_and_parse_decl("entity Plate {}");
        assert_eq!(decls.len(), 1);
    }

    #[test]
    fn test_parse_era_attributes_header_and_body() {
        let decls = lex_and_parse_decl(
            r#"
            era main : initial {
                : dt(1.0 <s>)
            }
            "#,
        );
        let era = match &decls[0] {
            Declaration::Era(era) => era,
            _ => panic!("expected era"),
        };
        assert!(era.attributes.iter().any(|attr| attr.name == "initial"));
        let dt = era.dt.as_ref().expect("expected dt");
        assert!(matches!(dt.kind, UntypedKind::Literal { .. }));
    }

    #[test]
    fn test_parse_era_dt_argument_error() {
        let decls = lex_and_parse_decl(
            r#"
            era main {
                : dt(1.0, 2.0)
            }
            "#,
        );
        let era = match &decls[0] {
            Declaration::Era(era) => era,
            _ => panic!("expected era"),
        };
        let dt = era.dt.as_ref().expect("expected dt");
        assert!(matches!(dt.kind, UntypedKind::ParseError(_)));
    }

    #[test]
    fn test_parse_multiple_dt_is_error() {
        let decls = lex_and_parse_decl(
            r#"
            era main {
                : dt(1.0 <s>)
                : dt(2.0 <s>)
            }
            "#,
        );
        let era = match &decls[0] {
            Declaration::Era(era) => era,
            _ => panic!("expected era"),
        };
        let dt = era.dt.as_ref().expect("expected dt");
        assert!(matches!(dt.kind, UntypedKind::ParseError(_)));
    }

    #[test]
    fn test_parse_era_dt_in_header() {
        let decls = lex_and_parse_decl(
            r#"
            era main : dt(1.0 <s>) {
                : initial
            }
            "#,
        );
        let era = match &decls[0] {
            Declaration::Era(era) => era,
            _ => panic!("expected era"),
        };
        assert!(era.attributes.iter().any(|attr| attr.name == "initial"));
        let dt = era.dt.as_ref().expect("expected dt");
        assert!(matches!(dt.kind, UntypedKind::Literal { .. }));
    }

    #[test]
    fn test_parse_era_dt_zero_args_is_error() {
        let decls = lex_and_parse_decl(
            r#"
            era main {
                : dt()
            }
            "#,
        );
        let era = match &decls[0] {
            Declaration::Era(era) => era,
            _ => panic!("expected era"),
        };
        let dt = era.dt.as_ref().expect("expected dt");
        assert!(matches!(dt.kind, UntypedKind::ParseError(_)));
    }

    #[test]
    fn test_parse_era_dt_header_and_body_is_error() {
        let decls = lex_and_parse_decl(
            r#"
            era main : dt(1.0 <s>) {
                : dt(2.0 <s>)
            }
            "#,
        );
        let era = match &decls[0] {
            Declaration::Era(era) => era,
            _ => panic!("expected era"),
        };
        let dt = era.dt.as_ref().expect("expected dt");
        assert!(matches!(dt.kind, UntypedKind::ParseError(_)));
    }
}
