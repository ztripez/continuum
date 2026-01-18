//! Name resolution pass
//!
//! Validates that all Path references in expressions resolve to declared symbols.
//!
//! # What This Pass Does
//!
//! 1. **Builds symbol table** - Collects all declarations (signals, fields, operators, types, etc.)
//! 2. **Validates Path references** - Ensures Signal(path), Field(path), etc. refer to real symbols
//! 3. **Scope checking** - Validates Local variables are in scope (from Let bindings)
//! 4. **Error reporting** - Produces CompileError for unresolved paths
//!
//! # What This Pass Does NOT Do
//!
//! - **No type resolution** - TypeExpr is not resolved to Type (that's `resolve::types`)
//! - **No type checking** - Type compatibility is validated later
//! - **No mutation** - This pass only validates; it doesn't modify the AST
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Resolution → Type Resolution → Validation
//!                       ^^^^^^
//!                     YOU ARE HERE
//! ```
//!
//! # Scoping Rules
//!
//! - **Global scope** - World-level declarations (signals, operators, fields, types, etc.)
//! - **Entity scope** - Per-entity members (e.g., `plate.velocity`)
//! - **Local scope** - Let-bound variables (shadowing allowed)
//! - **Built-ins** - Keywords like `prev`, `current`, `inputs`, `dt`, `self` (always valid)
//!
//! # Examples
//!
//! ```cdsl
//! signal temperature : Scalar<K>
//!
//! operator update {
//!     resolve {
//!         // Name resolution validates:
//!         // - `temperature` refers to the signal above ✓
//!         // - `x` is bound by let ✓
//!         let x = temperature + 10.0<K>
//!         x
//!     }
//! }
//!
//! field invalid_ref {
//!     measure {
//!         nonexistent  // ERROR: unresolved path
//!     }
//! }
//! ```

use crate::ast::{AggregateOp, Expr, Node, UntypedKind as ExprKind};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{EntityId, Path};
use std::collections::{HashMap, HashSet};

/// Symbol table for name resolution
///
/// Tracks all declared symbols in the world to validate Path references.
#[derive(Debug, Default)]
pub struct SymbolTable {
    /// Global symbols (world-level signals, operators, fields, etc.)
    globals: HashSet<Path>,

    /// User-defined type names
    types: HashSet<Path>,

    /// Per-entity members (entity_id → set of member paths)
    members: HashMap<EntityId, HashSet<Path>>,

    /// Config values
    config: HashSet<Path>,

    /// Const values
    consts: HashSet<Path>,
}

impl SymbolTable {
    /// Create an empty symbol table
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a global symbol (signal, operator, field, etc.)
    pub fn register_global(&mut self, path: Path) {
        self.globals.insert(path);
    }

    /// Register a user-defined type
    pub fn register_type(&mut self, path: Path) {
        self.types.insert(path);
    }

    /// Register a per-entity member
    pub fn register_member(&mut self, entity: EntityId, path: Path) {
        self.members.entry(entity).or_default().insert(path);
    }

    /// Register a config value
    pub fn register_config(&mut self, path: Path) {
        self.config.insert(path);
    }

    /// Register a const value
    pub fn register_const(&mut self, path: Path) {
        self.consts.insert(path);
    }

    /// Check if a path is a declared signal or global node
    pub fn has_global(&self, path: &Path) -> bool {
        self.globals.contains(path)
    }

    /// Check if a path is a declared type
    pub fn has_type(&self, path: &Path) -> bool {
        self.types.contains(path)
    }

    /// Check if a path is a declared config value
    pub fn has_config(&self, path: &Path) -> bool {
        self.config.contains(path)
    }

    /// Check if a path is a declared const value
    pub fn has_const(&self, path: &Path) -> bool {
        self.consts.contains(path)
    }
}

/// Build symbol table from world declarations
///
/// Collects all Node<I> declarations into a symbol table for name resolution.
///
/// # Parameters
///
/// - `globals`: Global nodes (world-level signals, operators, fields, etc.)
/// - `members`: Per-entity member nodes
///
/// # Returns
///
/// Symbol table containing all declared symbols.
pub fn build_symbol_table<I: crate::ast::Index>(
    globals: &[Node<I>],
    members: &HashMap<EntityId, Vec<Node<EntityId>>>,
) -> SymbolTable {
    let mut table = SymbolTable::new();

    // Register global nodes
    for node in globals {
        table.register_global(node.path.clone());
    }

    // Register per-entity members
    for (entity_id, entity_members) in members {
        for member in entity_members {
            table.register_member(entity_id.clone(), member.path.clone());
        }
    }

    table
}

/// Scope for tracking local variables
///
/// Tracks let-bound variables during expression validation.
/// Supports nested scopes (let inside let).
#[derive(Debug, Default)]
pub struct Scope {
    /// Stack of local variable scopes (innermost = last)
    locals: Vec<HashSet<String>>,
}

impl Scope {
    /// Create empty scope
    fn new() -> Self {
        Self::default()
    }

    /// Push a new scope level
    fn push(&mut self) {
        self.locals.push(HashSet::new());
    }

    /// Pop the current scope level
    fn pop(&mut self) {
        self.locals.pop();
    }

    /// Bind a local variable in the current scope
    fn bind(&mut self, name: String) {
        if let Some(current) = self.locals.last_mut() {
            current.insert(name);
        }
    }

    /// Check if a local variable is in scope
    fn has(&self, name: &str) -> bool {
        self.locals.iter().any(|scope| scope.contains(name))
    }
}

/// Validate expression name references
///
/// Recursively validates that all Path references in an expression refer to declared symbols.
///
/// # Parameters
///
/// - `expr`: Expression to validate
/// - `table`: Symbol table with all declarations
/// - `scope`: Current local variable scope
/// - `errors`: Accumulator for validation errors
///
/// # Errors
///
/// Adds CompileError to `errors` for:
/// - Unresolved signal paths
/// - Unresolved field paths
/// - Unresolved config/const paths
/// - Unresolved type paths (in Struct literals)
/// - Unresolved local variables
pub fn validate_expr(
    expr: &Expr,
    table: &SymbolTable,
    scope: &mut Scope,
    errors: &mut Vec<CompileError>,
) {
    match &expr.kind {
        // === Path references ===
        ExprKind::Signal(path) => {
            if !table.has_global(path) {
                errors.push(CompileError::new(
                    ErrorKind::UndefinedName,
                    expr.span,
                    format!("undefined signal '{}'", path),
                ));
            }
        }

        ExprKind::Field(path) => {
            if !table.has_global(path) {
                errors.push(CompileError::new(
                    ErrorKind::UndefinedName,
                    expr.span,
                    format!("undefined field '{}'", path),
                ));
            }
        }

        ExprKind::Config(path) => {
            if !table.has_config(path) {
                errors.push(CompileError::new(
                    ErrorKind::UndefinedName,
                    expr.span,
                    format!("undefined config value '{}'", path),
                ));
            }
        }

        ExprKind::Const(path) => {
            if !table.has_const(path) {
                errors.push(CompileError::new(
                    ErrorKind::UndefinedName,
                    expr.span,
                    format!("undefined const value '{}'", path),
                ));
            }
        }

        // === Local variables ===
        ExprKind::Local(name) => {
            if !scope.has(name) {
                errors.push(CompileError::new(
                    ErrorKind::UndefinedName,
                    expr.span,
                    format!("undefined variable '{}'", name),
                ));
            }
        }

        // === Let bindings (introduce scope) ===
        ExprKind::Let { name, value, body } => {
            // Validate value expression in current scope
            validate_expr(value, table, scope, errors);

            // Push new scope for body
            scope.push();
            scope.bind(name.clone());

            // Validate body in new scope
            validate_expr(body, table, scope, errors);

            // Pop scope
            scope.pop();
        }

        // === Struct construction ===
        ExprKind::Struct { ty, fields } => {
            // Validate type path exists
            if !table.has_type(ty) {
                errors.push(CompileError::new(
                    ErrorKind::UnknownType,
                    expr.span,
                    format!("unknown type '{}'", ty),
                ));
            }

            // Validate field value expressions
            for (_, field_expr) in fields {
                validate_expr(field_expr, table, scope, errors);
            }
        }

        // === Aggregates (introduce scope) ===
        ExprKind::Aggregate {
            op: _,
            entity: _,
            binding,
            body,
        } => {
            // Push new scope for body
            scope.push();
            scope.bind(binding.clone());

            // Validate body
            validate_expr(body, table, scope, errors);

            // Pop scope
            scope.pop();
        }

        ExprKind::Fold {
            entity: _,
            init,
            acc,
            elem,
            body,
        } => {
            // Validate init in current scope
            validate_expr(init, table, scope, errors);

            // Push new scope for body
            scope.push();
            scope.bind(acc.clone());
            scope.bind(elem.clone());

            // Validate body
            validate_expr(body, table, scope, errors);

            // Pop scope
            scope.pop();
        }

        // === Recursive cases (no scoping) ===
        ExprKind::Vector(elements) => {
            for elem in elements {
                validate_expr(elem, table, scope, errors);
            }
        }

        ExprKind::Binary { left, right, .. } => {
            validate_expr(left, table, scope, errors);
            validate_expr(right, table, scope, errors);
        }

        ExprKind::Unary { operand, .. } => {
            validate_expr(operand, table, scope, errors);
        }

        ExprKind::Call { args, .. } => {
            for arg in args {
                validate_expr(arg, table, scope, errors);
            }
        }

        ExprKind::KernelCall { args, .. } => {
            for arg in args {
                validate_expr(arg, table, scope, errors);
            }
        }

        ExprKind::FieldAccess { object, .. } => {
            validate_expr(object, table, scope, errors);
        }

        ExprKind::If {
            condition,
            then_branch,
            else_branch,
        } => {
            validate_expr(condition, table, scope, errors);
            validate_expr(then_branch, table, scope, errors);
            validate_expr(else_branch, table, scope, errors);
        }

        // === Literals and keywords (always valid) ===
        ExprKind::Literal { .. }
        | ExprKind::BoolLiteral(_)
        | ExprKind::Prev
        | ExprKind::Current
        | ExprKind::Inputs
        | ExprKind::Dt
        | ExprKind::Self_
        | ExprKind::Other
        | ExprKind::Payload
        | ExprKind::ParseError(_) => {
            // No validation needed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::RoleData;
    use crate::foundation::Span;

    fn make_path(s: &str) -> Path {
        Path::from_str(s)
    }

    fn make_node(path: &str) -> Node<()> {
        Node::new(
            make_path(path),
            Span::new(0, 0, 10, 1),
            RoleData::Signal,
            (),
        )
    }

    #[test]
    fn test_symbol_table_register_global() {
        let mut table = SymbolTable::new();
        table.register_global(make_path("temperature"));

        assert!(table.has_global(&make_path("temperature")));
        assert!(!table.has_global(&make_path("pressure")));
    }

    #[test]
    fn test_symbol_table_register_type() {
        let mut table = SymbolTable::new();
        table.register_type(make_path("Vector3"));

        assert!(table.has_type(&make_path("Vector3")));
        assert!(!table.has_type(&make_path("Vector2")));
    }

    #[test]
    fn test_build_symbol_table() {
        let globals = vec![make_node("temperature"), make_node("pressure")];

        let table = build_symbol_table(&globals, &HashMap::new());

        assert!(table.has_global(&make_path("temperature")));
        assert!(table.has_global(&make_path("pressure")));
        assert!(!table.has_global(&make_path("velocity")));
    }

    #[test]
    fn test_validate_signal_reference_valid() {
        let mut table = SymbolTable::new();
        table.register_global(make_path("temperature"));

        let expr = Expr::new(
            ExprKind::Signal(make_path("temperature")),
            Span::new(0, 0, 10, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_signal_reference_invalid() {
        let table = SymbolTable::new();

        let expr = Expr::new(
            ExprKind::Signal(make_path("nonexistent")),
            Span::new(0, 0, 10, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    }

    #[test]
    fn test_validate_local_variable_valid() {
        let table = SymbolTable::new();

        // let x = 10.0; x
        let expr = Expr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(Expr::new(
                    ExprKind::Literal {
                        value: 10.0,
                        unit: None,
                    },
                    Span::new(0, 0, 4, 1),
                )),
                body: Box::new(Expr::new(
                    ExprKind::Local("x".to_string()),
                    Span::new(0, 5, 6, 1),
                )),
            },
            Span::new(0, 0, 6, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_local_variable_out_of_scope() {
        let table = SymbolTable::new();

        // Reference to 'x' without let binding
        let expr = Expr::new(ExprKind::Local("x".to_string()), Span::new(0, 0, 1, 1));

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    }

    #[test]
    fn test_validate_nested_let() {
        let table = SymbolTable::new();

        // let x = 1.0; let y = 2.0; x + y
        let expr = Expr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(Expr::new(
                    ExprKind::Literal {
                        value: 1.0,
                        unit: None,
                    },
                    Span::new(0, 0, 3, 1),
                )),
                body: Box::new(Expr::new(
                    ExprKind::Let {
                        name: "y".to_string(),
                        value: Box::new(Expr::new(
                            ExprKind::Literal {
                                value: 2.0,
                                unit: None,
                            },
                            Span::new(0, 4, 7, 1),
                        )),
                        body: Box::new(Expr::new(
                            ExprKind::KernelCall {
                                kernel: continuum_kernel_types::KernelId::new("maths", "add"),
                                args: vec![
                                    Expr::new(
                                        ExprKind::Local("x".to_string()),
                                        Span::new(0, 8, 9, 1),
                                    ),
                                    Expr::new(
                                        ExprKind::Local("y".to_string()),
                                        Span::new(0, 10, 11, 1),
                                    ),
                                ],
                            },
                            Span::new(0, 8, 11, 1),
                        )),
                    },
                    Span::new(0, 4, 11, 1),
                )),
            },
            Span::new(0, 0, 11, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_struct_type_valid() {
        let mut table = SymbolTable::new();
        table.register_type(make_path("Vector3"));

        let expr = Expr::new(
            ExprKind::Struct {
                ty: make_path("Vector3"),
                fields: vec![
                    (
                        "x".to_string(),
                        Expr::new(
                            ExprKind::Literal {
                                value: 1.0,
                                unit: None,
                            },
                            Span::new(0, 0, 3, 1),
                        ),
                    ),
                    (
                        "y".to_string(),
                        Expr::new(
                            ExprKind::Literal {
                                value: 2.0,
                                unit: None,
                            },
                            Span::new(0, 4, 7, 1),
                        ),
                    ),
                    (
                        "z".to_string(),
                        Expr::new(
                            ExprKind::Literal {
                                value: 3.0,
                                unit: None,
                            },
                            Span::new(0, 8, 11, 1),
                        ),
                    ),
                ],
            },
            Span::new(0, 0, 11, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_struct_type_invalid() {
        let table = SymbolTable::new();

        let expr = Expr::new(
            ExprKind::Struct {
                ty: make_path("NonexistentType"),
                fields: vec![],
            },
            Span::new(0, 0, 10, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::UnknownType));
    }

    #[test]
    fn test_validate_aggregate_binding() {
        let mut table = SymbolTable::new();
        table.register_global(make_path("plates"));

        let expr = Expr::new(
            ExprKind::Aggregate {
                op: AggregateOp::Count,
                entity: EntityId::new("plates"),
                binding: "p".to_string(),
                body: Box::new(Expr::new(
                    ExprKind::Local("p".to_string()),
                    Span::new(0, 0, 1, 1),
                )),
            },
            Span::new(0, 0, 10, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_keywords_always_valid() {
        let table = SymbolTable::new();
        let mut scope = Scope::new();

        let keywords = vec![
            ExprKind::Prev,
            ExprKind::Current,
            ExprKind::Inputs,
            ExprKind::Dt,
            ExprKind::Self_,
            ExprKind::Other,
            ExprKind::Payload,
        ];

        for kind in keywords {
            let expr = Expr::new(kind, Span::new(0, 0, 4, 1));
            let mut errors = Vec::new();
            validate_expr(&expr, &table, &mut scope, &mut errors);
            assert!(errors.is_empty());
        }
    }
}
