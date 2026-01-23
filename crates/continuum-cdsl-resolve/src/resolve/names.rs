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

use crate::error::{CompileError, ErrorKind};
use continuum_cdsl_ast::foundation::{EntityId, Path};
use continuum_cdsl_ast::{Declaration, Expr, UntypedKind as ExprKind};
use std::collections::{HashMap, HashSet};

/// Symbol table for name resolution
///
/// Tracks all declared symbols in the world to validate Path references.
/// Symbols are separated by kind to enforce observer boundary (signals vs fields).
#[derive(Debug, Default)]
pub struct SymbolTable {
    /// Signals (authoritative state)
    signals: HashSet<Path>,

    /// Fields (observer-only)
    fields: HashSet<Path>,

    /// Operators (execution blocks)
    operators: HashSet<Path>,

    /// Impulses (external inputs)
    impulses: HashSet<Path>,

    /// Fractures (tension detectors)
    fractures: HashSet<Path>,

    /// Chronicles (observer-only recorders)
    chronicles: HashSet<Path>,

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

    /// Register a signal
    pub fn register_signal(&mut self, path: Path) {
        self.signals.insert(path);
    }

    /// Register a field
    pub fn register_field(&mut self, path: Path) {
        self.fields.insert(path);
    }

    /// Register an operator
    pub fn register_operator(&mut self, path: Path) {
        self.operators.insert(path);
    }

    /// Register an impulse
    pub fn register_impulse(&mut self, path: Path) {
        self.impulses.insert(path);
    }

    /// Register a fracture
    pub fn register_fracture(&mut self, path: Path) {
        self.fractures.insert(path);
    }

    /// Register a chronicle
    pub fn register_chronicle(&mut self, path: Path) {
        self.chronicles.insert(path);
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

    /// Check if a path is a declared signal
    pub fn has_signal(&self, path: &Path) -> bool {
        self.signals.contains(path)
    }

    /// Check if a path is a declared field
    pub fn has_field(&self, path: &Path) -> bool {
        self.fields.contains(path)
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

/// Build symbol table from world declarations.
///
/// Collects all declared symbols from the list of top-level declarations.
///
/// # Parameters
/// - `declarations`: The list of all top-level declarations from the parser.
///
/// # Returns
/// Symbol table containing all declared symbols, separated by kind.
pub fn build_symbol_table(declarations: &[Declaration]) -> SymbolTable {
    use continuum_cdsl_ast::RoleId;

    let mut table = SymbolTable::new();

    for decl in declarations {
        match decl {
            Declaration::Node(node) => match node.role_id() {
                RoleId::Signal => table.register_signal(node.path.clone()),
                RoleId::Field => table.register_field(node.path.clone()),
                RoleId::Operator => table.register_operator(node.path.clone()),
                RoleId::Impulse => table.register_impulse(node.path.clone()),
                RoleId::Fracture => table.register_fracture(node.path.clone()),
                RoleId::Chronicle => table.register_chronicle(node.path.clone()),
            },
            Declaration::Member(node) => {
                table.register_member(node.index.clone(), node.path.clone());
            }
            Declaration::Type(type_decl) => {
                table.register_type(Path::from(type_decl.name.as_str()));
            }
            Declaration::Const(entries) => {
                for entry in entries {
                    table.register_const(entry.path.clone());
                }
            }
            Declaration::Config(entries) => {
                for entry in entries {
                    table.register_config(entry.path.clone());
                }
            }
            _ => {}
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
    #[allow(dead_code)]
    fn new() -> Self {
        Self::default()
    }

    /// Push a new scope level
    fn push(&mut self) {
        self.locals.push(HashSet::new());
    }

    /// Pop the current scope level
    ///
    /// # Panics
    ///
    /// Panics if called with no active scope (programming error).
    fn pop(&mut self) {
        assert!(
            !self.locals.is_empty(),
            "Cannot pop scope: no active scope exists"
        );
        self.locals.pop();
    }

    /// Bind a local variable in the current scope
    ///
    /// # Panics
    ///
    /// Panics if called with no active scope (programming error).
    fn bind(&mut self, name: String) {
        assert!(
            !self.locals.is_empty(),
            "Cannot bind variable: no active scope exists"
        );
        self.locals.last_mut().unwrap().insert(name);
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
///
/// # Special Handling
///
/// **FieldAccess with Local roots**: When a FieldAccess expression has a Local
/// identifier as its root (e.g., `core.temp`), the root is NOT validated here.
/// This is because such expressions could be:
/// 1. Bare path references to signals/fields (`core.temp` → signal path)
/// 2. Actual field access on local variables (`obj.field`)
///
/// Distinguishing these cases requires type context, so validation is deferred
/// to type resolution (see `type_field_access()` in `expr_typing/helpers.rs`).
///
/// This deferral is safe because:
/// - ALL expressions must pass through type resolution before execution
/// - Type resolution validates bare paths against signal/field registries
/// - Invalid paths are caught and reported with proper diagnostics
pub fn validate_expr(
    expr: &Expr,
    table: &SymbolTable,
    scope: &mut Scope,
    errors: &mut Vec<CompileError>,
) {
    match &expr.kind {
        // === Path references ===
        ExprKind::Signal(path) => {
            if !table.has_signal(path) {
                errors.push(CompileError::new(
                    ErrorKind::UndefinedName,
                    expr.span,
                    format!("undefined signal '{}'", path),
                ));
            }
        }

        ExprKind::Field(path) => {
            if !table.has_field(path) {
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
            source,
            binding,
            body,
        } => {
            // Validate source
            validate_expr(source, table, scope, errors);

            // Push new scope for body
            scope.push();
            scope.bind(binding.clone());

            // Validate body
            validate_expr(body, table, scope, errors);

            // Pop scope
            scope.pop();
        }

        ExprKind::Fold {
            source,
            init,
            acc,
            elem,
            body,
        } => {
            // Validate source and init in current scope
            validate_expr(source, table, scope, errors);
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

        ExprKind::Filter { source, predicate } => {
            validate_expr(source, table, scope, errors);

            // Predicate has 'self' in scope
            scope.push();
            scope.bind("self".to_string());
            validate_expr(predicate, table, scope, errors);
            scope.pop();
        }

        ExprKind::Nearest { entity, position } => {
            // Validate entity reference
            // For now, let's just make sure it's a valid entity type in the future.
            // SymbolTable currently doesn't track entity IDs directly, but world nodes.
            // But we can check if it exists if we want.
            validate_expr(position, table, scope, errors);
        }

        ExprKind::Within {
            entity,
            position,
            radius,
        } => {
            validate_expr(position, table, scope, errors);
            validate_expr(radius, table, scope, errors);
        }

        ExprKind::Entity(_) | ExprKind::OtherInstances(_) | ExprKind::PairsInstances(_) => {
            // These are direct references to entity sets
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
            // FieldAccess chains are validated during type resolution, not name resolution.
            //
            // Rationale for deferring validation:
            // - Bare path resolution requires type context (signal/field registries)
            // - Name resolution doesn't distinguish `core.temp` (bare path) from `obj.field` (access)
            // - Premature validation would reject valid bare paths as undefined locals
            //
            // Guaranteed validation in type resolution (expr_typing/helpers.rs:226-279):
            // 1. Attempts bare path resolution via try_extract_path()
            // 2. If bare path fails, types object via type_expression() (validates object)
            // 3. Validates field exists on object's type (lines 252-279)
            // 4. Returns error if field doesn't exist
            //
            // This is safe because ALL expressions must pass through type resolution.

            // Only validate the object if it's NOT a bare Local identifier that could be a bare path root.
            // Bare path roots like `core` in `core.temp` should be resolved during type resolution,
            // not rejected here as undefined variables.
            if !matches!(object.kind, ExprKind::Local(_)) {
                validate_expr(object, table, scope, errors);
            }
            // If object is Local, skip validation - will be validated as bare path in type resolution
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

        // === Parse errors ===
        ExprKind::ParseError(msg) => {
            // Parser errors should be reported during name resolution
            errors.push(CompileError::new(
                ErrorKind::Syntax,
                expr.span,
                format!("parse error: {}", msg),
            ));
        }

        // === Literals and keywords (always valid) ===
        ExprKind::Literal { .. }
        | ExprKind::BoolLiteral(_)
        | ExprKind::StringLiteral(_)
        | ExprKind::Prev
        | ExprKind::Current
        | ExprKind::Inputs
        | ExprKind::Self_
        | ExprKind::Other
        | ExprKind::Payload => {
            // No validation needed
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use continuum_cdsl_ast::foundation::{AggregateOp, Span};
    use continuum_cdsl_ast::{Node, RoleData};

    fn make_path(s: &str) -> Path {
        Path::from_path_str(s)
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
    fn test_symbol_table_register_signal() {
        let mut table = SymbolTable::new();
        table.register_signal(make_path("temperature"));

        assert!(table.has_signal(&make_path("temperature")));
        assert!(!table.has_signal(&make_path("pressure")));
    }

    #[test]
    fn test_symbol_table_register_field() {
        let mut table = SymbolTable::new();
        table.register_field(make_path("elevation"));

        assert!(table.has_field(&make_path("elevation")));
        assert!(!table.has_field(&make_path("temperature")));
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
        let globals = vec![
            Declaration::Node(make_node("temperature")),
            Declaration::Node(make_node("pressure")),
        ];

        let table = build_symbol_table(&globals);

        assert!(table.has_signal(&make_path("temperature")));
        assert!(table.has_signal(&make_path("pressure")));
        assert!(!table.has_signal(&make_path("velocity")));
    }

    #[test]
    fn test_validate_signal_reference_valid() {
        let mut table = SymbolTable::new();
        table.register_signal(make_path("temperature"));

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
        let table = SymbolTable::new();
        let span = Span::new(0, 0, 10, 1);

        let expr = Expr::new(
            ExprKind::Aggregate {
                op: AggregateOp::Count,
                source: Box::new(Expr::new(ExprKind::Entity(EntityId::new("plates")), span)),
                binding: "p".to_string(),
                body: Box::new(Expr::new(
                    ExprKind::Local("p".to_string()),
                    Span::new(0, 0, 1, 1),
                )),
            },
            span,
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

    #[test]
    fn test_validate_config_reference() {
        let mut table = SymbolTable::new();
        table.register_config(make_path("world.gravity"));

        let expr = Expr::new(
            ExprKind::Config(make_path("world.gravity")),
            Span::new(0, 0, 10, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_const_reference() {
        let mut table = SymbolTable::new();
        table.register_const(make_path("math.pi"));

        let expr = Expr::new(
            ExprKind::Const(make_path("math.pi")),
            Span::new(0, 0, 10, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_field_reference() {
        let mut table = SymbolTable::new();
        table.register_field(make_path("elevation"));

        let expr = Expr::new(
            ExprKind::Field(make_path("elevation")),
            Span::new(0, 0, 10, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_multiple_unresolved_paths() {
        let table = SymbolTable::new();

        let expr = Expr::new(
            ExprKind::KernelCall {
                kernel: continuum_kernel_types::KernelId::new("maths", "add"),
                args: vec![
                    Expr::new(
                        ExprKind::Signal(make_path("missing_a")),
                        Span::new(0, 0, 1, 1),
                    ),
                    Expr::new(
                        ExprKind::Field(make_path("missing_b")),
                        Span::new(0, 2, 3, 1),
                    ),
                ],
            },
            Span::new(0, 0, 3, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert_eq!(errors.len(), 2);
        assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
        assert!(matches!(errors[1].kind, ErrorKind::UndefinedName));
    }

    #[test]
    fn test_validate_aggregate_scope_isolation() {
        let table = SymbolTable::new();
        let span = Span::new(0, 0, 5, 1);

        // aggregate(..., p) { p }; p  (p should be undefined here)
        let expr = Expr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(Expr::new(
                    ExprKind::Aggregate {
                        op: AggregateOp::Count,
                        source: Box::new(Expr::new(
                            ExprKind::Entity(EntityId::new("plates")),
                            span,
                        )),
                        binding: "p".to_string(),
                        body: Box::new(Expr::new(
                            ExprKind::Local("p".to_string()),
                            Span::new(0, 2, 3, 1),
                        )),
                    },
                    Span::new(0, 0, 3, 1),
                )),
                body: Box::new(Expr::new(
                    ExprKind::Local("p".to_string()),
                    Span::new(0, 4, 5, 1),
                )),
            },
            span,
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    }

    #[test]
    fn test_validate_fold_bindings() {
        let table = SymbolTable::new();
        let span = Span::new(0, 0, 5, 1);

        // fold(init, acc, elem) { acc + elem }
        let expr = Expr::new(
            ExprKind::Fold {
                source: Box::new(Expr::new(ExprKind::Entity(EntityId::new("plates")), span)),
                init: Box::new(Expr::new(
                    ExprKind::Literal {
                        value: 0.0,
                        unit: None,
                    },
                    Span::new(0, 0, 1, 1),
                )),
                acc: "acc".to_string(),
                elem: "elem".to_string(),
                body: Box::new(Expr::new(
                    ExprKind::KernelCall {
                        kernel: continuum_kernel_types::KernelId::new("maths", "add"),
                        args: vec![
                            Expr::new(ExprKind::Local("acc".to_string()), Span::new(0, 2, 3, 1)),
                            Expr::new(ExprKind::Local("elem".to_string()), Span::new(0, 4, 5, 1)),
                        ],
                    },
                    Span::new(0, 2, 5, 1),
                )),
            },
            span,
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert!(errors.is_empty());
    }

    #[test]
    fn test_validate_fold_scope_isolation() {
        let table = SymbolTable::new();
        let span = Span::new(0, 0, 5, 1);

        // let x = fold(0, acc, elem) { acc }; elem  (elem should be undefined here)
        let expr = Expr::new(
            ExprKind::Let {
                name: "x".to_string(),
                value: Box::new(Expr::new(
                    ExprKind::Fold {
                        source: Box::new(Expr::new(
                            ExprKind::Entity(EntityId::new("plates")),
                            span,
                        )),
                        init: Box::new(Expr::new(
                            ExprKind::Literal {
                                value: 0.0,
                                unit: None,
                            },
                            Span::new(0, 0, 1, 1),
                        )),
                        acc: "acc".to_string(),
                        elem: "elem".to_string(),
                        body: Box::new(Expr::new(
                            ExprKind::Local("acc".to_string()),
                            Span::new(0, 2, 3, 1),
                        )),
                    },
                    Span::new(0, 0, 3, 1),
                )),
                body: Box::new(Expr::new(
                    ExprKind::Local("elem".to_string()),
                    Span::new(0, 4, 5, 1),
                )),
            },
            span,
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::UndefinedName));
    }

    #[test]
    fn test_validate_parse_error_reported() {
        let table = SymbolTable::new();

        let expr = Expr::new(
            ExprKind::ParseError("expected expression".to_string()),
            Span::new(0, 0, 5, 1),
        );

        let mut errors = Vec::new();
        let mut scope = Scope::new();
        validate_expr(&expr, &table, &mut scope, &mut errors);

        assert_eq!(errors.len(), 1);
        assert!(matches!(errors[0].kind, ErrorKind::Syntax));
    }
}
