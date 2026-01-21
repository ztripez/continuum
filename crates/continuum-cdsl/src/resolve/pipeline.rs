//! Unified compiler pipeline for Continuum CDSL.
//!
//! This module orchestrates the various resolution and validation passes
//! to transform raw parsed declarations into a fully resolved [`World`].

use crate::ast::{CompiledWorld, Declaration, Era, KernelRegistry, Node, World};
use crate::desugar::desugar_declarations;
use crate::error::CompileError;
use crate::foundation::{EntityId, Path, Type};
use crate::resolve::blocks::compile_execution_blocks;
use crate::resolve::eras::resolve_eras;
use crate::resolve::expr_typing::{type_expression, TypingContext};
use crate::resolve::graph::compile_graphs;
use crate::resolve::names::{build_symbol_table, validate_expr, Scope};
use crate::resolve::strata::{resolve_cadences, resolve_strata};
use crate::resolve::structure::{validate_collisions, validate_cycles};
use crate::resolve::types::{
    project_entity_types, resolve_node_types, resolve_user_types, TypeTable,
};
use crate::resolve::uses::validate_uses;
use crate::resolve::validation::{validate_node, validate_seq_escape};
use indexmap::IndexMap;
use std::collections::HashMap;

macro_rules! apply_node_pass {
    ($errors:expr, $globals:expr, $members:expr, |$nodes:ident| $body:expr) => {{
        {
            let $nodes = $globals;
            if let Err(mut e) = $body {
                $errors.append(&mut e);
            }
        }
        {
            let $nodes = $members;
            if let Err(mut e) = $body {
                $errors.append(&mut e);
            }
        }
    }};
}

macro_rules! extend_node_pass {
    ($errors:expr, $globals:expr, $members:expr, |$nodes:ident| $body:expr) => {{
        {
            let $nodes = $globals;
            $errors.extend($body);
        }
        {
            let $nodes = $members;
            $errors.extend($body);
        }
    }};
}

macro_rules! validate_node_pass {
    ($errors:expr, $globals:expr, $members:expr, |$node:ident| $body:expr) => {{
        for $node in $globals {
            if let Err(mut e) = $body {
                $errors.append(&mut e);
            }
        }
        for $node in $members {
            if let Err(mut e) = $body {
                $errors.append(&mut e);
            }
        }
    }};
}

/// Compiles a list of raw declarations into a resolved [`CompiledWorld`].
///
/// This is the main entry point for the CDSL compiler. It runs all
/// resolution and validation passes in the correct order.
///
/// # Pipeline Order
/// 1. **Name Resolution** - Build symbol table and validate paths.
/// 2. **Structural Validation (Collisions)** - Check for path namespace conflicts.
/// 3. **Type Table Projection** - Synthesize hierarchical entity types.
/// 4. **Type Resolution** - Resolve TypeExpr to semantic Type for all nodes.
/// 5. **Stratum Resolution** - Assign strata and resolve cadences.
/// 6. **Era Resolution** - Validate eras and transition graph.
/// 7. **Block Compilation** - Type-check and compile execution blocks.
/// 8. **Structural Validation (Cycles)** - Detect circular dependencies.
/// 9. **Final Validation** - Semantic checks (purity, bounds, uses).
/// 10. **Graph Compilation** - Build deterministic execution DAGs.
///
/// # Parameters
/// - `declarations`: Parsed declarations in source order.
///
/// # Returns
/// Fully resolved [`CompiledWorld`] with typed nodes and execution graphs.
///
/// # Errors
/// Returns all compile errors encountered during resolution or validation,
/// including missing `world` declarations, invalid `dt` expressions, or
/// missing/conflicting `:initial` era declarations.
///
/// # Examples
/// ```rust
/// use continuum_cdsl::lexer::Token;
/// use continuum_cdsl::parser::parse_declarations;
/// use continuum_cdsl::resolve::pipeline::compile;
/// use logos::Logos;
///
/// let source = r#"
/// world demo { }
///
/// strata sim { : stride(1) }
///
/// era main : initial : dt(1.0 <s>) {
///     strata { sim: active }
/// }
///
/// signal counter : type Scalar : stratum(sim) {
///     resolve { prev }
/// }
/// "#;
///
/// let tokens: Vec<_> = Token::lexer(source)
///     .collect::<Result<Vec<_>, _>>()
///     .unwrap();
/// let decls = parse_declarations(&tokens, 0).unwrap();
/// let compiled = compile(decls).unwrap();
/// assert_eq!(compiled.world.metadata.path.to_string(), "demo");
/// ```
pub fn compile(declarations: Vec<Declaration>) -> Result<CompiledWorld, Vec<CompileError>> {
    let mut errors = Vec::new();

    // 1. Desugar
    let declarations = desugar_declarations(declarations);

    // 2. Group declarations by kind
    let mut world_decl = None;
    let mut global_nodes = Vec::new();
    let mut member_nodes = Vec::new();
    let mut entities = IndexMap::new();
    let mut strata = Vec::new();
    let mut era_decls = Vec::new();
    let mut _type_decls = Vec::new();
    let mut _const_entries = Vec::new();
    let mut _config_entries = Vec::new();

    for decl in &declarations {
        match decl {
            Declaration::World(w) => world_decl = Some(w.clone()),
            Declaration::Node(n) => global_nodes.push(n.clone()),
            Declaration::Member(m) => member_nodes.push(m.clone()),
            Declaration::Entity(e) => {
                entities.insert(e.path.clone(), e.clone());
            }
            Declaration::Stratum(s) => strata.push(s.clone()),
            Declaration::Era(e) => era_decls.push(e.clone()),
            Declaration::Type(t) => _type_decls.push(t.clone()),
            Declaration::Const(c) => _const_entries.extend(c.clone()),
            Declaration::Config(c) => _config_entries.extend(c.clone()),
        }
    }

    let Some(metadata) = world_decl else {
        return Err(vec![CompileError::new(
            crate::error::ErrorKind::Internal,
            crate::foundation::Span::new(0, 0, 0, 0),
            "No 'world' declaration found".to_string(),
        )]);
    };

    let world_span = metadata.span;
    let mut world = World::new(metadata);
    world.declarations = declarations;

    // 3. Name Resolution (Symbol Table Building)
    let symbol_table = build_symbol_table(&world.declarations);
    errors.extend(validate_collisions(&world.declarations));

    // 4. Type Resolution
    let mut type_table = TypeTable::new();

    // Register explicit user types
    if let Err(mut e) = resolve_user_types(&world.declarations, &mut type_table) {
        errors.append(&mut e);
    }

    // Synthesize hierarchical entity types
    if let Err(mut e) = project_entity_types(&world.declarations, &mut type_table) {
        errors.append(&mut e);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Resolve node types (TypeExpr -> Type)
    apply_node_pass!(errors, &mut global_nodes, &mut member_nodes, |nodes| {
        resolve_node_types(nodes, &type_table)
    });

    if !errors.is_empty() {
        return Err(errors);
    }

    // 5. Stratum Resolution
    if let Err(mut e) = resolve_cadences(&mut strata) {
        errors.append(&mut e);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Convert Strata list to Map for easier lookup
    let mut strata_map = IndexMap::new();
    for s in &strata {
        strata_map.insert(s.path.clone(), s.clone());
    }

    let strata_vec: Vec<_> = strata_map.values().cloned().collect();
    apply_node_pass!(errors, &mut global_nodes, &mut member_nodes, |nodes| {
        resolve_strata(nodes, &strata_vec)
    });

    if !errors.is_empty() {
        return Err(errors);
    }

    // 6. Era Resolution
    let registry = KernelRegistry::global();
    let signal_types = collect_node_types(&global_nodes, &member_nodes);

    // Context maps for typing
    let field_types = HashMap::new();
    let config_types = HashMap::new();
    let const_types = HashMap::new();

    let ctx = TypingContext::new(
        &type_table,
        &registry,
        &signal_types,
        &field_types,
        &config_types,
        &const_types,
    );

    let mut resolved_eras = IndexMap::new();
    let mut initial_candidates: Vec<(Path, crate::foundation::Span)> = Vec::new();
    for era_decl in &era_decls {
        let initial_attrs: Vec<_> = era_decl
            .attributes
            .iter()
            .filter(|attr| attr.name == "initial")
            .collect();
        if initial_attrs.len() > 1 {
            errors.push(CompileError::new(
                crate::error::ErrorKind::Conflict,
                era_decl.span,
                format!(
                    "era '{}' declares multiple :initial attributes",
                    era_decl.path
                ),
            ));
        }
        if let Some(attr) = initial_attrs.first() {
            if !attr.args.is_empty() {
                errors.push(CompileError::new(
                    crate::error::ErrorKind::InvalidCapability,
                    attr.span,
                    format!(
                        "initial attribute expects no arguments, got {}",
                        attr.args.len()
                    ),
                ));
            } else {
                initial_candidates.push((era_decl.path.clone(), attr.span));
            }
        }

        let dt = match &era_decl.dt {
            Some(expr) => match type_expression(expr, &ctx) {
                Ok(typed) => typed,
                Err(mut e) => {
                    errors.append(&mut e);
                    continue;
                }
            },
            None => {
                errors.push(CompileError::new(
                    crate::error::ErrorKind::Internal,
                    era_decl.span,
                    "Era missing dt expression".to_string(),
                ));
                continue;
            }
        };

        let mut era = Era::new(
            crate::foundation::EraId::new(era_decl.path.to_string()),
            era_decl.path.clone(),
            dt,
            era_decl.span,
        );
        era.doc = era_decl.doc.clone();
        resolved_eras.insert(era.path.clone(), era);
    }

    let initial_era =
        resolve_initial_era(&resolved_eras, &initial_candidates, world_span, &mut errors);

    if !errors.is_empty() {
        return Err(errors);
    }

    let mut eras_vec: Vec<_> = resolved_eras.values().cloned().collect();
    let stratum_ids: Vec<_> = strata_map.values().map(|s| s.id.clone()).collect();
    let era_errors = resolve_eras(&mut eras_vec, &stratum_ids);
    errors.extend(era_errors);

    if !errors.is_empty() {
        return Err(errors);
    }

    // Update resolved eras back into map
    for era in eras_vec {
        resolved_eras.insert(era.path.clone(), era);
    }

    // 7. Block Compilation
    validate_node_pass!(errors, &mut global_nodes, &mut member_nodes, |node| {
        compile_execution_blocks(node, &ctx)
    });

    // 8. Structural Validation (Cycles)
    errors.extend(validate_cycles(&global_nodes, &member_nodes));

    // 9. Validation
    // dedicated Name Validation (uses SymbolTable)
    let mut scope = Scope::default();
    for decl in &world.declarations {
        match decl {
            Declaration::Node(node) => {
                for (_, body) in &node.execution_blocks {
                    if let crate::ast::BlockBody::Expression(expr) = body {
                        validate_expr(expr, &symbol_table, &mut scope, &mut errors);
                    }
                }
            }
            Declaration::Member(node) => {
                for (_, body) in &node.execution_blocks {
                    if let crate::ast::BlockBody::Expression(expr) = body {
                        validate_expr(expr, &symbol_table, &mut scope, &mut errors);
                    }
                }
            }
            _ => {}
        }
    }

    extend_node_pass!(errors, &global_nodes, &member_nodes, |nodes| {
        validate_uses(nodes, &registry)
    });

    extend_node_pass!(errors, &global_nodes, &member_nodes, |nodes| {
        validate_seq_escape(nodes)
    });

    validate_node_pass!(errors, &global_nodes, &member_nodes, |node| {
        validate_node(node, &type_table, &registry)
    });

    if !errors.is_empty() {
        return Err(errors);
    }

    // Final construction
    world.globals = global_nodes
        .into_iter()
        .map(|n| (n.path.clone(), n))
        .collect();
    world.members = member_nodes
        .into_iter()
        .map(|n| (n.path.clone(), n))
        .collect();
    world.entities = entities;
    world.strata = strata_map;
    world.eras = resolved_eras;
    world.initial_era = initial_era;

    // 9. Graph Compilation
    let dag_set = match compile_graphs(&world) {
        Ok(ds) => ds,
        Err(mut e) => {
            errors.append(&mut e);
            return Err(errors);
        }
    };

    Ok(CompiledWorld::new(world, dag_set))
}

fn collect_node_types(globals: &[Node<()>], members: &[Node<EntityId>]) -> HashMap<Path, Type> {
    let mut map = HashMap::new();
    for n in globals {
        if let Some(ty) = &n.output {
            let ty: Type = ty.clone();
            map.insert(n.path.clone(), ty);
        }
    }
    for n in members {
        if let Some(ty) = &n.output {
            let ty: Type = ty.clone();
            map.insert(n.path.clone(), ty);
        }
    }
    map
}

fn resolve_initial_era(
    eras: &IndexMap<Path, Era>,
    initial_candidates: &[(Path, crate::foundation::Span)],
    world_span: crate::foundation::Span,
    errors: &mut Vec<CompileError>,
) -> Option<crate::foundation::EraId> {
    if initial_candidates.len() > 1 {
        let mut error = CompileError::new(
            crate::error::ErrorKind::Conflict,
            initial_candidates[0].1,
            "multiple eras marked :initial; exactly one era can be initial".to_string(),
        );
        for (_, span) in initial_candidates.iter().skip(1) {
            error = error.with_label(*span, "also marked :initial".to_string());
        }
        errors.push(error);
        return None;
    }

    if let Some((path, span)) = initial_candidates.first() {
        match eras.get(path) {
            Some(era) => return Some(era.id.clone()),
            None => {
                errors.push(CompileError::new(
                    crate::error::ErrorKind::Internal,
                    *span,
                    format!("initial era '{}' not found in resolved eras", path),
                ));
                return None;
            }
        }
    }

    if eras.is_empty() {
        errors.push(CompileError::new(
            crate::error::ErrorKind::CompilationFailed,
            world_span,
            "world declares no eras; add at least one era marked :initial".to_string(),
        ));
        return None;
    }

    let first_span = eras
        .values()
        .next()
        .map(|era| era.span)
        .unwrap_or(world_span);
    errors.push(CompileError::new(
        crate::error::ErrorKind::Conflict,
        first_span,
        "no era marked :initial; add :initial to exactly one era".to_string(),
    ));
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        Attribute, BlockBody, Entity, EraDecl, Expr, Node, RoleData, Stratum, UnitExpr, WorldDecl,
    };
    use crate::foundation::{EntityId, Span, StratumId};

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    #[test]
    fn test_compile_basic_world() {
        let span = test_span();
        let world_path = Path::from_path_str("terra");
        let metadata = WorldDecl {
            path: world_path.clone(),
            title: Some("Terra".to_string()),
            version: Some("1.0.0".to_string()),
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
        };

        let stratum_id = StratumId::new("fast");
        let stratum = Stratum::new(stratum_id.clone(), Path::from_path_str("fast"), span);

        let entity_id = EntityId::new("plate");
        let entity = Entity::new(entity_id.clone(), Path::from_path_str("plate"), span);

        let member_path = Path::from_path_str("plate.mass");
        let mut member = Node::new(member_path, span, RoleData::Signal, entity_id);
        member.type_expr = Some(crate::ast::TypeExpr::Scalar { unit: None });
        member.attributes.push(Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                crate::ast::UntypedKind::Signal(Path::from_path_str("fast")),
                span,
            )],
            span,
        });
        member.execution_blocks.push((
            "resolve".to_string(),
            BlockBody::Expression(Expr::literal(100.0, None, span)),
        ));

        let era = EraDecl {
            path: Path::from_path_str("main"),
            dt: Some(Expr::literal(
                1.0,
                Some(UnitExpr::Base("s".to_string())),
                span,
            )),
            strata_policy: Vec::new(),
            transitions: Vec::new(),
            attributes: vec![Attribute {
                name: "initial".to_string(),
                args: Vec::new(),
                span,
            }],
            span,
            doc: None,
        };

        let decls = vec![
            Declaration::World(metadata),
            Declaration::Stratum(stratum),
            Declaration::Entity(entity),
            Declaration::Era(era),
            Declaration::Member(member),
        ];

        let world = compile(decls).expect("Compilation failed");

        assert_eq!(world.world.metadata.path, world_path);
        assert!(world
            .world
            .entities
            .contains_key(&Path::from_path_str("plate")));
        assert!(world
            .world
            .members
            .contains_key(&Path::from_path_str("plate.mass")));

        let mass_node = world
            .world
            .members
            .get(&Path::from_path_str("plate.mass"))
            .unwrap();
        assert!(mass_node.output.is_some());
        assert_eq!(mass_node.executions.len(), 1);
    }

    #[test]
    fn test_missing_initial_era_is_error() {
        let span = test_span();
        let metadata = WorldDecl {
            path: Path::from_path_str("terra"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
        };

        let era = EraDecl {
            path: Path::from_path_str("main"),
            dt: Some(Expr::literal(
                1.0,
                Some(UnitExpr::Base("s".to_string())),
                span,
            )),
            strata_policy: Vec::new(),
            transitions: Vec::new(),
            attributes: Vec::new(),
            span,
            doc: None,
        };

        let decls = vec![Declaration::World(metadata), Declaration::Era(era)];
        let errors = compile(decls).expect_err("expected missing initial era error");
        assert!(errors
            .iter()
            .any(|e| e.message.contains("no era marked :initial")));
    }

    #[test]
    fn test_multiple_initial_era_is_error() {
        let span = test_span();
        let metadata = WorldDecl {
            path: Path::from_path_str("terra"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
        };

        let initial_attr = Attribute {
            name: "initial".to_string(),
            args: Vec::new(),
            span,
        };
        let era_a = EraDecl {
            path: Path::from_path_str("main"),
            dt: Some(Expr::literal(
                1.0,
                Some(UnitExpr::Base("s".to_string())),
                span,
            )),
            strata_policy: Vec::new(),
            transitions: Vec::new(),
            attributes: vec![initial_attr.clone()],
            span,
            doc: None,
        };
        let era_b = EraDecl {
            path: Path::from_path_str("alt"),
            dt: Some(Expr::literal(
                1.0,
                Some(UnitExpr::Base("s".to_string())),
                span,
            )),
            strata_policy: Vec::new(),
            transitions: Vec::new(),
            attributes: vec![initial_attr],
            span,
            doc: None,
        };

        let decls = vec![
            Declaration::World(metadata),
            Declaration::Era(era_a),
            Declaration::Era(era_b),
        ];
        let errors = compile(decls).expect_err("expected multiple initial eras error");
        assert!(errors
            .iter()
            .any(|e| e.message.contains("multiple eras marked :initial")));
    }

    #[test]
    fn test_multiple_initial_attributes_single_era_is_error() {
        let span = test_span();
        let metadata = WorldDecl {
            path: Path::from_path_str("terra"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
        };

        let initial_attr = Attribute {
            name: "initial".to_string(),
            args: Vec::new(),
            span,
        };
        let era = EraDecl {
            path: Path::from_path_str("main"),
            dt: Some(Expr::literal(
                1.0,
                Some(UnitExpr::Base("s".to_string())),
                span,
            )),
            strata_policy: Vec::new(),
            transitions: Vec::new(),
            attributes: vec![initial_attr.clone(), initial_attr],
            span,
            doc: None,
        };

        let decls = vec![Declaration::World(metadata), Declaration::Era(era)];
        let errors = compile(decls).expect_err("expected duplicate initial error");
        assert!(errors
            .iter()
            .any(|e| e.message.contains("declares multiple :initial attributes")));
    }

    #[test]
    fn test_initial_era_with_args_is_error() {
        let span = test_span();
        let metadata = WorldDecl {
            path: Path::from_path_str("terra"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
        };

        let era = EraDecl {
            path: Path::from_path_str("main"),
            dt: Some(Expr::literal(
                1.0,
                Some(UnitExpr::Base("s".to_string())),
                span,
            )),
            strata_policy: Vec::new(),
            transitions: Vec::new(),
            attributes: vec![Attribute {
                name: "initial".to_string(),
                args: vec![Expr::literal(1.0, None, span)],
                span,
            }],
            span,
            doc: None,
        };

        let decls = vec![Declaration::World(metadata), Declaration::Era(era)];
        let errors = compile(decls).expect_err("expected initial args error");
        assert!(errors
            .iter()
            .any(|e| e.message.contains("initial attribute expects no arguments")));
    }

    #[test]
    fn test_initial_era_is_resolved() {
        let span = test_span();
        let metadata = WorldDecl {
            path: Path::from_path_str("terra"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
        };

        let era = EraDecl {
            path: Path::from_path_str("main"),
            dt: Some(Expr::literal(
                1.0,
                Some(UnitExpr::Base("s".to_string())),
                span,
            )),
            strata_policy: Vec::new(),
            transitions: Vec::new(),
            attributes: vec![Attribute {
                name: "initial".to_string(),
                args: Vec::new(),
                span,
            }],
            span,
            doc: None,
        };

        let decls = vec![Declaration::World(metadata), Declaration::Era(era)];
        let compiled = compile(decls).expect("expected compile success");
        assert_eq!(
            compiled.world.initial_era,
            Some(crate::foundation::EraId::new("main"))
        );
    }

    #[test]
    fn test_no_eras_is_error() {
        let span = test_span();
        let metadata = WorldDecl {
            path: Path::from_path_str("terra"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
        };

        let decls = vec![Declaration::World(metadata)];
        let errors = compile(decls).expect_err("expected no-era error");
        assert!(errors
            .iter()
            .any(|e| e.message.contains("world declares no eras")));
    }

    #[test]
    fn test_missing_dt_is_error() {
        let span = test_span();
        let metadata = WorldDecl {
            path: Path::from_path_str("terra"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
        };

        let era = EraDecl {
            path: Path::from_path_str("main"),
            dt: None,
            strata_policy: Vec::new(),
            transitions: Vec::new(),
            attributes: vec![Attribute {
                name: "initial".to_string(),
                args: Vec::new(),
                span,
            }],
            span,
            doc: None,
        };

        let decls = vec![Declaration::World(metadata), Declaration::Era(era)];
        let errors = compile(decls).expect_err("expected missing dt error");
        assert!(errors
            .iter()
            .any(|e| e.message.contains("Era missing dt expression")));
    }
}
