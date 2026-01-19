//! Unified compiler pipeline for Continuum CDSL.
//!
//! This module orchestrates the various resolution and validation passes
//! to transform raw parsed declarations into a fully resolved [`World`].

use crate::ast::{Declaration, Era, KernelRegistry, Node, World};
use crate::desugar::desugar_declarations;
use crate::error::CompileError;
use crate::foundation::{EntityId, Path, Type};
use crate::resolve::blocks::compile_execution_blocks;
use crate::resolve::eras::resolve_eras;
use crate::resolve::expr_typing::{TypingContext, type_expression};
use crate::resolve::names::{Scope, build_symbol_table, validate_expr};
use crate::resolve::strata::{resolve_cadences, resolve_strata};
use crate::resolve::types::{
    TypeTable, project_entity_types, resolve_node_types, resolve_user_types,
};
use crate::resolve::uses::validate_uses;
use crate::resolve::validation::{validate_node, validate_seq_escape};
use std::collections::HashMap;

/// Compiles a list of raw declarations into a resolved [`World`].
///
/// This is the main entry point for the CDSL compiler. It runs all
/// resolution and validation passes in the correct order.
///
/// # Pipeline Order
/// 1. **Name Resolution** - Build symbol table and validate paths.
/// 2. **Type Table Projection** - Synthesize hierarchical entity types.
/// 3. **Type Resolution** - Resolve TypeExpr to semantic Type for all nodes.
/// 4. **Expression Typing** - (Deferred to Block Compilation)
/// 5. **Stratum Resolution** - Assign strata and resolve cadences.
/// 6. **Era Resolution** - Validate eras and transition graph.
/// 7. **Block Compilation** - Type-check and compile execution blocks.
/// 8. **Validation** - Final semantic validation (purity, bounds, uses).
pub fn compile(declarations: Vec<Declaration>) -> Result<World, Vec<CompileError>> {
    let mut errors = Vec::new();

    // 1. Desugar
    let declarations = desugar_declarations(declarations);

    // 2. Group declarations by kind
    let mut world_decl = None;
    let mut global_nodes = Vec::new();
    let mut member_nodes = Vec::new();
    let mut entities = HashMap::new();
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

    let mut world = World::new(metadata);
    world.declarations = declarations;

    // 3. Name Resolution (Symbol Table Building)
    let symbol_table = build_symbol_table(&world.declarations);

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
    if let Err(mut e) = resolve_node_types(&mut global_nodes, &type_table) {
        errors.append(&mut e);
    }
    if let Err(mut e) = resolve_node_types(&mut member_nodes, &type_table) {
        errors.append(&mut e);
    }

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
    let mut strata_map = HashMap::new();
    for s in &strata {
        strata_map.insert(s.path.clone(), s.clone());
    }

    let strata_vec: Vec<_> = strata_map.values().cloned().collect();
    if let Err(mut e) = resolve_strata(&mut global_nodes, &strata_vec) {
        errors.append(&mut e);
    }
    if let Err(mut e) = resolve_strata(&mut member_nodes, &strata_vec) {
        errors.append(&mut e);
    }

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

    let mut resolved_eras = HashMap::new();
    for era_decl in &era_decls {
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
    for node in &mut global_nodes {
        if let Err(mut e) = compile_execution_blocks(node, &ctx) {
            errors.append(&mut e);
        }
    }
    for node in &mut member_nodes {
        if let Err(mut e) = compile_execution_blocks(node, &ctx) {
            errors.append(&mut e);
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // 8. Validation
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

    errors.extend(validate_uses(&global_nodes, &registry));
    errors.extend(validate_uses(&member_nodes, &registry));

    errors.extend(validate_seq_escape(&global_nodes));
    errors.extend(validate_seq_escape(&member_nodes));

    for node in &global_nodes {
        if let Err(mut e) = validate_node(node, &type_table, &registry) {
            errors.append(&mut e);
        }
    }
    for node in &member_nodes {
        if let Err(mut e) = validate_node(node, &type_table, &registry) {
            errors.append(&mut e);
        }
    }

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

    Ok(world)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Attribute, BlockBody, Entity, Expr, Node, RoleData, Stratum, WorldDecl};
    use crate::foundation::{EntityId, Span, StratumId};

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    #[test]
    fn test_compile_basic_world() {
        let span = test_span();
        let world_path = Path::from_str("terra");
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
        let stratum = Stratum::new(stratum_id.clone(), Path::from_str("fast"), span);

        let entity_id = EntityId::new("plate");
        let entity = Entity::new(entity_id.clone(), Path::from_str("plate"), span);

        let member_path = Path::from_str("plate.mass");
        let mut member = Node::new(member_path, span, RoleData::Signal, entity_id);
        member.type_expr = Some(crate::ast::TypeExpr::Scalar { unit: None });
        member.attributes.push(Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                crate::ast::UntypedKind::Signal(Path::from_str("fast")),
                span,
            )],
            span,
        });
        member.execution_blocks.push((
            "resolve".to_string(),
            BlockBody::Expression(Expr::literal(100.0, None, span)),
        ));

        let decls = vec![
            Declaration::World(metadata),
            Declaration::Stratum(stratum),
            Declaration::Entity(entity),
            Declaration::Member(member),
        ];

        let world = compile(decls).expect("Compilation failed");

        assert_eq!(world.metadata.path, world_path);
        assert!(world.entities.contains_key(&Path::from_str("plate")));
        assert!(world.members.contains_key(&Path::from_str("plate.mass")));

        let mass_node = world.members.get(&Path::from_str("plate.mass")).unwrap();
        assert!(mass_node.output.is_some());
        assert_eq!(mass_node.executions.len(), 1);
    }
}
