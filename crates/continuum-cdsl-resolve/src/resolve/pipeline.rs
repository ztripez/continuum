//! Unified compiler pipeline for Continuum CDSL.
//!
//! This module orchestrates the various resolution and validation passes
//! to transform raw parsed declarations into a fully resolved [`World`].

use crate::desugar::desugar_declarations;
use crate::error::CompileError;
use crate::resolve::blocks::compile_execution_blocks;
use crate::resolve::entities::flatten_entity_members;
use crate::resolve::eras::resolve_eras;
use crate::resolve::expr_typing::{type_expression, TypingContext};
use crate::resolve::graph::compile_graphs;
use crate::resolve::integrators::validate_integrators;
use crate::resolve::names::{build_symbol_table, validate_expr, Scope};
use crate::resolve::strata::{resolve_cadences, resolve_strata};
use crate::resolve::structure::{validate_collisions, validate_cycles};
use crate::resolve::types::{
    project_entity_types, resolve_node_types, resolve_user_types, TypeTable,
};
use crate::resolve::uses::validate_uses;
use crate::resolve::validation::{validate_node, validate_seq_escape};
use continuum_cdsl_ast::foundation::{EntityId, Path, Type};
use continuum_cdsl_ast::{
    CompiledWorld, ConfigEntry, ConstEntry, Declaration, Era, Expr, KernelRegistry, Node, World,
    WorldDecl,
};
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
pub fn compile(declarations: Vec<Declaration>) -> Result<CompiledWorld, Vec<CompileError>> {
    let mut errors = Vec::new();

    // 1. Flatten entity nested members into standalone member declarations
    // This must happen BEFORE desugar so that member expressions get desugared
    let declarations = flatten_entity_members(declarations, &mut errors);

    // 2. Desugar
    let declarations = desugar_declarations(declarations);

    // 3. Group declarations by kind
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
            Declaration::Function(_) => {
                // Functions are resolved separately in expression context
            }
        }
    }

    let Some(mut metadata) = world_decl else {
        return Err(vec![CompileError::new(
            crate::error::ErrorKind::Internal,
            continuum_cdsl_ast::foundation::Span::new(0, 0, 0, 0),
            "No 'world' declaration found".to_string(),
        )]);
    };

    resolve_world_metadata(&mut metadata, &mut errors);

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

    // 5.5. Inject Debug Fields
    if world.metadata.debug {
        inject_debug_fields(&mut global_nodes, &mut member_nodes);
    }

    // 6. Era Resolution
    let registry = KernelRegistry::global();
    let signal_types = collect_node_types(&global_nodes, &member_nodes);

    // Context maps for typing
    let field_types = HashMap::new();
    let config_types = match collect_config_types(&_config_entries, &type_table) {
        Ok(types) => types,
        Err(mut e) => {
            errors.append(&mut e);
            HashMap::new()
        }
    };
    let const_types = match collect_const_types(&_const_entries, &type_table) {
        Ok(types) => types,
        Err(mut e) => {
            errors.append(&mut e);
            HashMap::new()
        }
    };

    let ctx = TypingContext::new(
        &type_table,
        &registry,
        &signal_types,
        &field_types,
        &config_types,
        &const_types,
    );

    let mut resolved_eras = IndexMap::new();
    let mut initial_candidates: Vec<(Path, continuum_cdsl_ast::foundation::Span)> = Vec::new();
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
            continuum_cdsl_ast::foundation::EraId::new(era_decl.path.to_string()),
            era_decl.path.clone(),
            dt,
            era_decl.span,
        );
        era.doc = era_decl.doc.clone();

        // Convert parsed strata policy entries to resolved StratumPolicy
        for entry in &era_decl.strata_policy {
            let stratum_id =
                continuum_cdsl_ast::foundation::StratumId::new(entry.stratum.to_string());

            // Parse state_name into active flag
            let active = match entry.state_name.to_lowercase().as_str() {
                "active" => true,
                "gated" => false,
                other => {
                    errors.push(CompileError::new(
                        crate::error::ErrorKind::InvalidCapability,
                        entry.span,
                        format!(
                            "unknown stratum state '{}', expected 'active' or 'gated'",
                            other
                        ),
                    ));
                    continue;
                }
            };

            era.strata_policy.push(continuum_cdsl_ast::StratumPolicy {
                stratum: stratum_id,
                active,
                cadence_override: entry.stride,
            });
        }

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
                    if let continuum_cdsl_ast::BlockBody::Expression(expr) = body {
                        validate_expr(expr, &symbol_table, &mut scope, &mut errors);
                    }
                }
            }
            Declaration::Member(node) => {
                for (_, body) in &node.execution_blocks {
                    if let continuum_cdsl_ast::BlockBody::Expression(expr) = body {
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

    extend_node_pass!(errors, &mut global_nodes, &mut member_nodes, |nodes| {
        validate_integrators(nodes, &registry)
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

/// Generic helper to collect types from entries with optional inference.
///
/// This helper deduplicates the type collection logic for config and const entries.
///
/// # Type Parameters
///
/// * `T` - Entry type (ConfigEntry or ConstEntry)
/// * `F` - Function type for extracting the value expression for inference
///
/// # Arguments
///
/// * `entries` - Slice of entries to process
/// * `type_table` - Type table for resolving explicit TypeExpr
/// * `get_value_expr` - Function that extracts the value expression for inference
/// * `entry_kind` - String describing the entry kind ("Config" or "Const") for error messages
fn collect_types_generic<T, F>(
    entries: &[T],
    type_table: &TypeTable,
    get_value_expr: F,
    entry_kind: &str,
) -> Result<HashMap<Path, Type>, Vec<CompileError>>
where
    T: HasTypeInfo,
    F: Fn(&T) -> Option<&Expr>,
{
    use crate::resolve::types::{infer_type_from_expr, resolve_type_expr};
    use continuum_cdsl_ast::TypeExpr;

    let mut map = HashMap::new();
    let mut errors = Vec::new();

    for entry in entries {
        let ty = if matches!(entry.type_expr(), TypeExpr::Infer) {
            // Type inference from value expression (literal expressions only)
            if let Some(value_expr) = get_value_expr(entry) {
                match infer_type_from_expr(value_expr, entry.span()) {
                    Ok(ty) => ty,
                    Err(e) => {
                        errors.push(e);
                        continue;
                    }
                }
            } else {
                errors.push(CompileError::new(
                    crate::error::ErrorKind::TypeMismatch,
                    entry.span(),
                    format!(
                        "{} entry '{}' requires explicit type or default value for inference",
                        entry_kind,
                        entry.path()
                    ),
                ));
                continue;
            }
        } else {
            // Explicit type
            match resolve_type_expr(entry.type_expr(), type_table, entry.span()) {
                Ok(ty) => ty,
                Err(e) => {
                    errors.push(e);
                    continue;
                }
            }
        };

        map.insert(entry.path().clone(), ty);
    }

    if errors.is_empty() {
        Ok(map)
    } else {
        Err(errors)
    }
}

/// Trait for extracting type information from config/const entries.
trait HasTypeInfo {
    fn type_expr(&self) -> &continuum_cdsl_ast::TypeExpr;
    fn span(&self) -> continuum_cdsl_ast::foundation::Span;
    fn path(&self) -> &Path;
}

impl HasTypeInfo for ConfigEntry {
    fn type_expr(&self) -> &continuum_cdsl_ast::TypeExpr {
        &self.type_expr
    }
    fn span(&self) -> continuum_cdsl_ast::foundation::Span {
        self.span
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl HasTypeInfo for ConstEntry {
    fn type_expr(&self) -> &continuum_cdsl_ast::TypeExpr {
        &self.type_expr
    }
    fn span(&self) -> continuum_cdsl_ast::foundation::Span {
        self.span
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

/// Collect types for config entries.
///
/// Resolves TypeExpr to Type for each config entry and builds a lookup map.
/// If TypeExpr is Infer, infers type from default value expression.
///
/// # Phase Boundary Safety
///
/// This function is called during type collection phase, BEFORE full expression typing.
/// However, it's safe to call `infer_type_from_expr` (via `collect_types_generic`) here because:
///
/// 1. **Literal-only inference**: `infer_type_from_expr` ONLY handles literal expressions:
///    - `BoolLiteral` → `Bool`
///    - `Literal{value, unit}` → `Scalar<unit>`  
///    - `Vector([elements])` → `Vector<dim>`
///    - All other expressions → Error (requires explicit type)
///
/// 2. **No registry dependencies**: Literal type inference doesn't require signal/field/const registries
///
/// 3. **Fail loudly**: Complex expressions that would create circular dependencies are rejected
///    with clear error messages
///
/// This is NOT a phase boundary violation - it's literal type inference, not full expression typing.
fn collect_config_types(
    entries: &[ConfigEntry],
    type_table: &TypeTable,
) -> Result<HashMap<Path, Type>, Vec<CompileError>> {
    collect_types_generic(
        entries,
        type_table,
        |entry| entry.default.as_ref(),
        "Config",
    )
}

/// Collect types for const entries.
///
/// Resolves TypeExpr to Type for each const entry and builds a lookup map.
/// If TypeExpr is Infer, infers type from value expression.
///
/// # Phase Boundary Safety
///
/// This function is called during type collection phase, BEFORE full expression typing.
/// However, it's safe to call `infer_type_from_expr` (via `collect_types_generic`) here because:
///
/// 1. **Literal-only inference**: `infer_type_from_expr` ONLY handles literal expressions:
///    - `BoolLiteral` → `Bool`
///    - `Literal{value, unit}` → `Scalar<unit>`  
///    - `Vector([elements])` → `Vector<dim>`
///    - All other expressions → Error (requires explicit type)
///
/// 2. **No registry dependencies**: Literal type inference doesn't require signal/field/const registries
///
/// 3. **Fail loudly**: Complex expressions that would create circular dependencies are rejected
///    with clear error messages
///
/// This is NOT a phase boundary violation - it's literal type inference, not full expression typing.
fn collect_const_types(
    entries: &[ConstEntry],
    type_table: &TypeTable,
) -> Result<HashMap<Path, Type>, Vec<CompileError>> {
    collect_types_generic(entries, type_table, |entry| Some(&entry.value), "Const")
}

fn resolve_world_metadata(metadata: &mut WorldDecl, errors: &mut Vec<CompileError>) {
    for attr in &metadata.attributes {
        match attr.name.as_str() {
            "title" => {
                if attr.args.len() != 1 {
                    errors.push(CompileError::new(
                        crate::error::ErrorKind::InvalidCapability,
                        attr.span,
                        "title attribute expects 1 argument".to_string(),
                    ));
                    continue;
                }
                match &attr.args[0].kind {
                    continuum_cdsl_ast::UntypedKind::StringLiteral(s) => {
                        metadata.title = Some(s.clone())
                    }
                    _ => errors.push(CompileError::new(
                        crate::error::ErrorKind::TypeMismatch,
                        attr.args[0].span,
                        "title attribute expects a string literal".to_string(),
                    )),
                }
            }
            "version" => {
                if attr.args.len() != 1 {
                    errors.push(CompileError::new(
                        crate::error::ErrorKind::InvalidCapability,
                        attr.span,
                        "version attribute expects 1 argument".to_string(),
                    ));
                    continue;
                }
                match &attr.args[0].kind {
                    continuum_cdsl_ast::UntypedKind::StringLiteral(s) => {
                        metadata.version = Some(s.clone())
                    }
                    _ => errors.push(CompileError::new(
                        crate::error::ErrorKind::TypeMismatch,
                        attr.args[0].span,
                        "version attribute expects a string literal".to_string(),
                    )),
                }
            }
            "description" => {
                if attr.args.len() != 1 {
                    errors.push(CompileError::new(
                        crate::error::ErrorKind::InvalidCapability,
                        attr.span,
                        "description attribute expects 1 argument".to_string(),
                    ));
                    continue;
                }
                match &attr.args[0].kind {
                    continuum_cdsl_ast::UntypedKind::StringLiteral(s) => {
                        metadata.description = Some(s.clone())
                    }
                    _ => errors.push(CompileError::new(
                        crate::error::ErrorKind::TypeMismatch,
                        attr.args[0].span,
                        "description attribute expects a string literal".to_string(),
                    )),
                }
            }
            "debug" => {
                if metadata.debug {
                    errors.push(CompileError::new(
                        crate::error::ErrorKind::Conflict,
                        attr.span,
                        "duplicate :debug attribute in world block".to_string(),
                    ));
                    continue;
                }
                if !attr.args.is_empty() {
                    errors.push(CompileError::new(
                        crate::error::ErrorKind::InvalidCapability,
                        attr.span,
                        "debug attribute expects no arguments".to_string(),
                    ));
                    continue;
                }
                metadata.debug = true;
            }
            _ => {
                errors.push(CompileError::new(
                    crate::error::ErrorKind::Internal,
                    attr.span,
                    format!("Unknown world attribute: {}", attr.name),
                ));
            }
        }
    }
}

fn inject_debug_fields(global_nodes: &mut Vec<Node<()>>, member_nodes: &mut Vec<Node<EntityId>>) {
    use continuum_cdsl_ast::RoleId;

    let mut debug_globals = Vec::new();
    for node in global_nodes.iter() {
        if node.role_id() == RoleId::Signal {
            debug_globals.push(create_debug_node(node, ()));
        }
    }
    global_nodes.extend(debug_globals);

    let mut debug_members = Vec::new();
    for node in member_nodes.iter() {
        if node.role_id() == RoleId::Signal {
            debug_members.push(create_debug_node(node, node.index.clone()));
        }
    }
    member_nodes.extend(debug_members);
}

fn create_debug_node<I: continuum_cdsl_ast::Index>(source: &Node<I>, index: I) -> Node<I> {
    use continuum_cdsl_ast::{BlockBody, RoleData};

    let mut debug_path = Path::from("debug");
    for segment in &source.path.segments {
        debug_path.segments.push(segment.clone());
    }

    let mut debug_node = Node::new(
        debug_path,
        source.span,
        RoleData::Field {
            reconstruction: None,
        },
        index,
    );
    // Copy resolved metadata
    debug_node.type_expr = source.type_expr.clone();
    debug_node.output = source.output.clone();
    debug_node.stratum = source.stratum.clone();
    debug_node.doc = source.doc.clone();
    debug_node.title = source.title.as_ref().map(|t| format!("Debug: {}", t));
    debug_node.file = source.file.clone();

    // Create measure block: signal.path
    let expr = Expr::new(
        continuum_cdsl_ast::UntypedKind::Signal(source.path.clone()),
        source.span,
    );
    debug_node
        .execution_blocks
        .push(("measure".to_string(), BlockBody::Expression(expr)));

    debug_node
}

fn resolve_initial_era(
    eras: &IndexMap<Path, Era>,
    initial_candidates: &[(Path, continuum_cdsl_ast::foundation::Span)],
    world_span: continuum_cdsl_ast::foundation::Span,
    errors: &mut Vec<CompileError>,
) -> Option<continuum_cdsl_ast::foundation::EraId> {
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
    use continuum_cdsl_ast::foundation::{EntityId, Span, StratumId};
    use continuum_cdsl_ast::{
        Attribute, BlockBody, Entity, EraDecl, Expr, Node, RoleData, Stratum, UnitExpr, WorldDecl,
        WorldPolicy,
    };

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
            debug: false,
            policy: WorldPolicy::default(),
        };

        let stratum_id = StratumId::new("fast");
        let stratum = Stratum::new(stratum_id.clone(), Path::from_path_str("fast"), span);

        let entity_id = EntityId::new("plate");
        let entity = Entity::new(entity_id.clone(), Path::from_path_str("plate"), span);

        let member_path = Path::from_path_str("plate.mass");
        let mut member = Node::new(member_path, span, RoleData::Signal, entity_id);
        member.type_expr = Some(continuum_cdsl_ast::TypeExpr::Scalar {
            unit: None,
            bounds: None,
        });
        member.attributes.push(Attribute {
            name: "stratum".to_string(),
            args: vec![Expr::new(
                continuum_cdsl_ast::UntypedKind::Signal(Path::from_path_str("fast")),
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
            debug: false,
            policy: WorldPolicy::default(),
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
            debug: false,
            policy: WorldPolicy::default(),
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
            debug: false,
            policy: WorldPolicy::default(),
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
            debug: false,
            policy: WorldPolicy::default(),
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
            debug: false,
            policy: WorldPolicy::default(),
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
            Some(continuum_cdsl_ast::foundation::EraId::new("main"))
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
            debug: false,
            policy: WorldPolicy::default(),
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
            debug: false,
            policy: WorldPolicy::default(),
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

    #[test]
    fn test_inject_debug_fields() {
        let span = test_span();
        let metadata = WorldDecl {
            path: Path::from_path_str("terra"),
            title: None,
            version: None,
            warmup: None,
            attributes: vec![],
            span,
            doc: None,
            debug: true,
            policy: WorldPolicy::default(),
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

        let stratum_path = Path::from_path_str("sim");
        let stratum = Stratum::new(StratumId::new("sim"), stratum_path.clone(), span);

        let signal_path = Path::from_path_str("gravity");
        let mut signal = Node::new(signal_path.clone(), span, RoleData::Signal, ());
        signal.type_expr = Some(continuum_cdsl_ast::TypeExpr::Scalar {
            unit: None,
            bounds: None,
        });
        signal.attributes.push(Attribute {
            name: "strata".to_string(),
            args: vec![Expr::new(
                continuum_cdsl_ast::UntypedKind::Signal(Path::from_path_str("sim")),
                span,
            )],
            span,
        });
        signal.execution_blocks.push((
            "resolve".to_string(),
            BlockBody::Expression(Expr::literal(9.8, None, span)),
        ));

        let entity_id = EntityId::new("plate");
        let entity = Entity::new(entity_id.clone(), Path::from_path_str("plate"), span);

        let member_path = Path::from_path_str("plate.mass");
        let mut member = Node::new(
            member_path.clone(),
            span,
            RoleData::Signal,
            entity_id.clone(),
        );
        member.type_expr = Some(continuum_cdsl_ast::TypeExpr::Scalar {
            unit: None,
            bounds: None,
        });
        member.attributes.push(Attribute {
            name: "strata".to_string(),
            args: vec![Expr::new(
                continuum_cdsl_ast::UntypedKind::Signal(Path::from_path_str("sim")),
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
            Declaration::Era(era),
            Declaration::Stratum(stratum),
            Declaration::Node(signal),
            Declaration::Entity(entity),
            Declaration::Member(member),
        ];

        let world = compile(decls).expect("Compilation failed");

        // Should have gravity signal and debug.gravity field
        assert!(world
            .world
            .globals
            .contains_key(&Path::from_path_str("gravity")));
        assert!(world
            .world
            .globals
            .contains_key(&Path::from_path_str("debug.gravity")));

        let debug_node = world
            .world
            .globals
            .get(&Path::from_path_str("debug.gravity"))
            .unwrap();
        assert_eq!(debug_node.role_id(), continuum_cdsl_ast::RoleId::Field);
        assert_eq!(debug_node.executions.len(), 1);
        assert_eq!(
            debug_node.executions[0].phase,
            continuum_cdsl_ast::foundation::Phase::Measure
        );

        // Verify member signal
        assert!(world
            .world
            .members
            .contains_key(&Path::from_path_str("plate.mass")));
        assert!(world
            .world
            .members
            .contains_key(&Path::from_path_str("debug.plate.mass")));

        let debug_member = world
            .world
            .members
            .get(&Path::from_path_str("debug.plate.mass"))
            .unwrap();
        assert_eq!(debug_member.role_id(), continuum_cdsl_ast::RoleId::Field);
        assert_eq!(debug_member.index, EntityId::new("plate"));
    }
}

#[cfg(test)]
mod config_const_tests {
    use super::*;
    use continuum_cdsl_ast::foundation::{KernelType, Shape, Span, Type, Unit};
    use continuum_cdsl_ast::{ConfigEntry, ConstEntry, Expr, TypeExpr, UntypedKind};

    fn test_span() -> Span {
        Span::new(0, 0, 0, 1)
    }

    fn make_literal(value: f64) -> Expr {
        Expr::new(UntypedKind::Literal { value, unit: None }, test_span())
    }

    fn scalar_type() -> Type {
        Type::Kernel(KernelType {
            shape: Shape::Scalar,
            unit: Unit::dimensionless(),
            bounds: None,
        })
    }

    #[test]
    fn test_collect_config_types_explicit() {
        let entries = vec![ConfigEntry {
            path: Path::from_path_str("test.value"),
            type_expr: TypeExpr::Scalar {
                unit: None,
                bounds: None,
            },
            default: Some(make_literal(42.0)),
            span: test_span(),
            doc: None,
        }];

        let type_table = TypeTable::new();
        let result = collect_config_types(&entries, &type_table);

        // Should succeed with explicit type
        assert!(result.is_ok(), "Explicit type should resolve");
    }

    #[test]
    fn test_collect_config_types_infer_from_literal() {
        let entries = vec![ConfigEntry {
            path: Path::from_path_str("test.value"),
            type_expr: TypeExpr::Infer,
            default: Some(make_literal(42.0)),
            span: test_span(),
            doc: None,
        }];

        let type_table = TypeTable::new();
        let result = collect_config_types(&entries, &type_table);

        assert!(result.is_ok(), "Type inference from literal should succeed");
        let types = result.unwrap();
        assert_eq!(types.len(), 1);
        assert_eq!(types[&Path::from_path_str("test.value")], scalar_type());
    }

    #[test]
    fn test_collect_config_types_infer_no_default() {
        let entries = vec![ConfigEntry {
            path: Path::from_path_str("test.value"),
            type_expr: TypeExpr::Infer,
            default: None,
            span: test_span(),
            doc: None,
        }];

        let type_table = TypeTable::new();
        let result = collect_config_types(&entries, &type_table);

        // Should fail - cannot infer without default
        assert!(result.is_err(), "Inference without default should fail");
        let errors = result.unwrap_err();
        assert_eq!(errors.len(), 1);
        assert!(errors[0]
            .message
            .contains("requires explicit type or default"));
    }

    #[test]
    fn test_collect_config_types_multiple_entries() {
        let entries = vec![
            ConfigEntry {
                path: Path::from_path_str("config.a"),
                type_expr: TypeExpr::Infer,
                default: Some(make_literal(1.0)),
                span: test_span(),
                doc: None,
            },
            ConfigEntry {
                path: Path::from_path_str("config.b"),
                type_expr: TypeExpr::Infer,
                default: Some(make_literal(2.0)),
                span: test_span(),
                doc: None,
            },
        ];

        let type_table = TypeTable::new();
        let result = collect_config_types(&entries, &type_table);

        assert!(result.is_ok());
        let types = result.unwrap();
        assert_eq!(types.len(), 2);
    }

    #[test]
    fn test_collect_const_types_infer_from_literal() {
        let entries = vec![ConstEntry {
            path: Path::from_path_str("const.pi"),
            type_expr: TypeExpr::Infer,
            value: make_literal(3.14159),
            span: test_span(),
            doc: None,
        }];

        let type_table = TypeTable::new();
        let result = collect_const_types(&entries, &type_table);

        assert!(result.is_ok(), "Const type inference should succeed");
        let types = result.unwrap();
        assert_eq!(types.len(), 1);
        assert_eq!(types[&Path::from_path_str("const.pi")], scalar_type());
    }

    #[test]
    fn test_collect_const_types_bool_literal() {
        let entries = vec![ConstEntry {
            path: Path::from_path_str("const.enabled"),
            type_expr: TypeExpr::Infer,
            value: Expr::new(UntypedKind::BoolLiteral(true), test_span()),
            span: test_span(),
            doc: None,
        }];

        let type_table = TypeTable::new();
        let result = collect_const_types(&entries, &type_table);

        assert!(result.is_ok());
        let types = result.unwrap();
        assert_eq!(types[&Path::from_path_str("const.enabled")], Type::Bool);
    }

    #[test]
    fn test_collect_const_types_complex_expr_fails() {
        // Type inference only works for literals, not complex expressions
        let entries = vec![ConstEntry {
            path: Path::from_path_str("const.sum"),
            type_expr: TypeExpr::Infer,
            value: Expr::new(UntypedKind::Local("x".to_string()), test_span()),
            span: test_span(),
            doc: None,
        }];

        let type_table = TypeTable::new();
        let result = collect_const_types(&entries, &type_table);

        // Should fail - cannot infer from complex expressions
        assert!(result.is_err(), "Complex expression inference should fail");
        let errors = result.unwrap_err();
        assert!(errors[0]
            .message
            .contains("Cannot infer type from complex expression"));
    }

    #[test]
    fn test_has_type_info_trait_config() {
        let entry = ConfigEntry {
            path: Path::from_path_str("test"),
            type_expr: TypeExpr::Infer,
            default: None,
            span: test_span(),
            doc: None,
        };

        assert_eq!(entry.path(), &Path::from_path_str("test"));
        assert!(matches!(entry.type_expr(), TypeExpr::Infer));
    }

    #[test]
    fn test_has_type_info_trait_const() {
        let entry = ConstEntry {
            path: Path::from_path_str("test"),
            type_expr: TypeExpr::Infer,
            value: make_literal(1.0),
            span: test_span(),
            doc: None,
        };

        assert_eq!(entry.path(), &Path::from_path_str("test"));
        assert!(matches!(entry.type_expr(), TypeExpr::Infer));
    }
}
