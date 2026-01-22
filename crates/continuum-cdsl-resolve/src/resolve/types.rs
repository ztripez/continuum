//! Type resolution for CDSL type and unit syntax.
//!
//! Translates parsed [`TypeExpr`] and [`UnitExpr`] nodes into semantic [`Type`]
//! and [`Unit`] values used by later validation passes.
//!
//! # Pipeline Position
//!
//! ```text
//! Parse → Desugar → Name Resolution → Type Resolution → Validation
//!                                         ^^^^^^
//!                                      YOU ARE HERE
//! ```

use crate::ast::{Declaration, Expr, ExprKind, TypeExpr};
use crate::error::{CompileError, ErrorKind};
use crate::foundation::{Bounds, EntityId, Path, Shape, Span, Type, UserType, UserTypeId};
use crate::resolve::units::resolve_unit_expr;
use std::collections::HashMap;

/// Registry of user-defined types keyed by fully-qualified [`Path`].
#[derive(Debug, Default)]
pub struct TypeTable {
    /// Map from type path to UserType definition
    types: HashMap<Path, UserType>,

    /// Map from type path to TypeId (for quick lookups)
    type_ids: HashMap<Path, UserTypeId>,
}

impl TypeTable {
    /// Creates a new, empty [`TypeTable`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a user-defined type in the table.
    pub fn register(&mut self, user_type: UserType) {
        let path = user_type.name().clone();
        let id = user_type.id().clone();
        self.types.insert(path.clone(), user_type);
        self.type_ids.insert(path, id);
    }

    /// Looks up a user type by [`Path`].
    pub fn get(&self, path: &Path) -> Option<&UserType> {
        self.types.get(path)
    }

    /// Looks up a user type identifier by [`Path`].
    pub fn get_id(&self, path: &Path) -> Option<UserTypeId> {
        self.type_ids.get(path).cloned()
    }

    /// Tests whether a user type exists in the table.
    pub fn contains(&self, path: &Path) -> bool {
        self.types.contains_key(path)
    }

    /// Iterate over all registered user types.
    pub fn iter(&self) -> impl Iterator<Item = &UserType> {
        self.types.values()
    }

    /// Look up a user type by its [`UserTypeId`].
    pub fn get_by_id(&self, id: &UserTypeId) -> Option<&UserType> {
        self.types.values().find(|user_type| user_type.id() == id)
    }

    /// Retrieves a mutable reference to a registered [`UserType`] by its [`Path`].
    pub fn get_mut(&mut self, path: &Path) -> Option<&mut UserType> {
        self.types.get_mut(path)
    }
}

/// Synthesizes [`UserType`] definitions from [`Entity`](crate::ast::Entity) and
/// [`Member`](crate::ast::Declaration::Member) declarations.
///
/// This pass performs hierarchical projection:
/// 1. Group members by entity.
/// 2. For each entity, recursively build nested [`UserType`]s for hierarchical paths.
/// 3. Use relative names for fields (e.g., `self.velocity` instead of `self.plate.velocity`).
///
/// # Example
///
/// Member `plate.physics.velocity` projects to:
/// - `UserType("plate")` with field `physics: UserType("plate.physics")`
/// - `UserType("plate.physics")` with field `velocity: Vector<3, m/s>`
pub fn project_entity_types(
    declarations: &[Declaration],
    type_table: &mut TypeTable,
) -> Result<(), Vec<CompileError>> {
    let mut errors = Vec::new();
    let mut entity_members = HashMap::new();
    let mut entity_paths = Vec::new();

    // 1. Identify entities and register empty UserTypes
    // This unblocks circular references during member type resolution
    for decl in declarations {
        match decl {
            Declaration::Entity(entity) => {
                entity_paths.push(entity.path.clone());
                let type_id = UserTypeId::from(entity.path.to_string());
                type_table.register(UserType::new(type_id, entity.path.clone(), vec![]));
            }
            Declaration::Member(node) => {
                entity_members
                    .entry(node.index.clone())
                    .or_insert_with(Vec::new)
                    .push(node);
            }
            _ => {}
        }
    }

    // 2. Hierarchical projection for each entity
    for path in entity_paths {
        let entity_id = EntityId::new(path.to_string());
        if let Some(members) = entity_members.get(&entity_id) {
            if let Err(mut e) = project_hierarchical_entity(&path, members, type_table) {
                errors.append(&mut e);
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Recursively build nested types for an entity's hierarchical members.
fn project_hierarchical_entity<I: crate::ast::Index>(
    root_path: &Path,
    members: &[&crate::ast::Node<I>],
    type_table: &mut TypeTable,
) -> Result<(), Vec<CompileError>> {
    let mut errors = Vec::new();

    // Map from type path -> list of fields (field_name, field_type)
    let mut type_fields: HashMap<Path, Vec<(String, Type)>> = HashMap::new();

    for node in members {
        let mut current_path = root_path.clone();

        // The member path segments after the entity root
        // e.g. plate.physics.velocity -> ["physics", "velocity"]
        let relative_segments = &node.path.segments()[root_path.len()..];

        if relative_segments.is_empty() {
            errors.push(CompileError::new(
                ErrorKind::Internal,
                node.span,
                format!(
                    "Member path '{}' does not start with entity path '{}'",
                    node.path, root_path
                ),
            ));
            continue;
        }

        // Traverse segments
        for i in 0..relative_segments.len() {
            let segment_name = &relative_segments[i];
            let is_leaf = i == relative_segments.len() - 1;

            if is_leaf {
                // Resolve member type
                let ty = match &node.type_expr {
                    Some(expr) => match resolve_type_expr(expr, type_table, node.span) {
                        Ok(ty) => ty,
                        Err(e) => {
                            errors.push(e);
                            continue;
                        }
                    },
                    None => {
                        errors.push(CompileError::new(
                            ErrorKind::UnknownType,
                            node.span,
                            format!("Member '{}' is missing a type expression", node.path),
                        ));
                        continue;
                    }
                };

                // Add leaf field to its parent type
                let fields = type_fields.entry(current_path.clone()).or_default();
                if fields.iter().any(|(name, _)| name == segment_name) {
                    errors.push(CompileError::new(
                        ErrorKind::TypeMismatch,
                        node.span,
                        format!(
                            "Duplicate member name '{}' at path '{}'",
                            segment_name, current_path
                        ),
                    ));
                } else {
                    fields.push((segment_name.clone(), ty));
                }
            } else {
                // Internal segment -> nested UserType
                let parent_path = current_path.clone();
                current_path = current_path.append(segment_name);

                // Add the nested type as a field to its parent if not already added
                let parent_fields = type_fields.entry(parent_path).or_default();
                if !parent_fields.iter().any(|(name, _)| name == segment_name) {
                    parent_fields.push((
                        segment_name.clone(),
                        Type::User(UserTypeId::from(current_path.to_string())),
                    ));
                }

                // Ensure the nested type itself exists in the map
                type_fields.entry(current_path.clone()).or_default();
            }
        }
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    // Register all identified types in the table
    // Sort keys for deterministic registration order
    let mut sorted_paths: Vec<_> = type_fields.keys().cloned().collect();
    sorted_paths.sort();

    for path in sorted_paths {
        let fields = type_fields.remove(&path).unwrap();
        let type_id = UserTypeId::from(path.to_string());
        type_table.register(UserType::new(type_id, path, fields));
    }

    Ok(())
}

/// Registers all explicit `type` declarations in the [`TypeTable`].
pub fn resolve_user_types(
    declarations: &[Declaration],
    type_table: &mut TypeTable,
) -> Result<(), Vec<CompileError>> {
    let mut errors = Vec::new();

    // 1. First pass: Register all type names
    for decl in declarations {
        if let Declaration::Type(type_decl) = decl {
            let path = Path::from(type_decl.name.as_str());
            let type_id = UserTypeId::from(path.to_string());
            type_table.register(UserType::new(type_id, path, vec![]));
        }
    }

    // 2. Second pass: Resolve field types
    for decl in declarations {
        if let Declaration::Type(type_decl) = decl {
            let path = Path::from(type_decl.name.as_str());
            let mut fields = Vec::new();

            for field in &type_decl.fields {
                match resolve_type_expr(&field.type_expr, type_table, field.span) {
                    Ok(ty) => fields.push((field.name.clone(), ty)),
                    Err(e) => errors.push(e),
                }
            }

            if let Some(user_type) = type_table.get_mut(&path) {
                user_type.fields = fields;
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Resolves `type_expr` into `output` for all nodes in the world.
pub fn resolve_node_types<I: crate::ast::Index>(
    nodes: &mut [crate::ast::Node<I>],
    type_table: &TypeTable,
) -> Result<(), Vec<CompileError>> {
    let mut errors = Vec::new();

    for node in nodes {
        if let Some(type_expr) = &node.type_expr {
            match resolve_type_expr(type_expr, type_table, node.span) {
                Ok(ty) => {
                    node.output = Some(ty);
                    node.type_expr = None; // Clear as consumed
                }
                Err(e) => errors.push(e),
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// Resolves types for all constants and configurations.
pub fn resolve_const_config_types(
    declarations: &mut [Declaration],
    type_table: &TypeTable,
) -> Result<(), Vec<CompileError>> {
    let mut errors = Vec::new();

    for decl in declarations {
        match decl {
            Declaration::Const(entries) => {
                for entry in entries {
                    match resolve_type_expr(&entry.type_expr, type_table, entry.span) {
                        Ok(_) => {} // Validation only for now
                        Err(e) => errors.push(e),
                    }
                }
            }
            Declaration::Config(entries) => {
                for entry in entries {
                    match resolve_type_expr(&entry.type_expr, type_table, entry.span) {
                        Ok(_) => {} // Validation only for now
                        Err(e) => errors.push(e),
                    }
                }
            }
            _ => {}
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
/// Resolves a parsed [`TypeExpr`] into a semantic [`Type`].
pub fn resolve_type_expr(
    type_expr: &TypeExpr,
    type_table: &TypeTable,
    span: Span,
) -> Result<Type, CompileError> {
    match type_expr {
        TypeExpr::Scalar { unit, bounds } => {
            let resolved_unit = resolve_unit_expr(unit.as_ref(), span)?;

            // Resolve bounds expressions to constant values
            let resolved_bounds = match bounds {
                Some((min_expr, max_expr)) => {
                    let min_val = evaluate_const_expr(min_expr, span)?;
                    let max_val = evaluate_const_expr(max_expr, span)?;

                    // Validate bounds make sense
                    if min_val >= max_val {
                        return Err(CompileError::new(
                            ErrorKind::InvalidBounds,
                            span,
                            format!(
                                "Type bounds invalid: min ({}) must be less than max ({})",
                                min_val, max_val
                            ),
                        ));
                    }

                    Some(Bounds {
                        min: Some(min_val),
                        max: Some(max_val),
                    })
                }
                None => None,
            };

            Ok(Type::kernel(Shape::Scalar, resolved_unit, resolved_bounds))
        }

        TypeExpr::Vector { dim, unit } => {
            if *dim == 0 {
                return Err(CompileError::new(
                    ErrorKind::DimensionMismatch,
                    span,
                    "Vector dimension must be greater than 0".to_string(),
                ));
            }
            let resolved_unit = resolve_unit_expr(unit.as_ref(), span)?;
            Ok(Type::kernel(
                Shape::Vector { dim: *dim },
                resolved_unit,
                None,
            ))
        }

        TypeExpr::Matrix { rows, cols, unit } => {
            if *rows == 0 || *cols == 0 {
                return Err(CompileError::new(
                    ErrorKind::DimensionMismatch,
                    span,
                    "Matrix dimensions must be greater than 0".to_string(),
                ));
            }
            let resolved_unit = resolve_unit_expr(unit.as_ref(), span)?;
            Ok(Type::kernel(
                Shape::Matrix {
                    rows: *rows,
                    cols: *cols,
                },
                resolved_unit,
                None,
            ))
        }

        TypeExpr::User(path) => {
            let type_id = type_table.get_id(path).ok_or_else(|| {
                CompileError::new(
                    ErrorKind::UnknownType,
                    span,
                    format!("Unknown user type: {}", path),
                )
            })?;
            Ok(Type::user(type_id))
        }

        TypeExpr::Bool => Ok(Type::Bool),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_span() -> Span {
        Span::new(0, 10, 20, 1)
    }

    #[test]
    fn test_resolve_scalar_type() {
        let type_table = TypeTable::new();
        let scalar_type = TypeExpr::Scalar {
            unit: None,
            bounds: None,
        };
        let resolved = resolve_type_expr(&scalar_type, &type_table, test_span()).unwrap();
        assert!(resolved.is_kernel());
    }

    #[test]
    fn test_project_entity_types_nested() {
        use crate::ast::{Entity, Node, RoleData};

        let span = test_span();
        let entity_id = EntityId::new("plate");
        let entity_path = Path::from_path_str("plate");
        let entity = Entity::new(entity_id.clone(), entity_path.clone(), span);

        // plate.physics.velocity
        let member_path = Path::from_path_str("plate.physics.velocity");
        let mut member = Node::new(member_path, span, RoleData::Signal, entity_id);
        member.type_expr = Some(TypeExpr::Vector { dim: 3, unit: None });

        let decls = vec![Declaration::Entity(entity), Declaration::Member(member)];

        let mut table = TypeTable::new();
        project_entity_types(&decls, &mut table).unwrap();

        // plate should have field "physics"
        let plate_type = table.get(&entity_path).expect("plate type missing");
        assert_eq!(plate_type.field_count(), 1);
        let physics_field = plate_type.fields().first().unwrap();
        assert_eq!(physics_field.0, "physics");

        // physics should be a UserType
        if let Type::User(physics_id) = &physics_field.1 {
            let physics_type = table.get_by_id(physics_id).expect("physics type missing");
            assert_eq!(physics_type.name().to_string(), "plate.physics");

            // physics should have field "velocity"
            assert_eq!(physics_type.field_count(), 1);
            let velocity_field = physics_type.fields().first().unwrap();
            assert_eq!(velocity_field.0, "velocity");
            assert!(velocity_field.1.is_kernel());
        } else {
            panic!("Expected User type for physics field");
        }
    }

    #[test]
    fn test_project_entity_types_circular_references() {
        use crate::ast::{Entity, Node, RoleData};

        let span = test_span();

        // Entity A
        let id_a = EntityId::new("A");
        let path_a = Path::from_path_str("A");
        let entity_a = Entity::new(id_a.clone(), path_a.clone(), span);

        // Member A.b : B
        let mut member_ab = Node::new(Path::from_path_str("A.b"), span, RoleData::Signal, id_a);
        member_ab.type_expr = Some(TypeExpr::User(Path::from_path_str("B")));

        // Entity B
        let id_b = EntityId::new("B");
        let path_b = Path::from_path_str("B");
        let entity_b = Entity::new(id_b.clone(), path_b.clone(), span);

        // Member B.a : A
        let mut member_ba = Node::new(Path::from_path_str("B.a"), span, RoleData::Signal, id_b);
        member_ba.type_expr = Some(TypeExpr::User(Path::from_path_str("A")));

        let decls = vec![
            Declaration::Entity(entity_a),
            Declaration::Member(member_ab),
            Declaration::Entity(entity_b),
            Declaration::Member(member_ba),
        ];

        let mut table = TypeTable::new();
        project_entity_types(&decls, &mut table).expect("Circular references should resolve");

        let type_a = table.get(&path_a).unwrap();
        assert!(type_a.field("b").unwrap().is_user());
    }
}

/// Evaluate a constant expression to an f64 value.
///
/// This is used for type bounds which must be compile-time constants.
/// Currently supports:
/// - Numeric literals
///
/// Future: Support config values, const values, and simple arithmetic.
fn evaluate_const_expr(expr: &Expr, span: Span) -> Result<f64, CompileError> {
    match &expr.kind {
        ExprKind::Literal { value, unit } => {
            // Bounds must be dimensionless
            if unit.is_some() {
                return Err(CompileError::new(
                    ErrorKind::InvalidBounds,
                    span,
                    "Type bounds must be dimensionless numbers (no units)".to_string(),
                ));
            }
            Ok(*value)
        }
        _ => Err(CompileError::new(
            ErrorKind::InvalidBounds,
            span,
            "Type bounds must be constant literals (for now)".to_string(),
        )),
    }
}
