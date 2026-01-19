#[cfg(test)]
mod audit_tests {
    use crate::ast::{Declaration, Entity, Node, RoleData, TypeExpr};
    use crate::foundation::{EntityId, Path, Span};
    use crate::resolve::types::{TypeTable, project_entity_types};

    fn test_span() -> Span {
        Span::new(0, 0, 10, 1)
    }

    #[test]
    fn test_fail_loudly_on_missing_type_expr() {
        let span = test_span();
        let entity_path = Path::from_str("plate");
        let entity_id = EntityId::new("plate");
        let entity = Entity::new(entity_id.clone(), entity_path.clone(), span);

        let member_path = Path::from_str("plate.mass");
        let mut member = Node::new(member_path, span, RoleData::Signal, entity_id);
        // type_expr is None by default in Node::new
        member.type_expr = None;

        let decls = vec![Declaration::Entity(entity), Declaration::Member(member)];
        let mut table = TypeTable::new();

        // This currently succeeds but skips the member
        let result = project_entity_types(&decls, &mut table);
        assert!(
            result.is_ok(),
            "Expected it to succeed for now, but we want to fail loudly later"
        );

        let user_type = table.get(&entity_path).unwrap();
        assert_eq!(user_type.fields.len(), 0, "Member was silently skipped");
    }

    #[test]
    fn test_partial_update_on_error() {
        let span = test_span();
        let entity_path = Path::from_str("plate");
        let entity_id = EntityId::new("plate");
        let entity = Entity::new(entity_id.clone(), entity_path.clone(), span);

        // Good member
        let member_ok_path = Path::from_str("plate.mass");
        let mut member_ok = Node::new(member_ok_path, span, RoleData::Signal, entity_id.clone());
        member_ok.type_expr = Some(TypeExpr::Scalar { unit: None });

        // Bad member (Unknown user type)
        let member_bad_path = Path::from_str("plate.pos");
        let mut member_bad = Node::new(member_bad_path, span, RoleData::Signal, entity_id);
        member_bad.type_expr = Some(TypeExpr::User(Path::from_str("Unknown")));

        let decls = vec![
            Declaration::Entity(entity),
            Declaration::Member(member_ok),
            Declaration::Member(member_bad),
        ];
        let mut table = TypeTable::new();

        let result = project_entity_types(&decls, &mut table);
        assert!(result.is_err(), "Expected error due to Unknown type");

        let user_type = table.get(&entity_path).unwrap();
        // VIOLATION: The UserType was still updated with the partial field list!
        assert_eq!(
            user_type.fields.len(),
            1,
            "UserType has partial fields despite error!"
        );
        assert_eq!(user_type.fields[0].0, "mass");
    }
}
