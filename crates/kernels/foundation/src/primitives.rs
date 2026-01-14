//! Primitive type registry metadata shared across the stack.

/// Identifier for a primitive type definition in the registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrimitiveTypeId(&'static str);

impl PrimitiveTypeId {
    /// Create a new primitive type identifier.
    pub const fn new(name: &'static str) -> Self {
        Self(name)
    }

    /// Returns the canonical name for this primitive type.
    pub const fn name(self) -> &'static str {
        self.0
    }
}

/// High-level shape for a primitive type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveShape {
    Scalar,
    Vector { dim: u8 },
    Tensor,
    Grid,
    Seq,
}

/// Storage class for runtime value buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveStorageClass {
    Scalar,
    Vec2,
    Vec3,
    Vec4,
    Tensor,
    Grid,
    Seq,
}

/// Parameter kinds that can appear in primitive type declarations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrimitiveParamKind {
    Unit,
    Range,
    Magnitude,
    Rows,
    Cols,
    Width,
    Height,
    ElementType,
}

/// Parameter specification for a primitive type declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrimitiveParamSpec {
    /// Parameter name used in named arguments.
    pub name: &'static str,
    /// Parameter kind describing expected value shape.
    pub kind: PrimitiveParamKind,
    /// Positional index, if positional arguments are allowed.
    pub position: Option<usize>,
    /// Whether the parameter is optional.
    pub optional: bool,
}

/// Static definition for a primitive type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimitiveTypeDef {
    /// Identifier for this primitive type.
    pub id: PrimitiveTypeId,
    /// Canonical name used in DSL syntax.
    pub name: &'static str,
    /// Primitive shape category.
    pub shape: PrimitiveShape,
    /// Runtime storage class.
    pub storage: PrimitiveStorageClass,
    /// Parameter spec list for this type.
    pub params: &'static [PrimitiveParamSpec],
    /// Component labels (if vector-like).
    pub components: Option<&'static [&'static str]>,
}

const COMPONENTS_VEC2: [&str; 2] = ["x", "y"];
const COMPONENTS_VEC3: [&str; 3] = ["x", "y", "z"];
const COMPONENTS_VEC4: [&str; 4] = ["x", "y", "z", "w"];
const COMPONENTS_QUAT: [&str; 4] = ["w", "x", "y", "z"];

const PARAM_UNIT: PrimitiveParamSpec = PrimitiveParamSpec {
    name: "unit",
    kind: PrimitiveParamKind::Unit,
    position: Some(0),
    optional: true,
};

const PARAM_UNIT_TENSOR: PrimitiveParamSpec = PrimitiveParamSpec {
    name: "unit",
    kind: PrimitiveParamKind::Unit,
    position: Some(2),
    optional: false,
};

const PARAM_RANGE: PrimitiveParamSpec = PrimitiveParamSpec {
    name: "range",
    kind: PrimitiveParamKind::Range,
    position: Some(1),
    optional: true,
};

const PARAM_MAGNITUDE: PrimitiveParamSpec = PrimitiveParamSpec {
    name: "magnitude",
    kind: PrimitiveParamKind::Magnitude,
    position: None,
    optional: true,
};

const PARAM_ROWS: PrimitiveParamSpec = PrimitiveParamSpec {
    name: "rows",
    kind: PrimitiveParamKind::Rows,
    position: Some(0),
    optional: false,
};

const PARAM_COLS: PrimitiveParamSpec = PrimitiveParamSpec {
    name: "cols",
    kind: PrimitiveParamKind::Cols,
    position: Some(1),
    optional: false,
};

const PARAM_WIDTH: PrimitiveParamSpec = PrimitiveParamSpec {
    name: "width",
    kind: PrimitiveParamKind::Width,
    position: Some(0),
    optional: false,
};

const PARAM_HEIGHT: PrimitiveParamSpec = PrimitiveParamSpec {
    name: "height",
    kind: PrimitiveParamKind::Height,
    position: Some(1),
    optional: false,
};

const PARAM_ELEMENT_TYPE: PrimitiveParamSpec = PrimitiveParamSpec {
    name: "element_type",
    kind: PrimitiveParamKind::ElementType,
    position: Some(2),
    optional: false,
};

/// Registry of built-in primitive types.
pub static PRIMITIVE_TYPES: &[PrimitiveTypeDef] = &[
    PrimitiveTypeDef {
        id: PrimitiveTypeId::new("Scalar"),
        name: "Scalar",
        shape: PrimitiveShape::Scalar,
        storage: PrimitiveStorageClass::Scalar,
        params: &[PARAM_UNIT, PARAM_RANGE],
        components: None,
    },
    PrimitiveTypeDef {
        id: PrimitiveTypeId::new("Vec2"),
        name: "Vec2",
        shape: PrimitiveShape::Vector { dim: 2 },
        storage: PrimitiveStorageClass::Vec2,
        params: &[PARAM_UNIT, PARAM_MAGNITUDE],
        components: Some(&COMPONENTS_VEC2),
    },
    PrimitiveTypeDef {
        id: PrimitiveTypeId::new("Vec3"),
        name: "Vec3",
        shape: PrimitiveShape::Vector { dim: 3 },
        storage: PrimitiveStorageClass::Vec3,
        params: &[PARAM_UNIT, PARAM_MAGNITUDE],
        components: Some(&COMPONENTS_VEC3),
    },
    PrimitiveTypeDef {
        id: PrimitiveTypeId::new("Vec4"),
        name: "Vec4",
        shape: PrimitiveShape::Vector { dim: 4 },
        storage: PrimitiveStorageClass::Vec4,
        params: &[PARAM_UNIT, PARAM_MAGNITUDE],
        components: Some(&COMPONENTS_VEC4),
    },
    PrimitiveTypeDef {
        id: PrimitiveTypeId::new("Quat"),
        name: "Quat",
        shape: PrimitiveShape::Vector { dim: 4 },
        storage: PrimitiveStorageClass::Vec4,
        params: &[PARAM_MAGNITUDE],
        components: Some(&COMPONENTS_QUAT),
    },
    PrimitiveTypeDef {
        id: PrimitiveTypeId::new("Tensor"),
        name: "Tensor",
        shape: PrimitiveShape::Tensor,
        storage: PrimitiveStorageClass::Tensor,
        params: &[PARAM_ROWS, PARAM_COLS, PARAM_UNIT_TENSOR],
        components: None,
    },
    PrimitiveTypeDef {
        id: PrimitiveTypeId::new("Grid"),
        name: "Grid",
        shape: PrimitiveShape::Grid,
        storage: PrimitiveStorageClass::Grid,
        params: &[PARAM_WIDTH, PARAM_HEIGHT, PARAM_ELEMENT_TYPE],
        components: None,
    },
    PrimitiveTypeDef {
        id: PrimitiveTypeId::new("Seq"),
        name: "Seq",
        shape: PrimitiveShape::Seq,
        storage: PrimitiveStorageClass::Seq,
        params: &[PARAM_ELEMENT_TYPE],
        components: None,
    },
];

/// Lookup a primitive type definition by name.
pub fn primitive_type_by_name(name: &str) -> Option<&'static PrimitiveTypeDef> {
    PRIMITIVE_TYPES.iter().find(|def| def.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_quat_definition() {
        let def = primitive_type_by_name("Quat").expect("Quat definition missing");
        assert_eq!(def.shape, PrimitiveShape::Vector { dim: 4 });
        assert_eq!(def.storage, PrimitiveStorageClass::Vec4);
        assert_eq!(def.components, Some(&["w", "x", "y", "z"][..]));
    }
}
