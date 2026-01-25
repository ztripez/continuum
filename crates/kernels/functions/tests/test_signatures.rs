// Test KERNEL_SIGNATURES distributed slice population

// Import the library to trigger kernel registration
#[allow(unused_imports)]
use continuum_functions;
use continuum_kernel_types::KERNEL_SIGNATURES;

#[test]
fn test_kernel_signatures_populated() {
    println!("Testing KERNEL_SIGNATURES distributed slice...");
    println!("Total signatures registered: {}", KERNEL_SIGNATURES.len());

    // Assert exact count of all migrated kernels with type constraints
    // Note: Variadic functions (12 total) don't have compile-time signatures
    assert_eq!(
        KERNEL_SIGNATURES.len(),
        221,
        "Expected exactly 221 signatures, found {}",
        KERNEL_SIGNATURES.len()
    );

    // Group signatures by namespace
    let namespaces: std::collections::HashMap<&str, Vec<&str>> =
        KERNEL_SIGNATURES
            .iter()
            .fold(std::collections::HashMap::new(), |mut acc, sig| {
                acc.entry(sig.id.namespace.as_ref())
                    .or_insert_with(Vec::new)
                    .push(sig.id.name.as_ref());
                acc
            });
    assert!(
        namespaces.contains_key("matrix"),
        "Missing matrix namespace"
    );
    assert!(namespaces.contains_key("logic"), "Missing logic namespace");
    assert!(
        namespaces.contains_key("compare"),
        "Missing compare namespace"
    );
    assert!(namespaces.contains_key("dt"), "Missing dt namespace");
    assert!(namespaces.contains_key("rng"), "Missing rng namespace");
    assert!(namespaces.contains_key("quat"), "Missing quat namespace");
    assert!(
        namespaces.contains_key("tensor"),
        "Missing tensor namespace"
    );
    assert!(
        namespaces.contains_key(""),
        "Missing bare-name effect operations"
    );

    // Verify bare-name effect operations (empty namespace)
    let effect_ops = namespaces.get("").unwrap();
    assert!(effect_ops.contains(&"emit"), "emit not found");
    assert!(effect_ops.contains(&"spawn"), "spawn not found");
    assert!(effect_ops.contains(&"destroy"), "destroy not found");
    assert!(effect_ops.contains(&"log"), "log not found");
}

#[test]
fn test_representative_signatures() {
    use continuum_kernel_types::{KernelPurity, ShapeConstraint, UnitConstraint};

    // Test maths.add - should have shape/unit constraints
    let add_sig = KERNEL_SIGNATURES
        .iter()
        .find(|sig| sig.id.namespace == "maths" && sig.id.name == "add")
        .expect("maths.add not found");

    assert_eq!(add_sig.purity, KernelPurity::Pure);
    assert_eq!(add_sig.params.len(), 2);
    assert!(matches!(
        add_sig.params[0].shape,
        ShapeConstraint::Any { .. }
    ));
    assert!(matches!(
        add_sig.params[1].shape,
        ShapeConstraint::SameAs(0)
    ));

    // Test logic.and - boolean operations
    let and_sig = KERNEL_SIGNATURES
        .iter()
        .find(|sig| sig.id.namespace == "logic" && sig.id.name == "and")
        .expect("logic.and not found");

    assert_eq!(and_sig.purity, KernelPurity::Pure);
    assert_eq!(and_sig.params.len(), 2);

    // Test dt.integrate - simulation category
    let integrate_sig = KERNEL_SIGNATURES
        .iter()
        .find(|sig| sig.id.namespace == "dt" && sig.id.name == "integrate")
        .expect("dt.integrate not found");

    assert_eq!(integrate_sig.purity, KernelPurity::Pure);
    assert_eq!(integrate_sig.params.len(), 2); // prev, rate (dt is implicit)
    assert!(matches!(integrate_sig.params[0].unit, UnitConstraint::Any));

    // Test effect operation (bare name)
    let emit_sig = KERNEL_SIGNATURES
        .iter()
        .find(|sig| sig.id.namespace == "" && sig.id.name == "emit")
        .expect("emit not found");

    assert_eq!(emit_sig.purity, KernelPurity::Effect);
    assert!(
        emit_sig.id.namespace.is_empty(),
        "emit should have empty namespace"
    );

    // Test vector.dot_vec3 - should have specific shape constraints
    let dot_sig = KERNEL_SIGNATURES
        .iter()
        .find(|sig| sig.id.namespace == "vector" && sig.id.name == "dot_vec3")
        .expect("vector.dot_vec3 not found");

    assert_eq!(dot_sig.purity, KernelPurity::Pure);
    // Should have vector constraints (implementation may vary)
    assert!(dot_sig.params.len() >= 2);
}

#[test]
fn test_value_type_mapping_for_representative_kernels() {
    use continuum_kernel_types::ValueType;

    let find_value_type = |namespace: &str, name: &str| {
        KERNEL_SIGNATURES
            .iter()
            .find(|sig| sig.id.namespace == namespace && sig.id.name == name)
            .unwrap_or_else(|| panic!("kernel signature not found: {}.{}", namespace, name))
            .returns
            .value_type
    };

    assert_eq!(find_value_type("vector", "normalize_vec3"), ValueType::Vec3);
    assert_eq!(find_value_type("matrix", "mul_mat2"), ValueType::Mat2);
    assert_eq!(find_value_type("matrix", "mul_mat3"), ValueType::Mat3);
    assert_eq!(find_value_type("matrix", "mul_mat4"), ValueType::Mat4);
    assert_eq!(find_value_type("quat", "mul"), ValueType::Quat);
    assert_eq!(find_value_type("tensor", "transpose"), ValueType::Tensor);
    assert_eq!(find_value_type("compare", "eq"), ValueType::Bool);
    assert_eq!(find_value_type("rng", "uniform"), ValueType::Scalar);
}
