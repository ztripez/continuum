// Test KERNEL_SIGNATURES distributed slice population

// Import the library to trigger kernel registration
use continuum_functions;
use continuum_kernel_types::KERNEL_SIGNATURES;

#[test]
fn test_kernel_signatures_populated() {
    println!("Testing KERNEL_SIGNATURES distributed slice...");
    println!("Total signatures registered: {}", KERNEL_SIGNATURES.len());

    // Should have at least the logic and compare functions we migrated
    assert!(
        KERNEL_SIGNATURES.len() >= 10,
        "Expected at least 10 signatures (4 logic + 6 compare), found {}",
        KERNEL_SIGNATURES.len()
    );

    // Find logic.and signature
    let and_sig = KERNEL_SIGNATURES
        .iter()
        .find(|sig| sig.id.namespace == "logic" && sig.id.name == "and");

    assert!(and_sig.is_some(), "logic.and signature not found!");

    let sig = and_sig.unwrap();
    println!("\nFound logic.and signature:");
    println!("  Namespace: {}", sig.id.namespace);
    println!("  Name: {}", sig.id.name);
    println!("  Purity: {:?}", sig.purity);
    println!("  Params: {}", sig.params.len());

    assert_eq!(sig.params.len(), 2, "logic.and should have 2 parameters");

    for (i, param) in sig.params.iter().enumerate() {
        println!(
            "    Param {}: {} (shape: {:?}, unit: {:?})",
            i, param.name, param.shape, param.unit
        );
    }
    println!(
        "  Returns: shape={:?}, unit={:?}",
        sig.returns.shape, sig.returns.unit
    );

    println!("\nAll signatures:");
    for sig in KERNEL_SIGNATURES.iter() {
        println!(
            "  {}.{} ({} params)",
            sig.id.namespace,
            sig.id.name,
            sig.params.len()
        );
    }
}
