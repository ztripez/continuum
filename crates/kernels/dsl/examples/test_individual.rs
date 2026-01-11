//! Test Individual Files
//! Tests loading each file individually with validation

use std::path::Path;

use continuum_functions as _;

fn main() {
    let files = vec![
        "examples/terra/terra.cdsl",
        "examples/terra/atmosphere/atmosphere.cdsl",
        "examples/terra/ecology/ecology.cdsl",
        "examples/terra/geophysics/geophysics.cdsl",
        "examples/terra/hydrology/hydrology.cdsl",
        "examples/terra/stellar/stellar.cdsl",
    ];

    let mut failed = false;

    for file in files {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join(file);
        println!("Testing: {}", file);
        match continuum_dsl::load_file(&path) {
            Ok(_) => println!("  OK"),
            Err(e) => {
                println!("  ERROR: {}", e);
                failed = true;
            }
        }
    }

    if failed {
        std::process::exit(1);
    }
}
