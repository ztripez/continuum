use super::*;
use crate::Path;
use std::fs;
use tempfile::tempdir;

#[test]
fn test_full_compile_and_run() {
    let dir = tempdir().unwrap();
    let file_path = dir.path().join("world.cdsl");

    let source = r#"
world test_world {
}

strata sim {
    : stride(1)
}

era main : initial : dt(1.0 <s>) {
    strata { sim: active }
}

signal counter : type Scalar : stratum(sim) {
    resolve { prev + 1.0 }
}
"#;

    fs::write(&file_path, source).unwrap();

    // 1. Compile
    let compiled = compile(dir.path()).expect("Full compilation failed");

    assert_eq!(compiled.world.metadata.path.to_string(), "test_world");
    assert!(
        compiled
            .world
            .globals
            .contains_key(&Path::from_path_str("counter"))
    );
}

#[test]
fn test_compile_empty_directory_error() {
    let dir = tempdir().unwrap();
    let result = compile(dir.path());
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("No .cdsl files found"))
    );
}

#[test]
fn test_compile_multi_file_errors() {
    let dir = tempdir().unwrap();

    // File 1: Lexing error
    fs::write(dir.path().join("lex.cdsl"), "signal $bad").unwrap();

    // File 2: Parsing error
    fs::write(dir.path().join("parse.cdsl"), "world").unwrap();

    let result = compile(dir.path());
    assert!(result.is_err());
    let errors = result.unwrap_err();

    // Should have both errors
    assert!(errors.iter().any(|e| e.message.contains("Invalid token")));
    assert!(
        errors
            .iter()
            .any(|e| e.message.contains("found end of input"))
    );
}
