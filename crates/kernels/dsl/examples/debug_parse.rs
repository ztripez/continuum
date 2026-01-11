//! Debug Parse Example
//! Tests parsing of a specific file with detailed error reporting

use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let file_path = env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("examples/terra/atmosphere/atmosphere.cdsl")
    });

    println!("Parsing: {:?}", file_path);

    let source = fs::read_to_string(&file_path).unwrap();
    let (result, errors) = continuum_dsl::parse(&source);

    if !errors.is_empty() {
        println!("\n✗ Parse errors:");
        for err in &errors {
            // Get the span from the error
            let span = err.span();
            let start = span.start;
            let end = span.end;

            // Count lines and columns
            let mut line: usize = 1;
            let mut col: usize = 1;
            for (i, c) in source.char_indices() {
                if i >= start {
                    break;
                }
                if c == '\n' {
                    line += 1;
                    col = 1;
                } else {
                    col += 1;
                }
            }

            // Get the line content
            let lines: Vec<&str> = source.lines().collect();
            let line_content = lines.get(line - 1).unwrap_or(&"");

            println!("  Line {}, column {}: {}", line, col, err);
            println!("    | {}", line_content);
            println!("    | {}^", " ".repeat(col.saturating_sub(1)));

            // Show context around the error
            let context_start = start.saturating_sub(50);
            let context_end = (start + 50).min(source.len());
            println!("  Context: ...{}...", &source[context_start..context_end].replace('\n', "\\n"));
        }
    } else if let Some(unit) = result {
        println!("\n✓ Parsed successfully: {} items", unit.items.len());
    } else {
        println!("\n? No result and no errors?");
    }
}
