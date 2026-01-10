//! DSL Lint Tool
//!
//! Parses a single .cdsl file and reports detailed parse errors with line numbers.
//!
//! Usage: dsl-lint <file.cdsl>

use std::env;
use std::fs;
use std::process;

use tracing::{error, info};

fn main() {
    continuum_tools::init_logging();

    let args: Vec<String> = env::args().collect();

    if args.len() != 2 {
        eprintln!("Usage: {} <file.cdsl>", args[0]);
        process::exit(1);
    }

    let file_path = &args[1];

    let source = match fs::read_to_string(file_path) {
        Ok(s) => s,
        Err(e) => {
            error!("Error reading file '{}': {}", file_path, e);
            process::exit(1);
        }
    };

    info!("Linting: {}", file_path);
    info!("File size: {} bytes, {} lines", source.len(), source.lines().count());

    let (result, parse_errors) = continuum_dsl::parse(&source);

    if !parse_errors.is_empty() {
        error!("Parse errors found:");
        for err in &parse_errors {
            // Try to extract position from error
            let err_str = format!("{:?}", err);
            error!("  Error: {}", err);

            // If we can extract a byte offset, show context
            if let Some(span) = extract_span_from_error(&err_str) {
                show_error_context(&source, span, file_path);
            }
        }
        process::exit(1);
    }

    match result {
        Some(unit) => {
            info!("Successfully parsed {} items:", unit.items.len());
            for item in &unit.items {
                info!("  - {}", describe_item(&item.node));
            }

            // Run validation
            let validation_errors = continuum_dsl::validate(&unit);
            if !validation_errors.is_empty() {
                error!("Validation errors:");
                for err in &validation_errors {
                    error!("  - {}", err);
                }
                process::exit(1);
            }

            info!("No errors found.");
        }
        None => {
            error!("Failed to produce AST (no specific errors reported)");
            process::exit(1);
        }
    }
}

fn extract_span_from_error(err_str: &str) -> Option<(usize, usize)> {
    // Try to parse span from error debug output
    // Format is typically "at 12345..12350"
    if let Some(at_pos) = err_str.find("at ") {
        let rest = &err_str[at_pos + 3..];
        if let Some(dot_pos) = rest.find("..") {
            let start_str = &rest[..dot_pos];
            let end_rest = &rest[dot_pos + 2..];
            let end_str: String = end_rest.chars().take_while(|c| c.is_ascii_digit()).collect();

            if let (Ok(start), Ok(end)) = (start_str.parse::<usize>(), end_str.parse::<usize>()) {
                return Some((start, end));
            }
        }
    }
    None
}

fn show_error_context(source: &str, span: (usize, usize), file_path: &str) {
    let (start, _end) = span;

    // Find line number and column
    let mut line_num: usize = 1;
    let mut line_start = 0;

    for (i, c) in source.char_indices() {
        if i >= start {
            break;
        }
        if c == '\n' {
            line_num += 1;
            line_start = i + 1;
        }
    }

    let col = start - line_start + 1;

    // Get the line content
    let line_end = source[start..].find('\n').map(|p| start + p).unwrap_or(source.len());
    let line_content = &source[line_start..line_end];

    eprintln!();
    eprintln!("  --> {}:{}:{}", file_path, line_num, col);
    eprintln!("   |");
    eprintln!("{:>4} | {}", line_num, line_content);
    eprintln!("   | {}^", " ".repeat(col.saturating_sub(1)));
    eprintln!();

    // Show surrounding context
    let lines: Vec<&str> = source.lines().collect();
    let ctx_start = line_num.saturating_sub(3);
    let ctx_end = (line_num + 2).min(lines.len());

    eprintln!("  Context:");
    for (i, line) in lines.iter().enumerate().skip(ctx_start).take(ctx_end - ctx_start) {
        let marker = if i + 1 == line_num { ">>>" } else { "   " };
        eprintln!("{} {:>4} | {}", marker, i + 1, line);
    }
}

fn describe_item(item: &continuum_dsl::Item) -> String {
    match item {
        continuum_dsl::Item::ConstBlock(consts) => {
            format!("const block ({} constants)", consts.entries.len())
        }
        continuum_dsl::Item::ConfigBlock(config) => {
            format!("config block ({} entries)", config.entries.len())
        }
        continuum_dsl::Item::TypeDef(t) => format!("type {}", t.name.node),
        continuum_dsl::Item::FnDef(f) => format!("fn {}", f.path.node),
        continuum_dsl::Item::StrataDef(s) => format!("strata {}", s.path.node),
        continuum_dsl::Item::EraDef(e) => format!("era {}", e.name.node),
        continuum_dsl::Item::SignalDef(s) => format!("signal {}", s.path.node),
        continuum_dsl::Item::FieldDef(f) => format!("field {}", f.path.node),
        continuum_dsl::Item::OperatorDef(o) => format!("operator {}", o.path.node),
        continuum_dsl::Item::ImpulseDef(i) => format!("impulse {}", i.path.node),
        continuum_dsl::Item::FractureDef(f) => format!("fracture {}", f.path.node),
        continuum_dsl::Item::ChronicleDef(c) => format!("chronicle {}", c.path.node),
        continuum_dsl::Item::EntityDef(e) => format!("entity {}", e.path.node),
        continuum_dsl::Item::MemberDef(m) => format!("member {}", m.path.node),
        continuum_dsl::Item::WorldDef(w) => format!("world {}", w.path.node),
    }
}
