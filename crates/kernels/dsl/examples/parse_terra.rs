//! Parse Terra Example
//! Tests parsing of the terra world examples

use std::path::Path;

// Force linking of kernel functions crate so functions are registered
use continuum_functions as _;

fn main() {
    let terra_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples/terra");

    println!("Loading terra world from: {:?}", terra_path);

    match continuum_dsl::load_world(&terra_path) {
        Ok(result) => {
            println!("\n✓ Successfully loaded {} files", result.files.len());
            println!("  Total items: {}", result.unit.items.len());

            // Count item types
            let mut signals = 0;
            let mut fields = 0;
            let mut fractures = 0;
            let mut entities = 0;
            let mut members = 0;
            let mut strata = 0;
            let mut eras = 0;
            let mut impulses = 0;
            let mut functions = 0;
            let mut types = 0;
            let mut configs = 0;
            let mut consts = 0;
            let mut world = 0;

            for item in &result.unit.items {
                match &item.node {
                    continuum_dsl::ast::Item::SignalDef(_) => signals += 1,
                    continuum_dsl::ast::Item::FieldDef(_) => fields += 1,
                    continuum_dsl::ast::Item::FractureDef(_) => fractures += 1,
                    continuum_dsl::ast::Item::EntityDef(_) => entities += 1,
                    continuum_dsl::ast::Item::MemberDef(_) => members += 1,
                    continuum_dsl::ast::Item::StrataDef(_) => strata += 1,
                    continuum_dsl::ast::Item::EraDef(_) => eras += 1,
                    continuum_dsl::ast::Item::ImpulseDef(_) => impulses += 1,
                    continuum_dsl::ast::Item::FnDef(_) => functions += 1,
                    continuum_dsl::ast::Item::TypeDef(_) => types += 1,
                    continuum_dsl::ast::Item::ConfigBlock(_) => configs += 1,
                    continuum_dsl::ast::Item::ConstBlock(_) => consts += 1,
                    continuum_dsl::ast::Item::WorldDef(_) => world += 1,
                    continuum_dsl::ast::Item::ChronicleDef(_) => {}
                    continuum_dsl::ast::Item::OperatorDef(_) => {}
                    continuum_dsl::ast::Item::AnalyzerDef(_) => {}
                }
            }

            println!("\n  Item breakdown:");
            println!("    World definitions: {}", world);
            println!("    Strata: {}", strata);
            println!("    Eras: {}", eras);
            println!("    Signals: {}", signals);
            println!("    Fields: {}", fields);
            println!("    Fractures: {}", fractures);
            println!("    Entities: {}", entities);
            println!("    Members: {}", members);
            println!("    Impulses: {}", impulses);
            println!("    Functions: {}", functions);
            println!("    Types: {}", types);
            println!("    Config blocks: {}", configs);
            println!("    Const blocks: {}", consts);
        }
        Err(e) => {
            println!("\n✗ Failed to load terra world:");
            println!("{}", e);
            std::process::exit(1);
        }
    }
}
