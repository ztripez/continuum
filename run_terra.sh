#!/bin/bash
# Quick script to run Terra simulation using the library API

cd "$(dirname "$0")"

cat > /tmp/run_terra.rs << 'RUST'
use continuum_cdsl::compile;
use continuum_runtime::build_runtime;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let world_path = Path::new("examples/terra");
    
    println!("Compiling Terra world from: {}", world_path.display());
    let compiled = compile(world_path)?;
    
    println!("Building runtime...");
    let mut runtime = build_runtime(compiled);
    
    println!("Running 10 simulation steps...");
    for step in 1..=10 {
        runtime.tick();
        println!("  Step {} complete", step);
    }
    
    println!("Terra simulation complete!");
    Ok(())
}
RUST

echo "Compiling runner (this may take a few minutes)..."
rustc --edition 2021 /tmp/run_terra.rs \
    --extern continuum_cdsl=target/debug/libcontinuum_cdsl.rlib \
    --extern continuum_runtime=target/debug/libcontinuum_runtime.rlib \
    -L target/debug/deps \
    -o /tmp/run_terra 2>&1 | head -20

if [ -f /tmp/run_terra ]; then
    echo "Running Terra..."
    /tmp/run_terra
else
    echo "Failed to compile. Use: cargo run -p continuum_inspector -- examples/terra"
fi
