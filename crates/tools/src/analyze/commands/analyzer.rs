//! Analyzer subcommands: list and run

use clap::Subcommand;
use continuum_ir::CompiledWorld;
use continuum_lens::FieldSnapshot;
use std::path::PathBuf;

use crate::analyze::types::SnapshotRun;

#[derive(Subcommand, Debug)]
pub enum AnalyzerCommand {
    /// List all available analyzers in a world
    List {
        /// Path to compiled world file
        #[arg(value_name = "WORLD")]
        world: PathBuf,
    },

    /// Run an analyzer on snapshot data
    Run {
        /// Analyzer name (e.g., terra.hypsometric_integral)
        #[arg(value_name = "NAME")]
        name: String,

        /// Path to compiled world file
        #[arg(value_name = "WORLD")]
        world: PathBuf,

        /// Path to snapshot directory
        #[arg(value_name = "SNAPSHOTS")]
        snapshots: PathBuf,

        /// Specific tick to analyze (defaults to latest)
        #[arg(long)]
        tick: Option<u64>,
    },
}

pub fn run(cmd: AnalyzerCommand) -> Result<(), String> {
    match cmd {
        AnalyzerCommand::List { world } => run_list(world),
        AnalyzerCommand::Run {
            name,
            world,
            snapshots,
            tick,
        } => run_analyzer(&name, world, snapshots, tick),
    }
}

/// List all analyzers in a world
fn run_list(world_path: PathBuf) -> Result<(), String> {
    // Load the world file
    let world_str = std::fs::read_to_string(&world_path)
        .map_err(|e| format!("Failed to read world file {}: {}", world_path.display(), e))?;

    let compiled_world: CompiledWorld = serde_json::from_str(&world_str)
        .map_err(|e| format!("Failed to parse world file: {}", e))?;

    let analyzers = compiled_world.analyzers();

    if analyzers.is_empty() {
        println!("No analyzers defined in world");
        return Ok(());
    }

    println!("Available Analyzers:");
    println!("====================\n");

    for (id, analyzer) in analyzers {
        println!("{}  {}", id, id.path());
        if let Some(doc) = &analyzer.doc {
            println!("  Documentation: {}", doc);
        }
        println!("  Required Fields: {}", analyzer.required_fields.len());
        for field in &analyzer.required_fields {
            println!("    - {}", field);
        }
        println!();
    }

    Ok(())
}

/// Run an analyzer on snapshot data
fn run_analyzer(
    analyzer_name: &str,
    world_path: PathBuf,
    snapshots_path: PathBuf,
    tick: Option<u64>,
) -> Result<(), String> {
    // Load the world file
    let world_str = std::fs::read_to_string(&world_path)
        .map_err(|e| format!("Failed to read world file {}: {}", world_path.display(), e))?;

    let compiled_world: CompiledWorld = serde_json::from_str(&world_str)
        .map_err(|e| format!("Failed to parse world file: {}", e))?;

    // Get the analyzer
    let analyzer_id = continuum_foundation::AnalyzerId::from(analyzer_name);
    let analyzers = compiled_world.analyzers();
    let analyzer = analyzers
        .get(&analyzer_id)
        .ok_or_else(|| format!("Analyzer '{}' not found in world", analyzer_name))?;

    // Load snapshots
    let snapshot_run = SnapshotRun::load(snapshots_path)?;

    // Determine which tick to use
    let target_tick = if let Some(t) = tick {
        t
    } else {
        *snapshot_run
            .ticks()
            .last()
            .ok_or_else(|| "No snapshots found".to_string())?
    };

    // Get the snapshot for the target tick
    let _tick_data = snapshot_run
        .get_snapshot(target_tick)
        .ok_or_else(|| format!("No snapshot found for tick {}", target_tick))?;

    // Convert field data to FieldSnapshot format
    let mut snapshots = Vec::new();
    for field_id in &analyzer.required_fields {
        // For now, create a placeholder FieldSnapshot
        // In a full implementation, this would load actual field samples from the snapshot
        let field_snapshot = FieldSnapshot {
            field_id: field_id.clone(),
            tick: target_tick,
            samples: vec![], // Placeholder - would load from tick_data
        };
        snapshots.push((field_id.clone(), field_snapshot));
    }

    // Execute the analyzer
    let result = continuum_ir::execute_analyzer(analyzer, &snapshots)
        .map_err(|e| format!("Analyzer execution failed: {}", e))?;

    // Print results
    println!("Analyzer: {}", analyzer_name);
    println!("Tick: {}", target_tick);
    println!("\nResults:");
    println!(
        "{}",
        serde_json::to_string_pretty(&result.output).map_err(|e| e.to_string())?
    );

    if !result.validations.is_empty() {
        println!("\nValidations:");
        for (i, validation) in result.validations.iter().enumerate() {
            let status = if validation.passed {
                "✓ PASS"
            } else {
                "✗ FAIL"
            };
            println!("  [{}] {} ({})", i + 1, status, validation.severity);
            if let Some(msg) = &validation.message {
                println!("      {}", msg);
            }
        }
    }

    Ok(())
}
