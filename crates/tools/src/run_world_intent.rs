use std::path::PathBuf;

use continuum_cdsl::ast::{BinaryBundle, CompiledWorld};
use continuum_cdsl::compile;
use continuum_runtime::build_runtime;
use continuum_runtime::executor::{run_simulation, RunError, RunOptions, RunReport};

/// The source of a world to execute.
#[derive(Debug, Clone)]
pub enum WorldSource {
    /// A world root directory containing `*.cdsl` sources.
    Directory(PathBuf),
    /// A compiled binary bundle (`.cvm`).
    CompiledBundle(PathBuf),
    /// A compiled world serialized as JSON.
    CompiledJson(PathBuf),
}

impl WorldSource {
    /// Infer a world source from a path.
    pub fn from_path(path: PathBuf) -> Result<Self, RunWorldError> {
        if path.is_dir() {
            return Ok(Self::Directory(path));
        }

        let ext = path
            .extension()
            .and_then(|value| value.to_str())
            .ok_or_else(|| {
                RunWorldError::Io(format!(
                    "Cannot determine world source for path without extension: {}",
                    path.display()
                ))
            })?;

        match ext {
            "cvm" => Ok(Self::CompiledBundle(path)),
            "json" => Ok(Self::CompiledJson(path)),
            _ => Err(RunWorldError::Io(format!(
                "Unsupported world source extension: .{}",
                ext
            ))),
        }
    }

    pub fn load(&self) -> Result<CompiledWorld, RunWorldError> {
        match self {
            WorldSource::Directory(path) => compile(path).map_err(RunWorldError::from_compile),
            WorldSource::CompiledBundle(path) => {
                let data = std::fs::read(path)
                    .map_err(|err| RunWorldError::Io(format!("{}: {}", path.display(), err)))?;
                let bundle: BinaryBundle = bincode::deserialize(&data).map_err(|err| {
                    RunWorldError::Deserialize(format!("{}: {}", path.display(), err))
                })?;
                Ok(bundle.world)
            }
            WorldSource::CompiledJson(path) => {
                let data = std::fs::read_to_string(path)
                    .map_err(|err| RunWorldError::Io(format!("{}: {}", path.display(), err)))?;
                serde_json::from_str(&data).map_err(|err| {
                    RunWorldError::Deserialize(format!("{}: {}", path.display(), err))
                })
            }
        }
    }
}

/// Intent describing the execution lifecycle of a Continuum world.
pub struct RunWorldIntent {
    /// Where to load the world from.
    pub source: WorldSource,
    /// Runtime run options.
    pub options: RunOptions,
    /// Optional deterministic seed.
    pub seed: Option<u64>,
}

impl RunWorldIntent {
    /// Build a new intent with default run options.
    pub fn new(source: WorldSource, steps: u64) -> Self {
        let mut options = RunOptions::default();
        options.steps = steps;
        Self {
            source,
            options,
            seed: None,
        }
    }

    /// Load the compiled world from the source without executing.
    pub fn load(&self) -> Result<CompiledWorld, RunWorldError> {
        self.source.load()
    }

    /// Execute the intent and run the world simulation.
    pub fn execute(self) -> Result<RunReport, RunWorldError> {
        if self.options.steps == 0 {
            return Err(RunWorldError::Runtime(
                "Simulation requires at least 1 step".to_string(),
            ));
        }
        let compiled = self.source.load()?;
        let mut runtime = build_runtime(compiled);
        if let Some(seed) = self.seed {
            runtime.set_initial_seed(seed);
        }
        run_simulation(&mut runtime, self.options).map_err(RunWorldError::from_run_error)
    }
}

/// Errors emitted while executing a world intent.
#[derive(Debug)]
pub enum RunWorldError {
    /// Underlying IO failure.
    Io(String),
    /// Compilation failure for CDSL sources.
    Compile(String),
    /// Deserialize error for compiled worlds.
    Deserialize(String),
    /// Runtime execution failure.
    Runtime(String),
}

impl RunWorldError {
    fn from_compile(errors: Vec<continuum_cdsl::error::CompileError>) -> Self {
        let message = errors
            .iter()
            .map(|err| err.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        RunWorldError::Compile(message)
    }

    fn from_run_error(error: RunError) -> Self {
        RunWorldError::Runtime(error.to_string())
    }
}

impl std::fmt::Display for RunWorldError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunWorldError::Io(message) => write!(f, "io error: {}", message),
            RunWorldError::Compile(message) => write!(f, "compile error: {}", message),
            RunWorldError::Deserialize(message) => write!(f, "deserialize error: {}", message),
            RunWorldError::Runtime(message) => write!(f, "runtime error: {}", message),
        }
    }
}

impl std::error::Error for RunWorldError {}

/// Load a world source without executing it.
pub fn load_world(source: &WorldSource) -> Result<CompiledWorld, RunWorldError> {
    source.load()
}

/// Convenience helper to load and execute from a path.
pub fn run_world_from_path(
    path: PathBuf,
    options: RunOptions,
    seed: Option<u64>,
) -> Result<RunReport, RunWorldError> {
    let source = WorldSource::from_path(path)?;
    let intent = RunWorldIntent {
        source,
        options,
        seed,
    };
    intent.execute()
}
