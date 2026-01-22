# Running Terra Simulation

Terra is an Earth-like planet simulation in `examples/terra/`

## Method 1: Web Inspector (Recommended)

```bash
cargo run -p continuum_inspector -- examples/terra
```

This starts a web server with an interactive UI for:
- Running simulations step-by-step
- Visualizing fields and signals
- Inspecting world state

**Note:** First compile takes ~5-10 minutes due to the large codebase.

## Method 2: Programmatic (Library API)

```rust
use continuum_cdsl::compile;
use continuum_runtime::build_runtime;

let compiled = compile(Path::new("examples/terra"))?;
let mut runtime = build_runtime(compiled);

for _ in 0..100 {
    runtime.tick();
}
```

## Method 3: Run Tool (CLI)

```bash
# Using the tools library
cargo test -p continuum-tools run_terra --release -- --nocapture
```

## What's in Terra?

- **terra.cdsl** - World config, strata, eras
- **geophysics/** - Plate tectonics, crust dynamics
- **stellar/** - Orbital mechanics, solar radiation
- **atmosphere/** - Radiative transfer, clouds
- **hydrology/** - Water cycle, precipitation
- **ecology/** - Carbon cycle, biosphere
- **scenarios/** - Initial conditions (default.yaml, early_earth.yaml, etc.)

## Scenarios

Terra supports different initial conditions:

```bash
# Default modern Earth
cargo run -p continuum_inspector -- examples/terra --scenario default

# Early Earth (high volcanic activity)
cargo run -p continuum_inspector -- examples/terra --scenario early_earth

# Cold start (frozen planet)
cargo run -p continuum_inspector -- examples/terra --scenario cold_start
```

## Validation Tests

Check if Terra produces Earth-like results:

```bash
cargo test -p continuum-runtime terra_validation
```

Or use the validate-terra skill for comprehensive testing.
