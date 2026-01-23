# POC - Minimal Thermal Simulation

A minimal cooling simulation demonstrating the Continuum DSL and runtime.

## Quick Start

### Headless Mode (CLI)

Run for a fixed number of steps:

```bash
cargo run --bin continuum-run -- examples/poc --steps 10
```

### Interactive Mode (Web Inspector)

**Terminal 1 - Start simulation:**
```bash
cargo run --bin continuum-run -- examples/poc
```

**Terminal 2 - Start web inspector:**
```bash
cargo run --bin continuum_inspector
```

**Browser:**
Open http://localhost:8080

## What's in POC?

- **2 signals**: core.temp, surface.temp (with `initial` blocks)
- **1 derived signal**: temp_gradient
- **1 fracture**: core.decay_heat (emits energy when temp drops)
- **1 stratum**: thermal (stride=1)
- **2 eras**: early (hot), stable (cooled)

## Features Demonstrated

- Signal initial values (`initial { }` blocks)
- Signal resolution with `prev` (history access)
- Fracture detection and emission
- Era transitions
- Let expressions for readable calculations
- Assertions for invariant validation
