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

- **3 signals**: core.temp, surface.temp, temp_gradient (all with `initial` blocks)
- **1 fracture**: core.decay_heat (emits energy when temp drops)
- **1 impulse**: thermal_shock (external thermal event)
- **1 field**: avg_temp (observer output)
- **2 strata**: thermal, events (stride=1)
- **2 eras**: early (hot, dt=31.5Py), stable (cooled, dt=3.15Py)

## Features Demonstrated

- Signal initial values (`initial { }` blocks)
- Signal resolution with `prev` (history access)
- Fracture detection and emission in Collect phase
- Impulses for external events
- Fields for observer output
- Multiple strata with era activation
- Era transitions based on conditions
- Assertions for invariant validation
