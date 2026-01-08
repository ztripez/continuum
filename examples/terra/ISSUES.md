# Terra DSL Issues Log

This file tracks DSL features that are missing or broken, discovered during the terra reorganization.

## Fixed Issues

### 1. Unit strings with multiple slashes (e.g., `kg/m²/yr`)
- **Status**: FIXED
- **Example**: `Scalar<kg/m²/yr, 0..1>` now parses correctly
- The `unit_string` parser already supported this - issue was elsewhere

### 2. If-else expressions
- **Status**: FIXED
- **Example**: `if x > 0 { a } else { b }` works in resolve blocks
- Added `if_expr` parser to `expr.rs`

### 3. Logical operators (&& and ||)
- **Status**: FIXED
- **Example**: `if phase > 0.5 && phase < 1.5 { ... }` now parses
- Added `logical_and` and `logical_or` operators to expression parser

### 4. Complex unit expressions in types
- **Status**: FIXED
- **Supported**: `kg/m³`, `W/m²`, `m/s²`, `N/m²`, `kg/m²/yr`, `Pa*s`
- All compound units with `/` and `*` now parse correctly

## Features That Parse (Verified Working)

### 1. Vector constructors (vec2, vec3, vec4)
- **Status**: WORKS (as function calls)
- **Example**: `vec2(x, y)`, `vec3(x, y, z)` parse as function calls

### 2. Vector field access (.x, .y, .z, .w)
- **Status**: WORKS
- **Example**: `prev.x`, `signal.rotation.state.y` parse correctly

### 3. Modulo function
- **Status**: WORKS (as function call)
- **Example**: `mod(phase, 6.283)` parses as a function call

### 4. User-defined functions
- **Status**: WORKS
- **Example**: `fn.namespace.name(params) { body }` parses correctly

### 5. relax_to function
- **Status**: WORKS (as function call)
- **Example**: `relax_to(prev, target, tau)` parses as a function call

## Runtime Features (Need IR/Codegen)

The following features parse correctly but need implementation in IR and runtime:

### Built-in Functions
Functions like `vec2`, `vec3`, `mod`, `relax_to`, `decay`, `clamp`, `exp`, `pow`, `sqrt`, `abs`, etc.
need to be registered and implemented in the function registry.

### Cross-Module Signal Dependencies
The following signals reference signals from other modules:

#### hydrology depends on:
- `signal.surface.temp` (from geophysics)
- `signal.atmosphere.water_vapor` (from atmosphere)
- `signal.atmosphere.co2_ppmv` (from atmosphere)

#### ecology depends on:
- `signal.surface.temp` (from geophysics)
- `signal.crust.elevation` (from geophysics)
- `signal.atmosphere.surface_temp` (from atmosphere)
- `signal.atmosphere.co2_ppmv` (from atmosphere)
- `signal.hydrology.water_mass` (from hydrology)

#### atmosphere depends on:
- `signal.surface.temp` (from geophysics) - or defines its own

### Cross-Module Constant Access
- `const.hydrology.reference_water_mass_kg` - used by ecology
- All other constants are module-local

## Tools Available

- `dsl-lint <file.cdsl>` - Parse and validate a single DSL file with error context
- `world-load <world-dir>` - Load and parse all DSL files in a world
- `world-run <world-dir>` - Load, compile, and run a world simulation
