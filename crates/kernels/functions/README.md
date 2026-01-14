# Continuum Functions

Standard library of kernel functions.

This crate implements built-in primitives under namespaces like `maths.*`, `vector.*`, and `dt.*`.

## Categories

- **Math**: `sin`, `cos`, `pow`, `exp`, `lerp`, `clamp`...
- **Vector**: `dot`, `cross`, `normalize`, `magnitude`...
- **Physics**: `gravity`, `orbital_velocity`, `stefan_boltzmann`...
- **Reduction**: `sum`, `max`, `min`...

All functions in this crate are registered via `continuum_kernel_macros` and are available for lookup in `continuum_kernel_registry`.
