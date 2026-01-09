# Continuum Kernel Macros

Procedural macros for the Continuum kernel system.

This crate exports the `#[kernel_fn]` attribute macro, which simplifies the registration of engine-provided functions into the global kernel registry.

## Usage

```rust
use continuum_kernel_macros::kernel_fn;
use continuum_foundation::Dt;

/// Exponential decay toward zero
#[kernel_fn(name = "decay")]
pub fn decay(value: f64, halflife: f64, dt: Dt) -> f64 {
    value * 0.5_f64.powf(dt.0 / halflife)
}
```

The macro handles:
- Generating the `KernelDescriptor`.
- Wrapping the function to match the uniform `KernelImpl` signature.
- Registering the kernel in the `linkme` distributed slice.
