# Continuum Kernel Registry

Distributed registry for engine-provided kernel functions.

This crate provides the mechanism to collect, look up, and dispatch kernel functions. It uses `linkme` to allow kernels to be defined anywhere in the codebase and automatically registered at link time.

## Architecture

- **`KernelDescriptor`**: Metadata about a kernel (name, docs, arity, backend hints).
- **`KernelImpl`**: Uniform function pointer types (`PureFn`, `DtFn`).
- **`KERNELS`**: Distributed slice collecting all registered kernels.

## Features

- **Discovery**: `is_known("name")` checks if a kernel exists.
- **Dispatch**: `eval("name", args, dt)` invokes the kernel dynamically (used by the VM).
- **Metadata**: Provides introspection for the DSL compiler to validate function calls.
