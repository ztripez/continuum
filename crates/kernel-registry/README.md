# Continuum Kernel Registry

Distributed registry for engine-provided kernel functions.

This crate provides the mechanism to collect, look up, and dispatch kernel functions. It uses `linkme` to allow kernels to be defined anywhere in the codebase and automatically registered at link time.

## Architecture

- **`KernelDescriptor`**: Metadata about a kernel (name, docs, arity, backend hints).
- **`KernelImpl`**: Uniform function pointer types (`PureFn`, `DtFn`).
- **`KERNELS`**: Distributed slice collecting all registered kernels.

## Features

- **Discovery**: `is_known_in("namespace", "name")` checks if a kernel exists.
- **Dispatch**: `eval_in_namespace("namespace", "name", args, dt)` invokes the kernel dynamically.
- **Metadata**: Provides introspection for the DSL compiler to validate function calls.
