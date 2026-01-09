# Continuum VM

Stack-based bytecode virtual machine for efficient expression evaluation.

This crate executes the logic within signal resolvers and operators. It provides a compact, cache-friendly instruction set designed for deterministic floating-point computation.

## Features

- **Bytecode**: Flat `Op` enum (`Add`, `Mul`, `LoadSignal`, `Call`).
- **Compiler**: Compiles `continuum_ir::CompiledExpr` into `BytecodeChunk`.
- **Executor**: Stack-based interpreter loop.
- **Determinism**: Strict adherence to execution order and operations.

## Instruction Set

The VM supports:
- Arithmetic (`Add`, `Sub`, `Pow`...)
- Logic & Comparison (`Eq`, `Gt`, `And`...)
- Context Access (`LoadPrev`, `LoadDt`, `LoadInputs`)
- Kernel Dispatch (`Call`)
- Control Flow (`Jump`, `JumpIfZero`)
