# Continuum IR

Typed Intermediate Representation (IR) and lowering logic.

The IR acts as the semantic bridge between the user-facing DSL AST and the executable runtime DAG. It resolves types, infers dependencies, inlines functions, and simplifies expressions into a form ready for compilation.

## Role

1.  **Lowering**: Transforms `continuum_dsl::ast` into `continuum_ir::types`.
2.  **Type Checking**: Enforces unit consistency and type safety.
3.  **Dependency Inference**: Analyzes expressions to build explicit read/write sets for signals and operators.
4.  **Function Inlining**: Expands user-defined DSL functions into their call sites.

## Key Types

- **`CompiledWorld`**: The complete, resolved definition of a simulation world.
- **`CompiledSignal`**: A signal with fully resolved dependencies and bytecode-ready expression.
- **`CompiledExpr`**: A simplified expression tree (no sugar, fully typed).
