# Execution Lifecycle

This document defines the **lifecycle of a Continuum run**.

The lifecycle describes **when** major system stages occur, from loading a World
to advancing causal history under a specific Scenario.

It does **not** define execution logic or time semantics; those are covered elsewhere.

---

## Overview

A Continuum run proceeds through the following stages:

1. World loading
2. DSL compilation
3. Execution graph construction
4. Scenario application
5. Warmup (pre-causal)
6. Causal execution (ticks)

Only stages 5 and 6 involve computation.
Only stage 6 produces causal history.

---

## 1. World Loading

The system loads a World from disk.

This includes:
- locating `world.yaml`
- loading and merging all YAML configuration
- discovering all DSL source files

At this stage:
- no logic is executed
- no time advances
- no state exists

World loading establishes **causal structure**, not instantiation.

---

## 2. DSL Compilation

All DSL files are compiled.

This includes:
- parsing source files
- building the symbol table
- type checking
- lowering to typed IR
- inferring dependencies

The output of this stage is a **complete, validated intermediate representation**
of the World’s declared causal structure.

No execution occurs during compilation.

---

## 3. Execution Graph Construction

From the typed IR, the system builds **execution graphs**.

Graphs are constructed per:

```

(phase × stratum × era)

```

This stage includes:
- dependency analysis
- cycle detection
- topological ordering
- partitioning into parallel execution levels

The result is a fully determined execution plan.

At the end of this stage:
- execution order is fixed
- parallelism boundaries are known
- no runtime decisions about ordering remain

Graph construction is independent of Scenario.

---

## 4. Scenario Application

A **Scenario** is applied to the World to produce a concrete run configuration.

This stage includes:
- applying initial signal values
- applying signal parameters and constants
- **loading and freezing config/const values**
- validating scenario-defined bounds and assertions
- fixing all instantiation-time choices

### Config and Const Loading

During scenario application, configuration and constant values are extracted from
DSL declarations and frozen in the runtime execution context:

**Const Values** (`const {}` blocks):
- Global simulation constants that never change
- Loaded from `value` field in `ConstEntry` declarations
- NOT overridable by scenarios (immutable across all runs)
- Must be compile-time literals (Scalar or Vec3 with literal components)
- Example: `const { physics.stefan_boltzmann: 5.67e-8 }`

**Config Values** (`config {}` blocks):
- Configuration parameters with world-provided defaults
- Loaded from `default` field in `ConfigEntry` declarations
- Scenarios may override config values (but not const values)
- Must be compile-time literals (Scalar or Vec3 with literal components)
- Example: `config { thermal.decay_halflife: 1.42e17 }`

Both config and const values are:
- Extracted during `build_runtime()` function execution
- Stored in `BytecodePhaseExecutor.config_values` and `BytecodePhaseExecutor.const_values` HashMap fields
- Passed by reference to all `VMContext` instances during phase execution
- Frozen before warmup begins
- Available to all execution phases as immutable context
- Never recomputed or modified after loading

If a config or const declaration contains a non-literal expression (kernel calls,
references, operators), `build_runtime()` panics with a clear error message
(enforcing the Fail Loudly principle).

Scenario application:
- does not advance time
- does not execute logic
- does not produce causal history

After this stage, the run has a fully specified **initial state** with frozen
config/const values.

---

## 5. Warmup (Pre-Causal)

Warmup is a **bounded, deterministic pre-execution phase**.

Warmup exists to prepare values that must be **stable before causal history begins**.

Warmup may be used to:
- derive time unit tables
- stabilize reconstruction caches
- converge internal fixed-point values
- validate execution invariants

Warmup is **not time**.

During warmup:
- no ticks advance
- no causal history is produced
- no fields are emitted
- no observers are invoked

Warmup must be:
- deterministic
- bounded
- non-observable

If warmup produces divergent or unstable results, execution must fail.

For full warmup semantics and DSL syntax, see @execution/warmup.md.

---

## 6. Begin Causal Execution

After warmup completes successfully, causal execution begins.

From this point onward:
- ticks advance time
- execution phases run in order
- signals are resolved
- fields may be emitted
- observers may attach

Everything that happens after this point is part of the simulation’s causal history.

---

## Lifecycle Guarantees

The lifecycle guarantees that:

- World structure is fixed before instantiation
- Scenario configuration is fixed before execution
- No observer can influence execution before causality begins
- No time advancement occurs before warmup completes
- Execution order is fully determined before the first tick
- Failures before the first tick leave no partial history

---

## What the Lifecycle Is Not

The lifecycle is **not**:
- a runtime loop description
- a scheduling algorithm
- a visualization concern
- a domain-specific concept

It is the **structural envelope** within which execution occurs.

---

## Summary

- Lifecycle defines *when* things happen
- Worlds define structure
- Scenarios define instantiation
- Warmup is pre-causal and non-temporal
- Causal history begins only after warmup
- Execution order is fixed before time advances

Nothing that happens before the first tick is part of time.
