# Bytecode & VM Execution

This document specifies the **Continuum Bytecode VM**, the instruction set, and how DSL execution blocks are lowered and executed at runtime.

---

## 1. Overview

Continuum does not interpret DSL source or AST at runtime. Instead, execution blocks (e.g., `resolve { ... }`) are compiled into a compact, typed **bytecode** representation.

The bytecode VM is:
- **Stack-based**: Operations pop arguments from and push results to an evaluation stack.
- **Slot-aware**: Local bindings (`let`) use indexed slots for high-performance access.
- **Deterministic**: All operations are reproducible and explicit.
- **Phase-restricted**: Opcodes enforce architectural boundaries (e.g., `emit` is only allowed in `Collect`).

---

## 2. VM State

Each block execution maintains a `VMContext`:

| Component | Description |
|-----------|-------------|
| **Stack** | Evaluation stack for temporary values. |
| **Slots** | Indexed storage for local variables and loop bindings. |
| **Registers** | Fast access to context values (`prev`, `current`, `dt`, `self`). |
| **Channels** | Accumulators for side-effects (`emit` signal/field). |

---

## 3. Instruction Set

### 3.1 Stack Operations

| Opcode | Operands | Description |
|--------|----------|-------------|
| `PushLiteral` | `Value` | Pushes a constant value onto the stack. |
| `Load` | `Slot` | Pushes the value from the specified slot onto the stack. |
| `Store` | `Slot` | Pops the top value and stores it in the specified slot. |
| `Dup` | - | Duplicates the top stack value. |
| `Pop` | - | Discards the top stack value. |

### 3.2 Constructors

| Opcode | Operands | Description |
|--------|----------|-------------|
| `BuildVector` | `dim: u8` | Pops `dim` scalars and pushes a `VecN` value. |
| `BuildStruct` | `fields: Vec<String>` | Pops values for each field and pushes a `Map`. |

### 3.3 Control Flow & Iteration

| Opcode | Operands | Description |
|--------|----------|-------------|
| `Let` | `Slot` | Starts a local scope for the specified slot. |
| `EndLet` | - | Ends the current local scope. |
| `Aggregate` | `EntityId, Slot, BlockId, Op` | Iterates entity instances, executing `BlockId` for each. |
| `Fold` | `EntityId, AccSlot, ElemSlot, BlockId` | Stateful reduction over entity instances. |
| `FieldAccess` | `name: String` | Pops a `Map` and pushes the value of the named field. |

### 3.4 Temporal & Context

| Opcode | Description | Allowed Phases |
|--------|-------------|----------------|
| `LoadSignal` | Load signal value by Path. | `Resolve`, `Fracture`, `Measure` |
| `LoadPrev` | Load previous tick value of current signal. | `Resolve` |
| `LoadCurrent`| Load current resolved value. | `Fracture`, `Measure` |
| `LoadInputs` | Load accumulated impulse inputs. | `Resolve` |
| `LoadDt` | Load current timestep (dt). | All |
| `LoadSelf` | Load current entity identity. | All (Members only) |
| `LoadOther` | Load "other" entity in interaction. | `Fracture` |
| `LoadPayload`| Load impulse payload. | `Collect` |

### 3.5 Effects

| Opcode | Description | Phase |
|--------|-------------|-------|
| `Emit` | Accumulate value to a signal. | `Collect` |
| `EmitField`| Write value to a spatial field. | `Measure` |
| `Spawn` | Create a new entity instance. | `Fracture` |
| `Destroy` | Mark instance for destruction. | `Fracture` |

### 3.6 Observation

| Opcode | Description | Phase |
|--------|-------------|-------|
| `LoadField` | Read reconstructed field value. | `Measure` (Observers only) |

---

## 4. Execution Lifecycle

1. **Lowering**: The `Compiler` transforms a `TypedExpr` into a sequence of instructions.
2. **Registry**: Opcodes are validated against the `OpcodeMetadata` table for operand counts and phase compatibility.
3. **Dispatch**: The `BytecodeExecutor` retrieves the `Handler` for each instruction from the global registry.
4. **Execution**: The handler manipulates the stack and context. Errors (e.g., stack underflow, invalid types) are reported as `ExecutionError`.

---

## 5. Failure Modes (Fail Loudly)

The VM does not perform silent corrections:
- **Type Mismatch**: Runtime type checks trigger immediate errors.
- **Missing Data**: Accessing an uninitialized slot or missing signal results in a fault.
- **Phase Violation**: Using an effect opcode in a pure phase causes an assertion failure.
- **Stack Safety**: Underflow or leftover values at block return are treated as corrupted state.
