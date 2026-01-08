# Execution DAG

This document defines the **Execution DAG** (Directed Acyclic Graph) used by Continuum
to schedule and execute simulation logic.

The DAG is the **authoritative execution plan**.
It is the concrete representation of causality.

---

## 1. Why a DAG Exists

Simulation logic has dependencies.

- Signals depend on other signals
- Operators depend on resolved state
- Some work may run in parallel
- Some work must be ordered

The execution DAG exists to:
- make dependencies explicit
- derive execution order from causality
- enable safe parallelism
- guarantee determinism

If execution order is not derivable from the DAG, the system is incorrect.

---

## 2. What the DAG Represents

An execution DAG represents **work**, not data.

Each node represents a unit of execution.
Each edge represents a dependency that must be satisfied before execution.

The DAG is constructed from:
- DSL declarations
- inferred signal reads
- phase and stratum constraints
- era-specific activation rules

The DAG is **fully determined before execution begins**.

---

## 3. DAG Partitioning

Execution DAGs are constructed per:

```

(phase × stratum × era)

```

This partitioning ensures:
- clear phase boundaries
- independent multi-rate execution
- predictable gating and cadence
- smaller, analyzable graphs

There is no single global DAG.
There are many small, well-scoped DAGs.

---

## 4. Node Types

At minimum, the DAG contains the following node categories:

### Signal Resolution Nodes
- Resolve a single signal
- Read resolved signals from the previous tick
- Write exactly one resolved signal value

### Operator Nodes
- Execute an operator body
- May read resolved signals
- May emit signal inputs or fields depending on phase

Node behavior is determined by:
- phase
- stratum
- declared body

---

## 5. Dependency Edges

Edges in the DAG represent **causal necessity**.

An edge `A → B` means:
> Node B must not execute until node A has completed.

Dependencies are inferred from:
- signal reads
- operator inputs
- phase ordering
- stratum cadence rules

Authors do **not** declare dependencies manually.

---

## 6. Acyclicity Is Mandatory

The execution graph must be acyclic.

- Cycles represent causal contradictions
- Cycles are detected during graph construction
- Any cycle is a compile-time error

There is no runtime cycle resolution.
If a cycle exists, the model is invalid.

---

## 7. Topological Levels

The DAG is executed in **topological levels**.

A level is a set of nodes that:
- have no unsatisfied dependencies between them
- may execute in parallel

Execution proceeds as:
1. execute all nodes in level 0
2. barrier
3. execute all nodes in level 1
4. barrier
5. repeat until complete

Levels are **stably ordered**.
Parallelism never changes which nodes belong to which level.

---

## 8. Deterministic Parallelism

Parallel execution is permitted only when it cannot change meaning.

The DAG guarantees:
- no data races on authoritative state
- no ordering-dependent behavior
- stable reduction semantics

If two nodes can execute in parallel, they are causally independent.

Parallelism is an optimization, not a semantic feature.

---

## 9. Phase Barriers and DAG Boundaries

DAG execution respects phase boundaries.

- DAGs do not cross phase boundaries
- A full barrier exists between phases
- Data produced in a phase is not visible to earlier phases

Phase ordering is enforced **outside** the DAG.
The DAG never encodes phase transitions.

---

## 10. Interaction with Eras and Strata

Era and stratum configuration affects:
- which DAGs are active
- which nodes are scheduled
- how often DAGs execute

Era transitions:
- do not modify DAG structure
- select a different precomputed DAG set

The shape of a DAG is static within an era.

---

## 11. Failure and the DAG

Failures during DAG execution are fatal by default.

- Assertion failures
- Invalid values
- Budget overruns

A failure:
- halts execution at a defined boundary
- produces no partial causal history beyond that point
- leaves previous history intact

The DAG ensures failures are localized and diagnosable.

---

## 12. What the DAG Is Not

The execution DAG is **not**:
- a scripting language
- a dynamic task graph
- a heuristic scheduler
- an observer artifact

It is a **compiled, static execution plan** derived from causality.

---

## Summary

- The DAG is the authoritative execution plan
- Dependencies are inferred, not authored
- Cycles are illegal
- Parallelism is safe and deterministic
- Phase and stratum boundaries are enforced
- Execution order is fully known before time advances

If execution order feels implicit, ad-hoc, or timing-dependent,
the model is wrong.
```