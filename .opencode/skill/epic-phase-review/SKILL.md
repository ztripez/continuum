---
name: epic-phase-review
description: Mandatory review workflow for completed epic phases using specialized review agents
license: MIT
compatibility: opencode
metadata:
  domain: process
  applies_to: all_worlds
---

# Epic Phase Review Protocol

**MANDATORY**: When any phase of an epic is implemented, you MUST send it to review agents for feedback BEFORE marking it complete.

## When This Applies

- **After implementing** any phase within an epic (design, implementation, testing, documentation)
- **Before closing** the phase's beads issue
- **Regardless of confidence** - all phases get reviewed

## Review Agent Selection

Choose agents based on what was implemented:

### Code Implementation
- `rust-doc-enforcer` - Verify Rust documentation standards
- `fail-hard-officer` - Check for error masking patterns
- `abstraction-architect` - Identify abstraction opportunities
- `code-hygiene-auditor` - Audit KISS/DRY/YAGNI violations
- `architecture-guardian` - Verify alignment with project architecture

### DSL/Language Work
- `dsl-architect-reviewer` - Review lexer/parser/AST/IR implementations
- `continuum-architect` - Validate against core principles and execution model

### Simulation/Field Work
- `math-field-theorist` - Review mathematical formulations and algorithms
- `compute-optimizer` - Identify performance bottlenecks and GPU opportunities

### All Code Changes
- `qa-coverage-reviewer` - Verify test coverage and correctness

## Workflow

### 1. Identify What Changed
```bash
git diff main...HEAD --name-only
```

### 2. Select Appropriate Agents

Based on the changes:
- Rust code → `rust-doc-enforcer`, `fail-hard-officer`, `code-hygiene-auditor`
- New abstractions → `abstraction-architect`
- Cross-domain changes → `architecture-guardian`
- DSL implementation → `dsl-architect-reviewer`
- Field/sampling code → `math-field-theorist`
- Performance-critical → `compute-optimizer`
- Any code → `qa-coverage-reviewer`

### 3. Launch Review Agents

Use the `mcp_task` tool with appropriate `subagent_type`:

**Example - Documentation Review:**
```xml
<invoke name="mcp_task">
<parameter name="subagent_type">rust-doc-enforcer