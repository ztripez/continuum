# Missing DSL Features for Terra World

This file tracks DSL features that are documented but not yet implemented in the parser/compiler.

**Status**: Terra world runs successfully with workarounds for missing features.
**Tested**: Bytecode VM executes 18 signals, 8 fields, 4 fractures successfully.

## Parser Features

### 1. `let` expressions in resolve blocks
- **Status**: IMPLEMENTED
- **Documented in**: docs/dsl/syntax.md line 536
- **Example**:
  ```
  resolve {
      let decay_factor = 0.693147 * dt / config.thermal.decay_halflife
      prev * exp(-decay_factor) + sum(inputs)
  }
  ```
- **Current workaround**: ~~Inline all expressions (reduces readability)~~ Now works!

### 2. `dt_raw` keyword (raw timestep access)
- **Status**: IMPLEMENTED (requires explicit `: uses(dt_raw)` declaration)
- **Documented in**: docs/dsl/syntax.md line 287
- **Note**: `dt` does NOT exist by design. Use dt-robust operators like `decay(prev, halflife)` instead. Raw `dt_raw` access requires explicit opt-in via `: uses(dt_raw)` attribute.

### 3. Unicode unit superscripts
- **Status**: NOT IMPLEMENTED
- **Examples**: `<m³/kg/s²>`, `<W/m²/K⁴>`, `<kg/m³>`
- **Current workaround**: Use dimensionless units or simple units like `<m>`, `<K>`, `<Pa>`

### 4. `fn` (user-defined functions)
- **Status**: IMPLEMENTED
- **Documented in**: docs/dsl/functions.md
- **Example**:
  ```
  fn.isostasy.buoyancy_factor(crustal_density, mantle_density) {
      1.0 - crustal_density / mantle_density
  }
  ```
- **Note**: Functions are pure and inlined at call sites. They can access `const.*` and `config.*`, call other functions, but cannot access `prev`, `dt_raw`, or write to signals.

### 5. Complex unit expressions
- **Status**: LIMITED
- **Working**: Simple units like `<K>`, `<m>`, `<Pa>`, `<s>`, `<kg>`, `<Myr>`, `<kyr>`
- **Not working**: Compound units with superscripts or complex fractions

## Runtime/Compiler Features

### 6. Cross-strata signal dependencies
- **Status**: UNKNOWN - needs testing
- **Example**: thermal strata signal reading from tectonics strata signal

### 7. Negative literal values in ranges
- **Status**: NEEDS TESTING
- **Example**: `Scalar<m, -11000..9000>`

## Kernel Functions Needed

The following kernel functions are used in the terra DSL:
- `decay(value, halflife)` - IMPLEMENTED
- `clamp(value, min, max)` - IMPLEMENTED
- `abs(x)` - IMPLEMENTED
- `exp(x)` - IMPLEMENTED
- `sum(inputs)` - IMPLEMENTED (special form)

### 8. Signal-local config blocks
- **Status**: IMPLEMENTED
- **Documented in**: docs/dsl/syntax.md line 256-262
- **Example**:
  ```
  signal.core.temp {
      config {
          initial_temp: 5500 <K>
      }
      resolve {
          config.core.temp.initial_temp  # Access via signal-prefixed path
      }
  }
  ```
- **Note**: Local config/const are namespaced under the signal path. A local `initial_temp` in `signal.core.temp` becomes `config.core.temp.initial_temp` globally.

### 9. Assert severity as bare keyword vs quoted string
- **Status**: Parser expects bare keyword (`warn`, `error`, `fatal`)
- **Documentation shows**: `warning` (doesn't work - use `warn`)

## Notes

When implementing these features:
1. Add parser support in `crates/kernels/dsl/src/parser/`
2. Add AST nodes if needed in `crates/kernels/dsl/src/ast.rs`
3. Add lowering in `crates/kernels/ir/src/lower.rs`
4. Add compilation in `crates/kernels/ir/src/compile.rs`
5. Add tests

## Priority for Implementation

**High Priority** (most impactful for DSL usability):
1. ~~`let` expressions~~ - IMPLEMENTED
2. ~~`fn` user-defined functions~~ - IMPLEMENTED

**Medium Priority**:
3. ~~Signal-local config blocks~~ - IMPLEMENTED
4. Unicode unit superscripts - matches documentation

**Low Priority** (workarounds exist):
5. ~~`dt` raw access~~ - IMPLEMENTED (with `: dt_raw` declaration)
