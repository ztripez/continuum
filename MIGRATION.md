# DSL Migration Guide: Old → New Compiler

This document lists all syntax changes required to migrate existing `.cdsl` files from the old compiler to the new compiler (compiler-rewrite branch).

## Parser Fixes Applied

The new parser was updated to accept:
- Keywords (`config`, `const`, `signal`, `field`, `entity`, `dt`, `strata`, `type`) in path segments
- Keywords (`strata`, `type`, `entity`, `dt`) as attribute names

## Breaking Syntax Changes

### 1. Attribute Placement

**Old syntax:** Attributes inside signal/field body
```cdsl
signal core.temp {
    : Scalar<K, 100..10000>
    : strata(thermal)
    : title("Core Temperature")
    
    resolve { ... }
}
```

**New syntax:** All attributes in header (before `{`)
```cdsl
signal core.temp : type Scalar<K> : strata(thermal) : title("Core Temperature") {
    resolve { ... }
}
```

### 2. Type Annotations

**Old syntax:** Type as attribute
```cdsl
signal x {
    : Scalar<K>
    resolve { ... }
}
```

**New syntax:** Explicit `: type` prefix
```cdsl
signal x : type Scalar<K> {
    resolve { ... }
}
```

### 3. Type Bounds (NOT SUPPORTED)

**Old syntax:** Range bounds in type parameters
```cdsl
signal x : type Scalar<K, 100..10000> { ... }
```

**New syntax:** Bounds removed (not yet supported by parser)
```cdsl
signal x : type Scalar<K> { ... }
```

### 4. Transition Blocks

**Old syntax:** Block with `to:` field
```cdsl
era early {
    transition {
        to: era.stable
        when { ... }
    }
}
```

**New syntax:** Inline target path
```cdsl
era early {
    transition stable when { ... }
}
```

Note: Era references drop the `era.` prefix (keywords can't start paths in old position).

### 5. Signal References

**Old syntax:** `signal.` prefix
```cdsl
resolve {
    signal.core.temp + signal.surface.temp
}
```

**New syntax:** Direct signal name
```cdsl
resolve {
    core.temp + surface.temp
}
```

### 6. Config/Const References (NOT FULLY SUPPORTED)

**Old syntax:** Dotted paths
```cdsl
config {
    thermal.decay_halflife : type Scalar<s> = 1.42e17 <s>
}

resolve {
    config.thermal.decay_halflife
}
```

**Current workaround:** Flatten paths (no dots)
```cdsl
config {
    decay_halflife : type Scalar<s> = 1.42e17 <s>
}

resolve {
    decay_halflife
}
```

**Issue:** Parser doesn't support dotted path expressions yet. Namespace paths like `config.X.Y` don't parse in expressions.

### 7. Namespace Function Calls (NOT SUPPORTED)

**Old syntax:** Namespace function calls
```cdsl
resolve {
    let decayed = dt.decay(prev, halflife) in
    let abs_val = maths.abs(gradient) in
    maths.clamp(value, min, max)
}
```

**Current workaround:** Remove or inline
```cdsl
resolve {
    prev  // dt.decay not available
}
```

**Issue:** Dotted paths in expressions not yet supported.

### 8. Assertion Syntax (PARTIAL SUPPORT)

**Old syntax:** Severity levels and messages
```cdsl
assert {
    prev >= 100 <K> : fatal, "temperature too low"
    prev <= 10000 <K> : warn, "temperature high"
}
```

**New syntax:** Boolean expressions only (no severity/message)
```cdsl
assert {
    prev >= 100 <K>
}
```

Multiple assertions require separate `assert` blocks or may not be supported yet.

### 9. Fracture Emit Blocks (NOT WORKING)

**Old syntax:** Signal assignment in emit
```cdsl
fracture core.decay_heat {
    when { core.temp < 5498 <K> }
    emit { core.temp <- 5.0 <K> }
}
```

**Issue:** Parser tries to parse emit body as expression before trying statement, so `<-` operator fails. Semicolons don't help.

**Workaround:** Comment out fractures for now.

### 10. Chronicle Observe Blocks (NOT WORKING)

**Old syntax:** Event emission in chronicles
```cdsl
chronicle thermal.events {
    observe {
        when signal.core.temp < 4500 <K> {
            emit event.significant_cooling {
                core_temp: signal.core.temp
            }
        }
    }
}
```

**Issue:** `emit` keyword not expected in `when` block body.

**Workaround:** Comment out chronicles for now.

### 11. Path Collisions

**Issue:** Signal names can't share prefixes with stratum/entity names.

**Example conflict:**
```cdsl
strata thermal { }
signal thermal.gradient { }  // ← Collision!
```

**Solution:** Rename to avoid prefix overlap
```cdsl
strata thermal { }
signal temp_gradient { }  // ← OK
```

### 12. Unknown Units

Units like `Myr`, `kyr` are not recognized. Use base units (`s`, `m`, `kg`, etc.) or define derived units in world manifest.

## Summary of Required Changes

To migrate a `.cdsl` file:

1. ✅ **Move all attributes to header** (before `{`)
2. ✅ **Add `: type` prefix** to type annotations
3. ✅ **Remove type bounds** (`Scalar<K, min..max>` → `Scalar<K>`)
4. ✅ **Update transition syntax** (`transition { to: X }` → `transition X when`)
5. ✅ **Remove `signal.` prefix** from signal references
6. ⚠️ **Flatten config/const paths** (remove dots until path expressions supported)
7. ⚠️ **Remove namespace function calls** (`dt.decay`, `maths.abs`) until supported
8. ⚠️ **Simplify assertions** (remove severity/message)
9. ⚠️ **Comment out fractures** (emit syntax broken)
10. ⚠️ **Comment out chronicles** (observe syntax broken)
11. ✅ **Rename signals** to avoid path prefix collisions
12. ✅ **Replace unknown units** with base or defined units

## Known Parser Limitations

These are fundamental limitations of the new parser that need fixing:

1. **No dotted path expressions** - `config.X.Y`, `dt.decay()`, `maths.abs()` don't parse in expression contexts
2. **Emit block parser bug** - `choice()` tries expression before statement, can't backtrack
3. **Type bounds not implemented** - Range constraints in types not parsed
4. **Assertion metadata removed** - Severity levels and messages not supported
5. **Chronicle emit syntax unclear** - `emit` keyword not accepted in `when` blocks

## Files Modified in POC Migration

- `examples/poc/poc.cdsl` - Fully migrated (parses successfully, semantic errors remain)
- `crates/continuum-cdsl/src/parser.rs` - Added keyword support in paths/attributes

## Next Steps

Terra migration will require the same changes across ~1000 lines. Consider:

1. Fix path expression parser to support dotted paths
2. Fix emit block parser (reorder choice or improve lookahead)
3. Implement type bounds in type expression parser
4. Document chronicle/fracture emit syntax properly
5. Add assertion severity/message support if needed
