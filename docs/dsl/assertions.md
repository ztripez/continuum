# Assertions

Assertions validate invariants at runtime without modifying values.
They emit structured faults on failure.

## Syntax

```cdsl
signal.core.temp {
    : Scalar<K, 100..10000>
    : strata(thermal)

    resolve {
        decay(prev, config.thermal.decay_halflife)
    }

    assert {
        prev > 100 <K>
        prev < 10000 <K> : warn, "temperature approaching upper bound"
        prev >= 0 : fatal, "temperature cannot be negative"
    }
}
```

## Assertion Components

Each assertion has:

1. **Condition** (required): An expression that must evaluate to true
2. **Severity** (optional): `warn`, `error` (default), or `fatal`
3. **Message** (optional): A string describing the failure

### Severity Levels

| Severity | Behavior |
|----------|----------|
| `warn`   | Log warning, execution continues |
| `error`  | Emit fault, may halt based on policy |
| `fatal`  | Emit fault, always halts simulation |

## Compile-Time Warnings

### Missing Range Assertions

If a signal declares a range constraint in its type but has no assertions,
the compiler emits a warning:

```cdsl
signal.core.temp {
    : Scalar<K, 100..10000>  # Range declared here
    : strata(thermal)

    resolve { prev }
    # WARNING: signal 'core.temp' has a range constraint but no assertions
}
```

To resolve this warning, add assertions that validate the range:

```cdsl
signal.core.temp {
    : Scalar<K, 100..10000>
    : strata(thermal)

    resolve { prev }

    assert {
        prev >= 100 <K>
        prev <= 10000 <K>
    }
}
```

Signals without range constraints do not trigger this warning:

```cdsl
signal.core.counter {
    : Scalar<count>  # No range, no warning
    resolve { prev + 1 }
}
```

## Operator Assertions

Operators can also have assertions:

```cdsl
operator.validate_energy {
    : strata(physics)
    : phase(resolve)

    collect {
        let total = signal.kinetic + signal.potential in
        emit_signal(signal.total_energy, total)
    }

    assert {
        signal.total_energy >= 0 : error, "energy conservation violated"
    }
}
```

## Best Practices

1. **Match type ranges with assertions**: If you declare `Scalar<K, 100..10000>`,
   add assertions to validate those bounds at runtime.

2. **Use appropriate severity**:
   - `warn` for soft limits that may be exceeded temporarily
   - `error` for invariants that indicate simulation problems
   - `fatal` for impossible states that corrupt simulation integrity

3. **Provide meaningful messages**: Help diagnose issues by describing what
   went wrong and potentially what caused it.

4. **Validate at boundaries**: Assert constraints on values that come from
   external sources or complex computations.
