# Continuum DSL — Analyzers

This document defines the **analyzer primitive** for domain-specific analysis queries in CDSL.

---

## 1. Purpose of Analyzers

An `analyzer` is a **pure observer** that computes derived analysis results from field snapshots.

Analyzers:
- read field data post-hoc (after simulation completes)
- produce structured JSON results
- optionally include validation checks with severity levels
- never influence simulation causality
- can be invoked via CLI: `continuum analyze run <name> <snapshot>`

Analyzers replace hard-coded Rust domain analyzers, allowing world authors to define custom analysis without writing code.

---

## 2. What Analyzers Are (and Are Not)

### Analyzers **are**:
- pure observers (no state mutation)
- compile-time validated
- expressible in CDSL
- discoverable via CLI
- portable with worlds
- useful for CI validation

### Analyzers **are not**:
- simulation logic
- phase-aware executables
- causal influences
- part of the execution graph
- scriptable (deterministic structure, not runtime)

---

## 3. Anatomy of an Analyzer

An analyzer has four parts:

```cdsl
analyzer <namespace.name> {
    : doc "<description>"                           # optional
    : requires(fields: [<field1>, <field2>, ...])  # required
    
    : compute {                                      # required
        <expression that produces JSON>
    }
    
    : validate {                                     # optional
        <validation checks>
    }
}
```

### 3.1 Namespace and Name

Analyzers are namespaced:

```cdsl
analyzer terra.hypsometric_integral { ... }
analyzer hydrology.water_balance { ... }
analyzer physics.energy_conservation { ... }
```

Naming conventions:
- lowercase, dot-separated
- globally unique within a world
- should reflect the domain and analysis type
- should be short but descriptive

### 3.2 Documentation

```cdsl
analyzer terra.hypsometric_integral {
    : doc "Land/ocean ratio and elevation distribution"
    ...
}
```

The doc string is:
- required (enforced by linter)
- human-readable
- available in CLI output
- used for IDE hover text

### 3.3 Field Requirements

```cdsl
analyzer terra.hypsometric_integral {
    : requires(fields: [geophysics.elevation])
    ...
}
```

The `requires` clause declares field dependencies:
- all listed fields must exist in the world
- validated at compile time
- used for error messages if fields are missing
- can list multiple fields: `[field1, field2, field3]`

If a required field is missing at runtime:
```
Error: Analyzer terra.hypsometric_integral requires field geophysics.elevation, which was not captured in this snapshot
```

### 3.4 Compute Block

The compute block is an expression that produces a JSON-compatible value:

```cdsl
: compute {
    let samples = field.samples(geophysics.elevation)
    let above_sea = samples.filter(|s| s.value > 0.0).count()
    let total = samples.count()
    let integral = above_sea / total
    
    emit {
        integral: integral,
        land_fraction: integral,
        ocean_fraction: 1.0 - integral,
        sample_count: total,
        statistics: stats.compute(samples)
    }
}
```

Compute blocks:
- are pure expressions (no side effects)
- can use `let` bindings
- can call kernel functions from `stats.*`, `maths.*`, `vector.*`, etc.
- must produce a value compatible with JSON serialization
- can use `emit { ... }` to construct structured output
- have access to `field.samples(field_id)` to read field data

The `emit` syntax creates a JSON object:

```cdsl
emit {
    key1: value1,
    key2: value2,
    nested: emit {
        inner: value3
    }
}
```

### 3.5 Validation Block

```cdsl
: validate {
    check integral in 0.2..0.4
        : severity(warning)
        : message("Land fraction {integral*100:.1}% (Earth-like: 29%)")
    
    check ocean_fraction >= 0.5
        : severity(error)
        : message("Ocean coverage too low: {ocean_fraction:.1}%")
}
```

Validation blocks contain zero or more `check` statements.

Each check:
- evaluates a boolean condition
- has a severity level: `error`, `warning`, or `info`
- has an optional message with template substitution
- produces a ValidationResult in the output

Severity levels determine CI behavior:
- **error**: fails CI, non-zero exit code
- **warning**: logged but does not fail CI
- **info**: informational only

#### Message Templates

Messages support template substitution using `{name}` syntax:

```cdsl
: message("Land fraction {integral*100:.1}% (Earth-like: 29%)")
```

Template expressions:
- can reference variables from compute block (`integral`)
- support format specifiers (`.1` = 1 decimal place)
- support arithmetic: `{integral*100:.1}%`
- must be deterministic and type-correct

---

## 4. Field Sampling

Analyzers access field data through the `field.*` namespace:

### 4.1 Get All Samples

```cdsl
let samples = field.samples(geophysics.elevation)
```

Returns a `FieldSamples` collection. Type: `FieldSamples<f64>` (samples with f64 values).

Properties:
- `.count()` - number of samples
- `.sum()` - sum of all values
- `.min()` - minimum value
- `.max()` - maximum value
- `.filter(|s| ...)` - filter by predicate, returns new `FieldSamples`
- `.map(|s| ...)` - transform values, returns new `FieldSamples`

### 4.2 Filter Samples

```cdsl
let ocean = samples.filter(|s| s.value <= 0.0)
let land = samples.filter(|s| s.value > 0.0)
```

Predicates:
- receive a sample `s` with `.value` (f64) and `.position` (Vec3)
- return boolean
- are pure and deterministic

### 4.3 Position-Based Queries

```cdsl
let samples = field.samples(geophysics.elevation)
let by_latitude = samples.filter(|s| 
    let lat = util.latitude(s.position)
    lat >= -30.0 and lat <= 30.0
)
```

The `s.position` is a Vec3 in normalized spherical coordinates.

Helper functions in `util.*` namespace:
- `util.latitude(pos: Vec3) -> Scalar<1>` - latitude in degrees (-90..90)
- `util.longitude(pos: Vec3) -> Scalar<1>` - longitude in degrees (-180..180)
- `util.distance_to(pos: Vec3, target: Vec3) -> Scalar<m>` - great circle distance

---

## 5. Statistics Namespace

Analyzers have access to the `stats.*` kernel namespace for common computations:

### 5.1 Descriptive Statistics

```cdsl
let stats_result = stats.compute(samples)
```

Returns:
```json
{
    "count": 1000,
    "min": -4000.0,
    "max": 8848.0,
    "mean": 840.5,
    "median": 500.0,
    "std_dev": 1200.3,
    "variance": 1440720.09,
    "percentiles": {
        "p5": 100.0,
        "p25": 300.0,
        "p75": 1500.0,
        "p95": 3000.0
    }
}
```

Individual functions:
- `stats.count(samples)` - number of samples
- `stats.sum(samples)` - sum of values
- `stats.mean(samples)` - arithmetic mean
- `stats.median(samples)` - median
- `stats.std_dev(samples)` - standard deviation
- `stats.variance(samples)` - variance
- `stats.percentile(samples, p)` - p-th percentile (0-100)

### 5.2 Correlation

```cdsl
let elev_samples = field.samples(geophysics.elevation)
let water_samples = field.samples(hydrology.water)
let r = stats.correlation(elev_samples, water_samples)

check r < -0.5
    : severity(error)
    : message("Water-elevation correlation: {r:.3}")
```

Pearson correlation coefficient: -1.0 to 1.0.

### 5.3 Binning and Histograms

```cdsl
let age_bins = stats.histogram(samples, [0, 50, 100, 150, 200, 300, 1e10])
```

Returns histogram bin counts. With n+1 boundaries, returns n bins.

Example with age ranges:
```cdsl
let bins = stats.histogram(samples, [
    0,     // 0-50Ma
    50,    // 50-100Ma
    100,   // 100-150Ma
    150,   // 150-200Ma
    200,   // 200-300Ma
    300,   // 300+Ma
    1e10
])

emit {
    "0-50Ma": bins[0] / samples.count(),
    "50-100Ma": bins[1] / samples.count(),
    "100-150Ma": bins[2] / samples.count(),
    // ...
}
```

### 5.4 Weighted Operations

```cdsl
let weights = field.samples(hydrology.water).map(|s| s.value)
let weighted_mean = stats.weighted_mean(samples, weights)
```

---

## 6. Output Schema

The compute block must produce a value compatible with JSON serialization.

Allowed types:
- **Scalars**: `Scalar<unit>` → number
- **Vectors**: `Vec2`, `Vec3`, `Vec4` → [x, y, z, w] array
- **Objects**: `emit { key: value, ... }` → JSON object
- **Arrays**: `[value1, value2, ...]` → JSON array
- **Primitives**: `true`, `false`, `null`, numbers, strings

Examples:

```cdsl
# Simple scalar output
: compute { 
    stats.mean(field.samples(temperature))
}

# Structured output
: compute {
    emit {
        mean: stats.mean(samples),
        min: stats.min(samples),
        max: stats.max(samples),
        samples: samples.count()
    }
}

# Nested output
: compute {
    let elev = field.samples(geophysics.elevation)
    let ocean = elev.filter(|s| s.value < 0.0)
    let land = elev.filter(|s| s.value >= 0.0)
    
    emit {
        overall: stats.compute(elev),
        ocean: emit {
            count: ocean.count(),
            mean: stats.mean(ocean),
            std_dev: stats.std_dev(ocean)
        },
        land: emit {
            count: land.count(),
            mean: stats.mean(land),
            std_dev: stats.std_dev(land)
        }
    }
}

# Array output
: compute {
    let bins = stats.histogram(samples, [0, 50, 100, 150, 200, 300, 1e10])
    [
        bins[0] / samples.count(),
        bins[1] / samples.count(),
        bins[2] / samples.count(),
        bins[3] / samples.count(),
        bins[4] / samples.count(),
        bins[5] / samples.count()
    ]
}
```

---

## 7. Example Analyzers

### 7.1 Simple Statistics

```cdsl
analyzer terra.elevation_stats {
    : doc "Basic elevation statistics"
    : requires(fields: [geophysics.elevation])
    
    : compute {
        stats.compute(field.samples(geophysics.elevation))
    }
    
    : validate {
        check mean > 0.0
            : severity(info)
            : message("Mean elevation: {mean:.0}m")
    }
}
```

### 7.2 Correlation Analysis

```cdsl
analyzer terra.water_elevation_check {
    : doc "Correlation between water presence and elevation"
    : requires(fields: [geophysics.elevation, hydrology.water])
    
    : compute {
        let elev = field.samples(geophysics.elevation)
        let water = field.samples(hydrology.water)
        
        emit {
            correlation: stats.correlation(elev, water),
            sample_count: elev.count()
        }
    }
    
    : validate {
        check correlation < -0.5
            : severity(error)
            : message("Water-elevation correlation: {correlation:.3} (expected strongly negative)")
    }
}
```

### 7.3 Categorical Analysis

```cdsl
analyzer terra.plate_age_distribution {
    : doc "Plate age distribution by age bins"
    : requires(fields: [geophysics.plate_age])
    
    : compute {
        let samples = field.samples(geophysics.plate_age)
        let age_ma = samples.map(|s| s.value / 1e6)  // Convert to Ma
        
        let bins = stats.histogram(age_ma, [
            0, 50, 100, 150, 200, 300, 1e10
        ])
        
        let total = age_ma.count()
        
        emit {
            "0-50Ma": bins[0] / total,
            "50-100Ma": bins[1] / total,
            "100-150Ma": bins[2] / total,
            "150-200Ma": bins[3] / total,
            "200-300Ma": bins[4] / total,
            ">300Ma": bins[5] / total
        }
    }
}
```

### 7.4 Latitude Band Analysis

```cdsl
analyzer terra.latitude_distribution {
    : doc "Elevation distribution by latitude bands"
    : requires(fields: [geophysics.elevation])
    
    : compute {
        let samples = field.samples(geophysics.elevation)
        
        # Divide into 10-degree latitude bands (18 total)
        let band_count = 18
        let band_size = 180.0 / band_count
        
        # Compute mean elevation per band
        let band_means = []
        for band in 0..band_count {
            let band_min = -90.0 + band * band_size
            let band_max = band_min + band_size
            
            let band_samples = samples.filter(|s|
                let lat = util.latitude(s.position)
                lat >= band_min and lat < band_max
            )
            
            let mean = if band_samples.count() > 0 {
                stats.mean(band_samples)
            } else {
                0.0
            }
            
            band_means = push(band_means, mean)
        }
        
        emit {
            band_means: band_means,
            tropical_mean: (band_means[7] + band_means[8] + band_means[9] + band_means[10]) / 4.0,
            polar_mean: (band_means[0] + band_means[1] + band_means[2] + band_means[15] + band_means[16] + band_means[17]) / 6.0
        }
    }
}
```

---

## 8. Execution Context

Analyzers execute in a special context:

- **Input**: `TickData` snapshot (fields at a specific tick)
- **Output**: JSON result + validation results
- **Access**: read-only access to field data
- **No access to**: signals, entities, members, simulation state
- **Determinism**: fully deterministic (same input → same output)

At runtime, the analyzer executor:
1. Loads the snapshot
2. Creates an AnalyzerContext
3. Executes the compute block
4. Executes validation checks
5. Formats output
6. Returns result

---

## 9. CLI Integration

Analyzers are invoked via the `continuum analyze` command:

```bash
# List all analyzers in a world
continuum analyze list examples/terra

# Run a specific analyzer
continuum analyze run terra.hypsometric_integral snapshots/terra_run_001/

# Run all validations
continuum analyze validate snapshots/terra_run_001/

# Output in different formats
continuum analyze run terra.hypsometric_integral snapshots/terra_run_001/ --output json
continuum analyze run terra.hypsometric_integral snapshots/terra_run_001/ --output table
```

Exit codes:
- 0: success, all validations passed
- 1: validation errors occurred
- 2: analyzer not found
- 3: snapshot not found

---

## 10. Edge Cases and Error Handling

### 10.1 Missing Fields

If a required field is missing from the snapshot:

```
Error: Analyzer terra.hypsometric_integral requires field geophysics.elevation, which was not captured in this snapshot

Available fields: [hydrology.water, temperature.air]
```

The analyzer fails early and clearly.

### 10.2 Empty Samples

If a field has zero samples (shouldn't happen, but):

```cdsl
let mean = if samples.count() > 0 {
    stats.mean(samples)
} else {
    0.0  // default or error handling
}
```

The `stats.compute()` function returns all zeros if samples are empty.

### 10.3 Type Mismatches

If compute block produces non-serializable value:

```
Compilation error: analyzer terra.hypsometric_integral compute block must produce JSON-serializable value, got OpType (unsupported)
```

Compilation catches type errors.

### 10.4 Template Substitution Errors

If a validation message template references undefined variable:

```cdsl
: message("Value: {undefined_var}")  # ERROR at compile time
```

Compile-time error: "undefined_var not in scope"

---

## 11. Best Practices

1. **Always document analyzers**
   ```cdsl
   : doc "Clear, specific description of what this analyzer computes"
   ```

2. **Validate field dependencies**
   ```cdsl
   : requires(fields: [field1, field2])
   ```

3. **Use compute block for all logic** (not validation)
   ```cdsl
   : compute {
       let value = complex_calculation()
       emit { result: value }
   }
   
   : validate {
       check value > threshold
           : severity(error)
   }
   ```

4. **Make validation messages informative**
   ```cdsl
   : message("Correlation {r:.3} (threshold: {threshold})")
   ```

5. **Group related analyzers by domain**
   ```
   analyzer terra.hydrology.water_balance { ... }
   analyzer terra.hydrology.runoff { ... }
   analyzer terra.hydrology.infiltration { ... }
   ```

6. **Handle edge cases gracefully**
   ```cdsl
   let mean = if samples.count() > 0 {
       stats.mean(samples)
   } else {
       0.0
   }
   ```

---

## 12. Limitations

Analyzers **cannot**:
- mutate world state
- access simulation signals
- call phase-specific operations
- depend on execution order
- use randomness
- perform I/O

These constraints ensure analyzers are pure observers.

---

## 13. Future Extensions

Potential future enhancements:
- Analyzer composition (calling analyzers from analyzers)
- Caching results across analyzer runs
- Multi-snapshot analysis (time series)
- Parallel analyzer execution
- Custom output formatters

---

## Summary

Analyzers are declarative, observable analysis queries defined in CDSL.

They:
- read field snapshots post-hoc
- produce structured JSON results
- include optional validation checks
- are discoverable and portable
- enable world authors to define domain-specific analysis

Analyzers replace hard-coded Rust analyzers with declarative, composable CDSL definitions.
