---
name: validate-terra
description: Validate terra planetary simulation against Earth-like baselines using CDSL analyzers
license: MIT
compatibility: opencode
metadata:
  domain: simulation
  world: terra
---

# Validate Terra Simulation

This skill guides validation of the terra (planetary geophysics) simulation to ensure it produces Earth-like characteristics.

## Quick Reference

### Required Checks (Must Pass)

| Check | Analyzer | Expected | Severity |
|-------|----------|----------|----------|
| Water-Elevation Correlation | `terra.water_elevation_check` | r < -0.5 | error |

### Recommended Checks (Should Pass)

| Check | Analyzer | Expected | Severity |
|-------|----------|----------|----------|
| Land Fraction | `terra.hypsometric_integral` | 0.20 - 0.40 | warning |
| Isostasy Balance | `terra.isostasy_balance` | r > 0.5 | warning |

## Step-by-Step Validation

### 1. Run the Simulation

```bash
# Minimal test run (quick)
continuum run examples/terra --steps 100 --dt 10Myr --samples 1000

# Standard validation run
continuum run examples/terra --steps 1000 --dt 1Myr --samples 10000

# Full simulation
continuum run examples/terra --steps 10000 --dt 1Myr --samples 50000
```

### 2. Capture Field Snapshots

The simulation must capture these fields for validation:
- `elevation.map` - Crustal elevation
- `thickness.map` - Crustal thickness  
- `plates.age_map` - Plate age
- `hydrology.water_presence` - Water distribution

### 3. Run Analyzers

```bash
# List all available analyzers
continuum analyze list examples/terra

# Run individual analyzers
continuum analyze run terra.hypsometric_integral <snapshot_dir>
continuum analyze run terra.water_elevation_check <snapshot_dir>
continuum analyze run terra.isostasy_balance <snapshot_dir>
continuum analyze run terra.ocean_analysis <snapshot_dir>
continuum analyze run terra.plate_age_analysis <snapshot_dir>
continuum analyze run terra.latitude_distribution <snapshot_dir>

# Run all validations at once
continuum analyze validate <snapshot_dir>
```

### 4. Interpret Results

#### Hypsometric Integral (Land/Ocean Ratio)
```json
{
  "integral": 0.29,
  "land_fraction": 0.29,
  "ocean_fraction": 0.71
}
```
- **Pass**: `land_fraction` between 0.20 and 0.40
- **Earth value**: 0.29 (29% land)
- **Failure indicates**: Unrealistic topography generation

#### Water-Elevation Correlation
```json
{
  "correlation": -0.75,
  "sample_count": 10000
}
```
- **Pass**: `correlation` < -0.5 (strongly negative)
- **Earth value**: ~-0.8
- **Failure indicates**: Water not flowing to low areas (physics bug)

#### Isostasy Balance
```json
{
  "isostasy_correlation": 0.68,
  "elevation_stats": {...},
  "thickness_stats": {...}
}
```
- **Pass**: `isostasy_correlation` > 0.5 (positive)
- **Earth value**: ~0.7
- **Failure indicates**: Crust/elevation decoupled (isostasy not working)

## Earth-like Baseline Values

### Elevation Statistics
| Metric | Expected Range | Earth Value |
|--------|---------------|-------------|
| Min elevation | -12000 to -8000m | -11034m (Mariana) |
| Max elevation | 6000 to 10000m | 8849m (Everest) |
| Mean elevation | -2500 to 500m | -2500m |
| Std deviation | 1000 to 4000m | ~2500m |

### Crustal Thickness
| Type | Expected | Earth Value |
|------|----------|-------------|
| Oceanic crust | 5-12 km | 7 km |
| Continental crust | 30-45 km | 35 km |

### Ocean Depth Zones
| Zone | Expected Fraction | Description |
|------|------------------|-------------|
| Continental shelf | 5-12% | 0 to -200m |
| Slope | 5-15% | -200 to -2000m |
| Abyssal plain | 50-75% | -2000 to -6000m |
| Trenches | 1-5% | < -6000m |

### Plate Age Distribution
| Age Bin | Expected Fraction |
|---------|------------------|
| 0-50 Ma | 30-50% |
| 50-100 Ma | 20-40% |
| 100-150 Ma | 15-25% |
| 150-200 Ma | 5-15% |
| > 200 Ma | < 10% |

## CI Integration

Exit codes for CI pipelines:
- `0`: All validations passed
- `1`: Error-level validations failed (e.g., water_elevation_check)
- `2`: Required field not captured in snapshot

```bash
# CI validation command
continuum analyze validate <snapshot_dir> --severity error
if [ $? -ne 0 ]; then
  echo "Validation failed!"
  exit 1
fi
```

## Troubleshooting

### "Field not found" Error
- Ensure simulation captured the required field
- Check field name mapping (old names vs new CDSL names)
- Verify `--fields` argument includes required fields

### Correlation Values Near Zero
- Check sample count (need sufficient data points)
- Verify fields are at same tick
- Ensure field values have variance (not all zeros)

### Land Fraction Outside Range
- Check elevation generation algorithm
- Verify sea level is at 0m
- Check initial conditions in scenario

## Field Name Mapping

Old Rust names → New CDSL names:
| Old Name | New Name |
|----------|----------|
| `CrustElevationM` | `elevation.map` |
| `CrustThicknessM` | `thickness.map` |
| `PlateAge` | `plates.age_map` |
| `WaterPresence` | `hydrology.water_presence` |

## Reference Files

- Analyzer definitions: `examples/terra/analyzers.cdsl`
- Baseline expectations: `examples/terra/tests/terra_baselines.yaml`
- World guide: `examples/terra/AGENTS.md`
