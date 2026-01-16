# TERRA - Planetary Geophysics Simulation

This simulation models Earth-like planetary evolution including tectonics, crustal dynamics, hydrology, and related geophysical processes.

## Reference Implementation

The old Rust-based simulation is located at:
`/home/ztripez/Documents/code/sides/continuum-alpha/continuum-prime/`

Key reference files:
- Domain analyzers: `crates/tools/src/analyze/domain/terra.rs`
- Analysis helpers: `crates/tools/src/analyze/helpers.rs`
- Types: `crates/tools/src/analyze/types.rs`

## Stability Testing

The simulation is validated against Earth-like baselines defined in:
`tests/terra_baselines.yaml`

### Validation Checks (must pass for stable simulation)

1. **Land Fraction** (warning): 20-40% land coverage
   - Earth-like value: 29%
   - Check: `terra.hypsometric_integral`

2. **Water-Elevation Correlation** (error): r < -0.5
   - Water must accumulate in low-lying areas
   - Check: `terra.water_elevation_check`

3. **Isostasy Correlation** (warning): r > 0.5
   - Crust thickness should correlate with elevation
   - Check: `terra.isostasy_balance`

### Running Validation

```bash
# Run all terra analyzers
continuum analyze list examples/terra
continuum analyze run terra.hypsometric_integral <snapshot_dir>
continuum analyze run terra.water_elevation_check <snapshot_dir>
continuum analyze run terra.isostasy_balance <snapshot_dir>

# Run all validations
continuum analyze validate <snapshot_dir>
```

### Analyzer Definitions

See `analyzers.cdsl` for all 6 domain analyzers:
1. `terra.hypsometric_integral` - Land/ocean ratio
2. `terra.ocean_analysis` - Ocean depth zones
3. `terra.isostasy_balance` - Isostatic equilibrium
4. `terra.plate_age_analysis` - Plate age distribution
5. `terra.water_elevation_check` - Water-elevation correlation
6. `terra.latitude_distribution` - Elevation by latitude

## Field Mapping

Old Rust field names → New CDSL field names:
- `CrustElevationM` → `elevation.map`
- `CrustThicknessM` → `thickness.map`
- `PlateAge` → `plates.age_map`
- `WaterPresence` → `hydrology.water_presence`

## Expected Earth-like Values

| Metric | Expected Range | Earth Value |
|--------|---------------|-------------|
| Land fraction | 0.20 - 0.40 | 0.29 |
| Mean ocean depth | 3000 - 4500m | 3688m |
| Water-elev correlation | < -0.5 | ~-0.8 |
| Isostasy correlation | > 0.5 | ~0.7 |
| Oceanic crust thickness | 5-12 km | 7 km |
| Continental crust thickness | 30-45 km | 35 km |