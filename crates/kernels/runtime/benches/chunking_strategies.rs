//!
//! Tests different chunk sizes and population sizes to measure:
//! - Optimal chunk size for different population scales
//! - Parallel speedup over serial execution
//! - Auto chunk selection effectiveness

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};

use continuum_runtime::executor::member_executor::{
    ChunkConfig, MemberResolveContext, resolve_scalar_l1,
};
use continuum_runtime::soa_storage::MemberSignalBuffer;
use continuum_runtime::storage::{EntityStorage, SignalStorage};
use continuum_runtime::types::{Dt, Value};

fn create_test_signals() -> SignalStorage {
    let mut storage = SignalStorage::default();
    storage.init("test.signal".into(), Value::Scalar(1.0));
    storage
}

fn create_test_members(count: usize) -> MemberSignalBuffer {
    let mut buffer = MemberSignalBuffer::new();
    buffer.register_signal(
        "value".to_string(),
        continuum_runtime::soa_storage::ValueType::scalar(),
    );
    buffer.init_instances(count);
    buffer
}

/// Serial baseline for comparison
fn resolve_serial(prev_values: &[f64], dt: Dt) -> Vec<f64> {
    prev_values
        .iter()
        .enumerate()
        .map(|(idx, &prev)| {
            // Simple resolver: integrate with rate 1.0
            prev + dt.seconds() + (idx as f64 * 0.001)
        })
        .collect()
}

/// Benchmark different chunk sizes for a fixed population
fn bench_chunk_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_sizes");

    let population = 10_000;
    let prev_values: Vec<f64> = (0..population).map(|i| i as f64 * 0.1).collect();
    let signals = create_test_signals();
    let members = create_test_members(population);
    let entities = EntityStorage::default();
    let dt = Dt(0.016); // ~60fps

    group.throughput(Throughput::Elements(population as u64));

    // Test various fixed chunk sizes
    for chunk_size in [64, 128, 256, 512, 1024, 2048, 4096] {
        let config = ChunkConfig {
            chunk_size,
            min_chunk: 64,
            max_chunk: 4096,
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &chunk_size,
            |b, _| {
                b.iter(|| {
                    resolve_scalar_l1(
                        black_box(&prev_values),
                        |ctx: &MemberResolveContext<'_, f64>| {
                            ctx.prev + ctx.dt.seconds() + (ctx.index.0 as f64 * 0.001)
                        },
                        black_box(&signals),
                        black_box(&entities),
                        black_box(&members),
                        dt,
                        0.0, // sim_time
                        config,
                    )
                })
            },
        );
    }

    // Test auto chunk size
    let auto_config = ChunkConfig::auto(population);
    group.bench_function("auto", |b| {
        b.iter(|| {
            resolve_scalar_l1(
                black_box(&prev_values),
                |ctx: &MemberResolveContext<'_, f64>| {
                    ctx.prev + ctx.dt.seconds() + (ctx.index.0 as f64 * 0.001)
                },
                black_box(&signals),
                black_box(&entities),
                black_box(&members),
                dt,
                0.0, // sim_time
                auto_config,
            )
        })
    });

    group.finish();
}

/// Benchmark parallel vs serial execution across population sizes
fn bench_parallel_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("parallel_speedup");

    let dt = Dt(0.016);

    for population in [1_000, 5_000, 10_000, 50_000, 100_000] {
        let prev_values: Vec<f64> = (0..population).map(|i| i as f64 * 0.1).collect();
        let signals = create_test_signals();
        let members = create_test_members(population);
        let entities = EntityStorage::default();

        group.throughput(Throughput::Elements(population as u64));

        // Serial baseline
        group.bench_with_input(
            BenchmarkId::new("serial", population),
            &population,
            |b, _| b.iter(|| resolve_serial(black_box(&prev_values), dt)),
        );

        // Parallel L1 with auto chunk
        let config = ChunkConfig::auto(population);
        group.bench_with_input(
            BenchmarkId::new("l1_parallel", population),
            &population,
            |b, _| {
                b.iter(|| {
                    resolve_scalar_l1(
                        black_box(&prev_values),
                        |ctx: &MemberResolveContext<'_, f64>| {
                            ctx.prev + ctx.dt.seconds() + (ctx.index.0 as f64 * 0.001)
                        },
                        black_box(&signals),
                        black_box(&entities),
                        black_box(&members),
                        dt,
                        0.0, // sim_time
                        config,
                    )
                })
            },
        );
    }

    group.finish();
}

/// Benchmark complex resolver expression (more compute per instance)
fn bench_compute_heavy_resolver(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_heavy");

    let population = 10_000;
    let prev_values: Vec<f64> = (0..population).map(|i| i as f64 * 0.1).collect();
    let signals = create_test_signals();
    let members = create_test_members(population);
    let entities = EntityStorage::default();
    let dt = Dt(0.016);

    group.throughput(Throughput::Elements(population as u64));

    // Parallel L1 with auto chunk
    let config = ChunkConfig::auto(population);
    group.bench_function("heavy_l1", |b| {
        b.iter(|| {
            resolve_scalar_l1(
                black_box(&prev_values),
                |ctx: &MemberResolveContext<'_, f64>| {
                    // More compute: several additions and multiplications
                    ctx.prev * 0.5
                        + ctx.dt.seconds() * 2.0
                        + (ctx.index.0 as f64 * 0.001).sin().abs()
                },
                black_box(&signals),
                black_box(&entities),
                black_box(&members),
                dt,
                0.0, // sim_time
                config,
            )
        })
    });

    group.finish();
}

/// Benchmark optimal_chunk_size function itself
fn bench_chunk_size_heuristics(c: &mut Criterion) {
    let mut group = c.benchmark_group("chunk_heuristics");

    for population in [100, 1_000, 10_000, 100_000, 1_000_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(population),
            &population,
            |b, &pop| b.iter(|| ChunkConfig::auto(black_box(pop))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_chunk_sizes,
    bench_parallel_speedup,
    bench_compute_heavy_resolver,
    bench_chunk_size_heuristics,
);
criterion_main!(benches);
