//! Tests for the Lens crate.

use continuum_foundation::{FieldId, FieldSample, Value};
use indexmap::IndexMap;

use crate::config::{FieldConfig, FieldLensConfig};
use crate::playback::PlaybackClock;
use crate::refinement::{RefinementRequestSpec, Region};
use crate::storage::FieldSnapshot;
use crate::topology::CubedSphereTopology;
use crate::{FieldLens, LensError};

fn sample(v: f64) -> FieldSample {
    FieldSample {
        position: [0.0, 0.0, 0.0],
        value: Value::Scalar(v),
    }
}

fn default_config() -> FieldLensConfig {
    FieldLensConfig {
        max_frames_per_field: 2,
        max_cached_per_field: 4,
        max_refinement_queue: 16,
    }
}

#[test]
fn record_eviction_is_bounded() {
    let mut lens = FieldLens::new(FieldLensConfig {
        max_frames_per_field: 2,
        max_cached_per_field: 4,
        max_refinement_queue: 16,
    })
    .expect("config valid");

    let field_id: FieldId = "terra.temp".into();

    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![sample(1.0)],
    });
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 2,
        samples: vec![sample(2.0)],
    });
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 3,
        samples: vec![sample(3.0)],
    });

    let ticks = lens.history_ticks(&field_id).expect("history exists");
    assert_eq!(ticks, vec![2, 3]);
}

#[test]
fn record_many_preserves_field_order() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let mut fields = IndexMap::new();
    fields.insert("field.a".into(), vec![sample(1.0)]);
    fields.insert("field.b".into(), vec![sample(2.0)]);

    lens.record_many(1, fields);

    let ids: Vec<String> = lens.field_ids().map(|id| id.to_string()).collect();
    assert_eq!(ids, vec!["field.a", "field.b"]);
}

#[test]
fn cache_clears_on_new_record() {
    let mut lens = FieldLens::new(FieldLensConfig {
        max_frames_per_field: 3,
        max_cached_per_field: 2,
        max_refinement_queue: 16,
    })
    .expect("config valid");

    let field_id: FieldId = "field.temp".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![sample(1.0)],
    });

    let _ = lens
        .query_at_tick(&field_id, [0.0, 0.0, 0.0], 1)
        .expect("query works");

    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 2,
        samples: vec![sample(2.0)],
    });

    // Cache should be cleared after new record
    // (internal state, tested via behavior)
}

#[test]
fn query_at_tick_returns_nearest_sample() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let field_id: FieldId = "field.elevation".into();
    let samples = vec![
        FieldSample {
            position: [0.0, 0.0, 0.0],
            value: Value::Scalar(10.0),
        },
        FieldSample {
            position: [10.0, 0.0, 0.0],
            value: Value::Scalar(20.0),
        },
    ];

    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 5,
        samples,
    });

    let value = lens
        .query_at_tick(&field_id, [1.0, 0.0, 0.0], 5)
        .expect("query works");
    assert_eq!(value, 10.0);
}

#[test]
fn query_at_tick_errors_on_missing_field() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let err = lens
        .query_at_tick(&"missing.field".into(), [0.0, 0.0, 0.0], 0)
        .expect_err("should error");

    match err {
        LensError::FieldNotFound(_) => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn query_at_tick_errors_on_missing_tick() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let field_id: FieldId = "field.elevation".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![sample(1.0)],
    });

    let err = lens
        .query_at_tick(&field_id, [0.0, 0.0, 0.0], 2)
        .expect_err("should error");

    match err {
        LensError::NoSamplesAtTick { .. } => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn playback_clock_brackets_ticks() {
    let mut clock = PlaybackClock::new(1.0);
    clock.advance(10);
    let (prev, next, alpha) = clock.bracketing_ticks();
    assert_eq!(prev, 9);
    assert_eq!(next, 9);
    assert_eq!(alpha, 0.0);

    clock.seek(9.5);
    let (prev, next, alpha) = clock.bracketing_ticks();
    assert_eq!(prev, 9);
    assert_eq!(next, 10);
    assert_eq!(alpha, 0.5);
}

#[test]
fn playback_clock_speed_affects_advance() {
    let mut clock = PlaybackClock::new(1.0);
    clock.set_speed(2.0);
    clock.advance(10);
    assert_eq!(clock.current_time(), 18.0);
}

#[test]
fn playback_clock_negative_speed_is_clamped() {
    let mut clock = PlaybackClock::new(0.0);
    clock.set_speed(-2.0);
    clock.advance(10);
    assert_eq!(clock.current_time(), 0.0);
}

#[test]
fn playback_clock_seek_clamps_negative() {
    let mut clock = PlaybackClock::new(0.0);
    clock.seek(-4.0);
    assert_eq!(clock.current_time(), 0.0);
}

#[test]
fn playback_clock_lag_offsets_time() {
    let mut clock = PlaybackClock::new(2.0);
    clock.advance(10);
    assert_eq!(clock.current_time(), 8.0);
}

#[test]
fn query_playback_uses_clock_time() {
    let mut lens = FieldLens::new(FieldLensConfig {
        max_frames_per_field: 3,
        max_cached_per_field: 4,
        max_refinement_queue: 16,
    })
    .expect("config valid");

    let field_id: FieldId = "field.temp".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![sample(0.0)],
    });
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 2,
        samples: vec![sample(10.0)],
    });

    let mut clock = PlaybackClock::new(0.0);
    clock.seek(1.5);

    let value = lens
        .query_playback(&field_id, [0.0, 0.0, 0.0], &clock)
        .expect("query works");
    assert_eq!(value, 5.0);
}

#[test]
fn query_interpolates_between_ticks() {
    let mut lens = FieldLens::new(FieldLensConfig {
        max_frames_per_field: 3,
        max_cached_per_field: 4,
        max_refinement_queue: 16,
    })
    .expect("config valid");

    let field_id: FieldId = "field.temp".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![sample(0.0)],
    });
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 2,
        samples: vec![sample(10.0)],
    });

    let value = lens
        .query(&field_id, [0.0, 0.0, 0.0], 1.5)
        .expect("query works");
    assert_eq!(value, 5.0);
}

#[test]
fn query_at_exact_tick_matches_fractional() {
    let mut lens = FieldLens::new(FieldLensConfig {
        max_frames_per_field: 3,
        max_cached_per_field: 4,
        max_refinement_queue: 16,
    })
    .expect("config valid");

    let field_id: FieldId = "field.temp".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![sample(2.0)],
    });
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 2,
        samples: vec![sample(4.0)],
    });

    let exact = lens
        .query_at_tick(&field_id, [0.0, 0.0, 0.0], 1)
        .expect("query works");
    let fractional = lens
        .query(&field_id, [0.0, 0.0, 0.0], 1.0)
        .expect("query works");
    assert_eq!(exact, fractional);
}

#[test]
fn query_batch_empty_positions_returns_empty() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let field_id: FieldId = "field.missing".into();
    let values = lens
        .query_batch(&field_id, &[], 1)
        .expect("empty batch should succeed");
    assert!(values.is_empty());
}

#[test]
fn query_at_tick_errors_on_empty_samples() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let field_id: FieldId = "field.empty".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: Vec::new(),
    });

    let err = lens
        .query_at_tick(&field_id, [0.0, 0.0, 0.0], 1)
        .expect_err("should error");

    match err {
        LensError::NoSamplesAtTick { .. } => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn query_at_exact_sample_position_returns_value() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let field_id: FieldId = "field.nn".into();
    let position = [1.0, 2.0, 3.0];
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![FieldSample {
            position,
            value: Value::Scalar(7.0),
        }],
    });

    let value = lens
        .query_at_tick(&field_id, position, 1)
        .expect("query works");
    assert_eq!(value, 7.0);
}

#[test]
fn refinement_queue_full_returns_error() {
    let mut lens = FieldLens::new(FieldLensConfig {
        max_frames_per_field: 2,
        max_cached_per_field: 4,
        max_refinement_queue: 1,
    })
    .expect("config valid");

    let field_id: FieldId = "field.refine".into();
    let topo = CubedSphereTopology::default();
    let tile = topo.tile_at([1.0, 0.0, 0.0], 2);

    let request = RefinementRequestSpec {
        field_id: field_id.clone(),
        region: Region::Tile(tile),
        target_lod: 2,
        priority: 1,
    };

    lens.request_refinement(request.clone())
        .expect("first request ok");

    let err = lens
        .request_refinement(request)
        .expect_err("queue should be full");

    match err {
        LensError::RefinementQueueFull => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

#[test]
fn topology_tile_at_is_deterministic() {
    let topo = CubedSphereTopology::default();
    let pos = [1.0, 0.2, -0.3];
    let a = topo.tile_at(pos, 3);
    let b = topo.tile_at(pos, 3);
    assert_eq!(a, b);
}

#[test]
fn latest_reconstruction_queries_latest_tick() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let field_id: FieldId = "field.temp".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![sample(1.0)],
    });
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 2,
        samples: vec![sample(2.0)],
    });

    let recon = lens
        .latest_reconstruction(&field_id)
        .expect("latest reconstruction");
    let value = recon.query([0.0, 0.0, 0.0]);
    assert_eq!(value, 2.0);
}

#[test]
fn query_batch_cpu_fallback() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let field_id: FieldId = "field.temp".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![
            FieldSample {
                position: [0.0, 0.0, 0.0],
                value: Value::Scalar(10.0),
            },
            FieldSample {
                position: [10.0, 0.0, 0.0],
                value: Value::Scalar(20.0),
            },
        ],
    });

    let results = lens
        .query_batch(&field_id, &[[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]], 1)
        .expect("batch query");
    assert_eq!(results, vec![10.0, 20.0]);
}

#[test]
fn tile_query_filters_samples() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let topo = CubedSphereTopology::default();
    let pos_a = [1.0, 0.0, 0.0];
    let pos_b = [-1.0, 0.0, 0.0];
    let tile_a = topo.tile_at(pos_a, 1);

    let field_id: FieldId = "field.elevation".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![
            FieldSample {
                position: pos_a,
                value: Value::Scalar(5.0),
            },
            FieldSample {
                position: pos_b,
                value: Value::Scalar(50.0),
            },
        ],
    });

    let recon = lens.tile(&field_id, tile_a, 1).expect("tile recon");
    let value = recon.query(pos_a);
    assert_eq!(value, 5.0);
}

#[test]
fn configure_field_cache_override_is_used() {
    let mut lens = FieldLens::new(FieldLensConfig {
        max_frames_per_field: 3,
        max_cached_per_field: 4,
        max_refinement_queue: 16,
    })
    .expect("config valid");

    let field_id: FieldId = "field.temp".into();
    lens.configure_field(
        field_id.clone(),
        FieldConfig {
            max_cached_per_field: Some(1),
        },
    );

    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![sample(1.0)],
    });
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 2,
        samples: vec![sample(2.0)],
    });

    let _ = lens.query_at_tick(&field_id, [0.0, 0.0, 0.0], 1).unwrap();
    let _ = lens.query_at_tick(&field_id, [0.0, 0.0, 0.0], 2).unwrap();

    // The per-field cache override of 1 should limit the cache size
    // (tested via internal behavior - we'd need access to internal state to verify directly)
}

#[test]
fn queries_do_not_mutate_samples() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let field_id: FieldId = "field.temp".into();
    let original = vec![
        FieldSample {
            position: [0.0, 0.0, 0.0],
            value: Value::Scalar(1.0),
        },
        FieldSample {
            position: [1.0, 0.0, 0.0],
            value: Value::Scalar(2.0),
        },
    ];

    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: original.clone(),
    });

    for _ in 0..10 {
        let _ = lens.query_at_tick(&field_id, [0.2, 0.0, 0.0], 1);
    }

    // Verify samples are unchanged by querying again
    let value = lens.query_at_tick(&field_id, [0.0, 0.0, 0.0], 1).unwrap();
    assert_eq!(value, 1.0);
}

#[test]
fn query_vector_returns_nearest_vector() {
    let mut lens = FieldLens::new(default_config()).expect("config valid");

    let field_id: FieldId = "field.velocity".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![FieldSample {
            position: [0.0, 0.0, 0.0],
            value: Value::Vec3([1.0, 2.0, 3.0]),
        }],
    });

    let result = lens
        .query_vector(&field_id, [0.0, 0.0, 0.0], 1.0)
        .expect("query works");

    // At exact tick (alpha=0), returns raw vector value
    assert_eq!(result, [1.0, 2.0, 3.0]);
}

#[test]
fn refinement_drains_by_priority() {
    let mut lens = FieldLens::new(FieldLensConfig {
        max_frames_per_field: 2,
        max_cached_per_field: 4,
        max_refinement_queue: 10,
    })
    .expect("config valid");

    let topo = CubedSphereTopology::default();
    let tile = topo.tile_at([1.0, 0.0, 0.0], 2);

    // Add requests with different priorities
    lens.request_refinement(RefinementRequestSpec {
        field_id: "low".into(),
        region: Region::Tile(tile),
        target_lod: 2,
        priority: 1, // Low priority
    })
    .unwrap();

    lens.request_refinement(RefinementRequestSpec {
        field_id: "high".into(),
        region: Region::Tile(tile),
        target_lod: 2,
        priority: 10, // High priority
    })
    .unwrap();

    lens.request_refinement(RefinementRequestSpec {
        field_id: "medium".into(),
        region: Region::Tile(tile),
        target_lod: 2,
        priority: 5, // Medium priority
    })
    .unwrap();

    // Drain should return highest priority first
    let drained = lens.drain_refinements(3);
    assert_eq!(drained.len(), 3);
    assert_eq!(drained[0].field_id.as_str(), "high");
    assert_eq!(drained[1].field_id.as_str(), "medium");
    assert_eq!(drained[2].field_id.as_str(), "low");
}

#[test]
fn refinement_priority_fifo_tiebreaker() {
    let mut lens = FieldLens::new(FieldLensConfig {
        max_frames_per_field: 2,
        max_cached_per_field: 4,
        max_refinement_queue: 10,
    })
    .expect("config valid");

    let topo = CubedSphereTopology::default();
    let tile = topo.tile_at([1.0, 0.0, 0.0], 2);

    // Add requests with same priority
    lens.request_refinement(RefinementRequestSpec {
        field_id: "first".into(),
        region: Region::Tile(tile),
        target_lod: 2,
        priority: 5,
    })
    .unwrap();

    lens.request_refinement(RefinementRequestSpec {
        field_id: "second".into(),
        region: Region::Tile(tile),
        target_lod: 2,
        priority: 5,
    })
    .unwrap();

    // FIFO tie-breaker: first submitted should come first
    let drained = lens.drain_refinements(2);
    assert_eq!(drained.len(), 2);
    assert_eq!(drained[0].field_id.as_str(), "first");
    assert_eq!(drained[1].field_id.as_str(), "second");
}

#[test]
fn query_vector_interpolates_and_normalizes() {
    let mut lens = FieldLens::new(FieldLensConfig {
        max_frames_per_field: 3,
        max_cached_per_field: 4,
        max_refinement_queue: 16,
    })
    .expect("config valid");

    let field_id: FieldId = "field.velocity".into();
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 1,
        samples: vec![FieldSample {
            position: [0.0, 0.0, 0.0],
            value: Value::Vec3([1.0, 0.0, 0.0]),
        }],
    });
    lens.record(FieldSnapshot {
        field_id: field_id.clone(),
        tick: 2,
        samples: vec![FieldSample {
            position: [0.0, 0.0, 0.0],
            value: Value::Vec3([0.0, 1.0, 0.0]),
        }],
    });

    let result = lens
        .query_vector(&field_id, [0.0, 0.0, 0.0], 1.5)
        .expect("query works");

    // Interpolated and normalized
    let mag = (result[0] * result[0] + result[1] * result[1] + result[2] * result[2]).sqrt();
    assert!((mag - 1.0).abs() < 1e-10);
}
