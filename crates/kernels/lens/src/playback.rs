//! Playback clock for observer queries.

/// Playback clock for observer queries (fractional tick time).
///
/// Supports lagged playback for interpolation between ticks.
#[derive(Debug, Clone)]
pub struct PlaybackClock {
    current_time: f64,
    lag_ticks: f64,
    speed: f64,
}

impl PlaybackClock {
    /// Create a playback clock with a fixed lag (in ticks).
    pub fn new(lag_ticks: f64) -> Self {
        Self {
            current_time: 0.0,
            lag_ticks,
            speed: 1.0,
        }
    }

    /// Current playback time (fractional tick).
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Set playback speed multiplier (>= 0).
    pub fn set_speed(&mut self, speed: f64) {
        self.speed = speed.max(0.0);
    }

    /// Seek to a specific playback time (clamped to >= 0).
    pub fn seek(&mut self, time: f64) {
        self.current_time = time.max(0.0);
    }

    /// Advance playback time based on simulation tick and lag.
    pub fn advance(&mut self, sim_tick: u64) {
        let target_time = sim_tick as f64 - self.lag_ticks;
        self.current_time = target_time.max(0.0) * self.speed;
    }

    /// Get bracketing ticks and interpolation alpha.
    pub fn bracketing_ticks(&self) -> (u64, u64, f64) {
        let tick_prev = self.current_time.floor() as u64;
        let tick_next = self.current_time.ceil() as u64;
        let alpha = self.current_time.fract();
        (tick_prev, tick_next, alpha)
    }
}
