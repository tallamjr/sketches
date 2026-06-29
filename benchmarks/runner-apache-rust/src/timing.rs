//! Shared throughput protocol: R independent rounds, each one untimed warmup
//! pass then REPS_PER_ROUND timed reps (the round-sample is the median of those
//! reps). Report the median over round-samples, their population stddev, and a
//! deterministic nonparametric 95% bootstrap CI (ops/s).

use std::time::Instant;

pub fn median(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        (xs[n / 2 - 1] + xs[n / 2]) / 2.0
    }
}

pub fn stddev(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    let mean = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    var.sqrt()
}

/// Number of bootstrap resamples for the throughput confidence interval.
const BOOTSTRAP_RESAMPLES: usize = 2000;
/// Fixed SplitMix64 seed so the bootstrap CI is reproducible run to run.
const BOOTSTRAP_SEED: u64 = 0x9E3779B97F4A7C15;
/// Timed reps inside each independent round; the round-sample is their median.
pub const REPS_PER_ROUND: usize = 5;

/// A stabilised throughput measurement: the median over independent rounds, the
/// population stddev of the round-samples, and a nonparametric 95% bootstrap CI.
#[derive(Clone, Copy, Debug)]
pub struct Throughput {
    pub median: f64,
    pub stddev: f64,
    pub ci_low: f64,
    pub ci_high: f64,
}

/// SplitMix64 step. Deterministic, identical across the C++ and Python runners.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Nonparametric 95% bootstrap CI of the median over `samples`. Deterministic:
/// fixed-seed SplitMix64, B=2000 resamples, 2.5/97.5 nearest-rank percentiles.
/// Returns (median, median) when fewer than two samples are given.
pub fn bootstrap_ci(samples: &[f64]) -> (f64, f64) {
    let r = samples.len();
    if r <= 1 {
        let m = if r == 1 { samples[0] } else { 0.0 };
        return (m, m);
    }
    let mut state = BOOTSTRAP_SEED;
    let mut resample_medians = Vec::with_capacity(BOOTSTRAP_RESAMPLES);
    let mut draw = vec![0.0_f64; r];
    for _ in 0..BOOTSTRAP_RESAMPLES {
        for slot in draw.iter_mut() {
            let idx = (splitmix64(&mut state) % r as u64) as usize;
            *slot = samples[idx];
        }
        resample_medians.push(median(&mut draw));
    }
    resample_medians.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let b = BOOTSTRAP_RESAMPLES;
    let idx = |p: f64| -> usize {
        let i = (p * b as f64).floor() as isize;
        i.clamp(0, b as isize - 1) as usize
    };
    (resample_medians[idx(0.025)], resample_medians[idx(0.975)])
}

/// Run `rounds` independent rounds. Each round does one untimed warmup `body()`
/// then `reps_per_round` timed `body()` calls; the round-sample is the median of
/// its per-rep rates (ops/s). Reports the median over round-samples, their
/// population stddev, and the bootstrap CI over them.
pub fn timed_throughput_rounds<F: FnMut()>(
    rounds: usize,
    reps_per_round: usize,
    ops_per_rep: u64,
    mut body: F,
) -> Throughput {
    let mut round_samples = Vec::with_capacity(rounds);
    for _ in 0..rounds {
        body(); // untimed warmup
        let mut rates = Vec::with_capacity(reps_per_round);
        for _ in 0..reps_per_round {
            let start = Instant::now();
            body();
            let secs = start.elapsed().as_secs_f64();
            rates.push(if secs > 0.0 {
                ops_per_rep as f64 / secs
            } else {
                0.0
            });
        }
        round_samples.push(median(&mut rates));
    }
    let stddev = stddev(&round_samples);
    let (ci_low, ci_high) = bootstrap_ci(&round_samples);
    let median = median(&mut round_samples);
    Throughput {
        median,
        stddev,
        ci_low,
        ci_high,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn median_odd() {
        let mut v = vec![3.0, 1.0, 2.0];
        assert_eq!(median(&mut v), 2.0);
    }
    #[test]
    fn median_even() {
        let mut v = vec![4.0, 1.0, 2.0, 3.0];
        assert_eq!(median(&mut v), 2.5);
    }
    #[test]
    fn stddev_known() {
        let v = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        // population stddev = 2.0
        assert!((stddev(&v) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn bootstrap_ci_on_pinned_vector_is_deterministic() {
        let samples = [
            100.0, 110.0, 90.0, 105.0, 95.0, 120.0, 80.0, 115.0, 85.0, 100.0,
        ];
        let (lo1, hi1) = bootstrap_ci(&samples);
        let (lo2, hi2) = bootstrap_ci(&samples);
        // Deterministic across calls (fixed-seed PRNG).
        assert_eq!(lo1, lo2);
        assert_eq!(hi1, hi2);
        // Interval is ordered and brackets the sample median (100.0).
        assert!(lo1 <= hi1);
        assert!(lo1 <= 100.0 && 100.0 <= hi1);
        // Interval lies within the data range.
        assert!(lo1 >= 80.0 && hi1 <= 120.0);
        // PRINT the pinned values so they can be copied into the C++/Python parity
        // tests. Run with `-- --nocapture`.
        println!("PINNED_CI_LOW={lo1} PINNED_CI_HIGH={hi1}");
    }

    #[test]
    fn bootstrap_ci_single_sample_is_degenerate() {
        let (lo, hi) = bootstrap_ci(&[42.0]);
        assert_eq!(lo, 42.0);
        assert_eq!(hi, 42.0);
    }

    #[test]
    fn timed_throughput_rounds_reports_ordered_interval() {
        // A body doing a fixed amount of trivial work; we only assert structure,
        // not absolute rate. 3 rounds x 4 reps.
        let t = timed_throughput_rounds(3, 4, 1000, || {
            core::hint::black_box((0u64..1000).sum::<u64>());
        });
        assert!(t.median > 0.0);
        assert!(t.ci_low <= t.median + f64::EPSILON);
        assert!(t.median <= t.ci_high + f64::EPSILON);
        assert!(t.stddev >= 0.0);
    }
}
