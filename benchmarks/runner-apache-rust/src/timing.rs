//! Shared throughput protocol: one untimed warmup pass, then R timed reps;
//! report median and population stddev of per-rep throughput (ops/s).

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

/// Run `body` once untimed (if `warmup`), then `reps` timed reps. Each rep is
/// assumed to perform `ops_per_rep` operations. Returns (median, stddev) ops/s.
pub fn timed_throughput<F: FnMut()>(
    reps: usize,
    warmup: bool,
    ops_per_rep: u64,
    mut body: F,
) -> (f64, f64) {
    if warmup {
        body();
    }
    let mut rates = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        body();
        let secs = start.elapsed().as_secs_f64();
        rates.push(if secs > 0.0 {
            ops_per_rep as f64 / secs
        } else {
            0.0
        });
    }
    let s = stddev(&rates);
    (median(&mut rates), s)
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
}
