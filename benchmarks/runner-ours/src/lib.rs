//! Row-producing logic for the `ours` benchmark runner.
//!
//! Each sketch in this crate is run over the shared datasets and emitted as a
//! CSV row in the schema defined in `benchmarks/results/schema.md`:
//!
//! ```text
//! implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,throughput_stddev,bytes,live_bytes,estimate,exact,rel_error
//! ```
//!
//! This runner emits two `implementation` labels: `ours` for the crate default
//! (xxh3-backed) sketches and `ours-murmur3` for the MurmurHash3-backed variants
//! of the hash-based sketches (which isolate the hash effect). `run` produces the
//! synthetic-dataset rows; `run_tpch` produces the rows for a TPC-H column.

use std::path::Path;

pub mod counting_alloc;
pub mod timing;

use datasets::{exact_distinct, synthetic_distinct, tpch_column};
use sketches::bloom::BloomFilterGeneric;
use sketches::countmin::CountMinSketchGeneric;
use sketches::cpc::{CpcSketch, CpcSketchGeneric};
use sketches::hash::murmur3::Murmur3Hasher;
use sketches::hash::xxh3::Xxh3Hasher;
use sketches::hash::{Hashable, SketchHasher};
use sketches::hll::{HllSketch, HllSketchGeneric};
use sketches::quantiles::KllSketch;
use sketches::serialization::Serializable;
use sketches::theta::{ThetaSketch, ThetaSketchGeneric};

/// The exact CSV header line shared by all runners.
pub const HEADER: &str = "implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,throughput_stddev,bytes,live_bytes,estimate,exact,rel_error";

/// Implementation label for the crate default (xxh3-backed) sketches.
const IMPL_OURS: &str = "ours";
/// Implementation label for the murmur3-backed variants of the hash-based
/// sketches, emitted to isolate the hash effect from the impl effect.
const IMPL_MURMUR3: &str = "ours-murmur3";
const HOT_KEY: &str = "__hot__";
const HOT_KEY_COUNT: u64 = 1000;

/// Format a single CSV row. Optional fields are written as empty strings.
///
/// One parameter per schema column is the clearest mapping to the fixed CSV
/// header, so the argument count deliberately mirrors the schema.
#[allow(clippy::too_many_arguments)]
fn row(
    implementation: &str,
    sketch: &str,
    dataset: &str,
    op: &str,
    n: u64,
    reps: u64,
    throughput_median: f64,
    throughput_stddev: f64,
    bytes: Option<usize>,
    live_bytes: Option<usize>,
    estimate: Option<f64>,
    exact: Option<f64>,
    rel_error: Option<f64>,
) -> String {
    let bytes = bytes.map(|b| b.to_string()).unwrap_or_default();
    let live_bytes = live_bytes.map(|b| b.to_string()).unwrap_or_default();
    let estimate = estimate.map(|v| format!("{v:.6}")).unwrap_or_default();
    let exact = exact.map(|v| format!("{v:.6}")).unwrap_or_default();
    let rel_error = rel_error.map(|v| format!("{v:.6}")).unwrap_or_default();
    format!(
        "{implementation},{sketch},{dataset},{op},{n},{reps},{throughput_median:.6},{throughput_stddev:.6},{bytes},{live_bytes},{estimate},{exact},{rel_error}"
    )
}

/// HLL distinct-count row over a stream of `Hashable` items with known exact
/// cardinality, built with hasher `H`. Timed over `reps` warmup+timed reps,
/// rebuilding the sketch each rep so build+update throughput is measured on a
/// clean sketch. The same body serves both the xxh3 default and the murmur3
/// variant; only `H` and `impl_label` differ.
fn hll_row_with<H: SketchHasher, T: Hashable>(
    impl_label: &str,
    dataset: &str,
    items: &[T],
    exact: f64,
    reps: usize,
) -> String {
    let n = items.len() as u64;
    let (median, stddev) = timing::timed_throughput(reps, true, n, || {
        let mut sketch = HllSketchGeneric::<H>::new(12);
        for item in items {
            sketch.update(item);
        }
        core::hint::black_box(&sketch);
    });
    // Build one more populated sketch outside the timing loop to read the row's
    // estimate and serialised size, measuring the heap delta to build and hold
    // it as `live_bytes`. The transient timed sketches above are freed before
    // this measurement, so they do not perturb the delta.
    let (sketch, live) = counting_alloc::measure_live(|| {
        let mut s = HllSketchGeneric::<H>::new(12);
        for item in items {
            s.update(item);
        }
        s
    });
    let estimate = sketch.estimate();
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.to_bytes().len();
    row(
        impl_label,
        "hll",
        dataset,
        "distinct_count",
        n,
        reps as u64,
        median,
        stddev,
        Some(bytes),
        Some(live),
        Some(estimate),
        Some(exact),
        Some(rel_error),
    )
}

/// HLL row for the crate default (xxh3) hasher.
fn hll_row<T: Hashable>(dataset: &str, items: &[T], exact: f64, reps: usize) -> String {
    hll_row_with::<Xxh3Hasher, _>(IMPL_OURS, dataset, items, exact, reps)
}

/// HLL row for the murmur3 hasher, isolating the hash effect.
fn hll_row_murmur3<T: Hashable>(dataset: &str, items: &[T], exact: f64, reps: usize) -> String {
    hll_row_with::<Murmur3Hasher, _>(IMPL_MURMUR3, dataset, items, exact, reps)
}

/// CPC distinct-count row over a stream of `Hashable` items with known exact
/// cardinality, built with hasher `H`. Uses `lg_k = 12` to match the HLL/Theta
/// configuration. Bytes is the serialised length via the `Serializable` trait.
fn cpc_row_with<H: SketchHasher, T: Hashable>(
    impl_label: &str,
    dataset: &str,
    items: &[T],
    exact: f64,
    reps: usize,
) -> String {
    let n = items.len() as u64;
    let (median, stddev) = timing::timed_throughput(reps, true, n, || {
        let mut sketch = CpcSketchGeneric::<H>::new(12);
        for item in items {
            sketch.update(item);
        }
        core::hint::black_box(&sketch);
    });
    let (sketch, live) = counting_alloc::measure_live(|| {
        let mut s = CpcSketchGeneric::<H>::new(12);
        for item in items {
            s.update(item);
        }
        s
    });
    let estimate = sketch.estimate();
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.to_bytes().len();
    row(
        impl_label,
        "cpc",
        dataset,
        "distinct_count",
        n,
        reps as u64,
        median,
        stddev,
        Some(bytes),
        Some(live),
        Some(estimate),
        Some(exact),
        Some(rel_error),
    )
}

/// CPC row for the crate default (xxh3) hasher.
fn cpc_row<T: Hashable>(dataset: &str, items: &[T], exact: f64, reps: usize) -> String {
    cpc_row_with::<Xxh3Hasher, _>(IMPL_OURS, dataset, items, exact, reps)
}

/// CPC row for the murmur3 hasher, isolating the hash effect.
fn cpc_row_murmur3<T: Hashable>(dataset: &str, items: &[T], exact: f64, reps: usize) -> String {
    cpc_row_with::<Murmur3Hasher, _>(IMPL_MURMUR3, dataset, items, exact, reps)
}

/// Theta distinct-count row over a stream of `Hashable` items, built with
/// hasher `H`.
fn theta_row_with<H: SketchHasher, T: Hashable>(
    impl_label: &str,
    dataset: &str,
    items: &[T],
    exact: f64,
    reps: usize,
) -> String {
    let n = items.len() as u64;
    let (median, stddev) = timing::timed_throughput(reps, true, n, || {
        let mut sketch = ThetaSketchGeneric::<H>::new(4096);
        for item in items {
            sketch.update(item);
        }
        core::hint::black_box(&sketch);
    });
    let (sketch, live) = counting_alloc::measure_live(|| {
        let mut s = ThetaSketchGeneric::<H>::new(4096);
        for item in items {
            s.update(item);
        }
        s
    });
    let estimate = sketch.estimate();
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.to_bytes().len();
    row(
        impl_label,
        "theta",
        dataset,
        "distinct_count",
        n,
        reps as u64,
        median,
        stddev,
        Some(bytes),
        Some(live),
        Some(estimate),
        Some(exact),
        Some(rel_error),
    )
}

/// Theta row for the crate default (xxh3) hasher.
fn theta_row<T: Hashable>(dataset: &str, items: &[T], exact: f64, reps: usize) -> String {
    theta_row_with::<Xxh3Hasher, _>(IMPL_OURS, dataset, items, exact, reps)
}

/// Theta row for the murmur3 hasher, isolating the hash effect.
fn theta_row_murmur3<T: Hashable>(dataset: &str, items: &[T], exact: f64, reps: usize) -> String {
    theta_row_with::<Murmur3Hasher, _>(IMPL_MURMUR3, dataset, items, exact, reps)
}

/// Bloom build row, built with hasher `H`. A membership filter has no
/// cardinality estimate, so the estimate/exact/rel_error fields are left empty.
/// Bytes is the bit-array size in bytes (`num_bits / 8`), read from the
/// filter's public statistics.
fn bloom_row_with<H: SketchHasher, T: Hashable>(
    impl_label: &str,
    dataset: &str,
    items: &[T],
    reps: usize,
) -> String {
    let n = items.len() as u64;
    let (median, stddev) = timing::timed_throughput(reps, true, n, || {
        let mut filter = BloomFilterGeneric::<H>::new(n as usize, 0.01);
        for item in items {
            filter.add(item);
        }
        core::hint::black_box(&filter);
    });
    let (filter, live) = counting_alloc::measure_live(|| {
        let mut f = BloomFilterGeneric::<H>::new(n as usize, 0.01);
        for item in items {
            f.add(item);
        }
        f
    });
    let bytes = filter.statistics().num_bits / 8;
    row(
        impl_label,
        "bloom",
        dataset,
        "build",
        n,
        reps as u64,
        median,
        stddev,
        Some(bytes),
        Some(live),
        None,
        None,
        None,
    )
}

/// Bloom row for the crate default (xxh3) hasher.
fn bloom_row<T: Hashable>(dataset: &str, items: &[T], reps: usize) -> String {
    bloom_row_with::<Xxh3Hasher, _>(IMPL_OURS, dataset, items, reps)
}

/// Bloom row for the murmur3 hasher, isolating the hash effect.
fn bloom_row_murmur3<T: Hashable>(dataset: &str, items: &[T], reps: usize) -> String {
    bloom_row_with::<Murmur3Hasher, _>(IMPL_MURMUR3, dataset, items, reps)
}

/// Count-Min point-query row, built with hasher `H`. Each item is incremented
/// once, then a designated hot key is incremented `HOT_KEY_COUNT` times; the
/// query is for that key. Bytes is the backing table size (`total_cells * 8`,
/// the cells are `u64`), read from the sketch's public statistics.
fn countmin_row_with<H: SketchHasher, T: Hashable>(
    impl_label: &str,
    dataset: &str,
    items: &[T],
    reps: usize,
) -> String {
    let n = items.len() as u64;
    let total_ops = n + HOT_KEY_COUNT;
    let (median, stddev) = timing::timed_throughput(reps, true, total_ops, || {
        let mut sketch = CountMinSketchGeneric::<H>::new(2048, 5, false);
        for item in items {
            sketch.increment(item);
        }
        for _ in 0..HOT_KEY_COUNT {
            sketch.increment(HOT_KEY);
        }
        core::hint::black_box(&sketch);
    });
    let (sketch, live) = counting_alloc::measure_live(|| {
        let mut s = CountMinSketchGeneric::<H>::new(2048, 5, false);
        for item in items {
            s.increment(item);
        }
        for _ in 0..HOT_KEY_COUNT {
            s.increment(HOT_KEY);
        }
        s
    });
    let estimate = sketch.estimate(HOT_KEY) as f64;
    let exact = HOT_KEY_COUNT as f64;
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.statistics().total_cells * std::mem::size_of::<u64>();
    row(
        impl_label,
        "countmin",
        dataset,
        "point_query",
        total_ops,
        reps as u64,
        median,
        stddev,
        Some(bytes),
        Some(live),
        Some(estimate),
        Some(exact),
        Some(rel_error),
    )
}

/// Count-Min row for the crate default (xxh3) hasher.
fn countmin_row<T: Hashable>(dataset: &str, items: &[T], reps: usize) -> String {
    countmin_row_with::<Xxh3Hasher, _>(IMPL_OURS, dataset, items, reps)
}

/// Count-Min row for the murmur3 hasher, isolating the hash effect.
fn countmin_row_murmur3<T: Hashable>(dataset: &str, items: &[T], reps: usize) -> String {
    countmin_row_with::<Murmur3Hasher, _>(IMPL_MURMUR3, dataset, items, reps)
}

/// KLL median row over the synthetic numeric range. The exact median of
/// `0..n` mapped to `f64` is `n / 2`.
fn kll_synthetic_row(n: u64, reps: usize) -> String {
    let (median, stddev) = timing::timed_throughput(reps, true, n, || {
        let mut sketch = KllSketch::<f64>::new(200);
        for i in synthetic_distinct(n) {
            sketch.update(i as f64);
        }
        core::hint::black_box(&sketch);
    });
    let (mut sketch, live) = counting_alloc::measure_live(|| {
        let mut s = KllSketch::<f64>::new(200);
        for i in synthetic_distinct(n) {
            s.update(i as f64);
        }
        s
    });
    let estimate = sketch
        .quantile(0.5)
        .expect("KLL median over non-empty stream");
    let exact = n as f64 / 2.0;
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.to_bytes().len();
    row(
        IMPL_OURS,
        "kll",
        "synthetic",
        "quantile_median",
        n,
        reps as u64,
        median,
        stddev,
        Some(bytes),
        Some(live),
        Some(estimate),
        Some(exact),
        Some(rel_error),
    )
}

/// Produce all CSV lines for the synthetic dataset (header first).
pub fn run(n: u64, reps: usize) -> Vec<String> {
    let synthetic: Vec<u64> = synthetic_distinct(n).collect();
    let exact = n as f64;

    vec![
        HEADER.to_string(),
        // Default (xxh3) rows.
        hll_row("synthetic", &synthetic, exact, reps),
        cpc_row("synthetic", &synthetic, exact, reps),
        theta_row("synthetic", &synthetic, exact, reps),
        kll_synthetic_row(n, reps),
        bloom_row("synthetic", &synthetic, reps),
        countmin_row("synthetic", &synthetic, reps),
        // Murmur3 variants of the five hash-based sketches (no KLL: it is
        // comparison-based and uses no hasher). These isolate the hash effect.
        hll_row_murmur3("synthetic", &synthetic, exact, reps),
        cpc_row_murmur3("synthetic", &synthetic, exact, reps),
        theta_row_murmur3("synthetic", &synthetic, exact, reps),
        bloom_row_murmur3("synthetic", &synthetic, reps),
        countmin_row_murmur3("synthetic", &synthetic, reps),
    ]
}

/// The exact CSV header line for the multi-trial RMSE mode (`--trials`).
///
/// This schema is separate from [`HEADER`]; the two are never mixed in one
/// file. Its columns are documented in `benchmarks/results/rmse_schema.md`.
pub const RMSE_HEADER: &str =
    "implementation,sketch,lg_k,trials,n_per_trial,rmse,mean_rel_error,max_rel_error";

/// Format a single RMSE summary row from a slice of per-trial relative errors.
///
/// `rmse = sqrt(mean(rel_error^2))`, `mean_rel_error = mean(rel_error)`, and
/// `max_rel_error = max(rel_error)`.
fn rmse_row(
    implementation: &str,
    sketch: &str,
    lg_k: u8,
    trials: u64,
    n: u64,
    errors: &[f64],
) -> String {
    let count = errors.len() as f64;
    let rmse = (errors.iter().map(|e| e * e).sum::<f64>() / count).sqrt();
    let mean = errors.iter().sum::<f64>() / count;
    let max = errors.iter().copied().fold(0.0_f64, f64::max);
    format!("{implementation},{sketch},{lg_k},{trials},{n},{rmse:.6},{mean:.6},{max:.6}")
}

/// Run `trials` independent trials of `n` distinct items each (trial t over the
/// disjoint range [t*n, (t+1)*n)) and emit one RMSE summary row per sketch.
pub fn run_rmse(trials: u64, n: u64) -> Vec<String> {
    let (mut theta_errs, mut hll_errs, mut cpc_errs) = (Vec::new(), Vec::new(), Vec::new());
    let truth = n as f64;
    for t in 0..trials {
        let start = t * n;
        let mut theta = ThetaSketch::new(4096);
        let mut hll = HllSketch::new(12);
        let mut cpc = CpcSketch::new(12);
        for i in start..start + n {
            theta.update(&i);
            hll.update(&i);
            cpc.update(&i);
        }
        theta_errs.push((theta.estimate() - truth).abs() / truth);
        hll_errs.push((hll.estimate() - truth).abs() / truth);
        cpc_errs.push((cpc.estimate() - truth).abs() / truth);
    }
    vec![
        RMSE_HEADER.to_string(),
        rmse_row("ours", "theta", 12, trials, n, &theta_errs),
        rmse_row("ours", "hll", 12, trials, n, &hll_errs),
        rmse_row("ours", "cpc", 12, trials, n, &cpc_errs),
    ]
}

/// Produce the CSV lines (no header) for a single TPC-H column.
///
/// The `dataset` label identifies the column source. KLL is intentionally
/// omitted: a median over arbitrary strings is not meaningful, so the KLL
/// measurement only runs on the synthetic numeric stream.
pub fn run_tpch(
    path: &Path,
    col: usize,
    dataset: &str,
    reps: usize,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let values = tpch_column(path, col).map_err(Box::new)?;
    let exact = exact_distinct(values.iter().cloned()) as f64;

    Ok(vec![
        // Default (xxh3) rows.
        hll_row(dataset, &values, exact, reps),
        cpc_row(dataset, &values, exact, reps),
        theta_row(dataset, &values, exact, reps),
        bloom_row(dataset, &values, reps),
        countmin_row(dataset, &values, reps),
        // Murmur3 variants, isolating the hash effect (no KLL here).
        hll_row_murmur3(dataset, &values, exact, reps),
        cpc_row_murmur3(dataset, &values, exact, reps),
        theta_row_murmur3(dataset, &values, exact, reps),
        bloom_row_murmur3(dataset, &values, reps),
        countmin_row_murmur3(dataset, &values, reps),
    ])
}
