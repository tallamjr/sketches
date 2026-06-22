//! Row-producing logic for the `apache-rust` benchmark runner.
//!
//! This runner drives the upstream Apache `datasketches` crate over the same
//! shared datasets as `runner-ours`, emitting CSV rows in the identical schema
//! so that the reporter can join the two implementations row-for-row:
//!
//! ```text
//! implementation,sketch,dataset,op,n,throughput_ops_per_s,bytes,estimate,exact,rel_error
//! ```
//!
//! The `implementation` field is always `apache-rust` here. Only the four
//! sketches that both libraries provide are emitted (`hll`, `theta`, `bloom`,
//! `countmin`); the Apache Rust crate has no KLL, so no `kll` row is produced.
//!
//! `run` produces the synthetic-dataset rows; `run_tpch` produces the rows for
//! a single TPC-H column.

use std::path::Path;
use std::time::Instant;

use datasets::{exact_distinct, synthetic_distinct, tpch_column};
use datasketches::bloom::BloomFilterBuilder;
use datasketches::countmin::CountMinSketch;
use datasketches::cpc::CpcSketch;
use datasketches::hll::{HllSketch, HllType};
use datasketches::theta::ThetaSketch;

/// The exact CSV header line shared by all runners.
pub const HEADER: &str =
    "implementation,sketch,dataset,op,n,throughput_ops_per_s,bytes,estimate,exact,rel_error";

const IMPLEMENTATION: &str = "apache-rust";
const HOT_KEY: &str = "__hot__";
const HOT_KEY_COUNT: u64 = 1000;

/// Format a single CSV row. Optional fields are written as empty strings.
///
/// One parameter per schema column is the clearest mapping to the fixed CSV
/// header, so the argument count deliberately mirrors the schema.
#[allow(clippy::too_many_arguments)]
fn row(
    sketch: &str,
    dataset: &str,
    op: &str,
    n: u64,
    throughput: f64,
    bytes: Option<usize>,
    estimate: Option<f64>,
    exact: Option<f64>,
    rel_error: Option<f64>,
) -> String {
    let bytes = bytes.map(|b| b.to_string()).unwrap_or_default();
    let estimate = estimate.map(|v| format!("{v:.6}")).unwrap_or_default();
    let exact = exact.map(|v| format!("{v:.6}")).unwrap_or_default();
    let rel_error = rel_error.map(|v| format!("{v:.6}")).unwrap_or_default();
    format!(
        "{IMPLEMENTATION},{sketch},{dataset},{op},{n},{throughput:.6},{bytes},{estimate},{exact},{rel_error}"
    )
}

fn throughput(items: u64, elapsed_secs: f64) -> f64 {
    if elapsed_secs > 0.0 {
        items as f64 / elapsed_secs
    } else {
        0.0
    }
}

/// A stream item that the Apache sketch `update`/`insert` methods can consume
/// by value. Both `u64` (synthetic) and `String` (TPC-H) satisfy this.
trait Item: std::hash::Hash + Clone {}
impl<T: std::hash::Hash + Clone> Item for T {}

/// HLL distinct-count row over a stream of items with known exact cardinality.
///
/// Uses `lg_config_k = 12` (matching our `HllSketch::new(12)`) and the `Hll8`
/// target type, which is the fastest-update variant and the one the upstream
/// estimation tests exercise by default.
fn hll_row<T: Item>(dataset: &str, items: &[T], exact: f64) -> String {
    let n = items.len() as u64;
    let mut sketch = HllSketch::new(12, HllType::Hll8);
    let start = Instant::now();
    for item in items {
        // Apache `update<T: Hash>(value: T)` takes the value by value.
        sketch.update(item.clone());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let estimate = sketch.estimate();
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.serialize().len();
    row(
        "hll",
        dataset,
        "distinct_count",
        n,
        throughput(n, elapsed),
        Some(bytes),
        Some(estimate),
        Some(exact),
        Some(rel_error),
    )
}

/// CPC distinct-count row over a stream of items with known exact cardinality.
///
/// Uses `lg_k = 12` (matching our `CpcSketch::new(12)`). The Apache
/// `update<T: Hash>(value: T)` takes the value by value; bytes is the
/// serialized length.
fn cpc_row<T: Item>(dataset: &str, items: &[T], exact: f64) -> String {
    let n = items.len() as u64;
    let mut sketch = CpcSketch::new(12);
    let start = Instant::now();
    for item in items {
        sketch.update(item.clone());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let estimate = sketch.estimate();
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.serialize().len();
    row(
        "cpc",
        dataset,
        "distinct_count",
        n,
        throughput(n, elapsed),
        Some(bytes),
        Some(estimate),
        Some(exact),
        Some(rel_error),
    )
}

/// Theta distinct-count row over a stream of items.
///
/// Uses `lg_k = 12` (nominal k = 4096, matching our `ThetaSketch::new(4096)`).
/// Theta's serialized form lives on the compact sketch, so bytes is taken from
/// `compact(true).serialize()`.
fn theta_row<T: Item>(dataset: &str, items: &[T], exact: f64) -> String {
    let n = items.len() as u64;
    let mut sketch = ThetaSketch::builder().lg_k(12).build();
    let start = Instant::now();
    for item in items {
        sketch.update(item.clone());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let estimate = sketch.estimate();
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.compact(true).serialize().len();
    row(
        "theta",
        dataset,
        "distinct_count",
        n,
        throughput(n, elapsed),
        Some(bytes),
        Some(estimate),
        Some(exact),
        Some(rel_error),
    )
}

/// Bloom build row. A membership filter has no cardinality estimate, so the
/// estimate/exact/rel_error fields are left empty. The filter is sized for `n`
/// items at a 1% false-positive probability via the accuracy builder, matching
/// our `BloomFilter::new(n, 0.01, false)`. Bytes is the serialized length.
fn bloom_row<T: Item>(dataset: &str, items: &[T]) -> String {
    let n = items.len() as u64;
    let mut filter = BloomFilterBuilder::with_accuracy(n.max(1), 0.01).build();
    let start = Instant::now();
    for item in items {
        // Apache `insert<T: Hash>(item: T)` takes the value by value.
        filter.insert(item.clone());
    }
    let elapsed = start.elapsed().as_secs_f64();
    let bytes = filter.serialize().len();
    row(
        "bloom",
        dataset,
        "build",
        n,
        throughput(n, elapsed),
        Some(bytes),
        None,
        None,
        None,
    )
}

/// Count-Min point-query row. Each item is incremented once, then a designated
/// hot key is incremented `HOT_KEY_COUNT` times; the query is for that key.
///
/// Configured with `num_hashes = 5`, `num_buckets = 2048` to match our
/// `CountMinSketch::new(2048, 5, ...)`. The value type is `u64` so the estimate
/// is a non-negative frequency. Bytes is the serialized length.
fn countmin_row<T: Item>(dataset: &str, items: &[T]) -> String {
    let n = items.len() as u64;
    let mut sketch = CountMinSketch::<u64>::new(5, 2048);
    let start = Instant::now();
    for item in items {
        sketch.update(item.clone());
    }
    for _ in 0..HOT_KEY_COUNT {
        sketch.update(HOT_KEY);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let estimate = sketch.estimate(HOT_KEY) as f64;
    let exact = HOT_KEY_COUNT as f64;
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.serialize().len();
    let total_ops = n + HOT_KEY_COUNT;
    row(
        "countmin",
        dataset,
        "point_query",
        total_ops,
        throughput(total_ops, elapsed),
        Some(bytes),
        Some(estimate),
        Some(exact),
        Some(rel_error),
    )
}

/// The exact CSV header line for the multi-trial RMSE mode (`--trials`).
///
/// This schema is separate from [`HEADER`]; the two are never mixed in one
/// file. Its columns match `runner-ours` so the two implementations can be
/// compared row-for-row.
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
/// disjoint range [t*n, (t+1)*n)) and emit one RMSE summary row per sketch,
/// using the upstream Apache `datasketches` crate at `lg_k = 12`.
pub fn run_rmse(trials: u64, n: u64) -> Vec<String> {
    let (mut theta_errs, mut hll_errs, mut cpc_errs) = (Vec::new(), Vec::new(), Vec::new());
    let truth = n as f64;
    for t in 0..trials {
        let start = t * n;
        let mut theta = ThetaSketch::builder().lg_k(12).build();
        let mut hll = HllSketch::new(12, HllType::Hll8);
        let mut cpc = CpcSketch::new(12);
        for i in start..start + n {
            // Apache `update<T: Hash>(value: T)` takes the value by value; `u64`
            // is `Copy` so the same `i` feeds all three sketches.
            theta.update(i);
            hll.update(i);
            cpc.update(i);
        }
        theta_errs.push((theta.estimate() - truth).abs() / truth);
        hll_errs.push((hll.estimate() - truth).abs() / truth);
        cpc_errs.push((cpc.estimate() - truth).abs() / truth);
    }
    vec![
        RMSE_HEADER.to_string(),
        rmse_row("apache-rust", "theta", 12, trials, n, &theta_errs),
        rmse_row("apache-rust", "hll", 12, trials, n, &hll_errs),
        rmse_row("apache-rust", "cpc", 12, trials, n, &cpc_errs),
    ]
}

/// Produce all CSV lines for the synthetic dataset (header first).
pub fn run(n: u64) -> Vec<String> {
    let synthetic: Vec<u64> = synthetic_distinct(n).collect();
    let exact = n as f64;

    vec![
        HEADER.to_string(),
        hll_row("synthetic", &synthetic, exact),
        cpc_row("synthetic", &synthetic, exact),
        theta_row("synthetic", &synthetic, exact),
        bloom_row("synthetic", &synthetic),
        countmin_row("synthetic", &synthetic),
    ]
}

/// Produce the CSV lines (no header) for a single TPC-H column.
///
/// The `dataset` label identifies the column source. As with `runner-ours`,
/// KLL is intentionally absent (and Apache Rust has none regardless).
pub fn run_tpch(
    path: &Path,
    col: usize,
    dataset: &str,
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let values = tpch_column(path, col).map_err(Box::new)?;
    let exact = exact_distinct(values.iter().cloned()) as f64;
    // Pass borrowed `&str` views so the per-item `clone()` inside each timed
    // helper loop is a cheap Copy of the reference, not a String heap
    // allocation. The bytes hashed are identical to passing the owned String.
    let refs: Vec<&str> = values.iter().map(|s| s.as_str()).collect();

    Ok(vec![
        hll_row(dataset, &refs, exact),
        cpc_row(dataset, &refs, exact),
        theta_row(dataset, &refs, exact),
        bloom_row(dataset, &refs),
        countmin_row(dataset, &refs),
    ])
}
