//! Row-producing logic for the `ours` benchmark runner.
//!
//! Each sketch in this crate is run over the shared datasets and emitted as a
//! CSV row in the schema defined in `benchmarks/results/schema.md`:
//!
//! ```text
//! implementation,sketch,dataset,op,n,throughput_ops_per_s,bytes,estimate,exact,rel_error
//! ```
//!
//! The `implementation` field is always `ours` here. `run` produces the
//! synthetic-dataset rows; `run_tpch` produces the rows for a TPC-H column.

use std::path::Path;
use std::time::Instant;

use datasets::{exact_distinct, synthetic_distinct, tpch_column};
use sketches::bloom::BloomFilter;
use sketches::countmin::CountMinSketch;
use sketches::cpc::CpcSketch;
use sketches::hll::HllSketch;
use sketches::quantiles::KllSketch;
use sketches::serialization::Serializable;
use sketches::theta::ThetaSketch;

/// The exact CSV header line shared by all runners.
pub const HEADER: &str =
    "implementation,sketch,dataset,op,n,throughput_ops_per_s,bytes,estimate,exact,rel_error";

const IMPLEMENTATION: &str = "ours";
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

/// HLL distinct-count row over a stream of `Hashable` items with known exact
/// cardinality.
fn hll_row<T: sketches::hash::Hashable>(dataset: &str, items: &[T], exact: f64) -> String {
    let n = items.len() as u64;
    let mut sketch = HllSketch::new(12);
    let start = Instant::now();
    for item in items {
        sketch.update(item);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let estimate = sketch.estimate();
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.to_bytes().len();
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

/// CPC distinct-count row over a stream of `Hashable` items with known exact
/// cardinality. Uses `lg_k = 12` to match the HLL/Theta configuration. Bytes is
/// the serialised length via the `Serializable` trait.
fn cpc_row<T: sketches::hash::Hashable>(dataset: &str, items: &[T], exact: f64) -> String {
    let n = items.len() as u64;
    let mut sketch = CpcSketch::new(12);
    let start = Instant::now();
    for item in items {
        sketch.update(item);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let estimate = sketch.estimate();
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.to_bytes().len();
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

/// Theta distinct-count row over a stream of `Hashable` items.
fn theta_row<T: sketches::hash::Hashable>(dataset: &str, items: &[T], exact: f64) -> String {
    let n = items.len() as u64;
    let mut sketch = ThetaSketch::new(4096);
    let start = Instant::now();
    for item in items {
        sketch.update(item);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let estimate = sketch.estimate();
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.to_bytes().len();
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
/// estimate/exact/rel_error fields are left empty. Bytes is the bit-array size
/// in bytes (`num_bits / 8`), read from the filter's public statistics.
fn bloom_row<T: sketches::hash::Hashable>(dataset: &str, items: &[T]) -> String {
    let n = items.len() as u64;
    let mut filter = BloomFilter::new(n as usize, 0.01, false);
    let start = Instant::now();
    for item in items {
        filter.add(item);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let bytes = filter.statistics().num_bits / 8;
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
/// Bytes is the backing table size (`total_cells * 8`, the cells are `u64`),
/// read from the sketch's public statistics.
fn countmin_row<T: sketches::hash::Hashable>(dataset: &str, items: &[T]) -> String {
    let n = items.len() as u64;
    let mut sketch = CountMinSketch::new(2048, 5, false, false);
    let start = Instant::now();
    for item in items {
        sketch.increment(item);
    }
    for _ in 0..HOT_KEY_COUNT {
        sketch.increment(HOT_KEY);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let estimate = sketch.estimate(HOT_KEY) as f64;
    let exact = HOT_KEY_COUNT as f64;
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.statistics().total_cells * std::mem::size_of::<u64>();
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

/// KLL median row over the synthetic numeric range. The exact median of
/// `0..n` mapped to `f64` is `n / 2`.
fn kll_synthetic_row(n: u64) -> String {
    let mut sketch = KllSketch::<f64>::new(200);
    let start = Instant::now();
    for i in synthetic_distinct(n) {
        sketch.update(i as f64);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let estimate = sketch
        .quantile(0.5)
        .expect("KLL median over non-empty stream");
    let exact = n as f64 / 2.0;
    let rel_error = (estimate - exact).abs() / exact;
    let bytes = sketch.to_bytes().len();
    row(
        "kll",
        "synthetic",
        "quantile_median",
        n,
        throughput(n, elapsed),
        Some(bytes),
        Some(estimate),
        Some(exact),
        Some(rel_error),
    )
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
        kll_synthetic_row(n),
        bloom_row("synthetic", &synthetic),
        countmin_row("synthetic", &synthetic),
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
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let values = tpch_column(path, col).map_err(Box::new)?;
    let exact = exact_distinct(values.iter().cloned()) as f64;

    Ok(vec![
        hll_row(dataset, &values, exact),
        cpc_row(dataset, &values, exact),
        theta_row(dataset, &values, exact),
        bloom_row(dataset, &values),
        countmin_row(dataset, &values),
    ])
}
