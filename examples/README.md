# Sketches Examples

This directory contains runnable examples demonstrating the probabilistic data structures provided by this library. Every example listed below ships in this directory and can be run with the command shown.

## Rust Examples

Run any example with `cargo run --example <name>`. Add `--release` for the throughput-oriented demos.

- `simple_demo` - Minimal tour of Bloom, CPC, HLL, and Theta sketches.
  `cargo run --example simple_demo`
- `basic_usage` - Broader walkthrough of the core sketch types and their estimates.
  `cargo run --example basic_usage`
- `aod_demo` - Array of Doubles sketch: cardinality with associated double values, set operations, and merging.
  `cargo run --example aod_demo`
- `bloom_filter_demo` - Standard and counting Bloom filters for approximate membership.
  `cargo run --example bloom_filter_demo`
- `cardinality_optimization_demo` - Linear and hybrid counters compared against HLL for distinct counting.
  `cargo run --example cardinality_optimization_demo`
- `frequency_estimation_demo` - Count-Min and Count sketches for frequency and heavy-hitter estimation.
  `cargo run --example frequency_estimation_demo`
- `hll_comparison` - Compares the HyperLogLog variants (HLL, HLL++, sparse HLL++).
  `cargo run --example hll_comparison --release`
- `quantiles_demo` - Quantile estimation with the KLL sketch.
  `cargo run --example quantiles_demo`
- `reservoir_sampling_demo` - Reservoir sampling algorithms, including weighted sampling.
  `cargo run --example reservoir_sampling_demo`
- `tdigest_demo` - T-Digest quantile estimation, with a comparison against the KLL sketch.
  `cargo run --example tdigest_demo`

## Jupyter Notebooks

- `tpch_performance_analysis.ipynb` - A TPC-H business-intelligence notebook exploring sketch behaviour on TPC-H style data.

To open the notebook:

```bash
pip install -e .[dev]
jupyter lab examples/tpch_performance_analysis.ipynb
```
