# Sketches Examples

This directory contains examples demonstrating the usage of various probabilistic data structures implemented in this library.

## Rust Examples
- `basic_usage.rs` - Basic usage of all sketch types
- `hll_comparison.rs` - Comparing HLL variants
- `tpch_generate.rs` - Generate TPC-H benchmark data
- `tpch_benchmarks.rs` - Run benchmarks on TPC-H data

## Python Examples
- `tpch_test.py` - Comprehensive testing with TPC-H data

## Jupyter Notebooks
- `getting_started.ipynb` - Introduction to probabilistic data structures

## TPC-H Benchmarking

The library includes support for TPC-H benchmarking using [tpchgen-rs](https://github.com/clflushopt/tpchgen-rs), allowing standardised testing and performance comparison.

### Generate TPC-H Data
```bash
# Generate at scale factor 0.01 (10MB)
cargo run --example tpch_generate -- 0.01 tpch_data

# Generate at scale factor 0.1 (100MB)
cargo run --example tpch_generate -- 0.1 tpch_data
```

### Run Benchmarks
```bash
# Rust benchmarks
cargo run --example tpch_benchmarks --release

# Python tests
python examples/tpch_test.py 0.01 tpch_data
```

## Running Other Examples

### Rust Examples
```bash
cargo run --example basic_usage
cargo run --example hll_comparison --release
```

### Jupyter Notebooks
```bash
pip install -e .[dev]
jupyter lab examples/
```