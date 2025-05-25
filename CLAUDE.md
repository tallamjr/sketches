# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Rust library providing probabilistic data structures (sketches) for approximate distinct counting, with Python bindings via PyO3. The project is inspired by Polars' architecture and offers memory-efficient alternatives to exact counting.

## Development Commands

### Build & Install
```bash
# Python development install (recommended)
pip install -e .[dev]

# Build Python extension
maturin develop

# Build release wheel
maturin build --release
```

### Testing
```bash
# Run all Python tests
pytest

# Run specific Python test
pytest tests/test_sketches.py::test_hll_basic -v

# Run Rust tests
cargo test

# Run Rust tests with output
cargo test -- --nocapture
```

### Performance Testing

#### Phase Comparison Benchmarks
```bash
# Phase 1: Memory optimization baseline
cargo run --example phase1_benchmark --release --features optimized

# Phase 2: Throughput optimization
cargo run --example phase2_benchmark --release --features optimized

# Phase 3: System-level optimization  
cargo run --example phase3_benchmark --release --features optimized

# Phase 3 with real TPC-H data (realistic business workloads)
cargo run --example phase3_tpch_benchmark --release --features optimized
```

#### Comprehensive Performance Tests
```bash
# Compare HLL implementations and variants
cargo run --example hll_comparison --release --features optimized

# Real-world TPC-H benchmark performance
cargo run --example tpch_benchmarks --release --features optimized

# Basic usage patterns and performance
cargo run --example basic_usage --release --features optimized
```

#### Python Performance Testing
```bash
# Install with development optimizations
pip install -e .[dev]

# Run Python tests with performance timing
pytest tests/test_sketches.py -v --tb=short

# Test memory efficiency and usage patterns
pytest tests/test_memory.py -v

# Compare against Polars DataFrame performance
pytest tests/test_polars_vs_sketch.py -v

# Run notebook examples with performance measurement
pytest --nbmake examples/getting_started.ipynb
```

#### Performance Testing
Run benchmarks with `--release --features optimized` flags for optimal performance testing.

### Formatting
```bash
# Format Rust code
cargo fmt

# Format Python code  
black .
```

## Architecture

The codebase has a dual-purpose structure:
- Pure Rust library (`rlib`) for Rust users
- Python extension module (`cdylib`) via PyO3

Key architectural decisions:
- Each sketch type (HLL, Theta, CPC) is implemented in its own module
- Python bindings are defined in `src/lib.rs` using PyO3 macros
- Test data uses TPC-H CSV files located in `tests/data/`

## Sketch Implementations

Current status:
- **HyperLogLog (HLL)**: ✅ Complete with HLL++ variants
- **Theta**: ✅ Complete with set operations (union, intersection, difference)
- **CPC**: 🚧 Under development
- **PP**: 🔴 Placeholder

When implementing new sketches:
1. Create a new module in `src/` (e.g., `src/new_sketch.rs`)
2. Add Python bindings in `src/lib.rs`
3. Write tests in both Rust (`tests/`) and Python (`tests/test_sketches.py`)
4. Ensure serialization support for distributed computing use cases