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