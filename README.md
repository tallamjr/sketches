# sketches -- High-Performance Probabilistic Data Structures

[![Rust](https://img.shields.io/badge/rust-1.86%2B-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)

**Fast, memory-efficient probabilistic data structures for streaming analytics, cardinality estimation, quantile computation, and sampling.**

Python bindings for Rust-based implementations of HyperLogLog, T-Digest, Reservoir Sampling, and more via PyO3.

**[Algorithm Deep Dive](ALGORITHMS.md)**

## Features

| **Algorithm Category**     | **Implementation** | **Description**                                                 | **Status** | **Mergeable** |
| -------------------------- | ------------------ | --------------------------------------------------------------- | ---------- | ------------- |
| **Cardinality Estimation** | HyperLogLog (HLL)  | Industry-standard distinct counting with ~1% error              | Yes        | Yes           |
|                            | HyperLogLog++      | Enhanced HLL with bias correction and sparse mode               | Yes        | Yes           |
|                            | CPC Sketch         | Most compact serialisation for network transfer                 | Yes        | Yes           |
|                            | Linear Counter     | Optimal for small cardinalities (n < 1000)                      | Yes        | No            |
|                            | Hybrid Counter     | Auto-transitions from Linear to HLL                             | Yes        | No            |
| **Set Operations**         | Theta Sketch       | Union, intersection, difference with cardinality estimation     | Yes        | Yes           |
| **Sampling**               | Algorithm R        | Basic reservoir sampling for uniform random samples             | Yes        | Yes           |
|                            | Algorithm A        | Optimised reservoir sampling (19x faster for large streams)     | Yes        | Yes           |
|                            | Weighted Sampling  | Probability-proportional reservoir sampling                     | Yes        | Yes           |
|                            | Stream Sampling    | High-throughput sampling with batching                          | Yes        | No            |
| **Quantile Estimation**    | T-Digest           | Superior accuracy for extreme quantiles (p95, p99)              | Yes        | Yes           |
|                            | KLL Sketch         | Provable error bounds (~1.65% at k=200)                         | Yes        | Yes           |
| **Frequency Estimation**   | Count-Min Sketch   | Conservative frequency estimation with epsilon-delta guarantees | Yes        | Yes           |
|                            | Count Sketch       | Unbiased frequency estimation using median                      | Yes        | Yes           |
|                            | Frequent Items     | Top-K heavy hitters with Space-Saving algorithm                 | Yes        | Yes           |
| **Membership Testing**     | Bloom Filter       | Fast membership testing with configurable false positive rate   | Yes        | Yes           |
|                            | Counting Bloom     | Bloom filter with deletion support                              | Yes        | Yes           |
| **Multi-dimensional**      | Array of Doubles   | Tuple sketch for multi-dimensional aggregation                  | Yes        | Yes           |

**Mergeable** means two independently built sketches can be combined into one that represents the union of both input streams, without access to the original data. This is essential for distributed systems where data is partitioned across nodes -- each node builds a local sketch, then all sketches are merged into a single result.

## Table of Contents

<!-- mtoc-start -->

- [Choosing the Right Sketch](#choosing-the-right-sketch)
- [Background: Probabilistic Data Structures](#background-probabilistic-data-structures)
  - [The Cardinality Conundrum](#the-cardinality-conundrum)
  - [Database Superpowers: Query Planning and GROUP BY Operations](#database-superpowers-query-planning-and-group-by-operations)
  - [How HLL Works at a Glance](#how-hll-works-at-a-glance)
  - [Implications and the Big Picture](#implications-and-the-big-picture)
- [Memory Usage Comparison](#memory-usage-comparison)
- [Package Installation](#package-installation)
  - [Prerequisites](#prerequisites)
  - [From PyPI](#from-pypi)
  - [From Source](#from-source)
- [Getting Started](#getting-started)
  - [Quick Installation](#quick-installation)
  - [Development Workflow](#development-workflow)
- [Library Usage](#library-usage)
  - [Business Intelligence Examples](#business-intelligence-examples)
  - [HLL Sketch Example](#hll-sketch-example)
    - [Minimal Test with Polars](#minimal-test-with-polars)
  - [CPC Sketch Example](#cpc-sketch-example)
    - [Minimal Test with Polars](#minimal-test-with-polars-1)
  - [Bloom Filter Example](#bloom-filter-example)
  - [Frequency Estimation Example](#frequency-estimation-example)
  - [Quantile Estimation Example](#quantile-estimation-example)
  - [Sampling Example](#sampling-example)
  - [Linear and Hybrid Counter Example](#linear-and-hybrid-counter-example)
  - [Array of Doubles (AOD) Sketch Example](#array-of-doubles-aod-sketch-example)
- [Extending HLL++: Sparse Buffer, Variable-Length Encoding, and Hybrid Representation](#extending-hll-sparse-buffer-variable-length-encoding-and-hybrid-representation)
  - [Theta Sketch Example](#theta-sketch-example)
    - [Minimal Test with Polars](#minimal-test-with-polars-2)
- [Architecture](#architecture)
- [Performance](#performance)
  - [Accuracy (multi-trial RMSE)](#accuracy-multi-trial-rmse)
  - [Throughput](#throughput)
  - [Benchmark harness](#benchmark-harness)
  - [When to use each](#when-to-use-each)
- [TPC-H Business Intelligence Benchmarks](#tpc-h-business-intelligence-benchmarks)
- [Roadmap and Missing Features](#roadmap-and-missing-features)
  - [Not Yet Implemented](#not-yet-implemented)
  - [Current Development Priorities](#current-development-priorities)
- [License](#license)

<!-- mtoc-end -->

## Choosing the Right Sketch

Different problems call for different sketches. Use this guide to pick the right one for your use case.

| Problem                   | "How do I know if..."                        | Small Scale                   | Large Scale                                 | Distributed / Mergeable             |
| ------------------------- | -------------------------------------------- | ----------------------------- | ------------------------------------------- | ----------------------------------- |
| **Membership**            | "Is X in the set?"                           | `BloomFilter`                 | `CountingBloomFilter` (if deletions needed) | Yes -- union via bitwise OR         |
| **Cardinality**           | "How many unique items?"                     | `LinearCounter` (n < 1000)    | `HllSketch` / `HllPlusPlusSketch`           | Yes -- register-wise max            |
| **Cardinality + Set Ops** | "What's the overlap between A and B?"        | `ThetaSketch`                 | `ThetaSketch`                               | Yes -- union, intersect, difference |
| **Compact Cardinality**   | "Unique count with minimal serialised size?" | `CpcSketch`                   | `CpcSketch`                                 | Yes -- sketch merging               |
| **Frequency**             | "What are the top-K items?"                  | `FrequentStringsSketch`       | `CountMinSketch` / `CountSketch`            | Yes -- entry-wise addition          |
| **Quantiles**             | "What's the p99 latency?"                    | `KllSketch` (provable bounds) | `TDigest` (extreme quantile accuracy)       | Yes -- digest merging               |
| **Sampling**              | "Give me a random subset"                    | `ReservoirSamplerR`           | `ReservoirSamplerA` (19x faster)            | Partial -- merge samplers           |
| **Weighted Sampling**     | "Sample proportional to weight"              | `WeightedReservoirSampler`    | `VarOptSketch` (Horvitz-Thompson)           | Yes -- VarOpt merge                 |
| **Multi-dimensional**     | "Aggregate multiple metrics per key"         | `AodSketch`                   | `AodSketch`                                 | Yes -- summary merging              |

**Key trade-offs:**

- **HLL vs Theta**: HLL is more memory-efficient for pure cardinality. Theta supports set operations (union, intersection, difference).
- **HLL vs CPC**: CPC achieves ~40% smaller serialised size but is more complex. Use CPC when network transfer cost matters.
- **Count-Min vs Count Sketch**: Count-Min always overestimates (conservative). Count Sketch is unbiased but uses more space.
- **KLL vs T-Digest**: KLL has provable error bounds (~1.65% at k=200). T-Digest excels at extreme quantiles (p99, p99.9) but bounds are empirical.
- **Algorithm R vs A**: Both produce uniform samples. Algorithm A skips items probabilistically, making it ~19x faster for large streams.

## Package Installation

### Prerequisites

- Python 3.10+
- Optionally, for DataFrame examples: `polars` (`pip install polars`).

- Optionally, for memory measurement examples: `psutil` (`pip install psutil`).

### From PyPI

The package is not yet published to PyPI. Install from source for now (see below).

```bash
pip install rusty-sketches
```

### From Source

```bash
git clone https://github.com/tallamjr/sketches.git
cd sketches
pip install .
```

For an editable install with development dependencies:

```bash
pip install -e .[dev]
```

**For TPC-H Performance Analysis Notebook:**

```bash
# Install with visualisation dependencies
pip install -e .[dev]  # includes seaborn, matplotlib, pandas

# Run the comprehensive business intelligence benchmarks
pytest --nbmake examples/tpch_performance_analysis.ipynb
```

## Roadmap and Missing Features

While this library provides a comprehensive suite of probabilistic data structures, the following features are planned but not yet implemented:

### Not Yet Implemented

**Similarity Estimation (Largest Gap)**

- **MinHash** - Set similarity estimation via min-hash signatures (Jaccard similarity)
- **SimHash** - Locality-sensitive hashing for cosine similarity and near-duplicate detection
- **LSH Framework** - Approximate nearest-neighbour search using banded hashing

**Historical / Educational**

- **Probabilistic Counter (Flajolet-Martin)** - Original probabilistic counting algorithm
- **LogLog Counter** - Predecessor to HyperLogLog
- **q-digest** - Tree-based quantile estimation with range query support
- **Quotient Filter** - Cache-friendly alternative to Bloom filters
- **Cuckoo Filter** - Bloom filter alternative with deletion support and better locality

**Performance**

- **SIMD Acceleration** - The implementation is currently pure scalar Rust; vectorised register updates are a possible future direction.
- **Apache byte-compatibility** - We use our own compact codec and xxh3 hashing, so the official DataSketches serialisation format is not supported.

**Integration**

- **Polars Integration** - Custom expressions and DataFrame operations
- **Advanced Serialisation** - Network-optimised protocols beyond the current compact codec

### Current Development Priorities

1. **MinHash implementation** -- fills the similarity estimation gap (largest missing domain)
2. Polars custom expressions for seamless DataFrame integration
3. Performance optimisations for Python bindings (batch operations)

## License

This project is licensed under the MIT License (see `pyproject.toml`).
