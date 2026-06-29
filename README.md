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

## Performance

**Accuracy is measured by multi-trial RMSE, not single runs.** A single accuracy comparison against Apache is statistically meaningless: the spread between trials is larger than the gap between implementations. Earlier versions of this README quoted single-run figures, which have been removed. The numbers below come from the benchmark harness running 100 trials of 100,000 distinct items at `lg_k = 12` (4096 registers), and are regenerable with `make -C benchmarks rmse`.

### Accuracy (multi-trial RMSE)

The theoretical error floor for `lg_k = 12` is `1/sqrt(4096) = 0.0156`. Relative-error RMSE over 100 trials x 100,000 distinct items:

| Sketch    | Ours (RMSE) | Apache DataSketches (RMSE) | Verdict                         |
| --------- | ----------- | -------------------------- | ------------------------------- |
| HLL       | 0.0122      | 0.0129                     | Ours slightly better, below floor (via HIP) |
| Theta     | 0.0153      | 0.0144                     | Parity, at the floor            |
| CPC       | 0.0089      | 0.0084                     | Parity, below the floor         |

All three distinct counters are at parity-or-better against Apache DataSketches, and HLL and CPC sit below the `1/sqrt(k)` floor thanks to their HIP estimators.

![RMSE by sketch and implementation](assets/benchmarks/rmse_comparison.png)

CPC was previously broken (it reported around 173% error). It is now an ICON+HIP port: roughly 0.34% error on a synthetic stream and 1.17% on a real TPC-H column. HLL gained a HIP estimator that moved its RMSE from 0.0175 to 0.0122, taking it from worse-than-Apache to slightly-better-than-Apache and below the floor:

![HLL accuracy before and after HIP](assets/benchmarks/hll_rmse_before_after.png)

### Throughput

Throughput is now measured on a stabilised harness: each figure is the median over independent rounds with a 95% bootstrap confidence interval, so a real change is distinguishable from run-to-run noise. On this harness (N = 1,000,000, single machine), our xxh3-backed default beats hand-tuned Apache C++ on four of the five shared sketches and beats the Apache Rust crate on all five:

| Sketch   | Ours vs Apache C++   | Ours vs Apache Rust |
| -------- | -------------------- | ------------------- |
| CountMin | 3.3x ahead           | 3.3x ahead          |
| HLL      | 2.5x ahead           | 5.4x ahead          |
| Theta    | 1.9x ahead           | 4.0x ahead          |
| CPC      | 1.3x ahead           | 1.9x ahead          |
| Bloom    | 0.93x (near parity)  | 3.9x ahead          |

![Throughput by sketch and implementation](assets/benchmarks/throughput.png)

The win is driven by the hash. xxh3 is about 2.86x faster per call than the MurmurHash3 Apache uses (1.56 ns versus 4.47 ns for an 8-byte key), and the distinct-counter update is hash-bound. On equal hashing (the `ours-murmur3` bars in the plot) Apache C++'s sketch loops are faster, so the comparison rewards the hash choice rather than loop-level cleverness. Bloom is the one sketch where Apache C++'s blocked layout stays ahead even with our faster hash; we sit within about 6% of it. Absolute numbers are machine-dependent; regenerate them with the harness (`make -C benchmarks report`).

### Memory

Per-sketch live heap (the build-and-hold working footprint, measured by a counting allocator) is at parity with Apache across every sketch:

![Live memory by sketch and implementation](assets/benchmarks/memory.png)

### Benchmark harness

The `benchmarks/` directory contains everything needed to reproduce the numbers above:

- Three standalone runners (`runner-ours`, `runner-apache-rust`, `runner-cpp`) that emit one shared CSV schema over identical datasets.
- A Python reporter that prints comparison tables, renders the Tahoma-styled matplotlib plots shown above, and enforces a CI accuracy gate against per-sketch thresholds.
- A multi-trial RMSE mode that is the only accuracy comparison we treat as meaningful.

```bash
# Multi-trial RMSE comparison (ours vs apache-rust vs apache-cpp)
make -C benchmarks rmse

# Single-run comparison table plus throughput, memory and accuracy plots
make -C benchmarks report

# Accuracy gate (used in CI)
make -C benchmarks gate
```

### When to use each

**Choose this library for:**

- **Algorithm diversity**: sampling, frequency estimation, specialised sketches alongside the distinct counters.
- **Accuracy**: HLL, Theta and CPC are at parity-or-better than Apache DataSketches on multi-trial RMSE.
- **Memory safety**: Rust eliminates segfaults and memory leaks.

**Choose Apache DataSketches for:**

- **Byte-compatible interchange**: if you need to read or write the official DataSketches serialisation format, which this library deliberately does not.
- **Ecosystem integration**: proven stability and broad language bindings across the Apache ecosystem.

## TPC-H Business Intelligence Benchmarks

**Real-world performance analysis with 6M+ business records**

Comprehensive benchmarking against actual TPC-H business data to demonstrate realistic performance characteristics:

```bash
# Run the TPC-H performance analysis notebook
pytest --nbmake examples/tpch_performance_analysis.ipynb
```

**Key Business Intelligence Queries Tested:**

- **Distinct customers placing orders** (1.5M records)
- **Unique parts sold** (50K lineitem records)
- **Orders with line items** (distinct order counting)

**Findings with Real Data:**

- Low error on distinct-count business metrics (for example, CPC reports around 1.17% relative error on a real TPC-H column).
- Bounded, fixed-size memory regardless of stream length, versus the linear growth of exact counting.
- Scalability across cardinalities from thousands to millions of items.

**[TPC-H Analysis Notebook](examples/tpch_performance_analysis.ipynb)**

The notebook demonstrates practical business value including distinct counting for customer analytics, inventory management, and order processing with realistic error bounds and performance characteristics.

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
