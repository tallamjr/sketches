# 🎯 `sketches` - High-Performance Probabilistic Data Structures

[![Rust](https://img.shields.io/badge/rust-1.86%2B-orange.svg)](https://www.rust-lang.org)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)

**Fast, memory-efficient probabilistic data structures for streaming analytics, cardinality estimation, quantile computation, and sampling.**

Python bindings for Rust-based implementations of HyperLogLog, T-Digest, Reservoir Sampling, and more via PyO3.

> [!Note]
>
> This project layout is inspired by the Polars project. A high-performance exploration of probabilistic data structures
> using performant Rust with a Python-friendly interface. Built for production use with comprehensive algorithm implementations.
>
> 📖 **[Deep Algorithm Comparison →](ALGORITHMS.md)**

## Features

| **Algorithm Category**     | **Implementation** | **Description**                                               | **Status** |
| -------------------------- | ------------------ | ------------------------------------------------------------- | ---------- |
| **Cardinality Estimation** | HyperLogLog (HLL)  | Industry-standard distinct counting with ~1% error            | ✅         |
|                            | HyperLogLog++      | Enhanced HLL with bias correction and sparse mode             | ✅         |
|                            | CPC Sketch         | Most compact serialization for network transfer               | ✅         |
|                            | Linear Counter     | Optimal for small cardinalities (n < 1000)                    | ✅         |
|                            | Hybrid Counter     | Auto-transitions from Linear → HLL                            | ✅         |
| **Set Operations**         | Theta Sketch       | Union, intersection, difference with cardinality estimation   | ✅         |
| **Sampling**               | Algorithm R        | Basic reservoir sampling for uniform random samples           | ✅         |
|                            | Algorithm A        | Optimized reservoir sampling (19x faster for large streams)   | ✅         |
|                            | Weighted Sampling  | Probability-proportional reservoir sampling                   | ✅         |
|                            | Stream Sampling    | High-throughput sampling with batching                        | ✅         |
| **Quantile Estimation**    | T-Digest           | Superior accuracy for extreme quantiles (p95, p99)            | ✅         |
|                            | KLL Sketch         | Simplified implementation (~20-30% error bounds)              | ✅         |
| **Frequency Estimation**   | Count-Min Sketch   | Conservative frequency estimation with ε-δ guarantees         | ✅         |
|                            | Count Sketch       | Unbiased frequency estimation using median                    | ✅         |
|                            | Frequent Items     | Top-K heavy hitters with Space-Saving algorithm               | ✅         |
| **Membership Testing**     | Bloom Filter       | Fast membership testing with configurable false positive rate | ✅         |
|                            | Counting Bloom     | Bloom filter with deletion support                            | ✅         |
| **Multi-dimensional**      | Array of Doubles   | Tuple sketch for multi-dimensional aggregation                | ✅         |

## 🚀 Performance Benchmarks

**Rigorous comparison against Apache DataSketches (industry standard)**

We conducted comprehensive benchmarks comparing our Rust-based implementation with the official Apache DataSketches Python library across key performance metrics:

### Processing Throughput

```
HyperLogLog Updates (2M items):
┌─────────────────────┬──────────────┬─────────────────┬──────────┐
│ Implementation      │ Time         │ Throughput      │ Ratio    │
├─────────────────────┼──────────────┼─────────────────┼──────────┤
│ Apache DataSketches │ 0.29s        │ 7.1M items/sec │ 5.2x     │
│ Our Library         │ 1.51s        │ 1.3M items/sec │ baseline │
└─────────────────────┴──────────────┴─────────────────┴──────────┘
```

### Memory Efficiency

```
HyperLogLog Memory Usage (1M items):
┌─────────────────────┬──────────────┬─────────────────┐
│ Implementation      │ Memory Usage │ Efficiency      │
├─────────────────────┼──────────────┼─────────────────┤
│ Apache DataSketches │ 32 KB        │ 9x better       │
│ Our Library         │ 288 KB       │ baseline        │
└─────────────────────┴──────────────┴─────────────────┘
```

### Accuracy Comparison

```
HyperLogLog Error Rates:
┌──────────────┬─────────────┬─────────────┬──────────┐
│ Dataset Size │ Our Error   │ Apache Error│ Winner   │
├──────────────┼─────────────┼─────────────┼──────────┤
│ 1,000        │ 0.22%       │ 0.72%       │ Ours ✅   │
│ 10,000       │ 2.48%       │ 0.72%       │ Apache   │
│ 100,000      │ 1.27%       │ 1.23%       │ Tie      │
│ 1,000,000    │ 1.77%       │ 1.14%       │ Apache   │
└──────────────┴─────────────┴─────────────┴──────────┘
```

### Key Insights

**🎯 Accuracy**: Both libraries achieve excellent <3% error rates
**⚡ Speed**: Apache DataSketches is 5x faster (optimized C++ core)
**💾 Memory**: Apache DataSketches uses 9x less memory
**🔧 Features**: Our library provides 2x more algorithms (18 vs 9)
**🛡️ Safety**: Rust guarantees memory safety and eliminates entire bug classes

### When to Use Each

**Choose Our Library For:**

- **Algorithm diversity** - sampling, frequency estimation, specialized sketches
- **Rich analytics** - confidence bounds, statistics, merging operations
- **Memory safety** - Rust eliminates segfaults and memory leaks
- **Modern development** - excellent type safety and error messages

**Choose Apache DataSketches For:**

- **Maximum performance** - 5x faster processing
- **Memory constraints** - 9x lower memory usage
- **Production scale** - billions of items daily
- **Enterprise deployment** - proven stability

## 🚀 Optimization Roadmap

**Current Status:** Apache DataSketches leads with 5x throughput and 9x memory efficiency  
**Target:** Match or exceed Apache DataSketches performance across all metrics

### Phase 1: Memory Optimization (Target: 85% reduction)
- **Sparse Mode**: HashMap storage for <1K items → 90% memory reduction
- **Bit-Packed Storage**: 6-bit registers instead of 8-bit → 25% reduction
- **Custom Allocators**: jemalloc and object pooling → 15% performance boost

### Phase 2: SIMD Implementation (Target: 300% throughput)
- **AVX2/NEON Vectorization**: Process 8 items simultaneously
- **Vectorized Estimation**: Parallel 2^(-rho) computation
- **SIMD Hash Functions**: Batch string processing

### Phase 3: Algorithm Optimizations (Target: 20% improvement)
- **HLL++ Bias Correction**: Pre-computed correction tables
- **Branch-Free Operations**: Eliminate conditionals in hot paths
- **Cache Optimization**: Aligned data structures and prefetching

**Performance Targets:**
- Throughput: 15-25M items/sec (2-3x faster than Apache DataSketches)
- Memory: 25-30KB (90% reduction, matching Apache DataSketches)
- Accuracy: <1% error consistently

Both libraries excel at their core mission: enabling approximate analytics on massive datasets with bounded memory and excellent accuracy.

## 📊 TPC-H Business Intelligence Benchmarks

**Real-world performance analysis with 6M+ business records**

We provide comprehensive benchmarking against actual TPC-H business data to demonstrate realistic performance characteristics:

```bash
# Run the TPC-H performance analysis notebook
pytest --nbmake examples/tpch_performance_analysis.ipynb
```

**Key Business Intelligence Queries Tested:**

- **Distinct customers placing orders** (1.5M records)
- **Unique parts sold** (50K lineitem records)
- **Orders with line items** (distinct order counting)

**Performance Highlights with Real Data:**

- **Sub-5% error rates** on critical business metrics
- **1000x+ memory efficiency** vs traditional exact counting
- **Millions of items/sec throughput** for streaming analytics
- **Scalability analysis** from 1K to 50K+ items

> 📈 **[Interactive TPC-H Analysis →](examples/tpch_performance_analysis.ipynb)**

The notebook demonstrates practical business value including distinct counting for customer analytics, inventory management, and order processing with realistic error bounds and performance characteristics.

## Table of Contents

<!-- mtoc-start -->

* [Background: Probabilistic Data Structures](#background-probabilistic-data-structures)
  * [The Cardinality Conundrum](#the-cardinality-conundrum)
  * [Database Superpowers: Query Planning & `GROUP BY` Operations](#database-superpowers-query-planning--group-by-operations)
  * [How HLL Works at a Glance](#how-hll-works-at-a-glance)
  * [Implications & the Big Picture](#implications--the-big-picture)
* [Memory Usage Comparison](#memory-usage-comparison)
* [Package Installation](#package-installation)
  * [Prerequisites](#prerequisites)
  * [From PyPI (if available)](#from-pypi-if-available)
  * [From Source](#from-source)
* [Getting Started](#getting-started)
  * [Quick Installation](#quick-installation)
  * [Development Workflow](#development-workflow)
* [Library Usage](#library-usage)
  * [🎯 Business Intelligence Examples](#-business-intelligence-examples)
  * [HLL Sketch Example](#hll-sketch-example)
    * [Minimal Test with Polars](#minimal-test-with-polars)
  * [CPC Sketch Example](#cpc-sketch-example)
    * [Minimal Test with Polars](#minimal-test-with-polars-1)
  * [Bloom Filter Example](#bloom-filter-example)
  * [Frequency Estimation Example](#frequency-estimation-example)
  * [Quantile Estimation Example](#quantile-estimation-example)
  * [Sampling Example](#sampling-example)
  * [Linear & Hybrid Counter Example](#linear--hybrid-counter-example)
  * [Array of Doubles (AOD) Sketch Example](#array-of-doubles-aod-sketch-example)
* [Extending HLL++: Sparse Buffer, Variable-Length Encoding, and Hybrid Representation](#extending-hll-sparse-buffer-variable-length-encoding-and-hybrid-representation)
  * [Theta Sketch Example](#theta-sketch-example)
    * [Minimal Test with Polars](#minimal-test-with-polars-2)
* [Roadmap & Missing Features](#roadmap--missing-features)
  * [Not Yet Implemented](#not-yet-implemented)
  * [Current Development Priorities](#current-development-priorities)
* [License](#license)

<!-- mtoc-end -->

## Background: Probabilistic Data Structures

Probabilistic data structures such as HyperLogLog (HLL), Compressed Counting (CPC)
sketches, and Theta sketches provide approximate answers (e.g., cardinality
estimates) while using significantly less memory compared to exact methods.

For example, to count the number of unique elements in a dataset of millions of
items, a conventional approach (e.g., using a hash set or a DataFrame’s unique
operation) must store every unique value in memory, resulting in $O(N)$ space.

In contrast, an HLL sketch uses a fixed-size array of registers ($2^k$ registers,
each a few bits), requiring only $O(2^k)$ space, independent of $N$. With $k = 12$
(the default in this library), HLL needs just 4096 registers (≈3 KB of memory)
yet can estimate cardinalities of millions of items with only a few percent error.

### The Cardinality Conundrum

Imagine you’re the DBA for a high-traffic website tracking unique visitors (by IP)
every month. If 1.44 billion visits happen with an average of 10 pages each, that’s
~12 billion rows of IPs. Checking uniqueness exactly (sorting or hashing all) would
gobble up hundreds of gigabytes of RAM and take ages—yet we only need an estimate.

Enter **HyperLogLog**. It treats the input as a _stream_ of hashed values and
records only a tiny **"sketch"** of the data. By observing _leading-zero patterns_
in those hash values (a rare long run of zeros suggests many distinct inputs),
HLL collects these patterns across many "buckets" and applies a Harmonic-Mean
formula (with bias correction) to deliver an approximate count.

Why is this so cool? It uses a tiny amount of memory (e.g., 12 KB for 4096
counters) and still achieves ~1\% error! I know, awesome right?!

HLL excels when you need a fast, memory-frugal answer and can tolerate a small
error (e.g., ±2%). It’s _much_ cheaper than exact counting at Big Data scales,
and many systems
([Trino](https://trino.io/docs/current/functions/hyperloglog.html), [Redis’s
`PFCOUNT`](https://redis.io/docs/latest/develop/data-types/probabilistic/hyperloglogs/),
[PostgreSQL’s
`hyperloglog`](https://github.com/postgres/postgres/blob/master/src/backend/lib/hyperloglog.c)
extension) bake HLL directly into their engines.

### Database Superpowers: Query Planning & `GROUP BY` Operations

Approximate distinct counts guide query planners to choose efficient execution
strategies. For example, most SQL engines must decide between a hash-based
aggregation (fast but memory heavy) and a pipelined sort/group (low memory but
requires sorted input). A wrong guess by orders of magnitude wastes resources.

By feeding HLL-based estimates (e.g., "This group has ~10 million unique values")
into the optimiser, systems like [Vertica](https://en.wikipedia.org/wiki/Vertica), PostgreSQL, and Snowflake select
better plans and avoid costly spills to disk or full-table scans.

### How HLL Works at a Glance

1. Hash each input to a `32-bit` value.
2. Use the first $p$ bits to select one of $2^p$ registers.
3. Count leading zeros in the remaining bits (plus one) as the "rank".
4. Store the maximum rank per register.
5. Estimate cardinality via a bias-corrected harmonic mean across registers.

This was taken further by [Stefan Heule et. al](https://research.google/pubs/hyperloglog-in-practice-algorithmic-engineering-of-a-state-of-the-art-cardinality-estimation-algorithm/)
who introduced the HyperLogLog++ algorithm.

HLL++ refines the original algorithm with:

- **64-bit hashes** to reduce collisions at massive scales.
- **Improved bias correction** for small cardinalities (linear counting switch).
- **Sparse representation** for compact storage when few registers are non-zero.

This yields higher accuracy (error ∼0.5%) and graceful scaling from tiny to
trillion-element workloads.

### Implications & the Big Picture

HyperLogLog sketches let Big Data systems _reason about size cheaply_ and one
can sketch data, merge across partitions, and get fast, memory-efficient
distinct counts—trading a dash of accuracy for massive speed and scale.

## Memory Usage Comparison

The following code, which uses the `sketches` library implemented in this repo,
illustrates the memory savings when using an HLL sketch instead of an exact
method (e.g., a Python set or Polars unique) for counting unique values in a
large dataset. It uses `psutil` to measure process memory before and after
operations:

```python
# pip install psutil
import os
import psutil
import polars as pl
from sketches import HllSketch

process = psutil.Process(os.getpid())

# Generate a DataFrame with 100 million integer IDs
df = pl.DataFrame({"id": range(100_000_000)})

# Measure exact unique count using a Python set
values = df["id"].to_list()
start = process.memory_info().rss
unique_set = set(values)
set_mem = process.memory_info().rss - start
print(f"Memory used by Python set: {set_mem / (1024**2):.2f} MB")

# Measure memory for HLL sketch (lg_k=12 for 4096 buckets)
start = process.memory_info().rss
sketch = HllSketch(lg_k=12)
for v in values:
    sketch.update(str(v))
hll_mem = process.memory_info().rss - start
print(f"Memory used by HLL sketch: {hll_mem / 1024:.2f} KB")

# Optional: verify counts
print(f"Exact unique: {len(unique_set)}")
print(f"HLL estimate: {sketch.estimate():.2f}")
```

```
$ python memtest.py
Memory used by Python set: 2445.36 MB
Memory used by HLL sketch: 192.00 KB  # <--- 👀
Exact unique: 100000000
HLL estimate: 98559344.17

```

This example will typically show tens of megabytes for the Python set versus
just a few kilobytes for the HLL sketch, showcasing the memory efficiency of
probabilistic data structures.

## Package Installation

### Prerequisites

- Python 3.10+
- Optionally, for DataFrame examples: `polars` (`pip install polars`).

- Optionally, for memory measurement examples: `psutil` (`pip install psutil`).

### From PyPI (if available)

```bash
pip install rusty-sketches
```

### From Source

```bash
git clone https://github.com/tallamjr/sketches.git
cd sketches/py-sketches
pip install .
```

For an editable install with development dependencies:

```bash
pip install -e .[dev]
```

**For TPC-H Performance Analysis Notebook:**

```bash
# Install with visualization dependencies
pip install -e .[dev]  # includes seaborn, matplotlib, pandas

# Run the comprehensive business intelligence benchmarks
pytest --nbmake examples/tpch_performance_analysis.ipynb
```

## Getting Started

### Quick Installation

```bash
# Install from source with development dependencies
pip install -e .[dev]

# Or just install the package
pip install -e .
```

### Development Workflow

```bash
# Build Python extension
maturin develop

# Run Python tests
pytest

# Run Rust tests
cargo test

# Format code
cargo fmt && black .
```

## Library Usage

Import the available sketches:

```python
# Cardinality estimation
from sketches import (
    HllSketch, HllPlusPlusSketch, HllPlusPlusSparseSketch,
    CpcSketch, ThetaSketch,
    LinearCounter, HybridCounter
)

# Membership testing
from sketches import BloomFilter, CountingBloomFilter

# Frequency estimation
from sketches import (
    CountMinSketch, CountSketch,
    FrequentStringsSketch
)

# Quantile estimation
from sketches import KllSketch, TDigest, StreamingTDigest

# Sampling
from sketches import (
    ReservoirSamplerR, ReservoirSamplerA,
    WeightedReservoirSampler, StreamSampler
)

# Multi-dimensional
from sketches import AodSketch
```

### 🎯 Business Intelligence Examples

For comprehensive real-world usage patterns, see our **TPC-H Performance Analysis Notebook**:

- **`examples/tpch_performance_analysis.ipynb`** - Complete BI analysis with 6M+ records
- **Business queries**: Customer counting, inventory analysis, order processing
- **Performance comparisons**: HLL vs Theta vs CPC across different data sizes
- **Memory analysis**: Sketch efficiency vs exact counting approaches
- **Scalability testing**: Performance from 1K to 50K+ items

```bash
# Interactive exploration of business analytics use cases
jupyter notebook examples/tpch_performance_analysis.ipynb
```

### HLL Sketch Example

```python
from sketches import HllSketch

# Initialise HLL sketch (lg_k=12 for 4096 buckets)
sketch = HllSketch(lg_k=12)

# Add items
for item in ["apple", "banana", "orange", "apple"]:
    sketch.update(item)

# Estimate cardinality
estimate = sketch.estimate()
print(f"Estimated unique items: {estimate:.2f}")
```

#### Minimal Test with Polars

This example reads a CSV file from `tests/data` using [polars] and compares the actual number of unique elements in a column to the sketch estimate.

```python
import polars as pl
from sketches import HllSketch

# Read data (adjust the path as needed)
df = pl.read_csv("tests/data/customer.csv")

# Select column and cast to string
column = "c_custkey"
values = df[column].cast(str).to_list()

# Actual unique count
actual = df[column].n_unique()

# Create and populate the sketch
sketch = HllSketch(lg_k=12)
for v in values:
    sketch.update(v)

# Estimate
estimate = sketch.estimate()

print(f"Actual unique `{column}` values: {actual}")
print(f"Estimated unique values (HLL): {estimate:.2f}")
# Actual unique `c_custkey` values: 150000
# Estimated unique values (HLL): 147364.41
```

### CPC Sketch Example

```python
from sketches import CpcSketch

sketch = CpcSketch(lg_k=11)
for i in range(1000):
    sketch.update(str(i))
print(f"Estimated cardinality (CPC): {sketch.estimate():.2f}")
```

#### Minimal Test with Polars

This example reads a CSV file from `tests/data` using [polars] and compares the actual number of unique elements in a column to the sketch estimate.

```python
import polars as pl
from sketches import CpcSketch

# Read data (adjust the path as needed)
df = pl.read_csv("tests/data/customer.csv")

# Select column and cast to string
column = "c_custkey"
values = df[column].cast(str).to_list()

# Actual unique count
actual = df[column].n_unique()

# Create and populate the sketch
sketch = CpcSketch(lg_k=11)
for v in values:
    sketch.update(v)

# Estimate
estimate = sketch.estimate()

print(f"Actual unique `{column}` values: {actual}")
print(f"Estimated unique values (CPC): {estimate:.2f}")
  # Actual unique `c_custkey` values: 150000
  # Estimated unique values (CPC): 142882.05
```

### Bloom Filter Example

```python
from sketches import BloomFilter, CountingBloomFilter

# Standard Bloom Filter
bloom = BloomFilter(capacity=100000, error_rate=0.01)
bloom.add("apple")
bloom.add("banana")

print(bloom.contains("apple"))  # True
print(bloom.contains("orange"))  # False (probably)

# Counting Bloom Filter (supports deletion)
counting_bloom = CountingBloomFilter(capacity=100000, error_rate=0.01)
counting_bloom.add("apple")
counting_bloom.add("apple")
counting_bloom.remove("apple")
print(counting_bloom.contains("apple"))  # Still True (added twice, removed once)
```

### Frequency Estimation Example

```python
from sketches import CountMinSketch, FrequentStringsSketch

# Count-Min Sketch for frequency estimation
cm_sketch = CountMinSketch.with_error_bounds(epsilon=0.001, delta=0.01)
for word in ["the", "quick", "brown", "fox", "the", "the"]:
    cm_sketch.increment(word)

print(f"Frequency of 'the': {cm_sketch.estimate('the')}")  # ~3

# Frequent Items (Heavy Hitters)
freq_sketch = FrequentStringsSketch.with_error_rate(0.001, 0.99)
for item in data_stream:
    freq_sketch.update(item)

# Get top-10 most frequent items
top_items = freq_sketch.get_top_k(10)
for item, estimate, lower, upper in top_items:
    print(f"{item}: {estimate} (bounds: {lower}-{upper})")
```

### Quantile Estimation Example

```python
from sketches import KllSketch, TDigest

# KLL Sketch for quantiles (simplified implementation with ~20-30% error bounds)
kll = KllSketch.with_accuracy(epsilon=0.25, confidence=0.8)
for value in data_stream:
    kll.update(value)

print(f"Median: {kll.median()}")
print(f"95th percentile: {kll.q95()}")
print(f"99th percentile: {kll.q99()}")

# T-Digest for superior extreme quantile accuracy
tdigest = TDigest.with_accuracy(0.01)
for value in data_stream:
    tdigest.add(value)

print(f"99.9th percentile: {tdigest.p999()}")  # Very accurate for extremes
print(f"Trimmed mean (10%-90%): {tdigest.trimmed_mean(0.1, 0.9)}")
```

### Sampling Example

```python
from sketches import ReservoirSamplerR, WeightedReservoirSampler

# Uniform sampling with Algorithm R
sampler = ReservoirSamplerR(capacity=1000)
for item in large_stream:
    sampler.add(item)

sample = sampler.sample()  # 1000 uniformly sampled items

# Weighted sampling (probability proportional to weight)
weighted_sampler = WeightedReservoirSampler(capacity=100)
weighted_sampler.add_weighted("important", weight=10.0)
weighted_sampler.add_weighted("normal", weight=1.0)

weighted_sample = weighted_sampler.sample_with_weights()
```

### Linear & Hybrid Counter Example

```python
from sketches import LinearCounter, HybridCounter

# Linear Counter - optimal for small cardinalities
linear = LinearCounter.with_expected_cardinality(1000, error_rate=0.01)
for item in small_dataset:
    linear.update(item)

print(f"Estimate: {linear.estimate()}")
print(f"Should switch to HLL: {linear.should_transition_to_hll()}")

# Hybrid Counter - automatically transitions from Linear to HLL
hybrid = HybridCounter.with_range(max_expected_cardinality=1_000_000)
for item in growing_dataset:
    hybrid.update(item)

print(f"Mode: {hybrid.mode()}")  # "Linear" or "HyperLogLog"
print(f"Estimate: {hybrid.estimate()}")
```

### Array of Doubles (AOD) Sketch Example

```python
from sketches import AodSketch

# Tuple sketch for multi-dimensional aggregation
aod = AodSketch(capacity=4096, num_values=3)

# Update with key and associated values
aod.update("user123", [1.0, 5.5, 3.2])  # e.g., [clicks, time_spent, purchases]
aod.update("user456", [2.0, 3.1, 1.0])

# Get cardinality estimate
print(f"Unique users: {aod.estimate():.0f}")

# Aggregate statistics
sums = aod.column_sums()  # [total_clicks, total_time, total_purchases]
means = aod.column_means()  # Average per user
```

## Extending HLL++: Sparse Buffer, Variable-Length Encoding, and Hybrid Representation

Beyond the built-in dense and simple sparse sketches, HLL++ can be optimised further:

- **Unsorted Insertion Buffer**: For high-throughput updates, buffer `(index, rank)` pairs in a small `Vec`, and flush into the main map once full.

  ```rust
  struct SparseBuffer {
      p: u8,
      buffer: Vec<(usize, u8)>,      // unsorted bucket updates
      map: BTreeMap<usize, u8>,      // current sparse registers
  }
  impl SparseBuffer {
      fn update<T: Hash>(&mut self, item: &T) {
          let hash = hash64(item);
          let idx = (hash >> (64 - self.p)) as usize;
          let rank = (hash << self.p).leading_zeros().saturating_add(1) as u8;
          self.buffer.push((idx, rank));
          if self.buffer.len() > self.buffer.capacity() {
              self.flush();
          }
      }
      fn flush(&mut self) {
          for (idx, rank) in self.buffer.drain(..) {
              let entry = self.map.entry(idx).or_insert(0);
              if *entry < rank { *entry = rank; }
          }
      }
  }
  ```

- **Variable-Length Encoding**: Compact sparse pairs into `u32` words `(idx<<6)|rank`, delta-sort, then LEB128 encode:

  ```rust
  fn pack(j: usize, r: u8) -> u32 { ((j as u32) << 6) | (r as u32) }
  let mut packed: Vec<u32> = map.iter().map(|(&j,&r)| pack(j,r)).collect();
  packed.sort_unstable();
  let mut bytes = Vec::new();
  let mut prev = 0;
  for v in packed {
      let delta = v.wrapping_sub(prev);
      leb128::write::unsigned(&mut bytes, delta as u128).unwrap();
      prev = v;
  }
  ```

- **Hybrid Sparse→Dense Switch**: Start in sparse mode; once `map.len() > m/2`, materialise a dense `Vec<u8>` and switch to `HllPlusPlusSketch` for O(1) updates.
  ```rust
  if sparse.map.len() > (1 << p) / 2 {
      let mut dense = HllPlusPlusSketch::new(p);
      for (&j,&r) in &sparse.map { dense.registers[j] = r; }
      // adopt `dense` for further updates...
  }
  ```

These extensions deliver fast, memory-efficient, and scalable HLL++ sketches across workloads.

### Theta Sketch Example

```python
from sketches import ThetaSketch

s1 = ThetaSketch(k=4096)
s2 = ThetaSketch(k=4096)

# Add items
for x in ["a", "b", "c"]:
    s1.update(x)
for x in ["b", "c", "d"]:
    s2.update(x)

# Union
union = s1.union(s2)
print(f"Estimated union size: {union.estimate():.2f}")

# Intersection
intersection = s1.intersect(s2)
print(f"Estimated intersection size: {intersection.estimate():.2f}")

# Difference
difference = s1.difference(s2)
print(f"Estimated difference size: {difference.estimate():.2f}")
```

#### Minimal Test with Polars

This example reads a CSV file from `tests/data` using [polars], partitions the data into two subsets, and compares the actual union, intersection, and difference cardinalities to the sketch estimates.

```python
import polars as pl
from sketches import ThetaSketch

# Read data (adjust the path as needed)
df = pl.read_csv("tests/data/customer.csv")

# Select column
column = "c_custkey"

# Partition data into two subsets (e.g., even and odd keys)
df1 = df.filter(pl.col(column) % 2 == 0)
df2 = df.filter(pl.col(column) % 2 != 0)

values1 = df1[column].cast(str).to_list()
values2 = df2[column].cast(str).to_list()

# Create and populate the sketches
s1 = ThetaSketch(k=4096)
s2 = ThetaSketch(k=4096)
for v in values1:
    s1.update(v)
for v in values2:
    s2.update(v)

# Actual counts using Polars
actual_union = pl.concat([df1.select(column), df2.select(column)]).unique().height
actual_intersection = df1.select(column).join(df2.select(column), on=column, how="inner").height
actual_difference = df1.select(column).join(df2.select(column), on=column, how="anti").height

# Sketch estimates
union = s1.union(s2)
intersection = s1.intersect(s2)
difference = s1.difference(s2)

# Print results
print(f"Actual union size: {actual_union}")
print(f"Estimated union size (Theta): {union.estimate():.2f}")
print(f"Actual intersection size: {actual_intersection}")
print(f"Estimated intersection size (Theta): {intersection.estimate():.2f}")
print(f"Actual difference size: {actual_difference}")
print(f"Estimated difference size (Theta): {difference.estimate():.2f}")
# Actual union size: 150000
# Estimated union size (Theta): 146064.88
# Actual intersection size: 0
# Estimated intersection size (Theta): 0.00
# Actual difference size: 75000
# Estimated difference size (Theta): 76308.22
```

## Roadmap & Missing Features

While this library provides a comprehensive suite of probabilistic data structures, the following features are planned but not yet implemented:

### Not Yet Implemented
- **Probabilistic Counter (Flajolet–Martin)** - Historical algorithm for educational purposes
- **SIMD Acceleration** - Framework is ready but actual AVX2/NEON implementations pending
- **GPU Acceleration** - Metal/CUDA kernels for massive parallel processing
- **Polars Integration** - Custom expressions and DataFrame operations
- **Advanced Serialization** - Network-optimized protocols beyond basic byte arrays

### Current Development Priorities
1. Actual SIMD implementations to replace scalar fallbacks
2. GPU acceleration for batch operations
3. Polars custom expressions for seamless DataFrame integration
4. Performance optimizations for Python bindings (batch operations)

See [TODO.md](TODO.md) for the complete development roadmap.

## License

This project is licensed under the MIT License (see `pyproject.toml`).
