# `sketches`

Python bindings for Rust-based data sketch algorithms (CPC, HLL, Theta) via PyO3.

> [!Note]
>
> This project layout is inspired by the Polars project. I thought a mini
> project on probabilistic data structures would be cool way to play around with
> using performant Rust but with a nice Python feel — let's see how things turn
> out. Things will change, things will break, we're here just having fun
> _#forthevibes_

## Features

- **CpcSketch**: Compressed Counting Sketch for cardinality estimation.
- **HllSketch**: HyperLogLog sketch for cardinality estimation.
- **ThetaSketch**: K-minimum values sketch (Theta Sketch) supporting union, intersection, difference.

## Background: Probabilistic Data Structures

Probabilistic data structures such as HyperLogLog (HLL), Compressed Counting (CPC)
sketches, and Theta sketches provide approximate answers (e.g., cardinality
estimates) while using significantly less memory compared to exact methods.
For example, to count the number of unique elements in a dataset of millions of
items, a conventional approach (e.g., using a hash set or a DataFrame’s unique
operation) must store every unique value in memory, resulting in O(N) space.
In contrast, an HLL sketch uses a fixed-size array of registers (2^k registers,
each a few bits), requiring only O(2^k) space, independent of N. With k = 12
(the default in this library), HLL needs just 4096 registers (≈3 KB of memory)
yet can estimate cardinalities of millions of items with only a few percent error.

### A Cardinality Conundrum

Imagine you’re the DBA for a high-traffic website tracking unique visitors (by IP)
every month. If 1.44 billion visits happen with an average of 10 pages each, that’s
~12 billion rows of IPs. Checking uniqueness exactly (sorting or hashing all) would
gobble up hundreds of gigabytes of RAM and take ages—yet we only need an estimate.

Enter **HyperLogLog**. It treats the input as a stream of hashed values and
records only a tiny “sketch” of the data. By observing _leading-zero patterns_
in those hash values (a rare long run of zeros suggests many distinct inputs),
HLL collects these patterns across many buckets and applies a harmonic-mean
formula (with bias correction) to deliver an approximate count. The magic: **HLL
uses tiny memory** (e.g., 12 KB for 4096 counters) and still achieves ~1% error.

> **Tip:** HLL excels when you need a fast, memory-frugal answer and can tolerate
> a small error (e.g., ±2%). It’s _much_ cheaper than exact counting at Big Data
> scales, and many systems (Snowflake, Presto/Trino, Redis’s `PFCOUNT`,
> PostgreSQL’s `hyperloglog` extension) bake HLL directly into their engines.

### Database Superpowers: Query Planning & GROUP BY

Approximate distinct counts guide query planners to choose efficient execution
strategies. For example, most SQL engines must decide between a hash-based
aggregation (fast but memory heavy) and a pipelined sort/group (low memory but
requires sorted input). A wrong guess by orders of magnitude wastes resources.
By feeding HLL-based estimates (e.g., “This group has ~10 million unique values”)
into the optimizer, systems like Vertica, PostgreSQL, and Snowflake select
better plans and avoid costly spills or full-table scans.

### How HLL Works at a Glance

- Hash each input to a 64-bit value.
- Use the first p bits to select one of 2^p registers.
- Count leading zeros in the remaining bits (plus one) as the “rank.”
- Store the maximum rank per register.
- Estimate cardinality via a bias-corrected harmonic mean across registers.

### A Rust Example: `hll::HllSketch` in Action

```rust
use hll::HllSketch;

// Create a new HyperLogLog with precision p = 12 (4096 registers)
let mut hll = HllSketch::new(12);

// Add elements (hashing is internal)
for id in 1u64..100_000u64 {
    hll.update(&id);
}

// Merge another sketch from a different partition
let mut hll2 = HllSketch::new(12);
for id in 50_000u64..150_000u64 {
    hll2.update(&id);
}
hll.merge(&hll2);

// Get the approximate unique count
let estimate: f64 = hll.estimate();
println!("Estimated unique elements: {:.0}", estimate);
```

### HyperLogLog++: Beyond the Basics

HLL++ refines the original algorithm with:

- **64-bit hashes** to reduce collisions at massive scales.
- **Improved bias correction** for small cardinalities (linear counting switch).
- **Sparse representation** for compact storage when few registers are non-zero.

This yields higher accuracy (error ∼0.5%) and graceful scaling from tiny to
trillion-element workloads.

### Implications & the Big Picture

HyperLogLog sketches let Big Data systems _reason about size cheaply_. Whether you’re
building a Rust microservice with the `hll` crate or a Python app using this
library’s `HllSketch`, you can sketch your data, merge across partitions, and get
fast, memory-efficient distinct counts—trading a dash of accuracy for massive speed
and scale.

## Memory Usage Comparison

The following Python code illustrates the dramatic memory savings when using an
HLL sketch instead of an exact method (e.g., a Python set or Polars unique) for
counting unique values in a large dataset. It uses `psutil` to measure process
memory before and after operations:

```python
# pip install psutil
import os
import psutil
import polars as pl
from sketches import HllSketch

process = psutil.Process(os.getpid())

# Generate a DataFrame with 1 million integer IDs
df = pl.DataFrame({"id": range(1_000_000)})

# Measure exact unique count using a Python set
values = df["id"].to_list()
start = process.memory_info().rss
unique_set = set(values)
set_mem = process.memory_info().rss - start
print(f"Memory used by Python set: {set_mem / (1024**2):.2f} MB")

# Measure memory for HLL sketch (default lg_k=12)
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

This example will typically show tens of megabytes for the Python set versus
just a few kilobytes for the HLL sketch, showcasing the memory efficiency of
probabilistic data structures.

## Installation

### Prerequisites

- Python 3.10 or higher.
- Rust toolchain (rustc and cargo). Install from https://rustup.rs.
- Optionally, for DataFrame examples: `polars` (`pip install polars`).

- Optionally, for memory measurement examples: `psutil` (`pip install psutil`).

### From PyPI (if available)

```bash
pip install sketches
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

## Usage

Import the module:

```python
from sketches import CpcSketch, HllSketch, ThetaSketch
```

### HLL Sketch Example

```python
from sketches import HllSketch

# Initialise HLL sketch (default lg_k=12)
sketch = HllSketch()

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
sketch = HllSketch()
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

s1 = ThetaSketch()
s2 = ThetaSketch()

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
s1 = ThetaSketch()
s2 = ThetaSketch()
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

## License

This project is licensed under the Apache-2.0 License (see `pyproject.toml`).
