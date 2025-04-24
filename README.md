# `sketches`

Python bindings for Rust-based data sketch algorithms (CPC, HLL, Theta) via PyO3.

## Features

- **CpcSketch**: Compressed Counting Sketch for cardinality estimation.
- **HllSketch**: HyperLogLog sketch for cardinality estimation.
- **ThetaSketch**: K-minimum values sketch (Theta Sketch) supporting union, intersection, difference.

## Background: Probabilistic Data Structures

Probabilistic data structures such as HyperLogLog (HLL), Compressed Counting
(CPC) sketches, and Theta sketches provide approximate answers (e.g.,
cardinality estimates) while using significantly less memory compared to exact
methods. For example, to count the number of unique elements in a dataset of
millions of items, a conventional approach (e.g., using a hash set or a
DataFrame’s unique operation) must store every unique value in memory, resulting
in O(N) space. In contrast, an HLL sketch uses a fixed-size array of registers
(2^k registers, each a few bits), requiring only O(2^k) space, independent of N.
With k = 12 (the default in this library), HLL needs just 4096 registers × 6
bits ≈ 3 KB of memory, yet can estimate cardinalities of millions of items with
only a few percent error.

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
