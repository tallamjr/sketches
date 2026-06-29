# Usage guide

[Back to the README](../README.md)

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
    LinearCounter, HybridCounter,
    HllSketchMode, HllUnion
)

# Membership testing
from sketches import BloomFilter, CountingBloomFilter

# Frequency estimation
from sketches import (
    CountMinSketch, CountSketch,
    FrequentStringsSketch
)

# Quantile estimation
from sketches import KllSketch, TDigest, StreamingTDigest, ReqSketch

# Sampling
from sketches import (
    ReservoirSamplerR, ReservoirSamplerA,
    WeightedReservoirSampler, StreamSampler,
    VarOptSketch
)

# Multi-dimensional
from sketches import AodSketch, TupleSketch
```

### Business Intelligence Examples

For comprehensive real-world usage patterns, see the **TPC-H Performance Analysis Notebook**:

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

# Initialise HLL sketch
# lg_k=10: 1024 registers, ~6 KB,  ~3.2% error (fast, small)
# lg_k=12: 4096 registers, ~24 KB, ~1.6% error (default, good balance)
# lg_k=14: 16384 registers, ~96 KB, ~0.8% error (high accuracy)
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

# KLL Sketch for quantiles (provable error bounds)
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

### Linear and Hybrid Counter Example

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
