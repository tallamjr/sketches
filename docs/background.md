# Background: probabilistic data structures

[Back to the README](../README.md)

## Background: Probabilistic Data Structures

Probabilistic data structures such as HyperLogLog (HLL), Compressed Counting (CPC)
sketches, and Theta sketches provide approximate answers (e.g., cardinality
estimates) while using significantly less memory compared to exact methods.

For example, to count the number of unique elements in a dataset of millions of
items, a conventional approach (e.g., using a hash set or a DataFrame's unique
operation) must store every unique value in memory, resulting in $O(N)$ space.

In contrast, an HLL sketch uses a fixed-size array of registers ($2^k$ registers,
each a few bits), requiring only $O(2^k)$ space, independent of $N$. With $k = 12$
(the default in this library), HLL needs just 4096 registers (approx. 3 KB of memory)
yet can estimate cardinalities of millions of items with only a few percent error.

### The Cardinality Conundrum

Imagine you are the DBA for a high-traffic website tracking unique visitors (by IP)
every month. If 1.44 billion visits happen with an average of 10 pages each, that is
~12 billion rows of IPs. Checking uniqueness exactly (sorting or hashing all) would
consume hundreds of gigabytes of RAM and take an impractical amount of time -- yet we
only need an estimate.

**HyperLogLog** treats the input as a _stream_ of hashed values and records only a
tiny **"sketch"** of the data. By observing _leading-zero patterns_ in those hash
values (a rare long run of zeros suggests many distinct inputs), HLL collects these
patterns across many "buckets" and applies a Harmonic-Mean formula (with bias
correction) to deliver an approximate count.

HLL uses a tiny amount of memory (e.g., 12 KB for 4096 counters) and still achieves
~1% error. It excels when you need a fast, memory-frugal answer and can tolerate a
small error (e.g., +/-2%). It is _much_ cheaper than exact counting at Big Data
scales, and many systems
([Trino](https://trino.io/docs/current/functions/hyperloglog.html), [Redis's
`PFCOUNT`](https://redis.io/docs/latest/develop/data-types/probabilistic/hyperloglogs/),
[PostgreSQL's
`hyperloglog`](https://github.com/postgres/postgres/blob/master/src/backend/lib/hyperloglog.c)
extension) bake HLL directly into their engines.

### Database Superpowers: Query Planning and GROUP BY Operations

Approximate distinct counts guide query planners to choose efficient execution
strategies. For example, most SQL engines must decide between a hash-based
aggregation (fast but memory heavy) and a pipelined sort/group (low memory but
requires sorted input). A wrong guess by orders of magnitude wastes resources.

By feeding HLL-based estimates (e.g., "This group has ~10 million unique values")
into the optimiser, systems like [Vertica](https://en.wikipedia.org/wiki/Vertica), PostgreSQL, and Snowflake select
better plans and avoid costly spills to disk or full-table scans.

### How HLL Works at a Glance

1. Hash each input to a `64-bit` value.
2. Use the first $p$ bits to select one of $2^p$ registers.
3. Count leading zeros in the remaining bits (plus one) as the "rank".
4. Store the maximum rank per register.
5. Estimate cardinality via a bias-corrected harmonic mean across registers.

This was taken further by [Stefan Heule et al.](https://research.google/pubs/hyperloglog-in-practice-algorithmic-engineering-of-a-state-of-the-art-cardinality-estimation-algorithm/)
who introduced the HyperLogLog++ algorithm.

HLL++ refines the original algorithm with:

- **64-bit hashes** to reduce collisions at massive scales.
- **Improved bias correction** for small cardinalities (linear counting switch).
- **Sparse representation** for compact storage when few registers are non-zero.

This yields higher accuracy (error ~0.5%) and graceful scaling from tiny to
trillion-element workloads.

### Implications and the Big Picture

HyperLogLog sketches let Big Data systems _reason about size cheaply_ and one
can sketch data, merge across partitions, and get fast, memory-efficient
distinct counts -- trading a dash of accuracy for massive speed and scale.

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
Memory used by HLL sketch: 192.00 KB
Exact unique: 100000000
HLL estimate: 98559344.17

```

This example will typically show tens of megabytes for the Python set versus
just a few kilobytes for the HLL sketch, showcasing the memory efficiency of
probabilistic data structures.
