import os

import polars as pl
import psutil
from sketches import HllSketch

process = psutil.Process(os.getpid())

# Generate a DataFrame with 1 million integer IDs
df = pl.DataFrame({"id": range(1_000_000_00)})

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
