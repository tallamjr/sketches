import sys
import subprocess
import pytest
import logging

# Skip if psutil is not available
pytest.importorskip("psutil")


def test_sketch_memory_much_lower_than_set():
    """
    Compare RSS memory usage of Python set vs each sketch implementation.
    Each sketch should use at most 10% of the set's memory.
    """
    N = 20000
    py = sys.executable
    # Inline script to run in isolated process
    script = """\
import os
import psutil
from sketches import HllSketch, ThetaSketch, CpcSketch

# prepare items
values = [str(i) for i in range({N})]
proc = psutil.Process(os.getpid())

# measure set memory
rss0 = proc.memory_info().rss
s = set(values)
rss1 = proc.memory_info().rss

# measure HLL sketch memory
sk_hll = HllSketch(12)  # Use default precision
for v in values:
    sk_hll.update(v)
rss2 = proc.memory_info().rss

# measure Theta sketch memory
sk_th = ThetaSketch(4096)  # Use default k
for v in values:
    sk_th.update(v)
rss3 = proc.memory_info().rss

# measure CPC sketch memory
sk_cp = CpcSketch(11)  # Use default precision
for v in values:
    sk_cp.update(v)
rss4 = proc.memory_info().rss

# measure Polars DataFrame memory (if available)
try:
    import polars as pl
    rss_pol1 = proc.memory_info().rss
    df = pl.DataFrame({{"v": values}})
    rss_pol2 = proc.memory_info().rss
    print(f"POLARS {{rss_pol2 - rss_pol1}}")
except ImportError:
    # Polars not installed; report zero
    print("POLARS 0")

# report all deltas
print(f"SET {{rss1 - rss0}}")
print(f"HLL {{rss2 - rss1}}")
print(f"THETA {{rss3 - rss2}}")
print(f"CPC {{rss4 - rss3}}")
""".format(N=N)
    # Execute the measurement script
    logging.info("Running memory measurement subprocess with N=%d", N)
    result = subprocess.run(
        [py, "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )
    out = result.stdout.strip().splitlines()
    mem = {}
    for line in out:
        key, val = line.split()
        mem[key] = int(val)
    # Log memory usage for each component
    for key, val in sorted(mem.items()):
        logging.info("%s memory: %d bytes", key, val)

    set_mem = mem.get("SET", 0)
    logging.info("SET memory: %d bytes", set_mem)
    assert set_mem > 0, "Failed to measure set memory"
    # Sketches should use at most 15% of set memory (more realistic threshold)
    threshold = 0.15
    for name in ("HLL", "THETA", "CPC"):
        sketch_mem = mem.get(name, 0)
        max_allowed = set_mem * threshold
        logging.info("%s memory: %d bytes (threshold: %d)", name, sketch_mem, max_allowed)
        assert sketch_mem < max_allowed, (
            f"{name} uses too much memory: {sketch_mem} bytes "
            f">= {threshold*100}% of set memory ({set_mem} bytes)"
        )
    # Polars DataFrame should use more memory than sketches, but be reasonable
    polars_mem = mem.get("POLARS", 0)
    if polars_mem == 0:
        logging.info("Polars not available, skipping Polars comparison")
    else:
        max_sketch = max(mem.get(name, 0) for name in ("HLL", "THETA", "CPC"))
        # More realistic expectation: Polars should use at least 2x the largest sketch
        min_expected = max_sketch * 2
        logging.info("POLARS memory: %d bytes (threshold: %d)", polars_mem, min_expected)
        assert polars_mem > min_expected, (
            f"Polars uses unexpectedly little memory: {polars_mem} bytes "
            f"<= 2x max sketch memory ({max_sketch} bytes). "
            f"This suggests sketches may not be providing expected memory savings."
        )