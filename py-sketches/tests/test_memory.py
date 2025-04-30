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
sk_hll = HllSketch()
for v in values:
    sk_hll.update(v)
rss2 = proc.memory_info().rss

# measure Theta sketch memory
sk_th = ThetaSketch()
for v in values:
    sk_th.update(v)
rss3 = proc.memory_info().rss

# measure CPC sketch memory
sk_cp = CpcSketch()
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
    # Sketches should use at most 10% of set memory
    for name in ("HLL", "THETA", "CPC"):
        sketch_mem = mem.get(name, 0)
        logging.info("%s memory: %d bytes (threshold: %d)", name, sketch_mem, set_mem * 0.1)
        assert sketch_mem < set_mem * 0.1, (
            f"{name} uses too much memory: {sketch_mem} bytes "
            f">= 10% of set memory ({set_mem} bytes)"
        )
    # Polars DataFrame should use significantly more memory than sketches
    polars_mem = mem.get("POLARS", 0)
    if polars_mem == 0:
        pytest.skip("Polars not available in subprocess; skipping Polars memory assertion")
    max_sketch = max(mem.get(name, 0) for name in ("HLL", "THETA", "CPC"))
    logging.info("POLARS memory: %d bytes (threshold: %d)", polars_mem, max_sketch * 10)
    assert polars_mem > max_sketch * 10, (
        f"Polars uses too little memory: {polars_mem} bytes "
        f"<= 10x max sketch memory ({max_sketch} bytes)"
    )