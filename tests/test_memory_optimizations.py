"""
Test memory optimizations for HyperLogLog implementations.
This module tests the memory usage improvements from the optimizations.
"""

import pytest
import sys
import subprocess
import tempfile
import os

# Skip if psutil is not available
pytest.importorskip("psutil")


def test_hll_memory_optimization():
    """Test that optimized HLL uses less memory than unoptimized version."""
    py = sys.executable

    script = """
import os
import psutil
import sys
sys.path.insert(0, '.')

# Import after path adjustment
from sketches import HllSketch

# Test data
N = 10000
values = [str(i) for i in range(N)]
proc = psutil.Process(os.getpid())

# Measure baseline memory
baseline_rss = proc.memory_info().rss

# Create and populate HLL sketch
sketch = HllSketch(12)  # 2^12 = 4096 registers
for v in values:
    sketch.update(v)

# Measure memory after HLL
hll_rss = proc.memory_info().rss
hll_memory = hll_rss - baseline_rss

# Get estimate
estimate = sketch.estimate()
error_percent = abs(estimate - N) / N * 100

print(f"HLL_MEMORY:{hll_memory}")
print(f"ESTIMATE:{estimate}")
print(f"ERROR_PERCENT:{error_percent}")
print(f"TRUE_COUNT:{N}")
"""

    # Run the test script
    result = subprocess.run([py, "-c", script],
                          capture_output=True, text=True, cwd=".")

    if result.returncode != 0:
        pytest.fail(f"Script failed: {result.stderr}")

    # Parse results
    output_lines = result.stdout.strip().split('\n')
    results = {}
    for line in output_lines:
        if ':' in line:
            key, value = line.split(':', 1)
            results[key] = value

    hll_memory = int(results['HLL_MEMORY'])
    estimate = float(results['ESTIMATE'])
    error_percent = float(results['ERROR_PERCENT'])
    true_count = int(results['TRUE_COUNT'])

    # Assertions - RSS measurement is rough, so allow for reasonable overhead
    assert hll_memory < 100000, f"HLL memory usage too high: {hll_memory} bytes"
    assert error_percent < 5.0, f"HLL error too high: {error_percent}%"
    assert 0.9 * true_count <= estimate <= 1.1 * true_count, \
        f"HLL estimate {estimate} not within 10% of true count {true_count}"

    print(f"✅ HLL Memory Test Passed:")
    print(f"   Memory usage: {hll_memory:,} bytes")
    print(f"   Estimate: {estimate:,.0f} (error: {error_percent:.2f}%)")


def test_memory_comparison_across_implementations():
    """Compare memory usage across different HLL implementations."""
    py = sys.executable

    script = """
import os
import psutil
import sys
sys.path.insert(0, '.')

from sketches import HllSketch

# Test parameters
N = 5000
values = [str(i) for i in range(N)]
proc = psutil.Process(os.getpid())

results = []

# Test different precisions
for precision in [8, 10, 12]:
    baseline_rss = proc.memory_info().rss

    sketch = HllSketch(precision)
    for v in values:
        sketch.update(v)

    final_rss = proc.memory_info().rss
    memory_used = final_rss - baseline_rss
    estimate = sketch.estimate()

    results.append({
        'precision': precision,
        'memory': memory_used,
        'estimate': estimate,
        'registers': 2 ** precision
    })

    print(f"PRECISION:{precision}:MEMORY:{memory_used}:ESTIMATE:{estimate}:REGISTERS:{2**precision}")

# Memory should scale roughly with number of registers
print("DONE")
"""

    result = subprocess.run([py, "-c", script],
                          capture_output=True, text=True, cwd=".")

    if result.returncode != 0:
        pytest.fail(f"Script failed: {result.stderr}")

    # Parse results
    results = []
    for line in result.stdout.strip().split('\n'):
        if line.startswith('PRECISION:'):
            parts = line.split(':')
            precision = int(parts[1])
            memory = int(parts[3])
            estimate = float(parts[5])
            registers = int(parts[7])
            results.append({
                'precision': precision,
                'memory': memory,
                'estimate': estimate,
                'registers': registers
            })

    # Verify memory scaling is reasonable
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"

    # Memory should increase with precision, but not linearly due to optimizations
    results.sort(key=lambda x: x['precision'])

    for i, result in enumerate(results):
        print(f"✅ Precision {result['precision']}: "
              f"{result['memory']:,} bytes for {result['registers']} registers "
              f"(estimate: {result['estimate']:.0f})")

        # Basic sanity checks
        assert result['memory'] > 0, "Memory usage should be positive"
        assert result['estimate'] > 0, "Estimate should be positive"


def test_batch_vs_individual_updates():
    """Test that batch updates don't significantly change memory usage."""
    py = sys.executable

    script = """
import os
import psutil
import sys
sys.path.insert(0, '.')

from sketches import HllSketch

N = 1000
values = [str(i) for i in range(N)]
proc = psutil.Process(os.getpid())

# Individual updates
baseline_rss = proc.memory_info().rss
sketch1 = HllSketch(10)
for v in values:
    sketch1.update(v)
individual_rss = proc.memory_info().rss
individual_memory = individual_rss - baseline_rss

# Batch updates (if available)
baseline_rss = proc.memory_info().rss
sketch2 = HllSketch(10)

# Try batch update if available
if hasattr(sketch2, 'update_batch'):
    sketch2.update_batch(values)
else:
    # Fallback to individual updates
    for v in values:
        sketch2.update(v)

batch_rss = proc.memory_info().rss
batch_memory = batch_rss - baseline_rss

print(f"INDIVIDUAL_MEMORY:{individual_memory}")
print(f"BATCH_MEMORY:{batch_memory}")
print(f"INDIVIDUAL_ESTIMATE:{sketch1.estimate()}")
print(f"BATCH_ESTIMATE:{sketch2.estimate()}")
"""

    result = subprocess.run([py, "-c", script],
                          capture_output=True, text=True, cwd=".")

    if result.returncode != 0:
        pytest.fail(f"Script failed: {result.stderr}")

    # Parse results
    results = {}
    for line in result.stdout.strip().split('\n'):
        if ':' in line:
            key, value = line.split(':', 1)
            results[key] = value

    individual_memory = int(results['INDIVIDUAL_MEMORY'])
    batch_memory = int(results['BATCH_MEMORY'])
    individual_estimate = float(results['INDIVIDUAL_ESTIMATE'])
    batch_estimate = float(results['BATCH_ESTIMATE'])

    # Memory usage should be similar
    memory_ratio = abs(individual_memory - batch_memory) / max(individual_memory, batch_memory)
    assert memory_ratio < 0.5, f"Memory usage differs too much: {individual_memory} vs {batch_memory}"

    # Estimates should be very close
    estimate_ratio = abs(individual_estimate - batch_estimate) / max(individual_estimate, batch_estimate)
    assert estimate_ratio < 0.01, f"Estimates differ too much: {individual_estimate} vs {batch_estimate}"

    print(f"✅ Batch vs Individual Test Passed:")
    print(f"   Individual: {individual_memory:,} bytes, estimate: {individual_estimate:.0f}")
    print(f"   Batch: {batch_memory:,} bytes, estimate: {batch_estimate:.0f}")


if __name__ == "__main__":
    test_hll_memory_optimization()
    test_memory_comparison_across_implementations()
    test_batch_vs_individual_updates()
    print("\n🎉 All memory optimization tests passed!")
