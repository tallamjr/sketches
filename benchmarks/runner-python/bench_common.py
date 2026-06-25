"""Shared timing + memory protocol for the Python benchmark plane.

Mirrors the Rust runners: one untimed warmup pass, then R timed reps, reporting
median and population stddev of per-rep throughput; per-object heap delta via
tracemalloc.
"""
import time
import tracemalloc

HEADER = (
    "implementation,sketch,dataset,op,n,reps,"
    "throughput_median_ops_per_s,throughput_stddev,bytes,live_bytes,"
    "estimate,exact,rel_error"
)

def median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0

def stddev(xs):
    n = len(xs)
    if n == 0:
        return 0.0
    mean = sum(xs) / n
    return (sum((x - mean) ** 2 for x in xs) / n) ** 0.5

def timed_throughput(reps, warmup, ops_per_rep, body):
    if warmup:
        body()
    rates = []
    for _ in range(reps):
        start = time.perf_counter()
        body()
        secs = time.perf_counter() - start
        rates.append(ops_per_rep / secs if secs > 0 else 0.0)
    return median(rates), stddev(rates)

def measure_live(build):
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    try:
        obj = build()
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    return obj, max(0, after - before)
