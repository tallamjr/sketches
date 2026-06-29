"""Shared timing + memory protocol for the Python benchmark plane.

Mirrors the Rust runners: R independent rounds, each one untimed warmup pass
then REPS_PER_ROUND timed reps (the round-sample is the median of those reps);
report the median over round-samples, their population stddev, and a
deterministic nonparametric 95% bootstrap CI; per-object heap delta via
tracemalloc.
"""
import math
import time
import tracemalloc

HEADER = (
    "implementation,sketch,dataset,op,n,reps,"
    "throughput_median_ops_per_s,throughput_stddev,"
    "throughput_ci_low,throughput_ci_high,bytes,live_bytes,"
    "estimate,exact,rel_error"
)

_U64 = 0xFFFFFFFFFFFFFFFF
_BOOTSTRAP_SEED = 0x9E3779B97F4A7C15
_BOOTSTRAP_RESAMPLES = 2000

REPS_PER_ROUND = 5

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

def _splitmix64(state):
    state = (state + 0x9E3779B97F4A7C15) & _U64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _U64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _U64
    z = z ^ (z >> 31)
    return state, z

def bootstrap_ci(samples):
    r = len(samples)
    if r <= 1:
        m = samples[0] if r == 1 else 0.0
        return (m, m)
    state = _BOOTSTRAP_SEED
    resample_medians = []
    for _ in range(_BOOTSTRAP_RESAMPLES):
        draw = []
        for _ in range(r):
            state, z = _splitmix64(state)
            draw.append(samples[z % r])
        resample_medians.append(median(draw))
    resample_medians.sort()
    b = _BOOTSTRAP_RESAMPLES

    def idx(p):
        i = int(math.floor(p * b))
        return max(0, min(b - 1, i))

    return (resample_medians[idx(0.025)], resample_medians[idx(0.975)])

def timed_throughput_rounds(rounds, reps_per_round, ops_per_rep, body):
    round_samples = []
    for _ in range(rounds):
        body()  # untimed warmup
        rates = []
        for _ in range(reps_per_round):
            start = time.perf_counter()
            body()
            secs = time.perf_counter() - start
            rates.append(ops_per_rep / secs if secs > 0 else 0.0)
        round_samples.append(median(rates))
    ci = bootstrap_ci(round_samples)
    return (median(round_samples), stddev(round_samples), ci[0], ci[1])

def measure_live(build):
    tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    try:
        obj = build()
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    return obj, max(0, after - before)
