"""Round-trip and pickle tests for the Python serialisation API.

Every sketch whose Rust type implements serialisation must expose a symmetric
to_bytes / from_bytes pair and support pickling. These tests build a populated
sketch, serialise it, reconstruct it, and assert the reconstructed sketch is
equivalent to the original. They also assert that pickling round-trips and that
malformed input raises ValueError rather than panicking.
"""

import math
import pickle

import pytest

import sketches as ds


# ---------------------------------------------------------------------------
# Helpers: build a populated sketch and extract a comparable signal from it.
# ---------------------------------------------------------------------------


def _items(n):
    return [f"item-{i}" for i in range(n)]


def build_cpc():
    s = ds.CpcSketch(11)
    for x in _items(2000):
        s.update(x)
    return s


def build_hll():
    s = ds.HllSketch(12)
    for x in _items(2000):
        s.update(x)
    return s


def build_hllpp():
    s = ds.HllPlusPlusSketch(12)
    for x in _items(2000):
        s.update(x)
    return s


def build_theta():
    s = ds.ThetaSketch(4096)
    for x in _items(2000):
        s.update(x)
    return s


def build_kll():
    s = ds.KllSketch(200)
    for i in range(5000):
        s.update(float(i))
    return s


def build_linear():
    s = ds.LinearCounter(4096)
    for x in _items(1000):
        s.update(x)
    return s


def build_hybrid():
    s = ds.HybridCounter(1024, 12, 500)
    for x in _items(2000):
        s.update(x)
    return s


def build_frequent():
    # Distinct per-key frequencies so the top-k ordering is deterministic.
    s = ds.FrequentStringsSketch(64, False)
    for k in range(40):
        for _ in range(k + 1):
            s.update(f"k-{k:02d}")
    return s


def build_tdigest():
    s = ds.TDigest()
    for i in range(5000):
        s.add(float(i))
    return s


def build_req():
    s = ds.ReqSketch(12)
    for i in range(5000):
        s.update(float(i))
    return s


def build_tuple():
    s = ds.TupleSketch(4096)
    for i in range(2000):
        s.update(f"item-{i}", float(i))
    return s


def build_varopt():
    s = ds.VarOptSketch(100)
    for i in range(2000):
        s.update(f"item-{i}", 1.0 + (i % 7))
    return s


def build_aod():
    s = ds.AodSketch(4096, 2)
    for i in range(2000):
        s.update(f"item-{i}", [float(i), float(i * 2)])
    return s


def build_reservoir_r():
    s = ds.ReservoirSamplerR(50)
    for x in _items(2000):
        s.add(x)
    return s


def build_reservoir_a():
    s = ds.ReservoirSamplerA(50)
    for x in _items(2000):
        s.add(x)
    return s


def build_stream_sampler():
    s = ds.StreamSampler(50, 10)
    s.push_batch(_items(2000))
    s.flush()
    return s


def build_bloom():
    s = ds.BloomFilter(2000, 0.01)
    for x in _items(1000):
        s.add(x)
    return s


def build_counting_bloom():
    s = ds.CountingBloomFilter(2000, 0.01, 255)
    for x in _items(1000):
        s.add(x)
    return s


def build_countmin():
    s = ds.CountMinSketch(2048, 5, False)
    for k in range(40):
        for _ in range(k + 1):
            s.increment(f"k-{k:02d}")
    return s


def build_count_sketch():
    s = ds.CountSketch(2048, 5)
    for k in range(40):
        s.update(f"k-{k:02d}", k + 1)
    return s


def sig_bloom(s):
    # Membership of every added item must be preserved exactly.
    return ("bloom", tuple(s.contains(x) for x in _items(1000)))


def sig_countmin(s):
    # Estimate of every key must be preserved exactly.
    return ("countmin", tuple(s.estimate(f"k-{k:02d}") for k in range(40)))


def sig_count_sketch(s):
    return ("count_sketch", tuple(s.estimate(f"k-{k:02d}") for k in range(40)))


def sig_cardinality(s):
    return ("estimate", round(s.estimate(), 6))


def sig_kll(s):
    return ("quantile", round(s.quantile(0.5), 6), round(s.quantile(0.9), 6))


def sig_tdigest(s):
    return ("quantile", round(s.quantile(0.5), 6), round(s.quantile(0.9), 6))


def sig_req(s):
    return ("quantile", round(s.quantile(0.5), 6), round(s.quantile(0.9), 6))


def sig_frequent(s):
    return ("frequent", tuple(s.get_top_k(5)))


def sig_tuple(s):
    return ("tuple", round(s.estimate(), 6), s.num_retained())


def sig_varopt(s):
    return ("varopt", round(s.get_total_weight(), 6), s.get_num_samples())


def sig_aod(s):
    return ("aod", round(s.estimate(), 6), s.len(), tuple(round(v, 6) for v in s.column_sums()))


def sig_sample_reservoir(s):
    return ("sample", tuple(sorted(s.sample())), s.items_seen())


def sig_sample_stream(s):
    return ("sample", tuple(sorted(s.sample())), s.stats()["items_processed"])


def _equivalent(a, b, tol):
    """Compare two signatures. tol == 0 means exact; otherwise numeric values
    are compared with a relative/absolute tolerance (KLL reconstructs by
    weighted replay so its quantiles are statistically equivalent, not exact)."""
    if tol == 0.0:
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_equivalent(x, y, tol) for x, y in zip(a, b))
    return a == b


# (name, class, builder, signature, tolerance) for every serialisable sketch.
CASES = [
    ("CpcSketch", ds.CpcSketch, build_cpc, sig_cardinality, 0.0),
    ("HllSketch", ds.HllSketch, build_hll, sig_cardinality, 0.0),
    ("HllPlusPlusSketch", ds.HllPlusPlusSketch, build_hllpp, sig_cardinality, 0.0),
    ("ThetaSketch", ds.ThetaSketch, build_theta, sig_cardinality, 0.0),
    ("KllSketch", ds.KllSketch, build_kll, sig_kll, 0.05),
    ("LinearCounter", ds.LinearCounter, build_linear, sig_cardinality, 0.0),
    ("HybridCounter", ds.HybridCounter, build_hybrid, sig_cardinality, 0.0),
    ("FrequentStringsSketch", ds.FrequentStringsSketch, build_frequent, sig_frequent, 0.0),
    ("TDigest", ds.TDigest, build_tdigest, sig_tdigest, 0.0),
    ("ReqSketch", ds.ReqSketch, build_req, sig_req, 0.0),
    ("TupleSketch", ds.TupleSketch, build_tuple, sig_tuple, 0.0),
    ("VarOptSketch", ds.VarOptSketch, build_varopt, sig_varopt, 0.0),
    ("AodSketch", ds.AodSketch, build_aod, sig_aod, 0.0),
    ("ReservoirSamplerR", ds.ReservoirSamplerR, build_reservoir_r, sig_sample_reservoir, 0.0),
    ("ReservoirSamplerA", ds.ReservoirSamplerA, build_reservoir_a, sig_sample_reservoir, 0.0),
    ("StreamSampler", ds.StreamSampler, build_stream_sampler, sig_sample_stream, 0.0),
    ("BloomFilter", ds.BloomFilter, build_bloom, sig_bloom, 0.0),
    ("CountingBloomFilter", ds.CountingBloomFilter, build_counting_bloom, sig_bloom, 0.0),
    ("CountMinSketch", ds.CountMinSketch, build_countmin, sig_countmin, 0.0),
    ("CountSketch", ds.CountSketch, build_count_sketch, sig_count_sketch, 0.0),
]


@pytest.mark.parametrize("name,cls,builder,sig,tol", CASES, ids=[c[0] for c in CASES])
def test_to_from_bytes_roundtrip(name, cls, builder, sig, tol):
    s = builder()
    data = s.to_bytes()
    assert isinstance(data, (bytes, bytearray))
    s2 = cls.from_bytes(data)
    assert _equivalent(sig(s2), sig(s), tol), (
        f"{name} did not round-trip via to_bytes/from_bytes"
    )


@pytest.mark.parametrize("name,cls,builder,sig,tol", CASES, ids=[c[0] for c in CASES])
def test_pickle_roundtrip(name, cls, builder, sig, tol):
    s = builder()
    s2 = pickle.loads(pickle.dumps(s))
    assert isinstance(s2, cls)
    assert _equivalent(sig(s2), sig(s), tol), f"{name} did not round-trip via pickle"


@pytest.mark.parametrize(
    "cls",
    [
        ds.CpcSketch,
        ds.HllSketch,
        ds.ThetaSketch,
        ds.KllSketch,
        ds.TDigest,
        ds.BloomFilter,
        ds.CountingBloomFilter,
        ds.CountMinSketch,
        ds.CountSketch,
    ],
)
def test_from_bytes_garbage_raises_value_error(cls):
    with pytest.raises(ValueError):
        cls.from_bytes(b"garbage")
