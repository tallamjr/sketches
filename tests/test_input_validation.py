"""Input validation tests for the Python (PyO3) API.

These assert that invalid constructor and merge arguments raise a Python
``ValueError`` rather than panicking (an uncatchable ``PanicException``) or
aborting the interpreter (for example via an out-of-memory allocation).
"""

import pytest

import sketches


# --- Constructors that must reject out-of-range arguments --------------------


@pytest.mark.parametrize(
    "factory",
    [
        lambda: sketches.CpcSketch(3),
        lambda: sketches.CpcSketch(27),
        lambda: sketches.HllSketch(3),
        lambda: sketches.HllSketch(19),
        lambda: sketches.HllPlusPlusSketch(3),
        lambda: sketches.HllPlusPlusSketch(19),
        lambda: sketches.HllPlusPlusSparseSketch(3),
        lambda: sketches.HllPlusPlusSparseSketch(19),
        lambda: sketches.KllSketch(4),
        lambda: sketches.BloomFilter(0, 0.01),
        lambda: sketches.BloomFilter(100, 0.0),
        lambda: sketches.BloomFilter(100, 1.0),
        lambda: sketches.CountingBloomFilter(0, 0.01),
        lambda: sketches.CountingBloomFilter(100, 0.0),
        lambda: sketches.CountingBloomFilter(100, 1.0),
        lambda: sketches.CountMinSketch(0, 5, False),
        lambda: sketches.CountMinSketch(2048, 0, False),
        lambda: sketches.CountMinSketch.with_error_bounds(0.0, 0.01),
        lambda: sketches.CountMinSketch.with_error_bounds(0.01, 1.0),
        lambda: sketches.CountSketch(0, 5),
        lambda: sketches.CountSketch(2048, 0),
        lambda: sketches.LinearCounter(0),
        lambda: sketches.LinearCounter.with_expected_cardinality(0, 0.01),
        lambda: sketches.LinearCounter.with_expected_cardinality(1000, 0.0),
        lambda: sketches.HybridCounter(0, 12, 100),
        lambda: sketches.HybridCounter(1024, 3, 100),
        lambda: sketches.AodSketch(0, 1),
        lambda: sketches.AodSketch(4096, 0),
        lambda: sketches.ReqSketch(3),
        lambda: sketches.VarOptSketch(0),
        lambda: sketches.HllSketchMode(3),
        lambda: sketches.HllSketchMode(22),
        lambda: sketches.HllUnion(3),
        lambda: sketches.HllUnion(22),
    ],
)
def test_invalid_constructor_raises_value_error(factory):
    with pytest.raises(ValueError):
        factory()


def test_counting_bloom_zero_capacity_does_not_crash():
    """Regression: ``CountingBloomFilter(0)`` previously tried to allocate a
    ``usize::MAX`` byte vector and aborted the interpreter. It must now raise a
    catchable ``ValueError`` instead.
    """
    with pytest.raises(ValueError):
        sketches.CountingBloomFilter(0)
    with pytest.raises(ValueError):
        sketches.CountingBloomFilter(0, 0.01)


# --- Merges that must reject incompatible operands ---------------------------


def test_hll_merge_mismatched_precision_raises():
    a = sketches.HllSketch(12)
    b = sketches.HllSketch(10)
    with pytest.raises(ValueError):
        a.merge(b)


def test_hllpp_merge_mismatched_precision_raises():
    a = sketches.HllPlusPlusSketch(12)
    b = sketches.HllPlusPlusSketch(10)
    with pytest.raises(ValueError):
        a.merge(b)


def test_hllpp_sparse_merge_mismatched_precision_raises():
    a = sketches.HllPlusPlusSparseSketch(12)
    b = sketches.HllPlusPlusSparseSketch(10)
    with pytest.raises(ValueError):
        a.merge(b)


# --- Valid constructions and merges must still succeed -----------------------


def test_valid_constructions_succeed():
    assert sketches.CpcSketch(4) is not None
    assert sketches.CpcSketch(26) is not None
    assert sketches.HllSketch(4) is not None
    assert sketches.HllSketch(18) is not None
    assert sketches.HllPlusPlusSketch(12) is not None
    assert sketches.HllPlusPlusSparseSketch(12) is not None
    assert sketches.KllSketch(8) is not None
    assert sketches.BloomFilter(100, 0.01) is not None
    assert sketches.CountingBloomFilter(100, 0.01) is not None
    assert sketches.CountingBloomFilter(100) is not None
    assert sketches.CountMinSketch(2048, 5, False) is not None
    assert sketches.CountMinSketch.with_error_bounds(0.01, 0.01) is not None
    assert sketches.CountSketch(2048, 5) is not None
    assert sketches.LinearCounter(1024) is not None
    assert sketches.HybridCounter(1024, 12, 100) is not None
    assert sketches.AodSketch(4096, 1) is not None
    assert sketches.ReqSketch(12) is not None
    assert sketches.VarOptSketch(10) is not None
    assert sketches.HllSketchMode(12) is not None
    assert sketches.HllUnion(12) is not None


def test_valid_merge_succeeds():
    a = sketches.HllSketch(12)
    b = sketches.HllSketch(12)
    for i in range(1000):
        a.update(f"a_{i}")
        b.update(f"b_{i}")
    a.merge(b)
    assert a.estimate() > 0.0
