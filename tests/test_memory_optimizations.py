"""HLL memory-footprint tests via the bounded serialised size.

These replace earlier process-RSS measurements, which were too noisy to be
meaningful for a few-kilobyte sketch. The serialised size is stable and is the
quantity a user actually stores or transfers.
"""

from sketches import HllSketch


def test_hll_serialised_size_is_bounded():
    """An HLL at lg_k=12 stays small regardless of stream length, and stays
    accurate."""
    n = 10000
    sketch = HllSketch(12)
    for i in range(n):
        sketch.update(str(i))

    size = len(sketch.to_bytes())
    # lg_k=12 is 2^12 registers; the codec form is about 4129 bytes. Allow head
    # room but assert it is firmly bounded (it does not grow with the stream).
    assert size < 8192, f"HLL serialised size unexpectedly large: {size} bytes"

    estimate = sketch.estimate()
    error = abs(estimate - n) / n
    assert error < 0.05, f"HLL error too high: {error * 100:.2f}%"


def test_hll_serialised_size_scales_with_precision():
    """Serialised size grows with lg_k (more registers), monotonically."""
    n = 5000
    values = [str(i) for i in range(n)]
    sizes = []
    for precision in (8, 10, 12):
        sketch = HllSketch(precision)
        for v in values:
            sketch.update(v)
        sizes.append(len(sketch.to_bytes()))

    assert sizes[0] < sizes[1] < sizes[2], (
        f"HLL serialised size should grow with precision, got {sizes}"
    )


def test_hll_build_is_deterministic():
    """Two HLL sketches built from the same stream produce identical estimates
    and identical serialised bytes (fixed-seed hashing, order-independent max
    registers)."""
    n = 1000
    values = [str(i) for i in range(n)]

    a = HllSketch(10)
    b = HllSketch(10)
    for v in values:
        a.update(v)
    for v in values:
        b.update(v)

    assert a.estimate() == b.estimate()
    assert a.to_bytes() == b.to_bytes()
