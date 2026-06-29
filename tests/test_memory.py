"""Memory-efficiency tests based on the sketches' bounded serialised size.

Process RSS deltas are far too noisy to measure a few-kilobyte sketch (and a
Rust-owned heap is not visible to Python memory tools anyway), so these tests
compare each sketch's serialised size, a stable and bounded quantity, against
the cost of storing the exact distinct set. That is the property that matters:
a sketch holds a fixed, tiny summary regardless of how many items it has seen.
"""

import sys

from sketches import CpcSketch, HllSketch, ThetaSketch


def _exact_set_bytes(values):
    """Approximate bytes to store the exact distinct set: the set container plus
    every distinct element it holds."""
    distinct = set(values)
    return sys.getsizeof(distinct) + sum(sys.getsizeof(v) for v in distinct)


def _serialised_size(sketch, values):
    for v in values:
        sketch.update(v)
    return len(sketch.to_bytes())


def test_sketch_serialised_size_far_below_exact_set():
    """Each cardinality sketch serialises to a small fraction of the memory the
    exact distinct set requires."""
    n = 20000
    values = [str(i) for i in range(n)]
    exact = _exact_set_bytes(values)
    assert exact > 0, "failed to size the exact set"

    sizes = {
        "HLL": _serialised_size(HllSketch(12), values),
        "Theta": _serialised_size(ThetaSketch(4096), values),
        "CPC": _serialised_size(CpcSketch(11), values),
    }

    for name, size in sizes.items():
        assert size > 0, f"{name} serialised to zero bytes"
        assert size < 0.15 * exact, (
            f"{name} serialised size {size} bytes is not far below the exact "
            f"set ({exact} bytes); expected under 15%"
        )
