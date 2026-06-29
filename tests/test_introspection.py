import pytest
import sketches
import logging


# The HLL codec format wraps the 2^p register array with a fixed overhead:
# a 5-byte SketchHeader, a 3-byte payload preamble (lg_k, cur_min, mode), and a
# 25-byte HIP trailer (three f64 plus one flag byte). This format round-trips
# via from_bytes, unlike the old registers-only dump.
HLL_CODEC_OVERHEAD = 5 + 3 + 25


def test_hll_to_bytes_length_default():
    """
    Default HLL sketch (p=12) serialises its 2^12 registers plus codec overhead.
    """
    sk = sketches.HllSketch(12)  # Use default precision explicitly
    b = sk.to_bytes()
    logging.info("HLL default to_bytes length: %d bytes", len(b))
    assert isinstance(b, (bytes, bytearray))
    expected = (1 << 12) + HLL_CODEC_OVERHEAD
    assert len(b) == expected, f"Expected {expected} bytes, got {len(b)}"
    # The serialised form must round-trip back to an equivalent sketch.
    assert sketches.HllSketch.from_bytes(b).estimate() == sk.estimate()


def test_hll_to_bytes_length_custom():
    """
    HLL sketch with custom precision serialises 2^p registers plus codec overhead.
    """
    for p in (8, 10, 14):
        sk = sketches.HllSketch(p)
        b = sk.to_bytes()
        logging.info("HLL to_bytes length for p=%d: %d bytes", p, len(b))
        expected = (1 << p) + HLL_CODEC_OVERHEAD
        assert len(b) == expected, f"Expected {expected} bytes for p={p}, got {len(b)}"


def test_cpc_to_bytes_length():
    """
    CPC sketch uses compressed serialization, so empty sketches are small.
    Unlike HLL, CPC doesn't serialize full register arrays when empty.
    """
    # Test that CPC serialization works and is compact when empty
    sk = sketches.CpcSketch(11)  # Use default precision explicitly
    b = sk.to_bytes()
    logging.info("CPC default to_bytes length: %d bytes", len(b))
    assert isinstance(b, (bytes, bytearray))
    # Empty CPC sketches serialize compactly (much smaller than full register array)
    assert len(b) < 100, f"Empty CPC sketch should be small, got {len(b)} bytes"

    # Test with some data - should still be reasonably compact
    for i in range(100):
        sk.update(f"item_{i}")
    b_with_data = sk.to_bytes()
    logging.info("CPC with data to_bytes length: %d bytes", len(b_with_data))
    assert len(b_with_data) > len(b), "CPC with data should be larger than empty"
    assert len(b_with_data) < (1 << 11), f"CPC with data should be compressed, got {len(b_with_data)} bytes"

    # Test custom lg_k
    for lg in (9, 13):
        sk = sketches.CpcSketch(lg)
        b = sk.to_bytes()
        logging.info("CPC to_bytes length for lg_k=%d: %d bytes", lg, len(b))
        assert len(b) < 100, f"Empty CPC sketch should be small for lg_k={lg}, got {len(b)} bytes"
