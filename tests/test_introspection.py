import pytest
import sketches
import logging


def test_hll_to_bytes_length_default():
    """
    Default HLL sketch should have 2^12 registers (default precision p=12).
    """
    sk = sketches.HllSketch(12)  # Use default precision explicitly
    b = sk.to_bytes()
    logging.info("HLL default to_bytes length: %d bytes", len(b))
    assert isinstance(b, (bytes, bytearray))
    # default p = 12 => 4096 registers
    assert len(b) == 1 << 12, f"Expected 4096 bytes, got {len(b)}"


def test_hll_to_bytes_length_custom():
    """
    HLL sketch with custom precision should have 2^p registers.
    """
    for p in (8, 10, 14):
        sk = sketches.HllSketch(p)
        b = sk.to_bytes()
        logging.info("HLL to_bytes length for p=%d: %d bytes", p, len(b))
        assert len(b) == 1 << p, f"Expected {1<<p} bytes for p={p}, got {len(b)}"


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