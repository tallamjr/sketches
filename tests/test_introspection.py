import pytest
import sketches
import logging


def test_hll_to_bytes_length_default():
    """
    Default HLL sketch should have 2^12 registers (default precision p=12).
    """
    sk = sketches.HllSketch()
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
    CPC sketch reuses HLL internals; number of registers equals 2^lg_k.
    """
    # default lg_k = 11 => 2048 registers
    sk = sketches.CpcSketch()
    b = sk.to_bytes()
    logging.info("CPC default to_bytes length: %d bytes", len(b))
    assert isinstance(b, (bytes, bytearray))
    assert len(b) == 1 << 11, f"Expected 2048 bytes, got {len(b)}"
    # custom lg_k
    for lg in (9, 13):
        sk = sketches.CpcSketch(lg)
        b = sk.to_bytes()
        logging.info("CPC to_bytes length for lg_k=%d: %d bytes", lg, len(b))
        assert len(b) == 1 << lg, f"Expected {1<<lg} bytes for lg_k={lg}, got {len(b)}"