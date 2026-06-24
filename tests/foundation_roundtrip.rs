use sketches::hll::HllSketch;
use sketches::serialization::Serializable;
use sketches::theta::ThetaSketch;

#[test]
fn hll_roundtrip_preserves_estimate() {
    let mut s = HllSketch::new(12);
    for i in 0u64..50_000 {
        s.update(&i);
    }
    let bytes = Serializable::to_bytes(&s);
    assert_eq!(&bytes[0..2], &[0x53, 0x4B]); // MAGIC
    let back = HllSketch::from_bytes(&bytes).unwrap();
    assert!((s.estimate() - back.estimate()).abs() < 1e-6);
}

#[test]
fn theta_roundtrip_preserves_estimate() {
    let mut s = ThetaSketch::new(4096);
    for i in 0u64..50_000 {
        s.update(&i);
    }
    let bytes = Serializable::to_bytes(&s);
    assert_eq!(&bytes[0..2], &[0x53, 0x4B]);
    let back = <ThetaSketch as Serializable>::from_bytes(&bytes).unwrap();
    assert!((s.estimate() - back.estimate()).abs() < 1e-6);
}
