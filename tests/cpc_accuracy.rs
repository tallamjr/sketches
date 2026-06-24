//! Real-data accuracy tests for the CPC sketch.
//!
//! These exercise the rewritten coupon collection, flavour transitions, and
//! HIP estimation against large distinct-value streams. The thresholds match
//! the standard error implied by lg_k = 12 (roughly 1 / sqrt(2^12) ~= 1.5%).

use sketches::cpc::CpcSketch;

fn rel_err(est: f64, truth: f64) -> f64 {
    (est - truth).abs() / truth
}

#[test]
fn cpc_synthetic_under_2pct() {
    let mut s = CpcSketch::new(12);
    for i in 0u64..1_000_000 {
        s.update(&i);
    }
    let e = rel_err(s.estimate(), 1_000_000.0);
    assert!(
        e < 0.02,
        "cpc synthetic rel_error {e} (target <0.02; was ~1.73 when broken)"
    );
}

#[test]
fn cpc_accuracy_sweeps_cardinality() {
    // CPC must be accurate across the flavour transitions, not just at one size.
    for &n in &[1_000u64, 50_000, 500_000] {
        let mut s = CpcSketch::new(12);
        for i in 0..n {
            s.update(&i);
        }
        let e = rel_err(s.estimate(), n as f64);
        assert!(e < 0.03, "cpc n={n} rel_error {e}");
    }
}

#[test]
fn cpc_union_matches_combined_cardinality() {
    // Two disjoint halves of a distinct stream; the union cardinality is the sum.
    let mut a = CpcSketch::new(12);
    let mut b = CpcSketch::new(12);
    for i in 0u64..400_000 {
        a.update(&i);
    }
    for i in 400_000u64..1_000_000 {
        b.update(&i);
    }
    let mut u = sketches::cpc::CpcUnion::new(12);
    u.update(&a);
    u.update(&b);
    let e = (u.estimate() - 1_000_000.0).abs() / 1_000_000.0;
    assert!(e < 0.03, "cpc union rel_error {e}");
}

#[test]
fn cpc_merge_matches_combined_cardinality() {
    // In-place merge of disjoint ranges must estimate the combined distinct count.
    let mut a = CpcSketch::new(12);
    let mut b = CpcSketch::new(12);
    for i in 0u64..350_000 {
        a.update(&i);
    }
    for i in 350_000u64..900_000 {
        b.update(&i);
    }
    a.merge(&b);
    let e = rel_err(a.estimate(), 900_000.0);
    assert!(e < 0.03, "cpc merge rel_error {e}");
}

#[test]
fn cpc_roundtrip_preserves_estimate_on_real_volume() {
    let mut s = CpcSketch::new(12);
    for i in 0u64..500_000 {
        s.update(&i);
    }
    let bytes = s.to_bytes();
    assert_eq!(&bytes[0..2], &[0x53, 0x4B]); // MAGIC
    let back = CpcSketch::from_bytes(&bytes).unwrap();
    assert!((s.estimate() - back.estimate()).abs() < 1e-6);
    assert_eq!(s.num_coupons(), back.num_coupons());
}
