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
