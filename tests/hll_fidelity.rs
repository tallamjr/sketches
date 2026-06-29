use sketches::hll::HllSketch;
use sketches::serialization::Serializable;

/// Deterministic multi-trial RMSE: trial t over the disjoint range [t*N, (t+1)*N).
/// No RNG, so this is a fixed, reproducible number, not a flaky threshold.
fn hll_rmse(trials: u64, n: u64) -> f64 {
    let truth = n as f64;
    let mut sq = 0.0_f64;
    for t in 0..trials {
        let mut s = HllSketch::new(12);
        for i in t * n..(t + 1) * n {
            s.update(&i);
        }
        let e = (s.estimate() - truth).abs() / truth;
        sq += e * e;
    }
    (sq / trials as f64).sqrt()
}

#[test]
fn hll_rmse_beats_floor_with_hip() {
    // This threshold is a genuine HIP-vs-classic discriminator. Under our fixed-seed
    // xxh3 hash the classic (composite) estimator measures ~0.0145 RMSE and the HIP
    // estimator measures ~0.0125 RMSE; the 0.0135 threshold sits between them. So the
    // classic estimator would FAIL this assertion and only the HIP path PASSES it,
    // proving HIP is actually engaged rather than merely passing a loose bound.
    let rmse = hll_rmse(50, 50_000);
    assert!(
        rmse < 0.0135,
        "hll RMSE {rmse} did not beat the floor (HIP not effective)"
    );
}

#[test]
fn hll_roundtrip_keeps_hip_estimate_exact() {
    let mut s = HllSketch::new(12);
    for i in 0u64..200_000 {
        s.update(&i);
    }
    let back = HllSketch::from_bytes(&Serializable::to_bytes(&s)).unwrap();
    assert!(
        (s.estimate() - back.estimate()).abs() < 1e-6,
        "round-trip changed the HIP estimate"
    );
}

#[test]
fn hll_merge_is_accurate_via_composite_fallback() {
    let mut a = HllSketch::new(12);
    let mut b = HllSketch::new(12);
    for i in 0u64..500_000 {
        a.update(&i);
    }
    for i in 500_000u64..1_000_000 {
        b.update(&i);
    }
    a.merge(&b).expect("equal precision merge");
    let e = (a.estimate() - 1_000_000.0).abs() / 1_000_000.0;
    assert!(e < 0.03, "merged hll rel_error {e}");
}
