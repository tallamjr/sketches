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
    // Classic HLL++ estimator RSE ~= 1.04/sqrt(4096) ~= 0.0163 (above the floor).
    // With HIP the RMSE drops below the 1/sqrt(4096) ~= 0.0156 floor; assert < 0.015.
    let rmse = hll_rmse(50, 50_000);
    assert!(
        rmse < 0.015,
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
