use sketches::countmin::CountMinSketch;
use sketches::hll::HllSketch;
use sketches::quantiles::KllSketch;
use sketches::theta::ThetaSketch;

fn rel_err(est: f64, truth: f64) -> f64 {
    (est - truth).abs() / truth
}

#[test]
fn hll_synthetic_under_2pct() {
    let mut s = HllSketch::new(12);
    for i in 0u64..1_000_000 {
        s.update(&i);
    }
    assert!(
        rel_err(s.estimate(), 1_000_000.0) < 0.02,
        "hll err {}",
        rel_err(s.estimate(), 1_000_000.0)
    );
}

#[test]
fn theta_synthetic_under_3pct() {
    let mut s = ThetaSketch::new(4096);
    for i in 0u64..1_000_000 {
        s.update(&i);
    }
    assert!(
        rel_err(s.estimate(), 1_000_000.0) < 0.03,
        "theta err {}",
        rel_err(s.estimate(), 1_000_000.0)
    );
}

#[test]
fn kll_median_within_rank_error() {
    let mut s = KllSketch::<f64>::new(200);
    for i in 0u64..100_000 {
        s.update(i as f64);
    }
    let q = s.quantile(0.5).expect("non-empty sketch has a median");
    assert!(
        (q - 50_000.0).abs() < 0.03 * 100_000.0,
        "kll median {q} off by too much"
    );
}

#[test]
fn countmin_never_underestimates() {
    let mut c = CountMinSketch::new(2048, 5, false);
    for _ in 0..500 {
        c.increment(&"hot");
    }
    for i in 0u64..5_000 {
        c.increment(&i);
    }
    assert!(
        c.estimate(&"hot") >= 500,
        "count-min must never underestimate"
    );
}
