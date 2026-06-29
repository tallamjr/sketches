import bench_common as bc

PINNED_CI_LOW = 90.0
PINNED_CI_HIGH = 110.0

def test_bootstrap_ci_matches_pinned_vector():
    samples = [100.0, 110.0, 90.0, 105.0, 95.0, 120.0, 80.0, 115.0, 85.0, 100.0]
    lo, hi = bc.bootstrap_ci(samples)
    assert abs(lo - PINNED_CI_LOW) < 1e-6
    assert abs(hi - PINNED_CI_HIGH) < 1e-6
    assert bc.bootstrap_ci([42.0]) == (42.0, 42.0)

def test_median_and_stddev():
    assert bc.median([3.0, 1.0, 2.0]) == 2.0
    assert abs(bc.stddev([2, 4, 4, 4, 5, 5, 7, 9]) - 2.0) < 1e-9

def test_measure_live_nonzero():
    obj, live = bc.measure_live(lambda: [0] * 100000)
    assert live > 0
    assert len(obj) == 100000

def test_header_matches_rust():
    assert bc.HEADER == (
        "implementation,sketch,dataset,op,n,reps,"
        "throughput_median_ops_per_s,throughput_stddev,"
        "throughput_ci_low,throughput_ci_high,bytes,live_bytes,"
        "estimate,exact,rel_error"
    )
