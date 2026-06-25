import statistics
import bench_common as bc

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
        "throughput_median_ops_per_s,throughput_stddev,bytes,live_bytes,"
        "estimate,exact,rel_error"
    )
