//! Smoke test: the emitted rows match the shared schema and the Apache HLL
//! row is present and accurate.

#[test]
fn apache_rust_header_matches_ours() {
    assert_eq!(
        runner_apache_rust::HEADER,
        "implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,throughput_stddev,throughput_ci_low,throughput_ci_high,bytes,live_bytes,estimate,exact,rel_error"
    );
}

#[test]
fn emits_apache_rust_rows() {
    let lines = runner_apache_rust::run(10_000, 5);
    assert!(lines[0].starts_with("implementation,sketch,dataset,op,n,"));
    // every data row has 15 comma-separated fields matching the header
    for line in &lines[1..] {
        assert_eq!(
            line.split(',').count(),
            15,
            "data row should have 15 columns: {line}"
        );
    }
    let hll = lines
        .iter()
        .find(|l| l.contains("apache-rust,hll,") && l.contains("distinct_count"))
        .expect("apache hll row");
    let rel_error: f64 = hll.rsplit(',').next().unwrap().parse().unwrap();
    assert!(rel_error < 0.05, "apache hll rel_error {rel_error}");
}
