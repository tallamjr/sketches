//! Smoke test: the emitted rows match the shared schema and HLL is accurate.

#[test]
fn header_has_new_columns() {
    assert_eq!(
        runner_ours::HEADER,
        "implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,throughput_stddev,bytes,live_bytes,estimate,exact,rel_error"
    );
}

#[test]
fn emits_reps_and_stddev_columns() {
    let rows = runner_ours::run(1000, 5);
    assert!(runner_ours::HEADER.contains("throughput_stddev"));
    // skip the header line; every data row has 13 comma-separated fields
    for r in &rows[1..] {
        assert_eq!(
            r.split(',').count(),
            13,
            "data row should have 13 columns: {r}"
        );
    }
}

#[test]
fn emits_valid_schema_and_reasonable_hll() {
    let lines = runner_ours::run(10_000, 5);

    assert!(lines[0].starts_with(
        "implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,throughput_stddev,bytes,live_bytes,estimate,exact,rel_error"
    ));

    // every data row has 13 comma-separated fields matching the header
    let header_fields = lines[0].split(',').count();
    assert_eq!(header_fields, 13, "header should have 13 columns");
    for line in &lines[1..] {
        assert_eq!(
            line.split(',').count(),
            13,
            "data row should have 13 columns: {line}"
        );
    }

    // find the HLL synthetic row and assert rel_error < 0.05
    let hll = lines
        .iter()
        .find(|l| l.contains(",hll,") && l.contains("distinct_count"))
        .expect("hll row");
    let rel_error: f64 = hll.rsplit(',').next().unwrap().parse().unwrap();
    assert!(rel_error < 0.05, "hll rel_error {rel_error}");
}
