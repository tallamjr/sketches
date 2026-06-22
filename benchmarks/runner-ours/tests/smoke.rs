//! Smoke test: the emitted rows match the shared schema and HLL is accurate.

#[test]
fn emits_valid_schema_and_reasonable_hll() {
    let lines = runner_ours::run(10_000);

    assert!(lines[0].starts_with(
        "implementation,sketch,dataset,op,n,throughput_ops_per_s,bytes,estimate,exact,rel_error"
    ));

    // find the HLL synthetic row and assert rel_error < 0.05
    let hll = lines
        .iter()
        .find(|l| l.contains(",hll,") && l.contains("distinct_count"))
        .expect("hll row");
    let rel_error: f64 = hll.rsplit(',').next().unwrap().parse().unwrap();
    assert!(rel_error < 0.05, "hll rel_error {rel_error}");
}
