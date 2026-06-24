//! Smoke test: the emitted rows match the shared schema and the Apache HLL
//! row is present and accurate.

#[test]
fn emits_apache_rust_rows() {
    let lines = runner_apache_rust::run(10_000);
    assert!(lines[0].starts_with("implementation,sketch,dataset,op,n,"));
    let hll = lines
        .iter()
        .find(|l| l.contains("apache-rust,hll,") && l.contains("distinct_count"))
        .expect("apache hll row");
    let rel_error: f64 = hll.rsplit(',').next().unwrap().parse().unwrap();
    assert!(rel_error < 0.05, "apache hll rel_error {rel_error}");
}
