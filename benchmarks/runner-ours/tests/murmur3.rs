//! Murmur3 isolation rows: every hash-based sketch emits both an `ours` (xxh3)
//! and an `ours-murmur3` row, and the comparison-based KLL emits neither
//! murmur3 variant.

#[test]
fn header_has_ci_columns() {
    let rows = runner_ours::run(100, 2);
    assert_eq!(
        rows[0],
        "implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,throughput_stddev,throughput_ci_low,throughput_ci_high,bytes,live_bytes,estimate,exact,rel_error"
    );
}

#[test]
fn murmur3_rows_present() {
    let rows = runner_ours::run(1000, 3);

    // The set of `implementation` labels (first CSV field) must contain both
    // the xxh3 default (`ours`) and the murmur3 variant (`ours-murmur3`).
    let impls: std::collections::HashSet<_> = rows[1..]
        .iter()
        .map(|r| r.split(',').next().unwrap().to_string())
        .collect();
    assert!(impls.contains("ours"), "missing `ours` rows");
    assert!(
        impls.contains("ours-murmur3"),
        "missing `ours-murmur3` rows"
    );

    // murmur3 is only emitted for the five hash-based sketches, never for kll.
    let mm_sketches: std::collections::HashSet<_> = rows
        .iter()
        .filter(|r| r.starts_with("ours-murmur3,"))
        .map(|r| r.split(',').nth(1).unwrap().to_string())
        .collect();
    assert!(
        !mm_sketches.contains("kll"),
        "kll must not have a murmur3 row"
    );
    // The murmur3 set is exactly the five hash-based sketches.
    let expected: std::collections::HashSet<String> = ["hll", "cpc", "theta", "bloom", "countmin"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(mm_sketches, expected, "murmur3 sketch set mismatch");
}
