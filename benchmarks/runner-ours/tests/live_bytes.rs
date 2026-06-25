//! `live_bytes` heap-delta test, isolated in its own integration-test binary.
//!
//! `measure_live` reads a single process-wide allocation counter, so a heap
//! delta is only stable when no other thread allocates or frees concurrently.
//! Cargo runs the tests within one binary in parallel by default; this file
//! holds exactly one test so its `run` call sees no concurrent allocator
//! traffic from sibling tests. Production runs (the binary's `main`) are
//! single-threaded, so they are stable for the same reason.

#[test]
fn live_bytes_is_nonzero() {
    let rows = runner_ours::run(10_000, 3);
    // `live_bytes` is column index 9 (0-based) in the 13-column schema. Assert
    // it is a positive number, not an exact value: the heap size is
    // allocator-dependent.
    for r in &rows[1..] {
        let lb: u64 = r
            .split(',')
            .nth(9)
            .unwrap_or("")
            .parse()
            .unwrap_or_else(|_| panic!("live_bytes not parseable in row: {r}"));
        assert!(lb > 0, "live_bytes zero in row: {r}");
    }
}
