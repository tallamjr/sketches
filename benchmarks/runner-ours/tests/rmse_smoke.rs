#[test]
fn rmse_rows_are_well_formed_and_near_theoretical() {
    // 20 trials x 10k is enough to land theta/hll near the 1/sqrt(4096) ~= 1.56% floor.
    let lines = runner_ours::run_rmse(20, 10_000);
    assert_eq!(
        lines[0],
        "implementation,sketch,lg_k,trials,n_per_trial,rmse,mean_rel_error,max_rel_error"
    );
    // one row per distinct-count sketch
    for s in ["theta", "hll", "cpc"] {
        let row = lines
            .iter()
            .find(|l| l.contains(&format!("ours,{s},")))
            .expect("row");
        let rmse: f64 = row.split(',').nth(5).unwrap().parse().unwrap();
        // RMSE must be positive and within a generous factor of the ~1.56% theoretical floor.
        assert!(
            rmse > 0.0 && rmse < 0.06,
            "{s} rmse {rmse} out of expected band"
        );
    }
}
