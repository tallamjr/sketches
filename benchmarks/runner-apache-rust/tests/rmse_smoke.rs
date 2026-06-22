#[test]
fn apache_rmse_rows_well_formed() {
    let lines = runner_apache_rust::run_rmse(20, 10_000);
    assert!(lines[0].starts_with("implementation,sketch,lg_k,trials,n_per_trial,rmse,"));
    for s in ["theta", "hll", "cpc"] {
        let row = lines
            .iter()
            .find(|l| l.contains(&format!("apache-rust,{s},")))
            .expect("row");
        let rmse: f64 = row.split(',').nth(5).unwrap().parse().unwrap();
        assert!(rmse > 0.0 && rmse < 0.06, "{s} rmse {rmse}");
    }
}
