// tests/tpch_validation.rs
use csv::ReaderBuilder;
use std::collections::HashSet;
use std::error::Error;

use cpc::CpcSketch;
use hll::HllSketch;
use theta::ThetaSketch;

fn load_column(path: &str, column: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(path)?;
    let headers = rdr.headers()?.clone();
    let idx = headers
        .iter()
        .position(|h| h == column)
        .ok_or("Column not found")?;
    let mut vals = Vec::new();
    for record in rdr.records() {
        let rec = record?;
        vals.push(rec.get(idx).unwrap().to_string());
    }
    Ok(vals)
}

#[test]
fn test_cpc_on_lineitem_orderkeys() -> Result<(), Box<dyn Error>> {
    let data = load_column("tests/data/lineitem.csv", "l_orderkey")?;
    let truth: HashSet<_> = data.iter().cloned().collect();
    let true_count = truth.len() as f64;

    let mut sk = CpcSketch::new(12);
    for v in &data {
        sk.update(v);
    }
    let est = sk.estimate();
    let rel_err =
        ((est - true_count).abs() / true_count).max((true_count - est).abs() / true_count);
    assert!(
        rel_err < 0.05,
        "CPC err >5%: est {}, true {}",
        est,
        true_count
    );
    Ok(())
}

#[test]
fn test_hll_on_lineitem_orderkeys() -> Result<(), Box<dyn Error>> {
    let data = load_column("tests/data/lineitem.csv", "l_orderkey")?;
    let truth: HashSet<_> = data.iter().cloned().collect();
    let true_count = truth.len() as f64;

    let mut sk = HllSketch::new(12);
    for v in &data {
        sk.update(v);
    }
    let est = sk.estimate();
    let rel_err =
        ((est - true_count).abs() / true_count).max((true_count - est).abs() / true_count);
    assert!(
        rel_err < 0.05,
        "HLL err >5%: est {}, true {}",
        est,
        true_count
    );
    Ok(())
}

#[test]
fn test_theta_union_and_intersection() -> Result<(), Box<dyn Error>> {
    let data = load_column("tests/data/lineitem.csv", "l_orderkey")?;
    let split = data.len() / 2;
    let (p1, p2) = (&data[..split], &data[split..]);

    let s1: HashSet<_> = p1.iter().cloned().collect();
    let s2: HashSet<_> = p2.iter().cloned().collect();
    let true_u = s1.union(&s2).count() as f64;
    let true_i = s1.intersection(&s2).count() as f64;

    let mut sk1 = ThetaSketch::new(1024);
    let mut sk2 = ThetaSketch::new(1024);
    for v in p1 {
        sk1.update(v);
    }
    for v in p2 {
        sk2.update(v);
    }

    let est_u = sk1.union(&sk2).estimate();
    let err_u = ((est_u - true_u).abs() / true_u).max((true_u - est_u).abs() / true_u);
    assert!(err_u < 0.10, "Theta union err >10%");

    let est_i = sk1.intersect(&sk2).estimate();
    let err_i = ((est_i - true_i).abs() / true_i).max((true_i - est_i).abs() / true_i);
    assert!(err_i < 0.15, "Theta intersect err >15%");

    Ok(())
}
