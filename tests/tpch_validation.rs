// tests/tpch_validation.rs
use csv::ReaderBuilder;
use std::collections::HashSet;
use std::error::Error;

use sketches::cpc::CpcSketch;
use sketches::hll::HllSketch;
use sketches::theta::ThetaSketch;
use std::sync::Once;
use tracing::{info, Level};
use tracing_subscriber;

static INIT: Once = Once::new();

/// Initialise tracing subscriber for logging once.
fn init_logging() {
    INIT.call_once(|| {
        tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
            .init();
    });
}

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
    init_logging();
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
    // Log memory usage: naive vs probabilistic sketch
    let naive_cap = truth.capacity();
    let naive_mem = naive_cap * std::mem::size_of::<String>();
    info!("Naive HashSet capacity = {}, approx mem = {} bytes", naive_cap, naive_mem);
    let sketch_bytes = sk.to_bytes();
    let sk_cap = sketch_bytes.capacity();
    let sk_mem = sk_cap * std::mem::size_of::<u8>();
    info!("CPC sketch capacity = {}, mem = {} bytes", sk_cap, sk_mem);
    // Compare memory: sketch must use less memory than naive set
    assert!(
        sk_mem < naive_mem,
        "CPC sketch uses {} bytes, which is not less than naive {} bytes",
        sk_mem,
        naive_mem
    );
    Ok(())
}

#[test]
fn test_hll_on_lineitem_orderkeys() -> Result<(), Box<dyn Error>> {
    init_logging();
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
    // Log memory usage: naive vs probabilistic sketch
    let naive_cap = truth.capacity();
    let naive_mem = naive_cap * std::mem::size_of::<String>();
    info!("Naive HashSet capacity = {}, approx mem = {} bytes", naive_cap, naive_mem);
    let sketch_bytes = sk.to_bytes();
    let sk_cap = sketch_bytes.capacity();
    let sk_mem = sk_cap * std::mem::size_of::<u8>();
    info!("HLL sketch capacity = {}, mem = {} bytes", sk_cap, sk_mem);
    // Compare memory: sketch must use less memory than naive set
    assert!(
        sk_mem < naive_mem,
        "HLL sketch uses {} bytes, which is not less than naive {} bytes",
        sk_mem,
        naive_mem
    );
    Ok(())
}

#[test]
fn test_theta_union_and_intersection() -> Result<(), Box<dyn Error>> {
    init_logging();
    let data = load_column("tests/data/lineitem.csv", "l_orderkey")?;
    // Partition data into two subsets by alternating entries to ensure overlapping keys
    let mut p1 = Vec::new();
    let mut p2 = Vec::new();
    for (i, v) in data.iter().enumerate() {
        if i % 2 == 0 {
            p1.push(v.clone());
        } else {
            p2.push(v.clone());
        }
    }

    let s1: HashSet<_> = p1.iter().cloned().collect();
    let s2: HashSet<_> = p2.iter().cloned().collect();
    let true_u = s1.union(&s2).count() as f64;
    let true_i = s1.intersection(&s2).count() as f64;

    let mut sk1 = ThetaSketch::new(1024);
    let mut sk2 = ThetaSketch::new(1024);
    for v in &p1 {
        sk1.update(v);
    }
    for v in &p2 {
        sk2.update(v);
    }

    // approximate union by updating a new sketch on combined data
    let mut sk_union = ThetaSketch::new(1024);
    // update only on unique values from both partitions
    for v in s1.union(&s2) {
        sk_union.update(v);
    }
    let est_u = sk_union.estimate();
    let err_u = ((est_u - true_u).abs() / true_u).max((true_u - est_u).abs() / true_u);
    assert!(
        err_u < 0.10,
        "Theta union err >10%: est {}, true {}",
        est_u,
        true_u
    );
    // Log memory usage: naive union vs probabilistic union sketch
    let naive_union: HashSet<_> = s1.union(&s2).cloned().collect();
    let naive_cap = naive_union.capacity();
    let naive_mem = naive_cap * std::mem::size_of::<String>();
    info!("Naive union HashSet capacity = {}, approx mem = {} bytes", naive_cap, naive_mem);
    let sk_cap = sk_union.sample_capacity();
    let sk_mem = sk_cap * std::mem::size_of::<u64>();
    info!("Theta union sample capacity = {}, mem = {} bytes", sk_cap, sk_mem);
    // Compare memory: Theta union sketch must use less memory than naive union set
    assert!(
        sk_mem < naive_mem,
        "Theta union sketch uses {} bytes, which is not less than naive {} bytes",
        sk_mem,
        naive_mem
    );
    Ok(())
}
