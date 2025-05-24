use sketches::{HllSketch, HllPlusPlusSketch, HllPlusPlusSparseSketch};
use std::time::Instant;

fn benchmark_sketch<F>(name: &str, mut update_fn: F, n: usize) -> (f64, u128)
where
    F: FnMut(&str),
{
    let start = Instant::now();
    
    for i in 0..n {
        update_fn(&format!("element_{}", i));
    }
    
    let duration = start.elapsed();
    let estimate = 0.0; // Would need to return sketch to get estimate
    
    println!("{}: {:.2?} for {} updates", name, duration, n);
    
    (estimate, duration.as_micros())
}

fn main() {
    println!("=== HyperLogLog Variants Comparison ===\n");
    
    let sizes = vec![1000, 10000, 100000, 1000000];
    
    for &n in &sizes {
        println!("\nTesting with {} unique elements:", n);
        println!("-".repeat(50));
        
        // Standard HLL
        let mut hll = HllSketch::new(14);
        let hll_start = Instant::now();
        for i in 0..n {
            hll.update(&format!("element_{}", i));
        }
        let hll_duration = hll_start.elapsed();
        let hll_estimate = hll.estimate();
        
        // HLL++
        let mut hll_pp = HllPlusPlusSketch::new(14);
        let hll_pp_start = Instant::now();
        for i in 0..n {
            hll_pp.update(&format!("element_{}", i));
        }
        let hll_pp_duration = hll_pp_start.elapsed();
        let hll_pp_estimate = hll_pp.estimate();
        
        // HLL++ Sparse
        let mut hll_pp_sparse = HllPlusPlusSparseSketch::new(14);
        let hll_pp_sparse_start = Instant::now();
        for i in 0..n {
            hll_pp_sparse.update(&format!("element_{}", i));
        }
        let hll_pp_sparse_duration = hll_pp_sparse_start.elapsed();
        let hll_pp_sparse_estimate = hll_pp_sparse.estimate();
        
        // Results
        println!("Standard HLL:");
        println!("  Time: {:.2?}", hll_duration);
        println!("  Estimate: {:.0} (error: {:.2}%)", 
                 hll_estimate, 
                 ((hll_estimate - n as f64).abs() / n as f64) * 100.0);
        
        println!("HLL++:");
        println!("  Time: {:.2?}", hll_pp_duration);
        println!("  Estimate: {:.0} (error: {:.2}%)", 
                 hll_pp_estimate, 
                 ((hll_pp_estimate - n as f64).abs() / n as f64) * 100.0);
        
        println!("HLL++ Sparse:");
        println!("  Time: {:.2?}", hll_pp_sparse_duration);
        println!("  Estimate: {:.0} (error: {:.2}%)", 
                 hll_pp_sparse_estimate, 
                 ((hll_pp_sparse_estimate - n as f64).abs() / n as f64) * 100.0);
    }
    
    println!("\n=== Memory Usage Comparison ===");
    println!("Standard HLL: {} bytes", std::mem::size_of::<HllSketch>());
    println!("HLL++: {} bytes", std::mem::size_of::<HllPlusPlusSketch>());
    println!("HLL++ Sparse: {} bytes", std::mem::size_of::<HllPlusPlusSparseSketch>());
}