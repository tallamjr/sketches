use sketches::hll::{AdaptiveHllPlusPlus, HllPlusPlusSketch, HllPlusPlusSparseSketch, HllSketch};
use std::time::Instant;

fn main() {
    println!("=== HyperLogLog Memory Optimization Benchmark ===\n");

    // Test different precisions and cardinalities
    let precisions = vec![8, 10, 12, 14];
    let cardinalities = vec![100, 1000, 10000, 100000];

    for &p in &precisions {
        println!("Precision p={} (2^{} = {} registers)", p, p, 1usize << p);
        println!("{}", "=".repeat(60));

        for &n in &cardinalities {
            println!("\nCardinality: {n} unique elements");
            benchmark_memory_usage(p, n);
        }
        println!();
    }

    // Specific memory efficiency test
    println!("\n=== Memory Efficiency Analysis ===");
    memory_efficiency_analysis();

    // Transition behavior analysis
    println!("\n=== Adaptive Transition Analysis ===");
    adaptive_transition_analysis();
}

fn benchmark_memory_usage(precision: u8, cardinality: usize) {
    // Generate test data
    let data: Vec<String> = (0..cardinality).map(|i| format!("item_{i}")).collect();

    // Standard HLL
    let mut hll_standard = HllSketch::new(precision);
    let start = Instant::now();
    for item in &data {
        hll_standard.update(item);
    }
    let hll_standard_time = start.elapsed();
    let hll_standard_memory = hll_standard.memory_usage();
    let hll_standard_estimate = hll_standard.estimate();

    // HLL++
    let mut hll_plus = HllPlusPlusSketch::new(precision);
    let start = Instant::now();
    for item in &data {
        hll_plus.update(item);
    }
    let hll_plus_time = start.elapsed();
    let hll_plus_memory = hll_plus.memory_usage();
    let hll_plus_estimate = hll_plus.estimate();

    // HLL++ Sparse
    let mut hll_sparse = HllPlusPlusSparseSketch::new(precision);
    let start = Instant::now();
    for item in &data {
        hll_sparse.update(item);
    }
    let hll_sparse_time = start.elapsed();
    let hll_sparse_memory = hll_sparse.memory_usage();
    let hll_sparse_estimate = hll_sparse.estimate();

    // Adaptive HLL++
    let mut hll_adaptive = AdaptiveHllPlusPlus::new(precision);
    let start = Instant::now();
    for item in &data {
        hll_adaptive.update(item);
    }
    let hll_adaptive_time = start.elapsed();
    let hll_adaptive_memory = hll_adaptive.memory_usage();
    let hll_adaptive_estimate = hll_adaptive.estimate();

    // Results
    println!("┌─────────────────┬──────────┬──────────┬──────────┬──────────────┐");
    println!("│ Implementation  │ Memory   │ Time     │ Estimate │ Error %      │");
    println!("├─────────────────┼──────────┼──────────┼──────────┼──────────────┤");
    println!(
        "│ Standard HLL    │ {:>7}B │ {:>7.2}ms│ {:>7.0}  │ {:>10.2}%  │",
        hll_standard_memory,
        hll_standard_time.as_secs_f64() * 1000.0,
        hll_standard_estimate,
        error_percentage(hll_standard_estimate, cardinality as f64)
    );
    println!(
        "│ HLL++           │ {:>7}B │ {:>7.2}ms│ {:>7.0}  │ {:>10.2}%  │",
        hll_plus_memory,
        hll_plus_time.as_secs_f64() * 1000.0,
        hll_plus_estimate,
        error_percentage(hll_plus_estimate, cardinality as f64)
    );
    println!(
        "│ HLL++ Sparse    │ {:>7}B │ {:>7.2}ms│ {:>7.0}  │ {:>10.2}%  │",
        hll_sparse_memory,
        hll_sparse_time.as_secs_f64() * 1000.0,
        hll_sparse_estimate,
        error_percentage(hll_sparse_estimate, cardinality as f64)
    );
    println!(
        "│ Adaptive HLL++  │ {:>7}B │ {:>7.2}ms│ {:>7.0}  │ {:>10.2}%  │",
        hll_adaptive_memory,
        hll_adaptive_time.as_secs_f64() * 1000.0,
        hll_adaptive_estimate,
        error_percentage(hll_adaptive_estimate, cardinality as f64)
    );
    println!("└─────────────────┴──────────┴──────────┴──────────┴──────────────┘");

    // Memory savings analysis
    let dense_baseline = hll_plus_memory.max(hll_standard_memory);
    println!("\nMemory Savings vs Dense Baseline ({dense_baseline} bytes):");
    println!(
        "  HLL++ Sparse: {:.1}% savings",
        100.0 * (1.0 - hll_sparse_memory as f64 / dense_baseline as f64)
    );
    println!(
        "  Adaptive HLL: {:.1}% savings",
        100.0 * (1.0 - hll_adaptive_memory as f64 / dense_baseline as f64)
    );

    if hll_adaptive.is_sparse() {
        println!(
            "  Adaptive is using: SPARSE mode ({} non-zero registers)",
            hll_adaptive.sparse_size().unwrap_or(0)
        );
    } else {
        println!("  Adaptive is using: DENSE mode");
    }
}

fn memory_efficiency_analysis() {
    let precision = 12;
    let test_sizes = vec![10, 50, 100, 200, 500, 1000, 2000, 5000];

    println!("Testing memory efficiency across different cardinalities (p={precision})");
    println!("┌─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐");
    println!("│ Cardinality │ Dense (B)   │ Sparse (B)  │ Adaptive (B)│ Best Choice │");
    println!("├─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤");

    for &size in &test_sizes {
        let data: Vec<String> = (0..size).map(|i| format!("test_{i}")).collect();

        let mut dense = HllPlusPlusSketch::new(precision);
        let mut sparse = HllPlusPlusSparseSketch::new(precision);
        let mut adaptive = AdaptiveHllPlusPlus::new(precision);

        for item in &data {
            dense.update(item);
            sparse.update(item);
            adaptive.update(item);
        }

        let dense_mem = dense.memory_usage();
        let sparse_mem = sparse.memory_usage();
        let adaptive_mem = adaptive.memory_usage();

        let best = if sparse_mem < dense_mem {
            "Sparse"
        } else {
            "Dense"
        };
        let adaptive_type = if adaptive.is_sparse() {
            "Sparse"
        } else {
            "Dense"
        };

        println!(
            "│ {:>11} │ {:>11} │ {:>11} │ {:>11} │ {:>11} │",
            size,
            dense_mem,
            sparse_mem,
            adaptive_mem,
            format!("{} ({})", best, adaptive_type)
        );
    }
    println!("└─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘");
}

fn adaptive_transition_analysis() {
    println!("Analyzing adaptive transition behavior (p=10, threshold=0.7)");

    let mut adaptive = AdaptiveHllPlusPlus::with_threshold_ratio(10, 0.7);
    let mut transition_point = None;

    for i in 1..=2000 {
        adaptive.update(&format!("item_{i}"));

        if i % 100 == 0 || i <= 10 {
            let memory = adaptive.memory_usage();
            let is_sparse = adaptive.is_sparse();
            let sparse_size = adaptive.sparse_size().unwrap_or(0);

            println!(
                "Items: {:>4} | Memory: {:>5}B | Mode: {:>6} | Registers: {:>4}",
                i,
                memory,
                if is_sparse { "SPARSE" } else { "DENSE" },
                sparse_size
            );

            if !is_sparse && transition_point.is_none() {
                transition_point = Some(i);
            }
        }
    }

    if let Some(transition) = transition_point {
        println!("\n🔄 Transition occurred around {transition} items");
    } else {
        println!("\n✅ Remained in sparse mode throughout the test");
    }
}

fn error_percentage(estimate: f64, true_value: f64) -> f64 {
    ((estimate - true_value).abs() / true_value) * 100.0
}

#[cfg(feature = "optimized")]
fn print_optimization_features() {
    println!("✅ Optimized features enabled:");
    println!("  - 6-bit packed registers");
    println!("  - Fast xxHash hashing");
    println!("  - SIMD batch operations");
    println!("  - CompactHashTable for sparse mode");
    println!("  - Rayon parallel processing");
}

#[cfg(not(feature = "optimized"))]
fn print_optimization_features() {
    println!("⚠️  Optimized features disabled:");
    println!("  - Using 8-bit unpacked registers");
    println!("  - Using standard Rust hasher");
    println!("  - Using BTreeMap for sparse mode");
    println!("  - Sequential processing only");
}
