use sketches::hll::HllSketch;
use sketches::linear::{HybridCounter, LinearCounter};
use std::time::Instant;

fn main() {
    println!("=== Cardinality Estimation Optimization Demo ===\n");

    // Linear Counter vs HyperLogLog Comparison
    println!("1. Linear Counter vs HyperLogLog for Small Cardinalities");
    println!("{}", "-".repeat(55));

    let test_sizes = [10, 50, 100, 200, 500, 1000];

    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<12}",
        "True", "Linear", "HLL", "Linear Err", "HLL Err"
    );
    println!("{}", "-".repeat(60));

    for &n in &test_sizes {
        let mut lc = LinearCounter::new(8192, false);
        let mut hll = HllSketch::new(12); // 2^12 = 4096 buckets

        // Add n unique items to both
        for i in 0..n {
            let item = format!("item_{}", i);
            lc.update(&item);
            hll.update(&item);
        }

        let lc_estimate = lc.estimate();
        let hll_estimate = hll.estimate();

        let lc_error = (lc_estimate - n as f64).abs() / n as f64 * 100.0;
        let hll_error = (hll_estimate - n as f64).abs() / n as f64 * 100.0;

        println!(
            "{:<8} {:<12.1} {:<12.1} {:<12.1}% {:<12.1}%",
            n, lc_estimate, hll_estimate, lc_error, hll_error
        );
    }

    println!("\nObservation: Linear Counter is more accurate for small cardinalities!");

    println!("\n{}", "=".repeat(60));

    // Memory Efficiency Analysis
    println!("\n2. Memory Usage Comparison");
    println!("{}", "-".repeat(25));

    let lc_8k = LinearCounter::new(8192, false);
    let lc_stats = lc_8k.statistics();

    println!("Linear Counter (8K bits): {} bytes", lc_stats.memory_usage);
    println!("HyperLogLog (lg_k=12):    ~{} bytes", 4096); // 2^12 buckets
    println!("HyperLogLog (lg_k=14):    ~{} bytes", 16384); // 2^14 buckets

    println!("\n{}", "=".repeat(60));

    // Hybrid Counter Demo
    println!("\n3. Hybrid Counter: Automatic Optimization");
    println!("{}", "-".repeat(40));

    let mut hybrid = HybridCounter::with_range(10000);
    println!("Created hybrid counter for max 10,000 items");

    // Track mode changes during data ingestion
    let checkpoints = [50, 100, 200, 500, 1000, 2000];

    println!(
        "\n{:<10} {:<15} {:<12} {:<10}",
        "Items", "Mode", "Estimate", "Error %"
    );
    println!("{}", "-".repeat(50));

    let mut items_added = 0;

    for &checkpoint in &checkpoints {
        // Add items up to checkpoint
        while items_added < checkpoint {
            hybrid.update(&format!("user_{}", items_added));
            items_added += 1;
        }

        let estimate = hybrid.estimate();
        let error = (estimate - checkpoint as f64).abs() / checkpoint as f64 * 100.0;

        println!(
            "{:<10} {:<15} {:<12.1} {:<10.1}%",
            checkpoint,
            hybrid.mode(),
            estimate,
            error
        );
    }

    let final_stats = hybrid.statistics();
    println!("\nFinal Statistics:");
    println!("  Mode: {}", final_stats.mode);
    println!(
        "  Memory: {:.1} KB",
        final_stats.memory_usage as f64 / 1024.0
    );
    if let Some(fill_ratio) = final_stats.fill_ratio {
        println!("  Fill ratio: {:.1}%", fill_ratio * 100.0);
    }

    println!("\n{}", "=".repeat(60));

    // Performance Benchmarking
    println!("\n4. Performance Benchmarking");
    println!("{}", "-".repeat(25));

    let n_items = 100000;
    println!("Benchmarking with {} items", n_items);

    // Linear Counter performance
    let start = Instant::now();
    let mut lc_perf = LinearCounter::new(16384, false);
    for i in 0..n_items {
        lc_perf.update(&format!("item_{}", i % 10000)); // Some duplicates
    }
    let lc_time = start.elapsed();

    // HyperLogLog performance
    let start = Instant::now();
    let mut hll_perf = HllSketch::new(12);
    for i in 0..n_items {
        hll_perf.update(&format!("item_{}", i % 10000));
    }
    let hll_time = start.elapsed();

    // Hybrid Counter performance
    let start = Instant::now();
    let mut hybrid_perf = HybridCounter::with_range(50000);
    for i in 0..n_items {
        hybrid_perf.update(&format!("item_{}", i % 10000));
    }
    let hybrid_time = start.elapsed();

    println!("\nPerformance Results:");
    println!(
        "  Linear Counter: {:?} ({:.0} items/sec)",
        lc_time,
        n_items as f64 / lc_time.as_secs_f64()
    );
    println!(
        "  HyperLogLog:    {:?} ({:.0} items/sec)",
        hll_time,
        n_items as f64 / hll_time.as_secs_f64()
    );
    println!(
        "  Hybrid Counter: {:?} ({:.0} items/sec)",
        hybrid_time,
        n_items as f64 / hybrid_time.as_secs_f64()
    );

    // Accuracy comparison
    println!("\nAccuracy Results (true cardinality: 10,000):");
    println!(
        "  Linear Counter: {:.1} (error: {:.1}%)",
        lc_perf.estimate(),
        (lc_perf.estimate() - 10000.0).abs() / 10000.0 * 100.0
    );
    println!(
        "  HyperLogLog:    {:.1} (error: {:.1}%)",
        hll_perf.estimate(),
        (hll_perf.estimate() - 10000.0).abs() / 10000.0 * 100.0
    );
    println!(
        "  Hybrid Counter: {:.1} (error: {:.1}%)",
        hybrid_perf.estimate(),
        (hybrid_perf.estimate() - 10000.0).abs() / 10000.0 * 100.0
    );

    println!("\n{}", "=".repeat(60));

    // Fill Ratio and Transition Analysis
    println!("\n5. Fill Ratio and Transition Analysis");
    println!("{}", "-".repeat(35));

    let mut lc_analysis = LinearCounter::new(4096, false);

    println!(
        "{:<8} {:<12} {:<10} {:<15}",
        "Items", "Fill Ratio", "Estimate", "Should Transition"
    );
    println!("{}", "-".repeat(50));

    for i in (100..=2000).step_by(200) {
        // Clear and re-add items
        lc_analysis.clear();
        for j in 0..i {
            lc_analysis.update(&format!("item_{}", j));
        }

        let stats = lc_analysis.statistics();
        println!(
            "{:<8} {:<12.1}% {:<10.1} {:<15}",
            i,
            stats.fill_ratio * 100.0,
            stats.estimated_cardinality,
            if stats.should_transition { "Yes" } else { "No" }
        );
    }

    println!("\n{}", "=".repeat(60));

    // Error Rate vs Bit Array Size
    println!("\n6. Error Rate vs Memory Trade-off");
    println!("{}", "-".repeat(30));

    let bit_sizes = [1024, 2048, 4096, 8192, 16384];
    let test_cardinality = 500;

    println!(
        "{:<10} {:<12} {:<10} {:<10}",
        "Bits", "Memory (KB)", "Estimate", "Error %"
    );
    println!("{}", "-".repeat(45));

    for &bits in &bit_sizes {
        let mut lc_size = LinearCounter::new(bits, false);

        // Add test items
        for i in 0..test_cardinality {
            lc_size.update(&format!("item_{}", i));
        }

        let estimate = lc_size.estimate();
        let error = (estimate - test_cardinality as f64).abs() / test_cardinality as f64 * 100.0;
        let memory_kb = lc_size.statistics().memory_usage as f64 / 1024.0;

        println!(
            "{:<10} {:<12.1} {:<10.1} {:<10.1}%",
            bits, memory_kb, estimate, error
        );
    }

    println!("\n{}", "=".repeat(60));

    // Real-world Use Cases
    println!("\n7. Real-world Applications");
    println!("{}", "-".repeat(25));

    println!("Linear Counter is ideal for:");
    println!("• Small-scale unique visitor counting (< 1K daily users)");
    println!("• Database cardinality estimation for small tables");
    println!("• IoT device counting in small networks");
    println!("• A/B test participant tracking");
    println!("• Cache key diversity monitoring");

    println!("\nHybrid Counter benefits:");
    println!("• Unknown cardinality ranges (could be small or large)");
    println!("• Growing datasets that start small");
    println!("• Resource-constrained environments");
    println!("• Streaming data with varying load patterns");

    println!("\nOptimal Configuration Guidelines:");
    println!("• Use Linear Counter when max cardinality < 1,000");
    println!("• Use HyperLogLog when cardinality > 10,000");
    println!("• Use Hybrid Counter for unknown/variable ranges");
    println!("• Size bit array ~10x expected cardinality for Linear Counter");

    println!("\n{}", "=".repeat(60));

    // Advanced: Custom Transition Strategies
    println!("\n8. Custom Transition Strategies");
    println!("{}", "-".repeat(30));

    // Conservative transition (higher threshold)
    let mut conservative = HybridCounter::new(8192, 12, 1000);

    // Aggressive transition (lower threshold)
    let mut aggressive = HybridCounter::new(4096, 12, 200);

    // Add same data to both
    for i in 0..800 {
        let item = format!("user_{}", i);
        conservative.update(&item);
        aggressive.update(&item);
    }

    println!("After 800 items:");
    println!(
        "  Conservative hybrid: {} (memory: {:.1} KB)",
        conservative.mode(),
        conservative.statistics().memory_usage as f64 / 1024.0
    );
    println!(
        "  Aggressive hybrid:   {} (memory: {:.1} KB)",
        aggressive.mode(),
        aggressive.statistics().memory_usage as f64 / 1024.0
    );

    println!("\nTrade-offs:");
    println!("• Conservative: Better accuracy for medium cardinalities, more memory");
    println!("• Aggressive: Lower memory usage, transitions early");

    println!("\nDemo completed successfully!");
}
