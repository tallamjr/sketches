use sketches::countmin::{CountMinSketch, CountSketch};
use std::collections::HashMap;

fn main() {
    println!("=== Frequency Estimation Demo ===\n");
    
    // Count-Min Sketch Demo
    println!("1. Count-Min Sketch");
    println!("{}", "-".repeat(20));
    
    let mut cm = CountMinSketch::new(1000, 5, false, false);
    println!("Created Count-Min sketch (1000 buckets, 5 hash functions)");
    
    // Simulate a data stream
    let data_stream = vec![
        "apple", "banana", "apple", "cherry", "apple", "banana",
        "date", "apple", "elderberry", "apple", "banana", "apple",
        "fig", "grape", "apple", "banana", "apple", "cherry"
    ];
    
    println!("Processing data stream with {} items", data_stream.len());
    
    // Process the stream
    for item in &data_stream {
        cm.increment(item);
    }
    
    // Count actual frequencies for comparison
    let mut actual_counts = HashMap::new();
    for item in &data_stream {
        *actual_counts.entry(*item).or_insert(0) += 1;
    }
    
    println!("\nFrequency estimation results:");
    println!("{:<12} {:<8} {:<10} {:<8}", "Item", "Actual", "Estimated", "Error");
    println!("{}", "-".repeat(40));
    
    for (item, &actual) in &actual_counts {
        let estimated = cm.estimate(item);
        let error = if actual > 0 { 
            ((estimated as f64 - actual as f64) / actual as f64 * 100.0).abs()
        } else { 
            0.0 
        };
        println!("{:<12} {:<8} {:<10} {:<7.1}%", item, actual, estimated, error);
    }
    
    // Test non-existent item
    let non_existent = cm.estimate(&"watermelon");
    println!("{:<12} {:<8} {:<10} {:<8}", "watermelon", 0, non_existent, "N/A");
    
    println!("\nSketch statistics:");
    let stats = cm.statistics();
    println!("  Dimensions: {}x{} = {} cells", stats.width, stats.depth, stats.total_cells);
    println!("  Fill ratio: {:.4}", stats.fill_ratio);
    println!("  Total count: {}", stats.total_count);
    
    println!("\n{}", "=".repeat(50));
    
    // Conservative Update Demo
    println!("\n2. Conservative Update Count-Min Sketch");
    println!("{}", "-".repeat(35));
    
    let mut cm_conservative = CountMinSketch::new(100, 5, false, true);
    println!("Created conservative Count-Min sketch");
    
    // Add the same data
    for item in &data_stream {
        cm_conservative.increment(item);
    }
    
    println!("\nComparison: Standard vs Conservative Update");
    println!("{:<12} {:<10} {:<13} {:<10}", "Item", "Standard", "Conservative", "Difference");
    println!("{}", "-".repeat(50));
    
    for (item, &actual) in &actual_counts {
        let standard_est = cm.estimate(item);
        let conservative_est = cm_conservative.estimate(item);
        let diff = if standard_est >= conservative_est {
            standard_est - conservative_est
        } else {
            conservative_est - standard_est
        };
        println!("{:<12} {:<10} {:<13} {:<10}", item, standard_est, conservative_est, diff);
    }
    
    println!("\n{}", "=".repeat(50));
    
    // Count Sketch Demo
    println!("\n3. Count Sketch (supports negative updates)");
    println!("{}", "-".repeat(40));
    
    let mut cs = CountSketch::new(1000, 5);
    println!("Created Count sketch (1000 buckets, 5 hash functions)");
    
    // Simulate positive and negative updates
    cs.update(&"balance", 100);   // Initial balance
    cs.update(&"balance", -30);   // Withdrawal
    cs.update(&"balance", 50);    // Deposit
    cs.update(&"balance", -20);   // Withdrawal
    
    cs.update(&"score", 75);      // Initial score
    cs.update(&"score", 25);      // Bonus points
    cs.update(&"score", -10);     // Penalty
    
    println!("\nBalance tracking (supports negative updates):");
    println!("Balance: {}", cs.estimate(&"balance"));    // Should be ~100
    println!("Score: {}", cs.estimate(&"score"));        // Should be ~90
    
    println!("\n{}", "=".repeat(50));
    
    // Error Bounds Demo
    println!("\n4. Count-Min Sketch with Error Bounds");
    println!("{}", "-".repeat(35));
    
    let cm_bounded = CountMinSketch::with_error_bounds(0.01, 0.01, false, false);
    let stats_bounded = cm_bounded.statistics();
    
    println!("Created sketch with 1% error rate and 1% failure probability");
    println!("Calculated dimensions: {}x{}", stats_bounded.width, stats_bounded.depth);
    println!("Theory: width ≈ e/ε ≈ {:.0}, depth ≈ ln(1/δ) ≈ {:.0}", 
             std::f64::consts::E / 0.01, (1.0 / 0.01_f64).ln());
    
    println!("\n{}", "=".repeat(50));
    
    // Heavy Hitters Demo
    println!("\n5. Heavy Hitters Detection");
    println!("{}", "-".repeat(25));
    
    let mut cm_heavy = CountMinSketch::new(500, 5, false, false);
    
    // Simulate Zipfian distribution (some items much more frequent)
    let zipf_data = [
        ("popular1", 1000), ("popular2", 800), ("popular3", 600),
        ("medium1", 200), ("medium2", 150), ("medium3", 100),
        ("rare1", 50), ("rare2", 30), ("rare3", 20), ("rare4", 10)
    ];
    
    for &(item, count) in &zipf_data {
        cm_heavy.update(&item, count);
    }
    
    println!("Added items with Zipfian distribution");
    
    let heavy_threshold = 300;
    let heavy_hitters = cm_heavy.heavy_hitters(heavy_threshold);
    
    println!("Heavy hitters (count ≥ {}):", heavy_threshold);
    for &(item, count) in &zipf_data {
        let estimated = cm_heavy.estimate(&item);
        if estimated >= heavy_threshold {
            println!("  {}: {} (estimated), {} (actual)", item, estimated, count);
        }
    }
    
    println!("\nDetected heavy hitter counts: {:?}", heavy_hitters);
    
    println!("\n{}", "=".repeat(50));
    
    // Performance Comparison
    println!("\n6. Performance Comparison");
    println!("{}", "-".repeat(25));
    
    let n_items = 100000;
    println!("Testing with {} updates", n_items);
    
    // Standard Count-Min Sketch
    let start = std::time::Instant::now();
    let mut cm_perf = CountMinSketch::new(10000, 5, false, false);
    for i in 0..n_items {
        cm_perf.increment(&format!("item_{}", i % 1000));
    }
    let standard_time = start.elapsed();
    
    // SIMD Count-Min Sketch (currently falls back to standard)
    let start = std::time::Instant::now();
    let mut cm_simd = CountMinSketch::new(10000, 5, true, false);
    for i in 0..n_items {
        cm_simd.increment(&format!("item_{}", i % 1000));
    }
    let simd_time = start.elapsed();
    
    println!("Standard implementation: {:?}", standard_time);
    println!("SIMD implementation: {:?}", simd_time);
    println!("Speedup: {:.2}x", standard_time.as_nanos() as f64 / simd_time.as_nanos() as f64);
    
    // Verify both give same results
    let test_item = "item_42";
    let standard_result = cm_perf.estimate(&test_item);
    let simd_result = cm_simd.estimate(&test_item);
    println!("Results match: {} (standard: {}, SIMD: {})", 
             standard_result == simd_result, standard_result, simd_result);
    
    println!("\n{}", "=".repeat(50));
    
    // Merge Demo
    println!("\n7. Sketch Merging");
    println!("{}", "-".repeat(15));
    
    let mut cm1 = CountMinSketch::new(1000, 5, false, false);
    let mut cm2 = CountMinSketch::new(1000, 5, false, false);
    
    // Add different data to each sketch
    for i in 0..500 {
        cm1.increment(&format!("common_item_{}", i % 100));
    }
    
    for i in 0..300 {
        cm2.increment(&format!("common_item_{}", i % 100));
    }
    
    let before_merge = cm1.estimate(&"common_item_0");
    cm1.merge(&cm2).unwrap();
    let after_merge = cm1.estimate(&"common_item_0");
    
    println!("Merged two sketches");
    println!("Item count before merge: {}", before_merge);
    println!("Item count after merge: {}", after_merge);
    println!("Expected increase: ~{}", 300 / 100); // Approximately
    
    println!("\nDemo completed successfully!");
}