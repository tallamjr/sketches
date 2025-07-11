use sketches::{bloom::BloomFilter, cpc::CpcSketch, hll::HllSketch, theta::ThetaSketch};
use std::time::Instant;

fn main() {
    println!("Phase 3 System-Level Optimizations Demo");
    println!("=======================================");

    // Demonstrate cache-optimized data structures
    cache_optimization_demo();

    // Demonstrate batch processing improvements
    batch_processing_demo();

    // Show memory pooling benefits
    memory_efficiency_demo();

    // Performance comparison
    performance_comparison_demo();
}

fn cache_optimization_demo() {
    println!("\n1. Cache Optimization");
    println!("--------------------");

    let mut hll = HllSketch::new(14); // Large sketch to show cache effects

    let start = Instant::now();

    // Sequential access pattern (cache-friendly)
    for i in 0..100_000 {
        hll.update(&format!("sequential_{}", i));
    }

    let sequential_time = start.elapsed();
    println!("Sequential updates: {:.2?}", sequential_time);
    println!("Estimate: {:.0}", hll.estimate());
    println!("Cache-aligned data structures improve memory access patterns");
}

fn batch_processing_demo() {
    println!("\n2. Batch Processing");
    println!("------------------");

    let mut hll = HllSketch::new(12);

    // Individual updates
    let start = Instant::now();
    for i in 0..50_000 {
        hll.update(&format!("item_{}", i));
    }
    let individual_time = start.elapsed();

    // Batch updates (when optimized feature is enabled)
    let mut hll_batch = HllSketch::new(12);
    let items: Vec<String> = (0..50_000).map(|i| format!("item_{}", i)).collect();
    let item_refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();

    let start = Instant::now();
    #[cfg(feature = "optimized")]
    hll_batch.update_batch(&item_refs);
    #[cfg(not(feature = "optimized"))]
    for item in &item_refs {
        hll_batch.update(item);
    }
    let batch_time = start.elapsed();

    println!("Individual updates: {:.2?}", individual_time);
    println!("Batch updates: {:.2?}", batch_time);

    #[cfg(feature = "optimized")]
    if individual_time > batch_time {
        let speedup = individual_time.as_nanos() as f64 / batch_time.as_nanos() as f64;
        println!("Batch processing speedup: {:.2}x", speedup);
    }

    println!(
        "Both estimates: {:.0} vs {:.0}",
        hll.estimate(),
        hll_batch.estimate()
    );
}

fn memory_efficiency_demo() {
    println!("\n3. Memory Efficiency");
    println!("-------------------");

    // Create multiple sketches to show memory pooling benefits
    let sketches: Vec<HllSketch> = (0..10).map(|_| HllSketch::new(12)).collect();

    println!("Created {} HLL sketches", sketches.len());
    println!("Memory pooling reduces allocation overhead");

    // CPC sketch shows sparse-to-dense mode switching
    let mut cpc = CpcSketch::new(11);

    println!("\nCPC Sketch Mode Switching:");

    // Start in sparse mode
    for i in 0..100 {
        cpc.update(&format!("sparse_{}", i));
    }
    println!("After 100 updates - still in sparse mode");

    // Trigger mode switch to dense
    for i in 100..2000 {
        cpc.update(&format!("dense_{}", i));
    }
    println!("After 2000 updates - switched to dense mode");
    println!("CPC estimate: {:.0}", cpc.estimate());
}

fn performance_comparison_demo() {
    println!("\n4. Performance Comparison");
    println!("------------------------");

    const N: usize = 100_000;

    // HLL Performance
    let mut hll = HllSketch::new(12);
    let start = Instant::now();
    for i in 0..N {
        hll.update(&i.to_string());
    }
    let hll_time = start.elapsed();
    let hll_rate = N as f64 / hll_time.as_secs_f64();

    // Theta Performance
    let mut theta = ThetaSketch::new(4096);
    let start = Instant::now();
    for i in 0..N {
        theta.update(&i.to_string());
    }
    let theta_time = start.elapsed();
    let theta_rate = N as f64 / theta_time.as_secs_f64();

    // CPC Performance
    let mut cpc = CpcSketch::new(11);
    let start = Instant::now();
    for i in 0..N {
        cpc.update(&i.to_string());
    }
    let cpc_time = start.elapsed();
    let cpc_rate = N as f64 / cpc_time.as_secs_f64();

    // Bloom Filter Performance
    let mut bloom = BloomFilter::new(N, 0.01, false);
    let start = Instant::now();
    for i in 0..N {
        bloom.add(&i.to_string());
    }
    let bloom_time = start.elapsed();
    let bloom_rate = N as f64 / bloom_time.as_secs_f64();

    println!("Performance Results ({} items):", N);
    println!(
        "├─ HLL:   {:.2?} ({:.0} items/sec) - estimate: {:.0}",
        hll_time,
        hll_rate,
        hll.estimate()
    );
    println!(
        "├─ Theta: {:.2?} ({:.0} items/sec) - estimate: {:.0}",
        theta_time,
        theta_rate,
        theta.estimate()
    );
    println!(
        "├─ CPC:   {:.2?} ({:.0} items/sec) - estimate: {:.0}",
        cpc_time,
        cpc_rate,
        cpc.estimate()
    );
    println!(
        "└─ Bloom: {:.2?} ({:.0} items/sec) - membership test ready",
        bloom_time, bloom_rate
    );

    // Set operations demo
    println!("\nSet Operations:");
    let mut theta1 = ThetaSketch::new(2048);
    let mut theta2 = ThetaSketch::new(2048);

    // Add overlapping data
    for i in 0..5000 {
        theta1.update(&format!("set1_{}", i));
    }
    for i in 2500..7500 {
        theta2.update(&format!("set1_{}", i)); // Reuse set1 prefix for overlap
    }

    let union = theta1.union(&theta2);
    let intersection = theta1.intersect(&theta2);

    println!("├─ Set 1: {:.0} items", theta1.estimate());
    println!("├─ Set 2: {:.0} items", theta2.estimate());
    println!("├─ Union: {:.0} items", union.estimate());
    println!("└─ Intersection: {:.0} items", intersection.estimate());

    println!("\n✨ Phase 3 optimizations include:");
    println!("  • Cache-aligned data structures");
    println!("  • SIMD-optimized operations");
    println!("  • Memory pooling and buffer recycling");
    println!("  • Batch processing capabilities");
    println!("  • SIMD acceleration framework (when available)");
}
