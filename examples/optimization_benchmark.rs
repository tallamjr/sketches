//! Benchmark to demonstrate the performance improvements from Phase 1 optimizations.

use sketches::bloom::BloomFilter;
use sketches::hll::HllSketch;
use std::time::Instant;

fn main() {
    println!("=== Phase 2 Optimization Benchmark ===\n");

    // Test parameters
    let precision = 12;
    let num_items = 1_000_000;
    let test_data: Vec<String> = (0..num_items).map(|i| format!("item_{}", i)).collect();

    println!(
        "Testing HLL Sketch with {} items (precision = {})",
        num_items, precision
    );

    // Test individual updates
    test_individual_updates(&test_data, precision);

    // Test batch updates (optimized feature only)
    #[cfg(feature = "optimized")]
    test_batch_updates(&test_data, precision);

    #[cfg(not(feature = "optimized"))]
    println!("Batch updates are only available with the 'optimized' feature enabled");

    println!("\n=== Bloom Filter Optimizations ===");
    test_bloom_filter_optimizations(&test_data);

    println!("\n=== Memory Usage Comparison ===");
    test_memory_usage(precision);

    println!("\n=== Hash Performance ===");
    test_hash_performance(&test_data);

    println!("\n=== SIMD Performance ===");
    test_simd_performance();

    println!("\n=== Feature Status ===");
    print_feature_status();
}

fn test_individual_updates(data: &[String], precision: u8) {
    println!("\n--- Individual Updates ---");

    let mut sketch = HllSketch::new(precision);
    let start = Instant::now();

    for item in data {
        sketch.update(item);
    }

    let duration = start.elapsed();
    let throughput = data.len() as f64 / duration.as_secs_f64();
    let estimate = sketch.estimate();
    let error = ((estimate - data.len() as f64) / data.len() as f64 * 100.0).abs();

    println!("Time: {:.2}s", duration.as_secs_f64());
    println!("Throughput: {:.0} items/sec", throughput);
    println!("Estimate: {:.0} (error: {:.2}%)", estimate, error);
}

#[cfg(feature = "optimized")]
fn test_batch_updates(data: &[String], precision: u8) {
    println!("\n--- Batch Updates (Optimized) ---");

    let mut sketch = HllSketch::new(precision);
    let batch_size = 10000;
    let start = Instant::now();

    for chunk in data.chunks(batch_size) {
        sketch.update_batch(chunk);
    }

    let duration = start.elapsed();
    let throughput = data.len() as f64 / duration.as_secs_f64();
    let estimate = sketch.estimate();
    let error = ((estimate - data.len() as f64) / data.len() as f64 * 100.0).abs();

    println!("Time: {:.2}s", duration.as_secs_f64());
    println!("Throughput: {:.0} items/sec", throughput);
    println!("Estimate: {:.0} (error: {:.2}%)", estimate, error);
}

fn test_memory_usage(precision: u8) {
    let sketch = HllSketch::new(precision);
    let registers_memory = sketch.to_bytes().len();

    println!("Standard sketch memory: {} bytes", registers_memory);

    #[cfg(feature = "optimized")]
    {
        // Packed registers use ~6 bits per register vs 8 bits
        let expected_packed_memory = (1 << precision) * 6 / 8; // 6 bits per register, packed
        println!("Optimized packed memory: ~{} bytes", expected_packed_memory);
        println!(
            "Memory reduction: ~{:.1}x",
            registers_memory as f64 / expected_packed_memory as f64
        );
    }

    #[cfg(not(feature = "optimized"))]
    println!("Optimized memory calculations require the 'optimized' feature");
}

fn test_hash_performance(data: &[String]) {
    let sample_size = 100_000;
    let sample_data = &data[..sample_size.min(data.len())];

    println!("Hashing {} items...", sample_size);

    #[cfg(feature = "optimized")]
    {
        use sketches::fast_hash;

        let start = Instant::now();
        let _hashes: Vec<u64> = sample_data
            .iter()
            .map(|item| fast_hash::fast_hash(item))
            .collect();
        let duration = start.elapsed();
        let throughput = sample_size as f64 / duration.as_secs_f64();

        println!("xxHash throughput: {:.0} hashes/sec", throughput);
    }

    // Test standard hash for comparison
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let start = Instant::now();
    let _hashes: Vec<u64> = sample_data
        .iter()
        .map(|item| {
            let mut hasher = DefaultHasher::new();
            item.hash(&mut hasher);
            hasher.finish()
        })
        .collect();
    let duration = start.elapsed();
    let throughput = sample_size as f64 / duration.as_secs_f64();

    println!("DefaultHasher throughput: {:.0} hashes/sec", throughput);
}

fn test_bloom_filter_optimizations(data: &[String]) {
    let capacity = 1_000_000;
    let error_rate = 0.01;
    let sample_data = &data[..100_000.min(data.len())];

    println!("Testing Bloom filter with {} items", sample_data.len());

    // Test without SIMD
    let mut bloom_no_simd = BloomFilter::new(capacity, error_rate, false);
    let start = Instant::now();

    #[cfg(feature = "optimized")]
    bloom_no_simd.add_batch(sample_data);

    #[cfg(not(feature = "optimized"))]
    for item in sample_data {
        bloom_no_simd.add(item);
    }

    let duration = start.elapsed();
    let throughput = sample_data.len() as f64 / duration.as_secs_f64();
    println!("Bloom filter (no SIMD): {:.0} items/sec", throughput);

    // Test with SIMD
    let mut bloom_with_simd = BloomFilter::new(capacity, error_rate, true);
    let start = Instant::now();

    #[cfg(feature = "optimized")]
    bloom_with_simd.add_batch(sample_data);

    #[cfg(not(feature = "optimized"))]
    for item in sample_data {
        bloom_with_simd.add(item);
    }

    let duration = start.elapsed();
    let throughput = sample_data.len() as f64 / duration.as_secs_f64();
    println!("Bloom filter (with SIMD): {:.0} items/sec", throughput);

    // Test query performance
    let start = Instant::now();

    #[cfg(feature = "optimized")]
    let _results = bloom_with_simd.contains_batch(sample_data);

    #[cfg(not(feature = "optimized"))]
    let _results: Vec<bool> = sample_data
        .iter()
        .map(|item| bloom_with_simd.contains(item))
        .collect();

    let duration = start.elapsed();
    let throughput = sample_data.len() as f64 / duration.as_secs_f64();
    println!("Bloom filter queries: {:.0} items/sec", throughput);
}

fn test_simd_performance() {
    #[cfg(feature = "optimized")]
    {
        use sketches::simd_ops::hyperloglog;
        use sketches::simd_ops::utils;

        println!("SIMD capabilities: {:?}", utils::simd_features());

        // Test SIMD leading zeros computation
        let test_values: Vec<u64> = (0..100_000).map(|i| (i as u64) << 32).collect();

        let start = Instant::now();
        let _results = hyperloglog::leading_zeros_batch(&test_values);
        let duration = start.elapsed();
        let throughput = test_values.len() as f64 / duration.as_secs_f64();

        println!("SIMD leading zeros: {:.0} operations/sec", throughput);

        // Test regular scalar computation for comparison
        let start = Instant::now();
        let _results: Vec<u8> = test_values
            .iter()
            .map(|&v| v.leading_zeros() as u8)
            .collect();
        let duration = start.elapsed();
        let scalar_throughput = test_values.len() as f64 / duration.as_secs_f64();

        println!(
            "Scalar leading zeros: {:.0} operations/sec",
            scalar_throughput
        );
        println!("SIMD speedup: {:.1}x", throughput / scalar_throughput);
    }

    #[cfg(not(feature = "optimized"))]
    println!("SIMD performance testing requires the 'optimized' feature");
}

fn print_feature_status() {
    println!(
        "Optimized features enabled: {}",
        cfg!(feature = "optimized")
    );

    #[cfg(feature = "optimized")]
    {
        use sketches::simd_ops::utils;
        println!("SIMD available: {}", utils::simd_available());
        println!("SIMD features: {:?}", utils::simd_features());
        println!("Memory allocator: jemalloc");
        println!("Hash function: xxHash");
        println!("Data structures: Compact/Packed");
    }

    #[cfg(not(feature = "optimized"))]
    {
        println!("Memory allocator: System default");
        println!("Hash function: SipHash (DefaultHasher)");
        println!("Data structures: Standard Vec/HashMap");
    }
}
