use sketches::bloom::{BloomFilter, CountingBloomFilter};

fn main() {
    println!("=== Bloom Filter Demo ===\n");
    
    // Standard Bloom Filter Demo
    println!("1. Standard Bloom Filter");
    println!("{}", "-".repeat(30));
    
    let mut bloom = BloomFilter::new(10000, 0.01, false);
    println!("Created Bloom filter for 10,000 items with 1% error rate");
    
    // Add some items
    let items = vec!["apple", "banana", "cherry", "date", "elderberry"];
    for item in &items {
        bloom.add(item);
        println!("Added: {}", item);
    }
    
    // Test membership
    println!("\nTesting membership:");
    for item in &items {
        println!("  {} in filter: {}", item, bloom.contains(item));
    }
    
    // Test false positives
    let test_items = vec!["grape", "kiwi", "mango", "orange", "pear"];
    println!("\nTesting items not added (might have false positives):");
    for item in &test_items {
        println!("  {} in filter: {}", item, bloom.contains(item));
    }
    
    // Show statistics
    let stats = bloom.statistics();
    println!("\nBloom Filter Statistics:");
    println!("  Total bits: {}", stats.num_bits);
    println!("  Hash functions: {}", stats.num_hash_functions);
    println!("  Bits set: {}", stats.bits_set);
    println!("  Fill ratio: {:.4}", stats.fill_ratio);
    println!("  False positive probability: {:.6}", stats.false_positive_probability);
    println!("  Using SIMD: {}", stats.uses_simd);
    
    println!("");
    println!("{}", "=".repeat(50));
    
    // Counting Bloom Filter Demo
    println!("\n2. Counting Bloom Filter (supports deletions)");
    println!("{}", "-".repeat(50));
    
    let mut counting_bloom = CountingBloomFilter::new(1000, 0.01, 255);
    println!("Created counting Bloom filter for 1,000 items");
    
    // Add items
    counting_bloom.add(&"item1");
    counting_bloom.add(&"item2");
    counting_bloom.add(&"item1"); // Add again
    
    println!("Added: item1, item2, item1 (again)");
    
    // Test membership
    println!("item1 in filter: {}", counting_bloom.contains(&"item1"));
    println!("item2 in filter: {}", counting_bloom.contains(&"item2"));
    println!("item3 in filter: {}", counting_bloom.contains(&"item3"));
    
    // Remove items
    println!("\nRemoving item1 once:");
    let removed = counting_bloom.remove(&"item1");
    println!("Removal successful: {}", removed);
    println!("item1 still in filter: {}", counting_bloom.contains(&"item1"));
    
    println!("\nRemoving item1 again:");
    let removed = counting_bloom.remove(&"item1");
    println!("Removal successful: {}", removed);
    println!("item1 still in filter: {}", counting_bloom.contains(&"item1"));
    
    println!("\nTrying to remove item3 (not in filter):");
    let removed = counting_bloom.remove(&"item3");
    println!("Removal successful: {}", removed);
    
    println!("");
    println!("{}", "=".repeat(50));
    
    // Performance comparison
    println!("\n3. Performance Comparison");
    println!("{}", "-".repeat(30));
    
    let n_items = 100000;
    println!("Testing with {} items", n_items);
    
    // Standard implementation
    let start = std::time::Instant::now();
    let mut bloom_standard = BloomFilter::new(n_items, 0.01, false);
    for i in 0..n_items {
        bloom_standard.add(&format!("item_{}", i));
    }
    let standard_time = start.elapsed();
    
    // SIMD implementation (currently falls back to standard)
    let start = std::time::Instant::now();
    let mut bloom_simd = BloomFilter::new(n_items, 0.01, true);
    for i in 0..n_items {
        bloom_simd.add(&format!("item_{}", i));
    }
    let simd_time = start.elapsed();
    
    println!("Standard implementation: {:?}", standard_time);
    println!("SIMD implementation: {:?}", simd_time);
    println!("Speedup: {:.2}x", standard_time.as_nanos() as f64 / simd_time.as_nanos() as f64);
    
    // Verify both give same results
    let test_item = "test_item";
    bloom_standard.add(&test_item);
    bloom_simd.add(&test_item);
    
    println!("Both contain test item: {}", 
             bloom_standard.contains(&test_item) && bloom_simd.contains(&test_item));
    
    println!("");
    println!("{}", "=".repeat(50));
    
    // False positive rate analysis
    println!("\n4. False Positive Rate Analysis");
    println!("{}", "-".repeat(40));
    
    let mut bloom_test = BloomFilter::new(10000, 0.01, false);
    
    // Add 10,000 items
    for i in 0..10000 {
        bloom_test.add(&format!("known_item_{}", i));
    }
    
    // Test with 10,000 new items
    let mut false_positives = 0;
    let test_count = 10000;
    
    for i in 0..test_count {
        if bloom_test.contains(&format!("unknown_item_{}", i)) {
            false_positives += 1;
        }
    }
    
    let actual_fp_rate = false_positives as f64 / test_count as f64;
    let expected_fp_rate = 0.01;
    
    println!("Expected false positive rate: {:.2}%", expected_fp_rate * 100.0);
    println!("Actual false positive rate: {:.2}%", actual_fp_rate * 100.0);
    println!("Difference: {:.4}%", (actual_fp_rate - expected_fp_rate).abs() * 100.0);
    
    let stats = bloom_test.statistics();
    println!("Theoretical FP rate: {:.6}%", stats.false_positive_probability * 100.0);
    
    println!("\nDemo completed successfully!");
}