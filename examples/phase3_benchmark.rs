//! Phase 3 system-level optimization benchmark.

use sketches::hll::HllSketch;
use std::time::Instant;

#[cfg(feature = "optimized")]
use sketches::compact_memory::{
    AdvancedSketchPool, BufferPool, CacheOptimizedPackedRegisters, MemoryUsage, Resettable,
};

fn main() {
    println!("=== Phase 3 System-Level Optimization Benchmark ===\n");

    let num_items = 2_000_000;
    let test_data: Vec<String> = (0..num_items).map(|i| format!("item_{}", i)).collect();

    println!("Testing with {} items", num_items);

    // Test cache optimization
    test_cache_optimization();

    // Test SIMD acceleration
    test_simd_acceleration(&test_data);

    // Test memory pooling
    test_memory_pooling();

    // Test buffer pooling
    test_buffer_pooling();

    // Test overall system performance
    test_system_performance(&test_data);

    println!("\n=== Phase 3 Summary ===");
    print_phase3_capabilities();
}

fn test_cache_optimization() {
    println!("\n--- Cache Optimization Test ---");

    #[cfg(feature = "optimized")]
    {
        let mut cache_registers = CacheOptimizedPackedRegisters::new(12, 6);
        let updates: Vec<(usize, u8)> = (0..10000).map(|i| (i % 4096, (i % 64) as u8)).collect();

        let start = Instant::now();
        cache_registers.update_max_batch_cache_optimized(&updates);
        let duration = start.elapsed();

        println!(
            "Cache-optimized batch updates: {:.2}ms",
            duration.as_millis()
        );
        println!("Cache statistics: {:?}", cache_registers.cache_statistics());

        // Test access pattern learning
        for i in 0..1000 {
            let _ = cache_registers.get(i % 100); // Create hot spots
        }

        println!(
            "After access pattern learning: {:?}",
            cache_registers.cache_statistics()
        );
    }

    #[cfg(not(feature = "optimized"))]
    println!("Cache optimization requires 'optimized' feature");
}

fn test_simd_acceleration(data: &[String]) {
    println!("\n--- SIMD Acceleration Test ---");

    #[cfg(feature = "optimized")]
    {
        use sketches::simd_ops::utils;

        // Check SIMD capabilities
        let simd_available = utils::simd_available();
        let simd_features = utils::simd_features();

        println!("SIMD available: {}", simd_available);
        println!("SIMD features: {:?}", simd_features);

        if simd_available {
            // Test SIMD-accelerated HLL operations
            let mut hll = HllSketch::new(12);
            let sample_data = &data[..100_000];

            let start = Instant::now();
            hll.update_batch(sample_data);
            let duration = start.elapsed();

            println!("SIMD HLL batch processing: {:.2}ms", duration.as_millis());
            println!("Items processed: {}", sample_data.len());
            println!(
                "Throughput: {:.0} items/sec",
                sample_data.len() as f64 / duration.as_secs_f64()
            );
            println!("Estimated cardinality: {:.0}", hll.estimate());
        } else {
            println!("SIMD optimizations not available on this platform");

            // Still test regular batch operations
            let mut hll = HllSketch::new(12);
            let sample_data = &data[..100_000];

            let start = Instant::now();
            hll.update_batch(sample_data);
            let duration = start.elapsed();

            println!("Standard batch processing: {:.2}ms", duration.as_millis());
            println!(
                "Throughput: {:.0} items/sec",
                sample_data.len() as f64 / duration.as_secs_f64()
            );
        }
    }

    #[cfg(not(feature = "optimized"))]
    println!("SIMD acceleration requires 'optimized' feature");
}

fn test_memory_pooling() {
    println!("\n--- Memory Pooling Test ---");

    #[cfg(feature = "optimized")]
    {
        // Mock sketch object for testing
        struct MockSketch {
            size: usize,
            data: Vec<u8>,
        }

        impl MockSketch {
            fn new(size: usize) -> Self {
                Self {
                    size,
                    data: vec![0; size],
                }
            }
        }

        impl Resettable for MockSketch {
            fn reset(&mut self) {
                self.data.fill(0);
            }
        }

        impl MemoryUsage for MockSketch {
            fn memory_usage(&self) -> usize {
                self.size
            }
        }

        let mut pool = AdvancedSketchPool::new(|| MockSketch::new(1024), 10);

        // Test pool operations
        let start = Instant::now();
        let mut objects = Vec::new();

        // Get objects from pool
        for i in 0..50 {
            let size = if i % 3 == 0 {
                512
            } else if i % 3 == 1 {
                2048
            } else {
                32768
            };
            objects.push(pool.get(size));
        }

        // Return objects to pool
        for obj in objects {
            pool.put(obj);
        }

        let duration = start.elapsed();
        println!("Pool operations: {:.2}µs", duration.as_micros());
        println!("Pool statistics: {:?}", pool.statistics());
        println!("Objects in pool: {}", pool.total_pooled_objects());
    }

    #[cfg(not(feature = "optimized"))]
    println!("Memory pooling requires 'optimized' feature");
}

fn test_buffer_pooling() {
    println!("\n--- Buffer Pooling Test ---");

    #[cfg(feature = "optimized")]
    {
        let mut buffer_pool = BufferPool::new(1024 * 1024); // 1MB max

        let start = Instant::now();

        // Simulate buffer allocation and reuse
        for _ in 0..1000 {
            // Get buffers
            let mut u8_buf = buffer_pool.get_u8_buffer(1024);
            let mut u64_buf = buffer_pool.get_u64_buffer(128);
            let mut hash_buf = buffer_pool.get_hash_buffer(256);

            // Simulate usage
            u8_buf.extend_from_slice(&[1, 2, 3, 4]);
            u64_buf.push(0x123456789abcdef0);
            hash_buf.push(0xdeadbeef);

            // Return buffers
            buffer_pool.return_u8_buffer(u8_buf);
            buffer_pool.return_u64_buffer(u64_buf);
            buffer_pool.return_hash_buffer(hash_buf);
        }

        let duration = start.elapsed();
        println!("Buffer pool operations: {:.2}µs", duration.as_micros());

        let (u8_count, u64_count, hash_count) = buffer_pool.buffer_pool_stats();
        println!(
            "Buffers in pool: {} u8, {} u64, {} hash",
            u8_count, u64_count, hash_count
        );
    }

    #[cfg(not(feature = "optimized"))]
    println!("Buffer pooling requires 'optimized' feature");
}

fn test_system_performance(data: &[String]) {
    println!("\n--- Overall System Performance ---");

    let sample_data = &data[..500_000]; // Use 500k items for comprehensive test

    // Test with all optimizations enabled
    let mut sketch = HllSketch::new(12);
    let start = Instant::now();

    #[cfg(feature = "optimized")]
    sketch.update_batch(sample_data);

    #[cfg(not(feature = "optimized"))]
    for item in sample_data {
        sketch.update(item);
    }

    let duration = start.elapsed();
    let throughput = sample_data.len() as f64 / duration.as_secs_f64();
    let estimate = sketch.estimate();
    let error = ((estimate - sample_data.len() as f64) / sample_data.len() as f64 * 100.0).abs();

    println!("System throughput: {:.0} items/sec", throughput);
    println!("Accuracy: {:.0} estimate ({:.2}% error)", estimate, error);
    println!("Processing time: {:.2}ms", duration.as_millis());

    // Memory efficiency test
    let memory_usage = sketch.to_bytes().len();
    let memory_per_item = memory_usage as f64 / sample_data.len() as f64;
    println!(
        "Memory usage: {} bytes ({:.3} bytes/item)",
        memory_usage, memory_per_item
    );
}

fn print_phase3_capabilities() {
    println!("Phase 3 optimizations enabled:");

    #[cfg(feature = "optimized")]
    {
        println!("✓ Cache-aligned data structures");
        println!("✓ Intelligent prefetching");
        println!("✓ SIMD acceleration framework");
        println!("✓ Advanced memory pooling");
        println!("✓ Buffer recycling");
        println!("✓ GIL-free Python operations");

        // GPU capabilities
        let gpu_devices = GpuManager::detect_devices();
        if !gpu_devices.is_empty() {
            println!("✓ {} GPU device(s) available", gpu_devices.len());
        } else {
            println!("○ No GPU devices detected");
        }
    }

    #[cfg(not(feature = "optimized"))]
    {
        println!("○ Optimizations disabled (enable with --features optimized)");
    }

    println!("\nSystem capabilities:");
    println!("  CPU cores: {}", num_cpus::get());
    println!("  Target architecture: {}", std::env::consts::ARCH);
    println!("  Operating system: {}", std::env::consts::OS);
}
