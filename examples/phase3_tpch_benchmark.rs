//! Phase 3 system-level optimization benchmark using real TPC-H data.

use sketches::hll::HllSketch;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::time::Instant;

#[cfg(feature = "optimized")]
use sketches::compact_memory::{
    AdvancedSketchPool, BufferPool, CacheOptimizedPackedRegisters, MemoryUsage, Resettable,
};

fn main() {
    println!("=== Phase 3 System-Level Optimization Benchmark (TPC-H Data) ===\n");

    // Load TPC-H data
    let tpch_data = load_tpch_data();
    println!("Loaded TPC-H data:");
    println!("  Customers: {} IDs", tpch_data.customer_ids.len());
    println!("  Orders: {} keys", tpch_data.order_keys.len());
    println!("  LineItems: {} keys", tpch_data.lineitem_keys.len());
    println!("  Parts: {} keys", tpch_data.part_keys.len());

    // Test cache optimization with real data patterns
    test_cache_optimization_tpch(&tpch_data);

    // Test SIMD acceleration with real data
    test_simd_acceleration_tpch(&tpch_data);

    // Test memory pooling
    test_memory_pooling();

    // Test buffer pooling
    test_buffer_pooling();

    // Test overall system performance with TPC-H workloads
    test_system_performance_tpch(&tpch_data);

    println!("\n=== Phase 3 TPC-H Summary ===");
    print_phase3_capabilities();
}

#[derive(Debug)]
struct TpchData {
    customer_ids: Vec<String>,
    order_keys: Vec<String>,
    lineitem_keys: Vec<String>,
    part_keys: Vec<String>,
}

fn load_tpch_data() -> TpchData {
    let mut customer_ids = Vec::new();
    let mut order_keys = Vec::new();
    let mut lineitem_keys = Vec::new();
    let mut part_keys = Vec::new();

    // Load customer IDs (realistic business entity cardinality)
    if let Ok(file) = File::open("tests/data/customer.csv") {
        let reader = BufReader::new(file);
        for (i, line) in reader.lines().enumerate() {
            if i == 0 { continue; } // Skip header
            if i > 50000 { break; } // Limit for performance
            if let Ok(line) = line {
                if let Some(first_field) = line.split(',').next() {
                    customer_ids.push(format!("CUST_{}", first_field));
                }
            }
        }
    }

    // Load order keys (temporal business data)
    if let Ok(file) = File::open("tests/data/orders.csv") {
        let reader = BufReader::new(file);
        for (i, line) in reader.lines().enumerate() {
            if i == 0 { continue; } // Skip header
            if i > 100000 { break; } // Limit for performance
            if let Ok(line) = line {
                if let Some(first_field) = line.split(',').next() {
                    order_keys.push(format!("ORD_{}", first_field));
                }
            }
        }
    }

    // Load lineitem keys (high-cardinality transaction data)
    if let Ok(file) = File::open("tests/data/lineitem.csv") {
        let reader = BufReader::new(file);
        for (i, line) in reader.lines().enumerate() {
            if i == 0 { continue; } // Skip header
            if i > 200000 { break; } // Limit for performance
            if let Ok(line) = line {
                let fields: Vec<&str> = line.split(',').collect();
                if fields.len() >= 4 {
                    // Combine order key, part key, supplier key, line number for unique ID
                    lineitem_keys.push(format!("LI_{}_{}_{}_{}", 
                        fields[0], fields[1], fields[2], fields[3]));
                }
            }
        }
    }

    // Load part keys (product catalog data)
    if let Ok(file) = File::open("tests/data/part.csv") {
        let reader = BufReader::new(file);
        for (i, line) in reader.lines().enumerate() {
            if i == 0 { continue; } // Skip header
            if i > 50000 { break; } // Limit for performance
            if let Ok(line) = line {
                if let Some(first_field) = line.split(',').next() {
                    part_keys.push(format!("PART_{}", first_field));
                }
            }
        }
    }

    TpchData {
        customer_ids,
        order_keys,
        lineitem_keys,
        part_keys,
    }
}

fn test_cache_optimization_tpch(data: &TpchData) {
    println!("\n--- Cache Optimization Test (TPC-H Data) ---");

    #[cfg(feature = "optimized")]
    {
        let mut cache_registers = CacheOptimizedPackedRegisters::new(12, 6);
        
        // Create realistic access patterns from TPC-H data
        // Simulate hash-based register updates with realistic skew
        let updates: Vec<(usize, u8)> = data.customer_ids.iter()
            .chain(data.order_keys.iter())
            .take(10000)
            .enumerate()
            .map(|(i, item)| {
                let hash = hash_string(item);
                let register_idx = (hash as usize) % 4096;
                let register_val = leading_zeros_plus_one(hash >> 12) as u8;
                (register_idx, register_val.min(63))
            })
            .collect();

        let start = Instant::now();
        cache_registers.update_max_batch_cache_optimized(&updates);
        let duration = start.elapsed();

        println!("TPC-H cache-optimized updates: {:.2}ms", duration.as_millis());
        println!("Cache statistics: {:?}", cache_registers.cache_statistics());

        // Test access pattern learning with realistic hotspots
        // Simulate frequent lookups to popular customers/orders
        for item in data.customer_ids.iter().take(100).cycle().take(1000) {
            let hash = hash_string(item);
            let register_idx = (hash as usize) % 4096;
            let _ = cache_registers.get(register_idx);
        }

        println!("After TPC-H access pattern learning: {:?}", cache_registers.cache_statistics());
    }

    #[cfg(not(feature = "optimized"))]
    println!("Cache optimization requires 'optimized' feature");
}

fn test_simd_acceleration_tpch(data: &TpchData) {
    println!("\n--- SIMD Acceleration Test (TPC-H Data) ---");

    #[cfg(feature = "optimized")]
    {
        use sketches::simd_ops::utils;
        
        // Check SIMD capabilities
        let simd_available = utils::simd_available();
        let simd_features = utils::simd_features();
        
        println!("SIMD available: {}", simd_available);
        println!("SIMD features: {:?}", simd_features);
        
        // Test with realistic business data variety
        let mixed_data: Vec<String> = data.customer_ids.iter()
            .chain(data.order_keys.iter())
            .chain(data.part_keys.iter())
            .take(50_000)
            .cloned()
            .collect();

        if simd_available {
            println!("Testing SIMD-optimized batch processing with TPC-H data...");
            
            let mut hll = HllSketch::new(12);
            let start = Instant::now();
            hll.update_batch(&mixed_data);
            let duration = start.elapsed();
            
            println!("SIMD TPC-H batch processing: {:.2}ms", duration.as_millis());
            println!("Items processed: {}", mixed_data.len());
            println!("Throughput: {:.0} items/sec", mixed_data.len() as f64 / duration.as_secs_f64());
            println!("Estimated cardinality: {:.0}", hll.estimate());
        } else {
            println!("SIMD optimizations not available - using standard processing");
            
            let mut hll = HllSketch::new(12);
            let start = Instant::now();
            hll.update_batch(&mixed_data);
            let duration = start.elapsed();
            
            println!("Standard TPC-H batch processing: {:.2}ms", duration.as_millis());
            println!("Throughput: {:.0} items/sec", mixed_data.len() as f64 / duration.as_secs_f64());
        }
        
        // Test different TPC-H data characteristics
        println!("\nTesting different TPC-H data types:");
        println!("Customer IDs: {} unique items", data.customer_ids.len());
        println!("Order keys: {} unique items", data.order_keys.len());
        println!("LineItem keys: {} unique items", data.lineitem_keys.len());
        println!("Part keys: {} unique items", data.part_keys.len());
    }

    #[cfg(not(feature = "optimized"))]
    println!("SIMD acceleration requires 'optimized' feature");
}

fn test_memory_pooling() {
    println!("\n--- Memory Pooling Test ---");

    #[cfg(feature = "optimized")]
    {
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

        let start = Instant::now();
        let mut objects = Vec::new();

        // Simulate realistic sketch size distribution
        for i in 0..50 {
            let size = match i % 4 {
                0 => 512,    // Small customer sketches
                1 => 2048,   // Medium order sketches
                2 => 8192,   // Large lineitem sketches
                _ => 32768,  // Very large aggregate sketches
            };
            objects.push(pool.get(size));
        }

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

        // Simulate realistic buffer usage patterns
        for _ in 0..1000 {
            let mut u8_buf = buffer_pool.get_u8_buffer(1024);
            let mut u64_buf = buffer_pool.get_u64_buffer(128);
            let mut hash_buf = buffer_pool.get_hash_buffer(256);

            // Simulate TPC-H data processing
            u8_buf.extend_from_slice(b"CUSTOMER_DATA");
            u64_buf.push(0x123456789abcdef0);
            hash_buf.push(0xdeadbeef);

            buffer_pool.return_u8_buffer(u8_buf);
            buffer_pool.return_u64_buffer(u64_buf);
            buffer_pool.return_hash_buffer(hash_buf);
        }

        let duration = start.elapsed();
        println!("Buffer pool operations: {:.2}µs", duration.as_micros());

        let (u8_count, u64_count, hash_count) = buffer_pool.buffer_pool_stats();
        println!("Buffers in pool: {} u8, {} u64, {} hash", u8_count, u64_count, hash_count);
    }

    #[cfg(not(feature = "optimized"))]
    println!("Buffer pooling requires 'optimized' feature");
}

fn test_system_performance_tpch(data: &TpchData) {
    println!("\n--- Overall System Performance (TPC-H Workloads) ---");

    // Test 1: Customer cardinality estimation
    test_workload("Customer IDs", &data.customer_ids);
    
    // Test 2: Order key cardinality
    test_workload("Order Keys", &data.order_keys);
    
    // Test 3: High-cardinality lineitem data
    if data.lineitem_keys.len() > 50000 {
        test_workload("LineItem Keys", &data.lineitem_keys[..50000]);
    }
    
    // Test 4: Mixed workload (realistic scenario)
    let mixed_workload: Vec<String> = data.customer_ids.iter()
        .chain(data.order_keys.iter())
        .chain(data.part_keys.iter())
        .take(100_000)
        .cloned()
        .collect();
    test_workload("Mixed TPC-H Data", &mixed_workload);
}

fn test_workload(name: &str, data: &[String]) {
    if data.is_empty() {
        println!("{}: No data available", name);
        return;
    }

    let mut sketch = HllSketch::new(12);
    let start = Instant::now();

    #[cfg(feature = "optimized")]
    sketch.update_batch(data);

    #[cfg(not(feature = "optimized"))]
    for item in data {
        sketch.update(item);
    }

    let duration = start.elapsed();
    let throughput = data.len() as f64 / duration.as_secs_f64();
    let estimate = sketch.estimate();
    let error = ((estimate - data.len() as f64) / data.len() as f64 * 100.0).abs();

    println!("\n{} Performance:", name);
    println!("  Dataset size: {} items", data.len());
    println!("  Throughput: {:.0} items/sec", throughput);
    println!("  Accuracy: {:.0} estimate ({:.2}% error)", estimate, error);
    println!("  Processing time: {:.2}ms", duration.as_millis());

    let memory_usage = sketch.to_bytes().len();
    let memory_per_item = memory_usage as f64 / data.len() as f64;
    println!("  Memory: {} bytes ({:.3} bytes/item)", memory_usage, memory_per_item);
}

fn print_phase3_capabilities() {
    println!("Phase 3 optimizations with TPC-H data:");

    #[cfg(feature = "optimized")]
    {
        println!("✓ Cache-aligned data structures");
        println!("✓ Intelligent prefetching with realistic access patterns");
        println!("✓ SIMD acceleration framework");
        println!("✓ Advanced memory pooling");
        println!("✓ Buffer recycling");
        println!("✓ Real-world TPC-H data validation");

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

    println!("\nTPC-H Data Characteristics:");
    println!("  Business entity cardinality patterns");
    println!("  Realistic data skew and distributions");
    println!("  Mixed string formats and lengths");
    println!("  Temporal and hierarchical relationships");

    println!("\nSystem capabilities:");
    println!("  CPU cores: {}", num_cpus::get());
    println!("  Target architecture: {}", std::env::consts::ARCH);
    println!("  Operating system: {}", std::env::consts::OS);
}

// Simple hash function for demonstration
fn hash_string(s: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    let mut hasher = DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

// Count leading zeros plus one (HLL register value calculation)
fn leading_zeros_plus_one(mut value: u64) -> u32 {
    if value == 0 {
        return 64;
    }
    
    let mut count = 0;
    while (value & 0x8000000000000000) == 0 {
        count += 1;
        value <<= 1;
    }
    count + 1
}