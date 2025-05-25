//! Comprehensive demonstration of T-Digest quantile estimation
//!
//! This example showcases:
//! - Basic T-Digest functionality for quantile estimation
//! - Comparison with KLL Sketch for different use cases
//! - Streaming quantile estimation with real-time updates
//! - Merging T-Digests from distributed sources
//! - Performance characteristics and accuracy validation

use rand::prelude::*;
use sketches::quantiles::KllSketch;
use sketches::tdigest::{StreamingTDigest, TDigest};
use std::time::Instant;

fn main() {
    println!("📊 T-Digest Quantile Estimation Demo\n");

    // Basic T-Digest demonstration
    basic_tdigest_demo();

    // Accuracy comparison between different algorithms
    accuracy_comparison_demo();

    // Streaming quantile estimation
    streaming_quantiles_demo();

    // Merging demonstration for distributed scenarios
    distributed_merge_demo();

    // Performance benchmarks
    performance_benchmark_demo();

    // Extreme quantile accuracy (where T-Digest excels)
    extreme_quantiles_demo();

    println!("\n✅ All T-Digest demos completed!");
}

fn basic_tdigest_demo() {
    println!("📈 Basic T-Digest Demonstration");
    println!("==============================");

    let mut digest = TDigest::new();

    // Add a normal distribution of values
    let mut rng = thread_rng();
    println!("Adding 10,000 normally distributed values...");

    for _ in 0..10_000 {
        let value: f64 = rng.r#gen_range(-3.0..3.0); // Rough normal distribution
        digest.add(value);
    }

    println!("Values processed: {}", digest.count());
    println!("Min value: {:.3}", digest.min().unwrap());
    println!("Max value: {:.3}", digest.max().unwrap());

    // Test common quantiles
    let quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99];
    println!("\nQuantile estimates:");
    for &q in &quantiles {
        if let Some(value) = digest.quantile(q) {
            println!("  P{:02.0}: {:.3}", q * 100.0, value);
        }
    }

    // Test rank estimation
    let test_values = [-2.0, -1.0, 0.0, 1.0, 2.0];
    println!("\nRank estimates:");
    for &value in &test_values {
        let rank = digest.rank(value);
        println!(
            "  Value {:.1} is at rank {:.3} ({}th percentile)",
            value,
            rank,
            (rank * 100.0) as u32
        );
    }

    println!();
}

fn accuracy_comparison_demo() {
    println!("🎯 Accuracy Comparison: T-Digest vs KLL Sketch");
    println!("==============================================");

    // Generate known distribution (uniform 0-1000)
    let data: Vec<f64> = (0..10000).map(|i| i as f64).collect();
    let mut shuffled_data = data.clone();
    shuffled_data.shuffle(&mut thread_rng());

    // True quantiles for comparison
    let true_quantiles = |q: f64| -> f64 { q * 9999.0 };

    // Test T-Digest
    let mut tdigest = TDigest::with_compression(200);
    for &value in &shuffled_data {
        tdigest.add(value);
    }

    // Test KLL Sketch
    let mut kll = KllSketch::new(200);
    for &value in &shuffled_data {
        kll.update(value);
    }

    println!("Quantile accuracy comparison (true vs estimated):");
    println!("Quantile | True Value | T-Digest | KLL | T-Digest Error | KLL Error");
    println!("---------|------------|----------|-----|----------------|----------");

    let test_quantiles = [0.5, 0.9, 0.95, 0.99, 0.999];
    for &q in &test_quantiles {
        let true_val = true_quantiles(q);
        let tdigest_val = tdigest.quantile(q).unwrap_or(0.0);
        let kll_val = kll.quantile(q).unwrap_or(0.0);

        let tdigest_error = ((tdigest_val - true_val) / true_val * 100.0).abs();
        let kll_error = ((kll_val - true_val) / true_val * 100.0).abs();

        println!(
            "P{:4.1}    | {:10.1} | {:8.1} | {:3.1} | {:13.3}% | {:7.3}%",
            q * 100.0,
            true_val,
            tdigest_val,
            kll_val,
            tdigest_error,
            kll_error
        );
    }

    println!("\nMemory usage comparison:");
    let tdigest_stats = tdigest.statistics();
    let kll_stats = kll.statistics();
    println!("  T-Digest: {} bytes", tdigest_stats.memory_usage);
    println!("  KLL:      {} bytes", kll_stats.memory_usage);

    println!();
}

fn streaming_quantiles_demo() {
    println!("🌊 Streaming Quantile Estimation");
    println!("================================");

    let mut streaming = StreamingTDigest::new(100, 1000);

    println!("Simulating real-time data stream...");

    // Simulate streaming data with changing distribution
    for batch in 0..10 {
        println!("Processing batch {}...", batch + 1);

        // Each batch has different characteristics
        let (mean, std_dev) = match batch {
            0..=2 => (100.0, 10.0), // Normal distribution around 100
            3..=5 => (200.0, 20.0), // Shift to 200
            6..=8 => (150.0, 50.0), // Wide distribution around 150
            _ => (75.0, 5.0),       // Tight distribution around 75
        };

        // Generate batch data
        let mut rng = thread_rng();
        for _ in 0..1000 {
            let value = rng.r#gen::<f64>() * std_dev + mean;
            streaming.add(value);
        }

        // Show current quantiles
        let median = streaming.quantile(0.5).unwrap_or(0.0);
        let p95 = streaming.quantile(0.95).unwrap_or(0.0);
        println!("  Current median: {:.1}, P95: {:.1}", median, p95);
    }

    println!("\nFinal streaming statistics:");
    let stats = streaming.statistics();
    println!("  Total items processed: {}", stats.count);
    println!("  Memory usage: {} bytes", stats.memory_usage);

    println!();
}

fn distributed_merge_demo() {
    println!("🌐 Distributed T-Digest Merging");
    println!("===============================");

    // Simulate data from multiple nodes/regions
    let mut node_digests = Vec::new();

    for node in 0..5 {
        let mut digest = TDigest::with_compression(100);

        // Each node processes different data ranges
        let base_value = node as f64 * 1000.0;

        println!(
            "Node {} processing data range {}-{}...",
            node,
            base_value as u32,
            (base_value + 1000.0) as u32
        );

        for i in 0..2000 {
            let value = base_value + (i as f64) + thread_rng().r#gen::<f64>() * 100.0;
            digest.add(value);
        }

        println!(
            "  Node {} local median: {:.1}",
            node,
            digest.median().unwrap_or(0.0)
        );

        node_digests.push(digest);
    }

    // Merge all digests at coordinator
    println!("\nMerging digests at coordinator...");
    let mut global_digest = TDigest::with_compression(100);

    for digest in &node_digests {
        global_digest.merge(digest);
    }

    println!("Global statistics after merge:");
    println!("  Total count: {}", global_digest.count());
    println!("  Global min: {:.1}", global_digest.min().unwrap_or(0.0));
    println!("  Global max: {:.1}", global_digest.max().unwrap_or(0.0));
    println!(
        "  Global median: {:.1}",
        global_digest.median().unwrap_or(0.0)
    );
    println!(
        "  Global P95: {:.1}",
        global_digest.quantile(0.95).unwrap_or(0.0)
    );

    println!();
}

fn performance_benchmark_demo() {
    println!("⚡ Performance Benchmarks");
    println!("========================");

    let data_sizes = [1_000, 10_000, 100_000, 1_000_000];
    let compressions = [50, 100, 200, 500];

    for &size in &data_sizes {
        println!("Benchmarking with {} data points:", size);

        // Generate test data
        let mut rng = thread_rng();
        let data: Vec<f64> = (0..size).map(|_| rng.r#gen::<f64>() * 1000.0).collect();

        for &compression in &compressions {
            let start = Instant::now();

            let mut digest = TDigest::with_compression(compression);
            for &value in &data {
                digest.add(value);
            }

            // Test quantile computation
            let _median = digest.median();
            let _p95 = digest.quantile(0.95);
            let _p99 = digest.quantile(0.99);

            let duration = start.elapsed();
            let stats = digest.statistics();

            println!(
                "  Compression {}: {:?} ({} bytes memory)",
                compression, duration, stats.memory_usage
            );
        }
        println!();
    }
}

fn extreme_quantiles_demo() {
    println!("🎯 Extreme Quantile Accuracy (T-Digest Specialty)");
    println!("=================================================");

    // Generate log-normal distribution (heavy tail)
    let mut rng = thread_rng();
    let mut values = Vec::new();

    for _ in 0..100_000 {
        // Log-normal: most values small, few very large
        let normal: f64 = rng.r#gen_range(-2.0..2.0);
        let log_normal = normal.exp();
        values.push(log_normal);
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // True extreme quantiles
    let true_quantiles: Vec<(f64, f64)> = [0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]
        .iter()
        .map(|&q| {
            let index = ((q * values.len() as f64) as usize).min(values.len() - 1);
            (q, values[index])
        })
        .collect();

    // Test T-Digest accuracy on extreme quantiles
    let mut digest = TDigest::with_compression(200);
    let mut shuffled_values = values.clone();
    shuffled_values.shuffle(&mut rng);

    for &value in &shuffled_values {
        digest.add(value);
    }

    println!("Extreme quantile accuracy for log-normal distribution:");
    println!("Quantile | True Value | T-Digest | Relative Error");
    println!("---------|------------|----------|---------------");

    for &(q, true_val) in &true_quantiles {
        let estimated = digest.quantile(q).unwrap_or(0.0);
        let error = ((estimated - true_val) / true_val * 100.0).abs();

        println!(
            "P{:6.2}  | {:10.3} | {:8.3} | {:12.2}%",
            q * 100.0,
            true_val,
            estimated,
            error
        );
    }

    println!("\nT-Digest excels at extreme quantiles due to adaptive compression!");
    println!("More centroids are allocated near the tails of the distribution.");

    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tdigest_basic_functionality() {
        let mut digest = TDigest::new();

        // Add values 1-100
        for i in 1..=100 {
            digest.add(i as f64);
        }

        assert_eq!(digest.count(), 100);
        assert_eq!(digest.min(), Some(1.0));
        assert_eq!(digest.max(), Some(100.0));

        // Test median
        let median = digest.median().unwrap();
        assert!((median - 50.5).abs() < 2.0); // Should be close to true median
    }

    #[test]
    fn test_tdigest_accuracy() {
        let mut digest = TDigest::with_accuracy(0.01);

        // Add uniform distribution 0-1000
        for i in 0..1000 {
            digest.add(i as f64);
        }

        // Test known quantiles
        let p50 = digest.quantile(0.5).unwrap();
        let p95 = digest.quantile(0.95).unwrap();

        // Should be close to true values (499.5 and 949.5)
        assert!((p50 - 499.5).abs() < 10.0);
        assert!((p95 - 949.5).abs() < 20.0);
    }

    #[test]
    fn test_streaming_tdigest() {
        let mut streaming = StreamingTDigest::new(100, 50);

        // Add values in batches
        for i in 0..200 {
            streaming.add(i as f64);
        }

        let median = streaming.quantile(0.5).unwrap();
        assert!((median - 99.5).abs() < 5.0);
    }

    #[test]
    fn test_tdigest_merge() {
        let mut digest1 = TDigest::new();
        let mut digest2 = TDigest::new();

        // Add different ranges
        for i in 0..50 {
            digest1.add(i as f64);
        }

        for i in 50..100 {
            digest2.add(i as f64);
        }

        // Merge
        digest1.merge(&digest2);

        assert_eq!(digest1.count(), 100);
        let median = digest1.median().unwrap();
        assert!((median - 49.5).abs() < 5.0); // Should be close to 49.5
    }
}
