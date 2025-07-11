//! Comprehensive demonstration of reservoir sampling algorithms
//!
//! This example showcases:
//! - Algorithm R (basic reservoir sampling)
//! - Algorithm A (optimized for large datasets)
//! - Weighted reservoir sampling
//! - Stream processing with batching
//! - Distributed sampling scenarios
//! - Performance comparisons between algorithms

use sketches::sampling::{
    ReservoirSamplerA, ReservoirSamplerR, SamplingStats, StreamSampler, WeightedReservoirSampler,
};
use std::collections::HashMap;
use std::time::Instant;

fn main() {
    println!("🎯 Reservoir Sampling Algorithms Demo\n");

    // Basic Algorithm R demonstration
    basic_algorithm_r_demo();

    // Algorithm A demonstration with performance comparison
    algorithm_a_performance_demo();

    // Weighted sampling demonstration
    weighted_sampling_demo();

    // Stream processing demonstration
    stream_processing_demo();

    // Distributed sampling scenario
    distributed_sampling_demo();

    // Statistical validation
    statistical_validation_demo();

    println!("\n✅ All reservoir sampling demos completed!");
}

fn basic_algorithm_r_demo() {
    println!("📝 Basic Algorithm R Demonstration");
    println!("==================================");

    let mut sampler = ReservoirSamplerR::new(5);

    // Simulate a stream of items
    println!("Adding items to stream...");
    for i in 1..=20 {
        sampler.add(format!("item_{}", i));
        if i % 5 == 0 {
            println!(
                "  After {} items: sample = {:?}",
                sampler.items_seen(),
                sampler.sample()
            );
        }
    }

    println!("Final sample: {:?}", sampler.sample());
    println!("Items processed: {}", sampler.items_seen());
    println!(
        "Sample size: {}/{}",
        sampler.sample().len(),
        sampler.capacity()
    );
    println!();
}

fn algorithm_a_performance_demo() {
    println!("⚡ Algorithm A Performance Comparison");
    println!("===================================");

    let sample_size = 100;
    let stream_size = 100_000;

    // Test Algorithm R
    let start = Instant::now();
    let mut sampler_r = ReservoirSamplerR::new(sample_size);
    for i in 0..stream_size {
        sampler_r.add(format!("item_{}", i));
    }
    let time_r = start.elapsed();

    // Test Algorithm A
    let start = Instant::now();
    let mut sampler_a = ReservoirSamplerA::new(sample_size);
    for i in 0..stream_size {
        sampler_a.add(format!("item_{}", i));
    }
    let time_a = start.elapsed();

    println!(
        "Processing {} items into sample of size {}:",
        stream_size, sample_size
    );
    println!("  Algorithm R: {:?}", time_r);
    println!("  Algorithm A: {:?}", time_a);
    println!(
        "  Speedup: {:.2}x",
        time_r.as_nanos() as f64 / time_a.as_nanos() as f64
    );

    // Verify both produce valid samples
    assert_eq!(sampler_r.sample().len(), sample_size);
    assert_eq!(sampler_a.sample().len(), sample_size);
    assert_eq!(sampler_r.items_seen(), stream_size);
    assert_eq!(sampler_a.items_seen(), stream_size);

    println!("✓ Both algorithms produced valid samples of correct size");
    println!();
}

fn weighted_sampling_demo() {
    println!("⚖️  Weighted Reservoir Sampling Demo");
    println!("===================================");

    let mut sampler = WeightedReservoirSampler::new(5);

    // Add items with different weights
    println!("Adding weighted items:");
    let weighted_items = vec![
        ("common", 1.0),       // Low weight
        ("rare", 10.0),        // High weight
        ("ultra_rare", 100.0), // Very high weight
        ("normal_1", 1.0),
        ("normal_2", 1.0),
        ("normal_3", 1.0),
        ("special", 5.0),    // Medium weight
        ("legendary", 50.0), // High weight
    ];

    for (item, weight) in &weighted_items {
        sampler.add_weighted(item.to_string(), *weight);
        println!("  Added '{}' with weight {}", item, weight);
    }

    println!("\nSample with weights: {:?}", sampler.sample_with_weights());
    println!("Total weight processed: {:.1}", sampler.total_weight());

    // Run multiple trials to see weight bias
    println!("\nRunning 1000 trials to validate weight bias:");
    let mut item_counts: HashMap<String, u32> = HashMap::new();

    for _ in 0..1000 {
        let mut trial_sampler = WeightedReservoirSampler::new(3);
        for (item, weight) in &weighted_items {
            trial_sampler.add_weighted(item.to_string(), *weight);
        }

        for item in trial_sampler.sample() {
            *item_counts.entry(item).or_insert(0) += 1;
        }
    }

    println!("Item selection frequencies:");
    for (item, count) in &item_counts {
        println!("  {}: {} times", item, count);
    }
    println!();
}

fn stream_processing_demo() {
    println!("🌊 Stream Processing Demo");
    println!("========================");

    let mut stream = StreamSampler::new(10, 3);

    println!("Processing stream in batches...");

    // Process data in batches
    let batches = vec![
        vec!["user_1", "user_2", "user_3", "user_4"],
        vec!["user_5", "user_6", "user_7"],
        vec!["user_8", "user_9", "user_10", "user_11", "user_12"],
        vec!["user_13", "user_14"],
    ];

    for (i, batch) in batches.iter().enumerate() {
        let string_batch = batch.iter().map(|s| s.to_string()).collect();
        stream.push_batch(string_batch);
        let stats = stream.stats();
        println!("  Batch {}: {} - {}", i + 1, batch.join(", "), stats);
    }

    // Flush remaining items
    stream.flush();
    let final_stats = stream.stats();
    println!("After flush: {}", final_stats);
    println!("Final sample: {:?}", stream.sample());
    println!();
}

fn distributed_sampling_demo() {
    println!("🌐 Distributed Sampling Demo");
    println!("============================");

    // Simulate multiple nodes collecting samples
    let mut node1 = ReservoirSamplerR::new(5);
    let mut node2 = ReservoirSamplerR::new(5);
    let mut node3 = ReservoirSamplerR::new(5);

    println!("Simulating distributed data collection...");

    // Node 1 processes data from region A
    for i in 1..=10 {
        node1.add(format!("region_A_item_{}", i));
    }
    println!("Node 1 sample: {:?}", node1.sample());

    // Node 2 processes data from region B
    for i in 1..=15 {
        node2.add(format!("region_B_item_{}", i));
    }
    println!("Node 2 sample: {:?}", node2.sample());

    // Node 3 processes data from region C
    for i in 1..=8 {
        node3.add(format!("region_C_item_{}", i));
    }
    println!("Node 3 sample: {:?}", node3.sample());

    // Merge samples at coordinator
    let mut coordinator = ReservoirSamplerR::new(5);
    coordinator.merge(&node1);
    coordinator.merge(&node2);
    coordinator.merge(&node3);

    println!("Merged global sample: {:?}", coordinator.sample());
    println!(
        "Total items processed across all nodes: {}",
        node1.items_seen() + node2.items_seen() + node3.items_seen()
    );
    println!("Coordinator processed: {} items", coordinator.items_seen());
    println!();
}

fn statistical_validation_demo() {
    println!("📊 Statistical Validation Demo");
    println!("==============================");

    // Test that sampling is uniform over many trials
    let trials = 10_000;
    let sample_size = 3;
    let population_size = 10;

    let mut selection_counts = vec![0; population_size];

    println!(
        "Running {} trials with population of {} items, sample size {}...",
        trials, population_size, sample_size
    );

    for _ in 0..trials {
        let mut sampler = ReservoirSamplerR::new(sample_size);

        // Add items 0-9
        for i in 0..population_size {
            sampler.add(i.to_string());
        }

        // Count selections
        for item_str in sampler.sample() {
            if let Ok(item) = item_str.parse::<usize>() {
                selection_counts[item] += 1;
            }
        }
    }

    // Expected count for each item
    let expected_count = trials * sample_size / population_size;
    println!("Expected selection count per item: {}", expected_count);

    println!("Actual selection counts:");
    for (item, count) in selection_counts.iter().enumerate() {
        let deviation = (*count as f64 - expected_count as f64) / expected_count as f64 * 100.0;
        println!(
            "  Item {}: {} times ({:+.1}% from expected)",
            item, count, deviation
        );
    }

    // Statistical test: chi-squared goodness of fit
    let chi_squared: f64 = selection_counts
        .iter()
        .map(|&count| {
            let diff = count as f64 - expected_count as f64;
            diff * diff / expected_count as f64
        })
        .sum();

    println!("Chi-squared statistic: {:.2}", chi_squared);

    // For 9 degrees of freedom, critical value at 95% confidence is ~16.92
    if chi_squared < 16.92 {
        println!("✓ Sampling appears to be uniformly distributed (p > 0.05)");
    } else {
        println!("⚠️  Sampling may not be perfectly uniform (p < 0.05)");
    }

    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_r_basic() {
        let mut sampler = ReservoirSamplerR::new(3);

        for i in 1..=10 {
            sampler.add(format!("item_{}", i));
        }

        assert_eq!(sampler.sample().len(), 3);
        assert_eq!(sampler.items_seen(), 10);
        assert!(sampler.is_full());
    }

    #[test]
    fn test_algorithm_a_basic() {
        let mut sampler = ReservoirSamplerA::new(3);

        for i in 1..=10 {
            sampler.add(format!("item_{}", i));
        }

        assert_eq!(sampler.sample().len(), 3);
        assert_eq!(sampler.items_seen(), 10);
        assert!(sampler.is_full());
    }

    #[test]
    fn test_weighted_sampling() {
        let mut sampler = WeightedReservoirSampler::new(2);

        sampler.add_weighted("heavy".to_string(), 100.0);
        sampler.add_weighted("light".to_string(), 0.1);

        assert_eq!(sampler.capacity(), 2);
        assert_eq!(sampler.total_weight(), 100.1);
    }

    #[test]
    fn test_merge_samplers() {
        let mut sampler1 = ReservoirSamplerR::new(3);
        let mut sampler2 = ReservoirSamplerR::new(3);

        sampler1.add("a".to_string());
        sampler1.add("b".to_string());

        sampler2.add("c".to_string());
        sampler2.add("d".to_string());

        sampler1.merge(&sampler2);

        assert_eq!(sampler1.sample().len(), 3);
        assert_eq!(sampler1.items_seen(), 4);
    }

    #[test]
    fn test_clear_functionality() {
        let mut sampler = ReservoirSamplerR::new(3);

        sampler.add("test".to_string());
        assert_eq!(sampler.items_seen(), 1);

        sampler.clear();
        assert_eq!(sampler.items_seen(), 0);
        assert_eq!(sampler.sample().len(), 0);
        assert!(!sampler.is_full());
    }
}
