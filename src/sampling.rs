//! Reservoir Sampling Algorithms
//!
//! This module implements various reservoir sampling algorithms for maintaining
//! a random sample from a stream of data.

use rand::prelude::*;
use rand::rngs::ThreadRng;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

/// Algorithm R: Basic reservoir sampling implementation
///
/// Maintains a uniform random sample of k items from a stream of n items
/// where n is not known in advance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirSamplerR<T: Clone> {
    /// The reservoir containing the current sample
    reservoir: Vec<T>,
    /// Maximum size of the reservoir
    capacity: usize,
    /// Number of items processed so far
    items_seen: usize,
    /// Random number generator
    #[serde(skip)]
    rng: ThreadRng,
}

impl<T: Clone> ReservoirSamplerR<T> {
    /// Create a new reservoir sampler with the given capacity
    pub fn new(capacity: usize) -> Self {
        Self {
            reservoir: Vec::with_capacity(capacity),
            capacity,
            items_seen: 0,
            rng: thread_rng(),
        }
    }

    /// Add an item to the reservoir using Algorithm R
    ///
    /// Time complexity: O(1)
    /// Space complexity: O(k) where k is the capacity
    pub fn add(&mut self, item: T) {
        self.items_seen += 1;

        if self.reservoir.len() < self.capacity {
            // Reservoir not full, just add the item
            self.reservoir.push(item);
        } else {
            // Reservoir full, decide whether to replace an existing item
            let random_index = self.rng.r#gen_range(0..self.items_seen);
            if random_index < self.capacity {
                self.reservoir[random_index] = item;
            }
        }
    }

    /// Get the current sample
    pub fn sample(&self) -> &[T] {
        &self.reservoir
    }

    /// Get the number of items seen so far
    pub fn items_seen(&self) -> usize {
        self.items_seen
    }

    /// Get the capacity of the reservoir
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if the reservoir is full
    pub fn is_full(&self) -> bool {
        self.reservoir.len() == self.capacity
    }

    /// Clear the reservoir and reset counters
    pub fn clear(&mut self) {
        self.reservoir.clear();
        self.items_seen = 0;
    }

    /// Merge two reservoir samplers (for distributed sampling)
    pub fn merge(&mut self, other: &Self) {
        for item in other.sample() {
            self.add(item.clone());
        }
    }
}

/// Algorithm A: Optimized reservoir sampling for large datasets
///
/// Uses Vitter's Algorithm A which skips items probabilistically to reduce
/// the number of random number generations for large datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirSamplerA<T: Clone> {
    /// The reservoir containing the current sample
    reservoir: Vec<T>,
    /// Maximum size of the reservoir
    capacity: usize,
    /// Number of items processed so far
    items_seen: usize,
    /// Number of items to skip until next potential replacement
    skip_count: usize,
    /// Random number generator
    #[serde(skip)]
    rng: ThreadRng,
}

impl<T: Clone> ReservoirSamplerA<T> {
    /// Create a new reservoir sampler with the given capacity
    pub fn new(capacity: usize) -> Self {
        let mut sampler = Self {
            reservoir: Vec::with_capacity(capacity),
            capacity,
            items_seen: 0,
            skip_count: 0,
            rng: thread_rng(),
        };
        sampler.compute_next_skip();
        sampler
    }

    /// Compute the number of items to skip until the next potential replacement
    fn compute_next_skip(&mut self) {
        if self.reservoir.len() < self.capacity {
            self.skip_count = 0;
            return;
        }

        // Use geometric distribution to determine skip count
        let u: f64 = self.rng.r#gen();
        let w = (-u.ln() / self.capacity as f64).exp();
        self.skip_count = ((self.items_seen as f64 + 1.0) * w).floor() as usize - self.items_seen;
    }

    /// Add an item to the reservoir using Algorithm A
    ///
    /// Time complexity: O(1) amortized
    /// Space complexity: O(k) where k is the capacity
    pub fn add(&mut self, item: T) {
        self.items_seen += 1;

        if self.reservoir.len() < self.capacity {
            // Reservoir not full, just add the item
            self.reservoir.push(item);
            if self.reservoir.len() == self.capacity {
                self.compute_next_skip();
            }
        } else if self.skip_count > 0 {
            // Skip this item
            self.skip_count -= 1;
        } else {
            // Replace a random item in the reservoir
            let random_index = self.rng.r#gen_range(0..self.capacity);
            self.reservoir[random_index] = item;
            self.compute_next_skip();
        }
    }

    /// Get the current sample
    pub fn sample(&self) -> &[T] {
        &self.reservoir
    }

    /// Get the number of items seen so far
    pub fn items_seen(&self) -> usize {
        self.items_seen
    }

    /// Get the capacity of the reservoir
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if the reservoir is full
    pub fn is_full(&self) -> bool {
        self.reservoir.len() == self.capacity
    }

    /// Clear the reservoir and reset counters
    pub fn clear(&mut self) {
        self.reservoir.clear();
        self.items_seen = 0;
        self.skip_count = 0;
        self.compute_next_skip();
    }

    /// Merge two reservoir samplers (for distributed sampling)
    pub fn merge(&mut self, other: &Self) {
        for item in other.sample() {
            self.add(item.clone());
        }
    }
}

/// Weighted Reservoir Sampling using Algorithm A-Res
///
/// Maintains a sample where each item has a weight, and the probability
/// of selection is proportional to the weight.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightedReservoirSampler<T: Clone> {
    /// Items with their weights and keys
    reservoir: Vec<(T, f64, f64)>, // (item, weight, key)
    /// Maximum size of the reservoir
    capacity: usize,
    /// Random number generator
    #[serde(skip)]
    rng: ThreadRng,
    /// Total weight processed
    total_weight: f64,
}

impl<T: Clone> WeightedReservoirSampler<T> {
    /// Create a new weighted reservoir sampler
    pub fn new(capacity: usize) -> Self {
        Self {
            reservoir: Vec::with_capacity(capacity),
            capacity,
            rng: thread_rng(),
            total_weight: 0.0,
        }
    }

    /// Add an item with weight using A-Res algorithm
    ///
    /// Each item gets a key = uniform_random^(1/weight)
    /// Items with highest keys are kept in the reservoir
    pub fn add_weighted(&mut self, item: T, weight: f64) {
        if weight <= 0.0 {
            return; // Invalid weight
        }

        self.total_weight += weight;

        // Generate key for this item
        let u: f64 = self.rng.r#gen();
        let key = u.powf(1.0 / weight);

        if self.reservoir.len() < self.capacity {
            // Reservoir not full, just add the item
            self.reservoir.push((item, weight, key));
            if self.reservoir.len() == self.capacity {
                // Sort by key (descending) when reservoir becomes full
                self.reservoir
                    .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            }
        } else {
            // Check if this item should replace the minimum key item
            let min_key = self.reservoir[self.capacity - 1].2;
            if key > min_key {
                // Replace the item with minimum key
                self.reservoir[self.capacity - 1] = (item, weight, key);
                // Re-sort to maintain order
                self.reservoir
                    .sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
            }
        }
    }

    /// Add an item with weight 1.0
    pub fn add(&mut self, item: T) {
        self.add_weighted(item, 1.0);
    }

    /// Get the current sample (items only)
    pub fn sample(&self) -> Vec<T> {
        self.reservoir
            .iter()
            .map(|(item, _, _)| item.clone())
            .collect()
    }

    /// Get the current sample with weights
    pub fn sample_with_weights(&self) -> Vec<(T, f64)> {
        self.reservoir
            .iter()
            .map(|(item, weight, _)| (item.clone(), *weight))
            .collect()
    }

    /// Get the total weight processed
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }

    /// Get the capacity of the reservoir
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Clear the reservoir
    pub fn clear(&mut self) {
        self.reservoir.clear();
        self.total_weight = 0.0;
    }
}

/// Stream processor for reservoir sampling
///
/// Provides a high-level interface for processing streams of data
pub struct StreamSampler<T: Clone> {
    sampler: ReservoirSamplerA<T>,
    batch_size: usize,
    buffer: VecDeque<T>,
}

impl<T: Clone> StreamSampler<T> {
    /// Create a new stream sampler
    pub fn new(capacity: usize, batch_size: usize) -> Self {
        Self {
            sampler: ReservoirSamplerA::new(capacity),
            batch_size,
            buffer: VecDeque::new(),
        }
    }

    /// Add items to the stream buffer
    pub fn push_batch(&mut self, items: Vec<T>) {
        for item in items {
            self.buffer.push_back(item);
        }
        self.process_buffer();
    }

    /// Process buffered items in batches
    fn process_buffer(&mut self) {
        while self.buffer.len() >= self.batch_size {
            for _ in 0..self.batch_size {
                if let Some(item) = self.buffer.pop_front() {
                    self.sampler.add(item);
                }
            }
        }
    }

    /// Flush remaining items in buffer
    pub fn flush(&mut self) {
        while let Some(item) = self.buffer.pop_front() {
            self.sampler.add(item);
        }
    }

    /// Get the current sample
    pub fn sample(&self) -> &[T] {
        self.sampler.sample()
    }

    /// Get statistics about the stream processing
    pub fn stats(&self) -> SamplingStats {
        SamplingStats {
            items_processed: self.sampler.items_seen(),
            sample_size: self.sampler.sample().len(),
            capacity: self.sampler.capacity(),
            buffer_size: self.buffer.len(),
        }
    }
}

/// Statistics for sampling operations
#[derive(Debug, Clone, PartialEq)]
pub struct SamplingStats {
    pub items_processed: usize,
    pub sample_size: usize,
    pub capacity: usize,
    pub buffer_size: usize,
}

impl fmt::Display for SamplingStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SamplingStats {{ processed: {}, sample: {}/{}, buffer: {} }}",
            self.items_processed, self.sample_size, self.capacity, self.buffer_size
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservoir_sampler_r_basic() {
        let mut sampler = ReservoirSamplerR::new(3);

        // Add items when reservoir is not full
        sampler.add(1);
        sampler.add(2);
        sampler.add(3);

        assert_eq!(sampler.sample().len(), 3);
        assert_eq!(sampler.items_seen(), 3);
        assert!(sampler.is_full());

        // Add more items - sample size should remain 3
        for i in 4..100 {
            sampler.add(i);
        }

        assert_eq!(sampler.sample().len(), 3);
        assert_eq!(sampler.items_seen(), 99);
    }

    #[test]
    fn test_reservoir_sampler_a_basic() {
        let mut sampler = ReservoirSamplerA::new(3);

        // Add items when reservoir is not full
        sampler.add(1);
        sampler.add(2);
        sampler.add(3);

        assert_eq!(sampler.sample().len(), 3);
        assert_eq!(sampler.items_seen(), 3);
        assert!(sampler.is_full());

        // Add more items - sample size should remain 3
        for i in 4..100 {
            sampler.add(i);
        }

        assert_eq!(sampler.sample().len(), 3);
        assert_eq!(sampler.items_seen(), 99);
    }

    #[test]
    fn test_weighted_reservoir_sampler() {
        let mut sampler = WeightedReservoirSampler::new(3);

        // Add items with different weights
        sampler.add_weighted("low".to_string(), 0.1);
        sampler.add_weighted("medium".to_string(), 1.0);
        sampler.add_weighted("high".to_string(), 10.0);

        assert_eq!(sampler.sample().len(), 3);
        assert_eq!(sampler.total_weight(), 11.1);

        // Add more items
        for i in 0..10 {
            sampler.add_weighted(format!("item_{}", i), 1.0);
        }

        assert_eq!(sampler.sample().len(), 3);
        assert_eq!(sampler.total_weight(), 21.1);
    }

    #[test]
    fn test_stream_sampler() {
        let mut stream = StreamSampler::new(5, 3);

        // Process items in batches
        stream.push_batch(vec![1, 2, 3, 4, 5]);
        stream.push_batch(vec![6, 7, 8]);
        stream.flush();

        let stats = stream.stats();
        assert_eq!(stats.sample_size, 5);
        assert_eq!(stats.items_processed, 8);
        assert_eq!(stats.buffer_size, 0);
    }

    #[test]
    fn test_sampler_merge() {
        let mut sampler1 = ReservoirSamplerR::new(3);
        let mut sampler2 = ReservoirSamplerR::new(3);

        sampler1.add(1);
        sampler1.add(2);
        sampler1.add(3);

        sampler2.add(4);
        sampler2.add(5);
        sampler2.add(6);

        sampler1.merge(&sampler2);

        assert_eq!(sampler1.sample().len(), 3);
        assert_eq!(sampler1.items_seen(), 6);
    }

    #[test]
    fn test_sampler_clear() {
        let mut sampler = ReservoirSamplerR::new(3);

        sampler.add(1);
        sampler.add(2);
        sampler.add(3);

        assert_eq!(sampler.sample().len(), 3);
        assert_eq!(sampler.items_seen(), 3);

        sampler.clear();

        assert_eq!(sampler.sample().len(), 0);
        assert_eq!(sampler.items_seen(), 0);
        assert!(!sampler.is_full());
    }

    #[test]
    fn test_uniform_distribution() {
        // Test that sampling produces approximately uniform distribution
        let mut counts = [0; 10];
        let trials = 1000;

        for _ in 0..trials {
            let mut sampler = ReservoirSamplerR::new(5);

            // Add items 0-9
            for i in 0..10 {
                sampler.add(i);
            }

            // Count how many times each item appears in samples
            for &item in sampler.sample() {
                counts[item] += 1;
            }
        }

        // Each item should appear roughly 500 times (5/10 * 1000)
        // Allow for statistical variation
        for &count in &counts {
            assert!(
                count > 300 && count < 700,
                "Count {} not in expected range",
                count
            );
        }
    }
}
