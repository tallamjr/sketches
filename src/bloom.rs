use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[cfg(feature = "optimized")]
use crate::fast_hash;
#[cfg(feature = "optimized")]
use crate::simd_ops::bloom;
#[cfg(feature = "optimized")]
use rayon::prelude::*;

/// Standard Bloom Filter implementation with optional SIMD optimizations
pub struct BloomFilter {
    bit_array: Vec<u64>,
    num_bits: usize,
    num_hash_functions: usize,
    use_simd: bool,
}

impl BloomFilter {
    /// Create a new Bloom filter
    ///
    /// # Arguments
    /// * `capacity` - Expected number of elements
    /// * `error_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    /// * `use_simd` - Whether to use SIMD optimizations (when available)
    pub fn new(capacity: usize, error_rate: f64, use_simd: bool) -> Self {
        assert!(capacity > 0, "Capacity must be greater than 0");
        assert!(
            error_rate > 0.0 && error_rate < 1.0,
            "Error rate must be between 0 and 1"
        );

        // Calculate optimal parameters
        let num_bits = Self::calculate_num_bits(capacity, error_rate);
        let num_hash_functions = Self::calculate_num_hash_functions(num_bits, capacity);

        // Use u64 chunks for better SIMD alignment
        let num_u64s = (num_bits + 63) / 64;

        BloomFilter {
            bit_array: vec![0u64; num_u64s],
            num_bits,
            num_hash_functions,
            use_simd,
        }
    }

    /// Calculate optimal number of bits
    fn calculate_num_bits(capacity: usize, error_rate: f64) -> usize {
        let bits = -(capacity as f64 * error_rate.ln()) / (2.0_f64.ln().powi(2));
        bits.ceil() as usize
    }

    /// Calculate optimal number of hash functions
    fn calculate_num_hash_functions(num_bits: usize, capacity: usize) -> usize {
        let k = (num_bits as f64 / capacity as f64) * 2.0_f64.ln();
        k.round().max(1.0) as usize
    }

    /// Add an element to the filter
    pub fn add<T: Hash>(&mut self, item: &T) {
        let hashes = self.hash_item(item);

        if self.use_simd && self.num_hash_functions >= 4 {
            self.set_bits_simd(&hashes);
        } else {
            self.set_bits_scalar(&hashes);
        }
    }

    /// Check if an element might be in the filter
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let hashes = self.hash_item(item);

        if self.use_simd && self.num_hash_functions >= 4 {
            self.check_bits_simd(&hashes)
        } else {
            self.check_bits_scalar(&hashes)
        }
    }

    /// Generate hash values for an item
    fn hash_item<T: Hash>(&self, item: &T) -> Vec<usize> {
        #[cfg(feature = "optimized")]
        {
            let hash1 = fast_hash::fast_hash(item);
            let hash2 = fast_hash::fast_hash_with_seed(&hash1, 0x517cc1b727220a95);

            // Use double hashing to generate multiple hash functions
            let mut hashes = Vec::with_capacity(self.num_hash_functions);
            for i in 0..self.num_hash_functions {
                let hash = hash1.wrapping_add((i as u64).wrapping_mul(hash2));
                hashes.push((hash as usize) % self.num_bits);
            }

            hashes
        }

        #[cfg(not(feature = "optimized"))]
        {
            let mut hasher1 = DefaultHasher::new();
            item.hash(&mut hasher1);
            let hash1 = hasher1.finish();

            let mut hasher2 = DefaultHasher::new();
            hash1.hash(&mut hasher2);
            let hash2 = hasher2.finish();

            // Use double hashing to generate multiple hash functions
            let mut hashes = Vec::with_capacity(self.num_hash_functions);
            for i in 0..self.num_hash_functions {
                let hash = hash1.wrapping_add((i as u64).wrapping_mul(hash2));
                hashes.push((hash as usize) % self.num_bits);
            }

            hashes
        }
    }

    /// Set bits using scalar operations
    fn set_bits_scalar(&mut self, positions: &[usize]) {
        for &pos in positions {
            let chunk_index = pos / 64;
            let bit_index = pos % 64;

            if chunk_index < self.bit_array.len() {
                self.bit_array[chunk_index] |= 1u64 << bit_index;
            }
        }
    }

    /// Check bits using scalar operations
    fn check_bits_scalar(&self, positions: &[usize]) -> bool {
        for &pos in positions {
            let chunk_index = pos / 64;
            let bit_index = pos % 64;

            if chunk_index >= self.bit_array.len() {
                return false;
            }

            if (self.bit_array[chunk_index] & (1u64 << bit_index)) == 0 {
                return false;
            }
        }
        true
    }

    /// Set bits using SIMD operations when available
    fn set_bits_simd(&mut self, positions: &[usize]) {
        #[cfg(feature = "optimized")]
        {
            bloom::set_bits(&mut self.bit_array, positions);
        }

        #[cfg(not(feature = "optimized"))]
        self.set_bits_scalar(positions);
    }

    /// Check bits using SIMD operations when available
    fn check_bits_simd(&self, positions: &[usize]) -> bool {
        #[cfg(feature = "optimized")]
        {
            let results = bloom::check_bits(&self.bit_array, positions);
            results.iter().all(|&x| x)
        }

        #[cfg(not(feature = "optimized"))]
        self.check_bits_scalar(positions)
    }

    /// Batch add multiple elements for better performance
    #[cfg(feature = "optimized")]
    pub fn add_batch<T: Hash + Sync>(&mut self, items: &[T]) {
        // Use parallel processing for large batches
        let all_positions: Vec<Vec<usize>> = if items.len() > 1000 {
            items.par_iter().map(|item| self.hash_item(item)).collect()
        } else {
            items.iter().map(|item| self.hash_item(item)).collect()
        };

        // Flatten all positions for SIMD processing
        let flat_positions: Vec<usize> = all_positions.into_iter().flatten().collect();

        // Use SIMD operations for setting bits
        if self.use_simd && flat_positions.len() >= 16 {
            bloom::set_bits(&mut self.bit_array, &flat_positions);
        } else {
            self.set_bits_scalar(&flat_positions);
        }
    }

    /// Batch add fallback for non-optimized builds
    #[cfg(not(feature = "optimized"))]
    pub fn add_batch<T: Hash>(&mut self, items: &[T]) {
        for item in items {
            self.add(item);
        }
    }

    /// Batch check multiple elements for better performance
    #[cfg(feature = "optimized")]
    pub fn contains_batch<T: Hash + Sync>(&self, items: &[T]) -> Vec<bool> {
        // Use parallel processing for large batches
        let all_positions: Vec<Vec<usize>> = if items.len() > 1000 {
            items.par_iter().map(|item| self.hash_item(item)).collect()
        } else {
            items.iter().map(|item| self.hash_item(item)).collect()
        };

        // Check each item's positions
        all_positions
            .into_iter()
            .map(|positions| {
                if self.use_simd && positions.len() >= 4 {
                    let results = bloom::check_bits(&self.bit_array, &positions);
                    results.iter().all(|&x| x)
                } else {
                    self.check_bits_scalar(&positions)
                }
            })
            .collect()
    }

    /// Batch check fallback for non-optimized builds
    #[cfg(not(feature = "optimized"))]
    pub fn contains_batch<T: Hash>(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Clear the filter
    pub fn clear(&mut self) {
        #[cfg(feature = "optimized")]
        {
            // Use parallel clearing for large filters
            if self.bit_array.len() > 1000 {
                self.bit_array.par_iter_mut().for_each(|chunk| *chunk = 0);
            } else {
                self.bit_array.fill(0);
            }
        }

        #[cfg(not(feature = "optimized"))]
        {
            for chunk in &mut self.bit_array {
                *chunk = 0;
            }
        }
    }

    /// Get the current false positive probability
    pub fn false_positive_probability(&self) -> f64 {
        let bits_set = self.count_set_bits();
        let fraction_set = bits_set as f64 / self.num_bits as f64;

        // Calculate actual false positive rate based on current state
        (1.0 - (-(self.num_hash_functions as f64) * fraction_set).exp())
            .powi(self.num_hash_functions as i32)
    }

    /// Count the number of set bits
    fn count_set_bits(&self) -> usize {
        self.bit_array
            .iter()
            .map(|&chunk| chunk.count_ones() as usize)
            .sum()
    }

    /// Get filter statistics
    pub fn statistics(&self) -> BloomFilterStats {
        let bits_set = self.count_set_bits();

        BloomFilterStats {
            num_bits: self.num_bits,
            num_hash_functions: self.num_hash_functions,
            bits_set,
            fill_ratio: bits_set as f64 / self.num_bits as f64,
            false_positive_probability: self.false_positive_probability(),
            uses_simd: self.use_simd,
        }
    }
}

/// Statistics about a Bloom filter
#[derive(Debug, Clone)]
pub struct BloomFilterStats {
    pub num_bits: usize,
    pub num_hash_functions: usize,
    pub bits_set: usize,
    pub fill_ratio: f64,
    pub false_positive_probability: f64,
    pub uses_simd: bool,
}

/// Counting Bloom Filter - allows deletions
pub struct CountingBloomFilter {
    counters: Vec<u8>,
    num_bits: usize,
    num_hash_functions: usize,
    max_count: u8,
}

impl CountingBloomFilter {
    /// Create a new counting Bloom filter
    pub fn new(capacity: usize, error_rate: f64, max_count: u8) -> Self {
        let num_bits = BloomFilter::calculate_num_bits(capacity, error_rate);
        let num_hash_functions = BloomFilter::calculate_num_hash_functions(num_bits, capacity);

        CountingBloomFilter {
            counters: vec![0u8; num_bits],
            num_bits,
            num_hash_functions,
            max_count,
        }
    }

    /// Add an element to the filter
    pub fn add<T: Hash>(&mut self, item: &T) {
        let hashes = self.hash_item(item);

        for &pos in &hashes {
            if pos < self.counters.len() && self.counters[pos] < self.max_count {
                self.counters[pos] += 1;
            }
        }
    }

    /// Remove an element from the filter
    pub fn remove<T: Hash>(&mut self, item: &T) -> bool {
        let hashes = self.hash_item(item);

        // Check if all positions have non-zero counts
        for &pos in &hashes {
            if pos >= self.counters.len() || self.counters[pos] == 0 {
                return false; // Item definitely not in filter
            }
        }

        // Decrement all counters
        for &pos in &hashes {
            if pos < self.counters.len() && self.counters[pos] > 0 {
                self.counters[pos] -= 1;
            }
        }

        true
    }

    /// Check if an element might be in the filter
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let hashes = self.hash_item(item);

        for &pos in &hashes {
            if pos >= self.counters.len() || self.counters[pos] == 0 {
                return false;
            }
        }
        true
    }

    /// Generate hash values for an item
    fn hash_item<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut hasher1 = DefaultHasher::new();
        item.hash(&mut hasher1);
        let hash1 = hasher1.finish();

        let mut hasher2 = DefaultHasher::new();
        hash1.hash(&mut hasher2);
        let hash2 = hasher2.finish();

        let mut hashes = Vec::with_capacity(self.num_hash_functions);
        for i in 0..self.num_hash_functions {
            let hash = hash1.wrapping_add((i as u64).wrapping_mul(hash2));
            hashes.push((hash as usize) % self.num_bits);
        }

        hashes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut filter = BloomFilter::new(1000, 0.01, false);

        // Add some items
        filter.add(&"hello");
        filter.add(&"world");
        filter.add(&123);

        // Check positive cases
        assert!(filter.contains(&"hello"));
        assert!(filter.contains(&"world"));
        assert!(filter.contains(&123));

        // Check negative cases (should be false, but might have false positives)
        assert!(!filter.contains(&"not_added"));
    }

    #[test]
    fn test_bloom_filter_simd() {
        let mut filter_simd = BloomFilter::new(1000, 0.01, true);
        let mut filter_standard = BloomFilter::new(1000, 0.01, false);

        let test_items = ["item1", "item2", "item3", "item4", "item5"];

        // Add same items to both filters
        for item in &test_items {
            filter_simd.add(item);
            filter_standard.add(item);
        }

        // Both should have same results
        for item in &test_items {
            assert_eq!(filter_simd.contains(item), filter_standard.contains(item));
        }
    }

    #[test]
    fn test_false_positive_rate() {
        let mut filter = BloomFilter::new(1000, 0.01, false);

        // Add 1000 items
        for i in 0..1000 {
            filter.add(&format!("item_{}", i));
        }

        // Check false positive rate with new items
        let mut false_positives = 0;
        let test_count = 10000;

        for i in 1000..(1000 + test_count) {
            if filter.contains(&format!("item_{}", i)) {
                false_positives += 1;
            }
        }

        let fp_rate = false_positives as f64 / test_count as f64;

        // Should be approximately 1% (with some tolerance)
        assert!(
            fp_rate < 0.05,
            "False positive rate {} is too high",
            fp_rate
        );
    }

    #[test]
    fn test_counting_bloom_filter() {
        let mut filter = CountingBloomFilter::new(1000, 0.01, 255);

        // Add items
        filter.add(&"test1");
        filter.add(&"test2");
        filter.add(&"test1"); // Add again

        // Check contains
        assert!(filter.contains(&"test1"));
        assert!(filter.contains(&"test2"));
        assert!(!filter.contains(&"test3"));

        // Remove item
        assert!(filter.remove(&"test1"));
        assert!(filter.contains(&"test1")); // Still there (added twice)

        assert!(filter.remove(&"test1"));
        assert!(!filter.contains(&"test1")); // Now removed

        // Try to remove non-existent item
        assert!(!filter.remove(&"test3"));
    }

    #[test]
    fn test_filter_parameters() {
        let filter = BloomFilter::new(10000, 0.001, false);
        let stats = filter.statistics();

        assert!(stats.num_bits > 0);
        assert!(stats.num_hash_functions > 0);
        assert_eq!(stats.bits_set, 0); // Empty filter
        assert_eq!(stats.fill_ratio, 0.0);
    }
}
