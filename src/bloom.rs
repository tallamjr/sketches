//! Bloom Filter and Counting Bloom Filter for approximate membership testing.
//!
//! A Bloom filter is a space-efficient probabilistic data structure that tests
//! whether an element is a member of a set. False positives are possible but
//! false negatives are not -- if the filter says "not present", the element is
//! definitely absent.
//!
//! # Error Bounds
//! - False positive probability: `P_fp ~= (1 - e^(-kn/m))^k`
//! - Optimal bits per element: `m/n = -ln(P_fp) / (ln(2))^2`
//! - Optimal hash count: `k = (m/n) * ln(2)`
//! - Example: 1M elements, 1% FP rate -> ~9.6 bits/element, k=7
//!
//! # Counting Bloom Filter
//! Replaces single bits with counters, enabling deletion at the cost of
//! ~4x more memory. Counter overflow is handled with saturation.
//!
//! # Common Uses
//! Web caches, spell checkers, database key lookups, network routing tables.
//!
//! # References
//! - Bloom, B. H. "Space/Time Trade-offs in Hash Coding with Allowable Errors."
//!   Communications of the ACM, 1970.

use crate::hash::xxh3::Xxh3Hasher;
use crate::hash::{DEFAULT_SEED, Hashable, SketchHasher, hash64_of};

/// Maximum number of hash positions handled on the stack per operation. Real
/// configurations use k of roughly 7; this cap is far above any practical k.
const MAX_HASHES: usize = 64;

/// Standard Bloom Filter implementation, generic over the hash backend.
pub struct BloomFilterGeneric<H: SketchHasher> {
    bit_array: Vec<u64>,
    num_bits: usize,
    num_hash_functions: usize,
    _hasher: core::marker::PhantomData<H>,
}

/// Standard Bloom Filter using the default `Xxh3Hasher` backend.
pub type BloomFilter = BloomFilterGeneric<Xxh3Hasher>;

impl<H: SketchHasher> BloomFilterGeneric<H> {
    /// Create a new Bloom filter
    ///
    /// # Arguments
    /// * `capacity` - Expected number of elements
    /// * `error_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    pub fn new(capacity: usize, error_rate: f64) -> Self {
        assert!(capacity > 0, "Capacity must be greater than 0");
        assert!(
            error_rate > 0.0 && error_rate < 1.0,
            "Error rate must be between 0 and 1"
        );

        // Calculate optimal parameters
        let num_bits = Self::calculate_num_bits(capacity, error_rate);
        let num_hash_functions = Self::calculate_num_hash_functions(num_bits, capacity);

        let num_u64s = num_bits.div_ceil(64);

        BloomFilterGeneric {
            bit_array: vec![0u64; num_u64s],
            num_bits,
            num_hash_functions,
            _hasher: core::marker::PhantomData,
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
    pub fn add<T: Hashable + ?Sized>(&mut self, item: &T) {
        debug_assert!(self.num_hash_functions <= MAX_HASHES);
        let mut buf = [0usize; MAX_HASHES];
        let positions = &mut buf[..self.num_hash_functions];
        self.hash_positions_into(item, positions);
        self.set_bits_scalar(positions);
    }

    /// Check if an element might be in the filter
    pub fn contains<T: Hashable + ?Sized>(&self, item: &T) -> bool {
        debug_assert!(self.num_hash_functions <= MAX_HASHES);
        let mut buf = [0usize; MAX_HASHES];
        let positions = &mut buf[..self.num_hash_functions];
        self.hash_positions_into(item, positions);
        self.check_bits_scalar(positions)
    }

    /// Fill `out[..num_hash_functions]` with this item's bit positions using
    /// double hashing with xxh3. `out` must be at least `num_hash_functions` long.
    fn hash_positions_into<T: Hashable + ?Sized>(&self, item: &T, out: &mut [usize]) {
        let hash1 = hash64_of(&H::default(), item, DEFAULT_SEED);
        let hash2 = hash64_of(&H::default(), item, 0x517cc1b727220a95);
        for (i, slot) in out.iter_mut().enumerate().take(self.num_hash_functions) {
            let hash = hash1.wrapping_add((i as u64).wrapping_mul(hash2));
            *slot = (hash as usize) % self.num_bits;
        }
    }

    /// Set bits using scalar operations
    fn set_bits_scalar(&mut self, positions: &[usize]) {
        for &pos in positions {
            let chunk_index = pos / 64;
            let bit_index = pos % 64;
            // SAFETY: every pos is `hash % num_bits`, so pos < num_bits and
            // chunk_index = pos/64 < num_bits.div_ceil(64) = bit_array.len().
            debug_assert!(chunk_index < self.bit_array.len());
            unsafe {
                *self.bit_array.get_unchecked_mut(chunk_index) |= 1u64 << bit_index;
            }
        }
    }

    /// Check bits using scalar operations
    fn check_bits_scalar(&self, positions: &[usize]) -> bool {
        for &pos in positions {
            let chunk_index = pos / 64;
            let bit_index = pos % 64;
            // SAFETY: every pos is `hash % num_bits`, so pos < num_bits and
            // chunk_index = pos/64 < num_bits.div_ceil(64) = bit_array.len().
            debug_assert!(chunk_index < self.bit_array.len());
            if (unsafe { *self.bit_array.get_unchecked(chunk_index) } & (1u64 << bit_index)) == 0 {
                return false;
            }
        }
        true
    }

    /// Batch add multiple elements
    pub fn add_batch<T: Hashable>(&mut self, items: &[T]) {
        for item in items {
            self.add(item);
        }
    }

    /// Batch check multiple elements
    pub fn contains_batch<T: Hashable>(&self, items: &[T]) -> Vec<bool> {
        items.iter().map(|item| self.contains(item)).collect()
    }

    /// Clear the filter
    pub fn clear(&mut self) {
        for chunk in &mut self.bit_array {
            *chunk = 0;
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
    pub fn add<T: Hashable + ?Sized>(&mut self, item: &T) {
        debug_assert!(self.num_hash_functions <= MAX_HASHES);
        let mut buf = [0usize; MAX_HASHES];
        let positions = &mut buf[..self.num_hash_functions];
        self.hash_positions_into(item, positions);

        for &pos in positions.iter() {
            // SAFETY: pos = hash % num_bits < num_bits == counters.len().
            debug_assert!(pos < self.counters.len());
            let counter = unsafe { self.counters.get_unchecked_mut(pos) };
            if *counter < self.max_count {
                *counter += 1;
            }
        }
    }

    /// Remove an element from the filter
    pub fn remove<T: Hashable + ?Sized>(&mut self, item: &T) -> bool {
        debug_assert!(self.num_hash_functions <= MAX_HASHES);
        let mut buf = [0usize; MAX_HASHES];
        let positions = &mut buf[..self.num_hash_functions];
        self.hash_positions_into(item, positions);

        // Check if all positions have non-zero counts
        for &pos in positions.iter() {
            // SAFETY: pos = hash % num_bits < num_bits == counters.len().
            debug_assert!(pos < self.counters.len());
            if unsafe { *self.counters.get_unchecked(pos) } == 0 {
                return false; // Item definitely not in filter
            }
        }

        // Decrement all counters
        for &pos in positions.iter() {
            // SAFETY: pos = hash % num_bits < num_bits == counters.len().
            debug_assert!(pos < self.counters.len());
            let counter = unsafe { self.counters.get_unchecked_mut(pos) };
            if *counter > 0 {
                *counter -= 1;
            }
        }

        true
    }

    /// Check if an element might be in the filter
    pub fn contains<T: Hashable + ?Sized>(&self, item: &T) -> bool {
        debug_assert!(self.num_hash_functions <= MAX_HASHES);
        let mut buf = [0usize; MAX_HASHES];
        let positions = &mut buf[..self.num_hash_functions];
        self.hash_positions_into(item, positions);

        for &pos in positions.iter() {
            // SAFETY: pos = hash % num_bits < num_bits == counters.len().
            debug_assert!(pos < self.counters.len());
            if unsafe { *self.counters.get_unchecked(pos) } == 0 {
                return false;
            }
        }
        true
    }

    /// Fill `out[..num_hash_functions]` with this item's bit positions using
    /// double hashing with xxh3. `out` must be at least `num_hash_functions` long.
    fn hash_positions_into<T: Hashable + ?Sized>(&self, item: &T, out: &mut [usize]) {
        let hash1 = hash64_of(&Xxh3Hasher, item, DEFAULT_SEED);
        let hash2 = hash64_of(&Xxh3Hasher, item, 0x517cc1b727220a95);
        for (i, slot) in out.iter_mut().enumerate().take(self.num_hash_functions) {
            let hash = hash1.wrapping_add((i as u64).wrapping_mul(hash2));
            *slot = (hash as usize) % self.num_bits;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_filter_basic() {
        let mut filter = BloomFilter::new(1000, 0.01);

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
    fn test_false_positive_rate() {
        let mut filter = BloomFilter::new(1000, 0.01);

        // Add 1000 items
        for i in 0..1000 {
            filter.add(&format!("item_{i}"));
        }

        // Check false positive rate with new items
        let mut false_positives = 0;
        let test_count = 10000;

        for i in 1000..(1000 + test_count) {
            if filter.contains(&format!("item_{i}")) {
                false_positives += 1;
            }
        }

        let fp_rate = false_positives as f64 / test_count as f64;

        // Should be approximately 1% (with some tolerance)
        assert!(fp_rate < 0.05, "False positive rate {fp_rate} is too high");
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
        let filter = BloomFilter::new(10000, 0.001);
        let stats = filter.statistics();

        assert!(stats.num_bits > 0);
        assert!(stats.num_hash_functions > 0);
        assert_eq!(stats.bits_set, 0); // Empty filter
        assert_eq!(stats.fill_ratio, 0.0);
    }

    #[test]
    fn murmur3_bloom_membership() {
        use crate::hash::murmur3::Murmur3Hasher;
        let mut f = BloomFilterGeneric::<Murmur3Hasher>::new(10_000, 0.01);
        for i in 0u64..1_000 {
            f.add(&i);
        }
        for i in 0u64..1_000 {
            assert!(f.contains(&i));
        }
    }

    #[test]
    fn bloom_fpr_reasonable_new_hash() {
        let mut b = BloomFilter::new(10_000, 0.01);
        for i in 0u64..10_000 {
            b.add(&i);
        }
        for i in 0u64..10_000 {
            assert!(b.contains(&i));
        }
        let mut fp = 0;
        for i in 10_000u64..20_000 {
            if b.contains(&i) {
                fp += 1;
            }
        }
        assert!(fp as f64 / 10_000.0 < 0.03, "fpr too high");
    }
}
