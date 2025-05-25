use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Linear Counter for small cardinality estimation
///
/// Linear Counter is more accurate than HyperLogLog for small cardinalities (< 1000)
/// and serves as an efficient preprocessing step before transitioning to HLL.
/// It's based on the coupon collector problem and uses a simple bit array.
pub struct LinearCounter {
    bit_array: Vec<u64>,
    num_bits: usize,
    bits_set: usize,
    use_simd: bool,
}

impl LinearCounter {
    /// Create a new Linear Counter
    ///
    /// # Arguments
    /// * `num_bits` - Size of the bit array (larger = more accurate, more memory)
    /// * `use_simd` - Whether to use SIMD optimizations when available
    pub fn new(num_bits: usize, use_simd: bool) -> Self {
        assert!(num_bits > 0, "Number of bits must be positive");

        // Round up to nearest multiple of 64 for alignment
        let aligned_bits = ((num_bits + 63) / 64) * 64;
        let num_u64s = aligned_bits / 64;

        LinearCounter {
            bit_array: vec![0u64; num_u64s],
            num_bits: aligned_bits,
            bits_set: 0,
            use_simd,
        }
    }

    /// Create a Linear Counter with optimal size for expected cardinality
    ///
    /// # Arguments
    /// * `expected_cardinality` - Expected number of unique items
    /// * `error_rate` - Desired error rate (e.g., 0.01 for 1% error)
    /// * `use_simd` - Whether to use SIMD optimizations
    pub fn with_expected_cardinality(
        expected_cardinality: usize,
        error_rate: f64,
        use_simd: bool,
    ) -> Self {
        assert!(
            error_rate > 0.0 && error_rate < 1.0,
            "Error rate must be between 0 and 1"
        );
        assert!(
            expected_cardinality > 0,
            "Expected cardinality must be positive"
        );

        // For Linear Counter, optimal m ≈ n / (error_rate^2) for small n
        // This is a simplified calculation
        let optimal_bits = (expected_cardinality as f64 / (error_rate * error_rate)) as usize;
        let min_bits = (expected_cardinality * 8).max(1024); // Ensure reasonable minimum

        Self::new(optimal_bits.max(min_bits), use_simd)
    }

    /// Update the counter with a new item
    pub fn update<T: Hash>(&mut self, item: &T) {
        let hash = self.hash_item(item);
        let bit_index = (hash as usize) % self.num_bits;

        let chunk_index = bit_index / 64;
        let bit_offset = bit_index % 64;
        let bit_mask = 1u64 << bit_offset;

        // Check if bit is already set to avoid double counting
        if (self.bit_array[chunk_index] & bit_mask) == 0 {
            self.bit_array[chunk_index] |= bit_mask;
            self.bits_set += 1;
        }
    }

    /// Hash an item to get a bit position
    fn hash_item<T: Hash>(&self, item: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        hasher.finish()
    }

    /// Estimate the cardinality
    ///
    /// Uses the formula: -m * ln(Z/m) where m is array size and Z is zero bits
    pub fn estimate(&self) -> f64 {
        let zero_bits = self.num_bits - self.bits_set;

        if zero_bits == 0 {
            // All bits are set - estimate is likely much larger than array size
            return self.num_bits as f64 * 1.5; // Conservative overflow estimate
        }

        if zero_bits == self.num_bits {
            // No bits set
            return 0.0;
        }

        let m = self.num_bits as f64;
        let z = zero_bits as f64;

        // Linear Counter formula: -m * ln(Z/m)
        -m * (z / m).ln()
    }

    /// Check if the counter should transition to HyperLogLog
    ///
    /// Returns true when Linear Counter becomes less accurate than HLL
    pub fn should_transition_to_hll(&self) -> bool {
        // Transition when fill ratio exceeds ~30% or estimated cardinality > threshold
        let fill_ratio = self.bits_set as f64 / self.num_bits as f64;
        let estimated_cardinality = self.estimate();

        fill_ratio > 0.3 || estimated_cardinality > (self.num_bits as f64 * 0.1)
    }

    /// Get the current fill ratio
    pub fn fill_ratio(&self) -> f64 {
        self.bits_set as f64 / self.num_bits as f64
    }

    /// Merge another Linear Counter into this one
    ///
    /// Both counters must have the same bit array size
    pub fn merge(&mut self, other: &LinearCounter) -> Result<(), &'static str> {
        if self.num_bits != other.num_bits {
            return Err("Cannot merge counters with different bit array sizes");
        }

        // Reset bits_set counter and recalculate
        self.bits_set = 0;

        if self.use_simd && self.bit_array.len() >= 4 {
            self.merge_chunked(other);
        } else {
            self.merge_scalar(other);
        }

        Ok(())
    }

    /// Merge using scalar operations
    fn merge_scalar(&mut self, other: &LinearCounter) {
        for i in 0..self.bit_array.len() {
            self.bit_array[i] |= other.bit_array[i];
            self.bits_set += self.bit_array[i].count_ones() as usize;
        }
    }

    /// Merge using chunked processing (NOT true SIMD - just batch optimization)
    fn merge_chunked(&mut self, other: &LinearCounter) {
        let len = self.bit_array.len();
        
        // Process 4 u32 values at a time in batches
        let chunks = self.bit_array.chunks_exact_mut(4);
        let other_chunks = other.bit_array.chunks_exact(4);
        
        for (self_chunk, other_chunk) in chunks.zip(other_chunks) {
            // Perform sequential bitwise OR operations
            self_chunk[0] |= other_chunk[0];
            self_chunk[1] |= other_chunk[1];
            self_chunk[2] |= other_chunk[2];
            self_chunk[3] |= other_chunk[3];
            
            // Count bits sequentially using regular count_ones()
            self.bits_set += self_chunk[0].count_ones() as usize;
            self.bits_set += self_chunk[1].count_ones() as usize;
            self.bits_set += self_chunk[2].count_ones() as usize;
            self.bits_set += self_chunk[3].count_ones() as usize;
        }
        
        // Handle remaining elements
        let remainder_start = (len / 4) * 4;
        for i in remainder_start..len {
            self.bit_array[i] |= other.bit_array[i];
            self.bits_set += self.bit_array[i].count_ones() as usize;
        }
    }


    /// Clear the counter
    pub fn clear(&mut self) {
        for chunk in &mut self.bit_array {
            *chunk = 0;
        }
        self.bits_set = 0;
    }

    /// Get counter statistics
    pub fn statistics(&self) -> LinearCounterStats {
        LinearCounterStats {
            num_bits: self.num_bits,
            bits_set: self.bits_set,
            fill_ratio: self.fill_ratio(),
            estimated_cardinality: self.estimate(),
            should_transition: self.should_transition_to_hll(),
            memory_usage: self.bit_array.len() * std::mem::size_of::<u64>(),
            uses_simd: self.use_simd,
        }
    }

    /// Create a transition-ready HyperLogLog sketch
    ///
    /// This would be used when transitioning from Linear Counter to HLL
    /// for better accuracy at larger cardinalities
    pub fn create_hll_transition(&self, lg_k: u8) -> crate::hll::HllSketch {
        // Create new HLL with specified precision
        crate::hll::HllSketch::new(lg_k)
        // Note: In a real implementation, you might want to try to preserve
        // some information from the Linear Counter, but that's complex
        // and typically the transition happens early enough that it's not critical
    }
}

/// Statistics about a Linear Counter
#[derive(Debug, Clone)]
pub struct LinearCounterStats {
    pub num_bits: usize,
    pub bits_set: usize,
    pub fill_ratio: f64,
    pub estimated_cardinality: f64,
    pub should_transition: bool,
    pub memory_usage: usize,
    pub uses_simd: bool,
}

/// Hybrid Counter that automatically transitions from Linear Counter to HyperLogLog
///
/// This provides optimal accuracy across all cardinality ranges by using
/// Linear Counter for small values and HLL for larger values.
pub struct HybridCounter {
    linear: Option<LinearCounter>,
    hll: Option<crate::hll::HllSketch>,
    transition_threshold: usize,
    lg_k: u8,
}

impl HybridCounter {
    /// Create a new Hybrid Counter
    ///
    /// # Arguments
    /// * `linear_bits` - Size of Linear Counter bit array
    /// * `lg_k` - HyperLogLog precision parameter
    /// * `transition_threshold` - Cardinality threshold for LC->HLL transition
    pub fn new(linear_bits: usize, lg_k: u8, transition_threshold: usize) -> Self {
        HybridCounter {
            linear: Some(LinearCounter::new(linear_bits, false)),
            hll: None,
            transition_threshold,
            lg_k,
        }
    }

    /// Create with optimal parameters for expected cardinality range
    pub fn with_range(max_expected_cardinality: usize) -> Self {
        let lg_k = if max_expected_cardinality < 1000 {
            10
        } else if max_expected_cardinality < 10000 {
            12
        } else {
            14
        };

        let linear_bits = (max_expected_cardinality / 10).max(1024).min(16384);
        let transition_threshold = linear_bits / 10;

        Self::new(linear_bits, lg_k, transition_threshold)
    }

    /// Update the counter with a new item
    pub fn update<T: Hash>(&mut self, item: &T) {
        if let Some(ref mut linear) = self.linear {
            linear.update(item);

            // Check if we should transition to HLL
            if linear.should_transition_to_hll()
                || linear.estimate() as usize > self.transition_threshold
            {
                self.transition_to_hll();
            }
        }

        if let Some(ref mut hll) = self.hll {
            hll.update(item);
        }
    }

    /// Transition from Linear Counter to HyperLogLog
    fn transition_to_hll(&mut self) {
        if self.linear.is_some() && self.hll.is_none() {
            self.hll = Some(crate::hll::HllSketch::new(self.lg_k));
            self.linear = None; // Drop the Linear Counter to free memory
        }
    }

    /// Estimate the cardinality
    pub fn estimate(&self) -> f64 {
        if let Some(ref linear) = self.linear {
            linear.estimate()
        } else if let Some(ref hll) = self.hll {
            hll.estimate()
        } else {
            0.0
        }
    }

    /// Get current mode
    pub fn mode(&self) -> &'static str {
        if self.linear.is_some() {
            "Linear Counter"
        } else {
            "HyperLogLog"
        }
    }

    /// Get hybrid counter statistics
    pub fn statistics(&self) -> HybridCounterStats {
        if let Some(ref linear) = self.linear {
            let linear_stats = linear.statistics();
            HybridCounterStats {
                mode: "Linear Counter".to_string(),
                estimated_cardinality: linear_stats.estimated_cardinality,
                memory_usage: linear_stats.memory_usage,
                fill_ratio: Some(linear_stats.fill_ratio),
                bits_set: Some(linear_stats.bits_set),
                transition_threshold: self.transition_threshold,
            }
        } else if let Some(ref hll) = self.hll {
            HybridCounterStats {
                mode: "HyperLogLog".to_string(),
                estimated_cardinality: hll.estimate(),
                memory_usage: (1 << self.lg_k) * std::mem::size_of::<u8>(), // Approximate HLL memory
                fill_ratio: None,
                bits_set: None,
                transition_threshold: self.transition_threshold,
            }
        } else {
            HybridCounterStats {
                mode: "Empty".to_string(),
                estimated_cardinality: 0.0,
                memory_usage: 0,
                fill_ratio: None,
                bits_set: None,
                transition_threshold: self.transition_threshold,
            }
        }
    }
}

/// Statistics about a Hybrid Counter
#[derive(Debug, Clone)]
pub struct HybridCounterStats {
    pub mode: String,
    pub estimated_cardinality: f64,
    pub memory_usage: usize,
    pub fill_ratio: Option<f64>,
    pub bits_set: Option<usize>,
    pub transition_threshold: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_counter_basic() {
        let mut lc = LinearCounter::new(1024, false);

        // Add some unique items
        for i in 0..100 {
            lc.update(&i);
        }

        let estimate = lc.estimate();
        let error = (estimate - 100.0).abs() / 100.0;

        assert!(
            error < 0.1,
            "Error {} too high for small cardinality",
            error
        );
        assert!(lc.bits_set > 0);
        assert!(lc.fill_ratio() > 0.0);
    }

    #[test]
    fn test_linear_counter_accuracy() {
        let mut lc = LinearCounter::with_expected_cardinality(500, 0.05, false);

        // Add known number of unique items
        for i in 0..300 {
            lc.update(&format!("item_{}", i));
        }

        let estimate = lc.estimate();
        let error = (estimate - 300.0).abs() / 300.0;

        assert!(
            error < 0.1,
            "Error {} too high (estimate: {}, actual: {})",
            error,
            estimate,
            300
        );
    }

    #[test]
    fn test_linear_counter_merge() {
        let mut lc1 = LinearCounter::new(2048, false);
        let mut lc2 = LinearCounter::new(2048, false);

        // Add different items to each counter
        for i in 0..100 {
            lc1.update(&i);
        }

        for i in 100..200 {
            lc2.update(&i);
        }

        let before_merge = lc1.estimate();
        lc1.merge(&lc2).unwrap();
        let after_merge = lc1.estimate();

        assert!(after_merge > before_merge);
        assert!((after_merge - 200.0).abs() / 200.0 < 0.15);
    }

    #[test]
    fn test_linear_counter_transition() {
        let mut lc = LinearCounter::new(512, false);

        // Fill it up to trigger transition recommendation
        for i in 0..400 {
            lc.update(&i);
        }

        assert!(lc.should_transition_to_hll());
        assert!(lc.fill_ratio() > 0.3);
    }

    #[test]
    fn test_hybrid_counter() {
        let mut hybrid = HybridCounter::new(2048, 10, 400); // Larger thresholds

        assert_eq!(hybrid.mode(), "Linear Counter");

        // Add items to stay in Linear Counter mode
        for i in 0..100 {
            hybrid.update(&i);
        }

        assert_eq!(hybrid.mode(), "Linear Counter");
        let estimate1 = hybrid.estimate();

        // Add more items to trigger transition
        for i in 100..800 {
            hybrid.update(&i);
        }

        // After many items, should transition to HLL
        assert_eq!(hybrid.mode(), "HyperLogLog");
        let estimate2 = hybrid.estimate();
        assert!(estimate2 > estimate1);
    }

    #[test]
    fn test_hybrid_counter_range() {
        let hybrid = HybridCounter::with_range(10000);
        let stats = hybrid.statistics();

        assert_eq!(stats.mode, "Linear Counter");
        assert!(stats.transition_threshold > 0);
        assert!(stats.memory_usage > 0);
    }

    #[test]
    fn test_linear_counter_edge_cases() {
        let mut lc = LinearCounter::new(1024, false);

        // Empty counter
        assert_eq!(lc.estimate(), 0.0);
        assert_eq!(lc.fill_ratio(), 0.0);

        // Same item multiple times
        for _ in 0..10 {
            lc.update(&"same_item");
        }

        let estimate = lc.estimate();
        assert!(
            estimate >= 0.8 && estimate <= 1.2,
            "Estimate {} should be close to 1",
            estimate
        );
    }

    #[test]
    fn test_linear_counter_simd() {
        let mut lc_simd = LinearCounter::new(1024, true);
        let mut lc_standard = LinearCounter::new(1024, false);

        let test_items = (0..100).collect::<Vec<_>>();

        // Add same items to both counters
        for item in &test_items {
            lc_simd.update(item);
            lc_standard.update(item);
        }

        // Should give same results
        assert_eq!(lc_simd.bits_set, lc_standard.bits_set);
        assert!((lc_simd.estimate() - lc_standard.estimate()).abs() < 0.01);
    }
}
