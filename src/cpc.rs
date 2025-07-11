use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// CPC (Compressed Probabilistic Counting) Sketch implementation
///
/// IMPORTANT LIMITATION: This implementation provides a CPC-like data structure
/// with proper sparse mode and transitions, but uses simplified HLL-style estimation
/// rather than the full CPC algorithm. The estimation method is NOT the complete
/// CPC methodology described in the original paper.
///
/// This provides reasonable cardinality approximations but should not be considered
/// a complete CPC implementation.
pub struct CpcSketch {
    lg_k: u8,
    k: usize,
    sparse_mode: bool,
    sparse_set: HashSet<u64>,
    table: Vec<u8>,
    num_coupons: usize, // Total number of items processed
}

impl CpcSketch {
    /// Create a new CPC sketch
    pub fn new(lg_k: u8) -> Self {
        assert!(lg_k >= 4 && lg_k <= 26, "lg_k must be between 4 and 26");

        let k = 1 << lg_k;

        CpcSketch {
            lg_k,
            k,
            sparse_mode: true,
            sparse_set: HashSet::new(),
            table: vec![0; k],
            num_coupons: 0,
        }
    }

    /// Update the sketch with an item
    pub fn update<T: Hash>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();

        self.num_coupons += 1;

        if self.sparse_mode {
            // In sparse mode, store the raw hash values
            let _was_new = self.sparse_set.insert(hash);

            // Check if we should transition from sparse to dense mode
            // CPC transitions when the sparse set size approaches k/4
            if self.sparse_set.len() >= self.k / 4 {
                self.switch_to_table_mode();
            }
        } else {
            // Dense mode - use bucket index and leading zeros
            let bucket = (hash & ((1 << self.lg_k) - 1)) as usize;
            let w = hash >> self.lg_k;

            // Count leading zeros in w, adding 1 (standard CPC approach)
            // For CPC, we need to be more careful about the rho calculation
            let rho = if w == 0 {
                // If w is 0, we've exhausted all bits - use a reasonable max value
                (64 - self.lg_k).min(32) as u8 + 1
            } else {
                (w.leading_zeros() + 1).min(32) as u8
            };

            // Update bucket with maximum rho value seen
            if rho > self.table[bucket] {
                self.table[bucket] = rho;
            }
        }
    }

    /// Switch from sparse to table mode
    fn switch_to_table_mode(&mut self) {
        self.sparse_mode = false;

        // Re-hash all items from sparse set into the table
        for &hash in &self.sparse_set {
            let bucket = (hash & ((1 << self.lg_k) - 1)) as usize;
            let w = hash >> self.lg_k;

            let rho = if w == 0 {
                (64 - self.lg_k).min(32) as u8 + 1
            } else {
                (w.leading_zeros() + 1).min(32) as u8
            };

            if rho > self.table[bucket] {
                self.table[bucket] = rho;
            }
        }

        self.sparse_set.clear();
    }

    /// Estimate cardinality using HLL approximation (not full CPC algorithm)
    ///
    /// WARNING: This implementation uses simplified HLL-style estimation rather than
    /// the full CPC algorithm. It provides reasonable approximations but does not
    /// implement the complete CPC estimation methodology.
    pub fn estimate(&self) -> f64 {
        if self.sparse_mode {
            // In sparse mode, we have exact count
            self.sparse_set.len() as f64
        } else {
            // Dense mode - use simplified HLL approximation
            self.estimate_hll_approximation()
        }
    }

    /// HLL-style approximation for cardinality estimation
    ///
    /// IMPORTANT: This function does NOT implement the CPC algorithm. It uses a
    /// simplified HLL-style harmonic mean approach for basic cardinality estimation.
    /// This provides reasonable approximations but is not the full CPC methodology.
    fn estimate_hll_approximation(&self) -> f64 {
        let k = self.k as f64;

        // Count the number of empty buckets (zeros)
        let num_zeros = self.table.iter().filter(|&&x| x == 0).count() as f64;

        // Use linear counting when we have many zeros (standard for small cardinalities)
        if num_zeros > 0.0 && num_zeros >= k * 0.1 {
            let linear_estimate = k * (k / num_zeros).ln();
            if linear_estimate.is_finite() && linear_estimate > 0.0 {
                return linear_estimate;
            }
        }

        // For larger cardinalities, use HLL-style harmonic mean calculation
        // WARNING: This is HLL algorithm, NOT CPC algorithm
        let mut harmonic_sum = 0.0;
        for &rho in &self.table {
            if rho > 0 {
                // Standard HLL harmonic mean: each rho contributes 2^(-rho)
                harmonic_sum += 2_f64.powi(-(rho as i32));
            } else {
                harmonic_sum += 1.0;
            }
        }

        if harmonic_sum > 0.0 {
            // Standard HLL formula (NOT CPC algorithm)
            let alpha = 0.7213 / (1.0 + 1.079 / k);
            let raw_estimate = alpha * k * k / harmonic_sum;

            // Apply reasonable bounds
            let min_bound = num_zeros.max(1.0);
            let max_bound = k * 1000.0; // Allow larger estimates than before

            raw_estimate.max(min_bound).min(max_bound)
        } else {
            k // Fallback if harmonic sum is invalid
        }
    }

    /// Get alpha constant for bias correction based on k
    fn get_alpha_constant(&self) -> f64 {
        match self.k {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / (self.k as f64)),
        }
    }

    /// Apply bias correction based on estimate range
    fn apply_bias_correction(&self, raw_estimate: f64, num_zeros: f64) -> f64 {
        let k = self.k as f64;

        // Small range correction (when estimate is small and we have zeros)
        if raw_estimate <= 2.5 * k && num_zeros > 0.0 {
            let linear_estimate = k * (k / num_zeros).ln();
            if linear_estimate.is_finite() && linear_estimate > 0.0 {
                linear_estimate
            } else {
                raw_estimate
            }
        }
        // Large range correction (prevent overflow in very large estimates)
        else if raw_estimate > (1.0 / 30.0) * (1u64 << 32) as f64 {
            let two_32 = (1u64 << 32) as f64;
            let corrected = -two_32 * (1.0 - raw_estimate / two_32).ln();
            if corrected.is_finite() && corrected > 0.0 {
                corrected
            } else {
                raw_estimate
            }
        }
        // Normal range - return raw estimate
        else {
            if raw_estimate.is_finite() && raw_estimate > 0.0 {
                raw_estimate
            } else {
                // Fallback if we have issues
                k
            }
        }
    }

    /// Merge another sketch
    pub fn merge(&mut self, other: &CpcSketch) {
        assert_eq!(
            self.lg_k, other.lg_k,
            "Cannot merge sketches with different lg_k"
        );

        // Update coupon count (approximation for merged sketches)
        self.num_coupons += other.num_coupons;

        // If either is in table mode, switch both to table mode
        if !self.sparse_mode || !other.sparse_mode {
            if self.sparse_mode {
                self.switch_to_table_mode();
            }

            // Merge tables
            if other.sparse_mode {
                // Add other's sparse items
                for &hash in &other.sparse_set {
                    let bucket = (hash & ((1 << self.lg_k) - 1)) as usize;
                    let w = hash >> self.lg_k;
                    let rho = if w == 0 {
                        (64 - self.lg_k).min(32) as u8 + 1
                    } else {
                        (w.leading_zeros() + 1).min(32) as u8
                    };

                    if rho > self.table[bucket] {
                        self.table[bucket] = rho;
                    }
                }
            } else {
                // Merge other's table
                for i in 0..self.k {
                    if other.table[i] > self.table[i] {
                        self.table[i] = other.table[i];
                    }
                }
            }
        } else {
            // Both in sparse mode
            for &hash in &other.sparse_set {
                self.sparse_set.insert(hash);

                // Check if we need to switch
                if self.sparse_set.len() >= self.k / 4 {
                    self.switch_to_table_mode();
                    break;
                }
            }
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Header
        bytes.push(self.lg_k);
        bytes.push(if self.sparse_mode { 0 } else { 1 });

        if self.sparse_mode {
            // Sparse mode
            bytes.extend_from_slice(&(self.sparse_set.len() as u32).to_le_bytes());
            for &hash in &self.sparse_set {
                bytes.extend_from_slice(&hash.to_le_bytes());
            }
        } else {
            // Table mode - simple RLE compression
            let mut i = 0;
            while i < self.table.len() {
                let val = self.table[i];
                let mut run = 1;

                while i + run < self.table.len() && self.table[i + run] == val && run < 255 {
                    run += 1;
                }

                if val == 0 && run > 2 {
                    bytes.push(0);
                    bytes.push(run as u8);
                } else {
                    for _ in 0..run {
                        bytes.push(val);
                    }
                }

                i += run;
            }
        }

        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpc_basic() {
        let mut sketch = CpcSketch::new(10);

        for i in 0..1000 {
            sketch.update(&i);
        }

        let estimate = sketch.estimate();
        let error = (estimate - 1000.0).abs() / 1000.0;

        // Our HLL-style approximation should have reasonable accuracy
        // Standard HLL accuracy is ~1.04/sqrt(2^precision) ≈ 3.2% for precision=10
        // Allow up to 10% error for our simplified HLL approximation
        assert!(
            error < 0.10,
            "HLL approximation error {:.1}% exceeds 10% tolerance",
            error * 100.0
        );
    }

    #[test]
    fn test_cpc_merge() {
        let mut sketch1 = CpcSketch::new(10);
        let mut sketch2 = CpcSketch::new(10);

        for i in 0..500 {
            sketch1.update(&i);
        }

        for i in 500..1000 {
            sketch2.update(&i);
        }

        sketch1.merge(&sketch2);

        let estimate = sketch1.estimate();
        let error = (estimate - 1000.0).abs() / 1000.0;

        // Merged sketch should maintain reasonable accuracy with HLL approximation
        assert!(
            error < 0.10,
            "HLL approximation merge error {:.1}% exceeds 10% tolerance",
            error * 100.0
        );
    }

    #[test]
    fn test_cpc_small() {
        let mut sketch = CpcSketch::new(8);

        for i in 0..50 {
            sketch.update(&format!("item_{}", i));
        }

        let estimate = sketch.estimate();
        let error = (estimate - 50.0).abs() / 50.0;

        assert!(error < 0.1, "Small cardinality error {} is too high", error);
    }
}
