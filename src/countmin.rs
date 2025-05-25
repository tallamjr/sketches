use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Count-Min Sketch for frequency estimation with optional SIMD optimizations
pub struct CountMinSketch {
    width: usize,
    depth: usize,
    table: Vec<Vec<u64>>,
    hash_seeds: Vec<u64>,
    use_simd: bool,
    conservative_update: bool,
}

impl CountMinSketch {
    /// Create a new Count-Min sketch
    /// 
    /// # Arguments
    /// * `width` - Number of buckets per row (affects accuracy)
    /// * `depth` - Number of hash functions/rows (affects probability of error)
    /// * `use_simd` - Whether to use SIMD optimizations when available
    /// * `conservative_update` - Whether to use conservative update (reduces overestimation)
    pub fn new(width: usize, depth: usize, use_simd: bool, conservative_update: bool) -> Self {
        assert!(width > 0 && depth > 0, "Width and depth must be positive");
        
        // Initialize hash seeds for different hash functions
        let mut hash_seeds = Vec::with_capacity(depth);
        let mut seed = 1u64;
        for _ in 0..depth {
            hash_seeds.push(seed);
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223); // Linear congruential generator
        }
        
        CountMinSketch {
            width,
            depth,
            table: vec![vec![0u64; width]; depth],
            hash_seeds,
            use_simd,
            conservative_update,
        }
    }
    
    /// Create a sketch with parameters calculated from error bounds
    /// 
    /// # Arguments
    /// * `epsilon` - Maximum relative error (e.g., 0.01 for 1% error)
    /// * `delta` - Probability of exceeding the error bound (e.g., 0.01 for 1% probability)
    /// * `use_simd` - Whether to use SIMD optimizations
    /// * `conservative_update` - Whether to use conservative update
    pub fn with_error_bounds(epsilon: f64, delta: f64, use_simd: bool, conservative_update: bool) -> Self {
        assert!(epsilon > 0.0 && epsilon < 1.0, "Epsilon must be between 0 and 1");
        assert!(delta > 0.0 && delta < 1.0, "Delta must be between 0 and 1");
        
        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;
        
        Self::new(width, depth, use_simd, conservative_update)
    }
    
    /// Update the count for an item
    pub fn update<T: Hash>(&mut self, item: &T, count: u64) {
        let hashes = self.hash_item(item);
        
        if self.use_simd && self.depth >= 4 {
            self.update_simd(&hashes, count);
        } else {
            self.update_scalar(&hashes, count);
        }
    }
    
    /// Increment the count for an item by 1
    pub fn increment<T: Hash>(&mut self, item: &T) {
        self.update(item, 1);
    }
    
    /// Estimate the frequency of an item
    pub fn estimate<T: Hash>(&self, item: &T) -> u64 {
        let hashes = self.hash_item(item);
        
        if self.use_simd && self.depth >= 4 {
            self.estimate_simd(&hashes)
        } else {
            self.estimate_scalar(&hashes)
        }
    }
    
    /// Generate hash values for an item
    fn hash_item<T: Hash>(&self, item: &T) -> Vec<usize> {
        let mut hashes = Vec::with_capacity(self.depth);
        
        for &seed in &self.hash_seeds {
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            item.hash(&mut hasher);
            let hash = hasher.finish();
            hashes.push((hash as usize) % self.width);
        }
        
        hashes
    }
    
    /// Update counts using scalar operations
    fn update_scalar(&mut self, hashes: &[usize], count: u64) {
        if self.conservative_update {
            // Conservative update: only increment if it doesn't exceed the minimum
            let current_min = self.estimate_scalar(hashes);
            for (row, &col) in hashes.iter().enumerate() {
                if self.table[row][col] == current_min {
                    self.table[row][col] = self.table[row][col].saturating_add(count);
                }
            }
        } else {
            // Standard update: increment all positions
            for (row, &col) in hashes.iter().enumerate() {
                self.table[row][col] = self.table[row][col].saturating_add(count);
            }
        }
    }
    
    /// Estimate frequency using scalar operations
    fn estimate_scalar(&self, hashes: &[usize]) -> u64 {
        let mut min_count = u64::MAX;
        
        for (row, &col) in hashes.iter().enumerate() {
            min_count = min_count.min(self.table[row][col]);
        }
        
        min_count
    }
    
    /// Update counts using SIMD operations (placeholder for now)
    /// TODO: Implement actual SIMD operations using std::simd when stable
    fn update_simd(&mut self, hashes: &[usize], count: u64) {
        // For now, fall back to scalar implementation
        // Future SIMD implementation would:
        // 1. Pack hash positions and counts into SIMD vectors
        // 2. Use gather/scatter operations for non-contiguous memory access
        // 3. Perform parallel saturating addition
        // 4. Handle conservative update logic with SIMD comparisons
        self.update_scalar(hashes, count);
    }
    
    /// Estimate frequency using SIMD operations (placeholder for now)
    fn estimate_simd(&self, hashes: &[usize]) -> u64 {
        // For now, fall back to scalar implementation
        // Future SIMD implementation would:
        // 1. Use gather operations to load counts from different rows
        // 2. Use SIMD minimum operations to find the minimum count
        // 3. Handle the reduction efficiently
        self.estimate_scalar(hashes)
    }
    
    /// Merge another Count-Min sketch into this one
    pub fn merge(&mut self, other: &CountMinSketch) -> Result<(), &'static str> {
        if self.width != other.width || self.depth != other.depth {
            return Err("Sketches must have the same dimensions");
        }
        
        // Element-wise addition of the tables
        for i in 0..self.depth {
            for j in 0..self.width {
                self.table[i][j] = self.table[i][j].saturating_add(other.table[i][j]);
            }
        }
        
        Ok(())
    }
    
    /// Get the total count of all items
    pub fn total_count(&self) -> u64 {
        // Return the minimum sum across all rows (most conservative estimate)
        let mut min_sum = u64::MAX;
        
        for row in &self.table {
            let sum: u64 = row.iter().sum();
            min_sum = min_sum.min(sum);
        }
        
        min_sum
    }
    
    /// Find heavy hitters (items with frequency above threshold)
    pub fn heavy_hitters(&self, threshold: u64) -> Vec<u64> {
        // This is a simplified version - in practice, you'd need to track
        // the actual items, not just their counts
        let mut heavy_counts = Vec::new();
        
        // Find unique counts that exceed threshold
        for row in &self.table {
            for &count in row {
                if count >= threshold && !heavy_counts.contains(&count) {
                    heavy_counts.push(count);
                }
            }
        }
        
        heavy_counts.sort_unstable();
        heavy_counts.reverse();
        heavy_counts
    }
    
    /// Get sketch statistics
    pub fn statistics(&self) -> CountMinStats {
        let mut total_cells = 0u64;
        let mut non_zero_cells = 0usize;
        let mut max_count = 0u64;
        let mut min_count = u64::MAX;
        
        for row in &self.table {
            for &count in row {
                total_cells += count;
                if count > 0 {
                    non_zero_cells += 1;
                }
                max_count = max_count.max(count);
                min_count = min_count.min(count);
            }
        }
        
        let fill_ratio = non_zero_cells as f64 / (self.width * self.depth) as f64;
        
        CountMinStats {
            width: self.width,
            depth: self.depth,
            total_cells: self.width * self.depth,
            non_zero_cells,
            fill_ratio,
            total_count: total_cells,
            max_count,
            min_count,
            uses_simd: self.use_simd,
            conservative_update: self.conservative_update,
        }
    }
    
    /// Clear the sketch
    pub fn clear(&mut self) {
        for row in &mut self.table {
            for count in row {
                *count = 0;
            }
        }
    }
}

/// Statistics about a Count-Min sketch
#[derive(Debug, Clone)]
pub struct CountMinStats {
    pub width: usize,
    pub depth: usize,
    pub total_cells: usize,
    pub non_zero_cells: usize,
    pub fill_ratio: f64,
    pub total_count: u64,
    pub max_count: u64,
    pub min_count: u64,
    pub uses_simd: bool,
    pub conservative_update: bool,
}

/// Count Sketch - similar to Count-Min but uses signed counters for better accuracy
pub struct CountSketch {
    width: usize,
    depth: usize,
    table: Vec<Vec<i64>>,
    hash_seeds: Vec<u64>,
    sign_seeds: Vec<u64>,
}

impl CountSketch {
    /// Create a new Count sketch
    pub fn new(width: usize, depth: usize) -> Self {
        assert!(width > 0 && depth > 0, "Width and depth must be positive");
        
        let mut hash_seeds = Vec::with_capacity(depth);
        let mut sign_seeds = Vec::with_capacity(depth);
        let mut seed = 1u64;
        
        for _ in 0..depth {
            hash_seeds.push(seed);
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            sign_seeds.push(seed);
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        }
        
        CountSketch {
            width,
            depth,
            table: vec![vec![0i64; width]; depth],
            hash_seeds,
            sign_seeds,
        }
    }
    
    /// Update the count for an item
    pub fn update<T: Hash>(&mut self, item: &T, count: i64) {
        for i in 0..self.depth {
            let hash_pos = self.hash_position(item, i);
            let sign = self.hash_sign(item, i);
            self.table[i][hash_pos] += count * sign;
        }
    }
    
    /// Estimate the frequency of an item
    pub fn estimate<T: Hash>(&self, item: &T) -> i64 {
        let mut estimates = Vec::with_capacity(self.depth);
        
        for i in 0..self.depth {
            let hash_pos = self.hash_position(item, i);
            let sign = self.hash_sign(item, i);
            estimates.push(self.table[i][hash_pos] * sign);
        }
        
        // Return the median estimate
        estimates.sort_unstable();
        estimates[estimates.len() / 2]
    }
    
    /// Generate hash position for an item in a specific row
    fn hash_position<T: Hash>(&self, item: &T, row: usize) -> usize {
        let mut hasher = DefaultHasher::new();
        self.hash_seeds[row].hash(&mut hasher);
        item.hash(&mut hasher);
        (hasher.finish() as usize) % self.width
    }
    
    /// Generate hash sign (+1 or -1) for an item in a specific row
    fn hash_sign<T: Hash>(&self, item: &T, row: usize) -> i64 {
        let mut hasher = DefaultHasher::new();
        self.sign_seeds[row].hash(&mut hasher);
        item.hash(&mut hasher);
        if hasher.finish() % 2 == 0 { 1 } else { -1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_countmin_basic() {
        let mut cm = CountMinSketch::new(1000, 5, false, false);
        
        // Add some items
        cm.increment(&"apple");
        cm.increment(&"banana");
        cm.increment(&"apple");
        cm.update(&"cherry", 3);
        
        // Check estimates
        assert_eq!(cm.estimate(&"apple"), 2);
        assert_eq!(cm.estimate(&"banana"), 1);
        assert_eq!(cm.estimate(&"cherry"), 3);
        assert_eq!(cm.estimate(&"not_added"), 0);
    }
    
    #[test]
    fn test_countmin_conservative() {
        let mut cm = CountMinSketch::new(100, 5, false, true);
        
        // Add items multiple times
        for _ in 0..10 {
            cm.increment(&"item1");
        }
        
        // Conservative update should give exact count for single item
        assert_eq!(cm.estimate(&"item1"), 10);
    }
    
    #[test]
    fn test_countmin_error_bounds() {
        let cm = CountMinSketch::with_error_bounds(0.01, 0.01, false, false);
        
        // Check that dimensions are reasonable
        assert!(cm.width > 250); // Should be around e/0.01 ≈ 271
        assert!(cm.depth >= 5);   // Should be around ln(1/0.01) ≈ 4.6
    }
    
    #[test]
    fn test_countmin_merge() {
        let mut cm1 = CountMinSketch::new(100, 5, false, false);
        let mut cm2 = CountMinSketch::new(100, 5, false, false);
        
        cm1.increment(&"item1");
        cm1.increment(&"item2");
        
        cm2.increment(&"item1");
        cm2.increment(&"item3");
        
        cm1.merge(&cm2).unwrap();
        
        assert_eq!(cm1.estimate(&"item1"), 2);
        assert_eq!(cm1.estimate(&"item2"), 1);
        assert_eq!(cm1.estimate(&"item3"), 1);
    }
    
    #[test]
    fn test_countmin_simd() {
        let mut cm_simd = CountMinSketch::new(100, 8, true, false);
        let mut cm_standard = CountMinSketch::new(100, 8, false, false);
        
        let items = ["item1", "item2", "item3", "item4"];
        
        // Add same items to both sketches
        for item in &items {
            cm_simd.increment(item);
            cm_standard.increment(item);
        }
        
        // Both should give same estimates
        for item in &items {
            assert_eq!(cm_simd.estimate(item), cm_standard.estimate(item));
        }
    }
    
    #[test]
    fn test_count_sketch_basic() {
        let mut cs = CountSketch::new(1000, 5);
        
        // Add some items
        cs.update(&"apple", 5);
        cs.update(&"banana", -2);
        cs.update(&"apple", 3);
        
        // Check estimates (should be exact for few items)
        assert_eq!(cs.estimate(&"apple"), 8);
        assert_eq!(cs.estimate(&"banana"), -2);
    }
    
    #[test]
    fn test_statistics() {
        let mut cm = CountMinSketch::new(100, 5, false, false);
        
        for i in 0..50 {
            cm.update(&format!("item_{}", i), (i % 10) as u64 + 1);
        }
        
        let stats = cm.statistics();
        assert_eq!(stats.width, 100);
        assert_eq!(stats.depth, 5);
        assert!(stats.fill_ratio > 0.0);
        assert!(stats.total_count > 0);
    }
}