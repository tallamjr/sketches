//! Count-Min Sketch and Count Sketch for frequency estimation.
//!
//! The Count-Min Sketch uses a 2D array of counters (depth x width) with
//! independent hash functions per row. Frequency is estimated by taking the
//! minimum across all rows, providing a conservative (always-overestimate) bound.
//!
//! # Error Bounds (Count-Min)
//! - Estimate error <= epsilon * N with probability >= 1 - delta
//! - Width: `w = ceil(e / epsilon)` (~2.718 / epsilon)
//! - Depth: `d = ceil(ln(1 / delta))`
//! - Example: epsilon=0.001, delta=0.01 -> w=2719, d=5
//!
//! # Count Sketch
//! Uses signed (+1/-1) counters and takes the median across rows.
//! Provides unbiased estimates (can under- or overestimate) but requires
//! more space than Count-Min for the same accuracy.
//!
//! # Common Uses
//! Network traffic monitoring, NLP word frequency, database query optimisation.
//!
//! # References
//! - Cormode, Muthukrishnan. "An Improved Data Stream Summary: The Count-Min
//!   Sketch and its Applications." Journal of Algorithms, 2005.
//! - Charikar, Chen, Farach-Colton. "Finding Frequent Items in Data Streams."
//!   Theoretical Computer Science, 2004.

use crate::hash::xxh3::Xxh3Hasher;
use crate::hash::{DEFAULT_SEED, Hashable, SketchHasher, hash64_of};
use core::marker::PhantomData;

/// Derive a per-row seed that is well-spread across rows.
/// Multiplying by a large odd constant (Fibonacci hashing) gives good mixing.
#[inline]
fn row_seed(base_seed: u64, row: usize) -> u64 {
    base_seed.wrapping_add((row as u64).wrapping_mul(0x9E3779B97F4A7C15))
}

/// Maximum number of rows handled on the stack per operation. Real
/// configurations use depth of roughly 5; this cap is far above any practical depth.
const MAX_HASHES: usize = 64;

/// Count-Min Sketch for frequency estimation
pub struct CountMinSketchGeneric<H: SketchHasher> {
    width: usize,
    depth: usize,
    table: Vec<Vec<u64>>,
    conservative_update: bool,
    _hasher: PhantomData<H>,
}

/// Count-Min Sketch using the default xxh3 backend.
pub type CountMinSketch = CountMinSketchGeneric<Xxh3Hasher>;

impl<H: SketchHasher> CountMinSketchGeneric<H> {
    /// Create a new Count-Min sketch
    ///
    /// # Arguments
    /// * `width` - Number of buckets per row (affects accuracy)
    /// * `depth` - Number of hash functions/rows (affects probability of error)
    /// * `conservative_update` - Whether to use conservative update (reduces overestimation)
    pub fn new(width: usize, depth: usize, conservative_update: bool) -> Self {
        assert!(width > 0 && depth > 0, "Width and depth must be positive");

        CountMinSketchGeneric {
            width,
            depth,
            table: vec![vec![0u64; width]; depth],
            conservative_update,
            _hasher: PhantomData,
        }
    }

    /// Create a sketch with parameters calculated from error bounds
    ///
    /// # Arguments
    /// * `epsilon` - Maximum relative error (e.g., 0.01 for 1% error)
    /// * `delta` - Probability of exceeding the error bound (e.g., 0.01 for 1% probability)
    /// * `conservative_update` - Whether to use conservative update
    pub fn with_error_bounds(epsilon: f64, delta: f64, conservative_update: bool) -> Self {
        assert!(
            epsilon > 0.0 && epsilon < 1.0,
            "Epsilon must be between 0 and 1"
        );
        assert!(delta > 0.0 && delta < 1.0, "Delta must be between 0 and 1");

        let width = (std::f64::consts::E / epsilon).ceil() as usize;
        let depth = (1.0 / delta).ln().ceil() as usize;

        Self::new(width, depth, conservative_update)
    }

    /// Update the count for an item
    pub fn update<T: Hashable + ?Sized>(&mut self, item: &T, count: u64) {
        debug_assert!(self.depth <= MAX_HASHES);
        let mut buf = [0usize; MAX_HASHES];
        let cols = &mut buf[..self.depth];
        self.hash_positions_into(item, cols);
        self.update_scalar(cols, count);
    }

    /// Increment the count for an item by 1
    pub fn increment<T: Hashable + ?Sized>(&mut self, item: &T) {
        self.update(item, 1);
    }

    /// Estimate the frequency of an item
    pub fn estimate<T: Hashable + ?Sized>(&self, item: &T) -> u64 {
        debug_assert!(self.depth <= MAX_HASHES);
        let mut buf = [0usize; MAX_HASHES];
        let cols = &mut buf[..self.depth];
        self.hash_positions_into(item, cols);
        self.estimate_scalar(cols)
    }

    /// Fill `out[..depth]` with this item's per-row column positions using one
    /// independent xxh3 hash per row. `out` must be at least `depth` long.
    ///
    /// Each row receives a distinct seed derived from DEFAULT_SEED, ensuring
    /// pairwise-independent hashes as required by the Count-Min error guarantee.
    fn hash_positions_into<T: Hashable + ?Sized>(&self, item: &T, out: &mut [usize]) {
        for (row, slot) in out.iter_mut().enumerate().take(self.depth) {
            let h = hash64_of(&H::default(), item, row_seed(DEFAULT_SEED, row));
            *slot = (h % self.width as u64) as usize;
        }
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

    /// Merge another Count-Min sketch into this one
    pub fn merge(&mut self, other: &CountMinSketchGeneric<H>) -> Result<(), &'static str> {
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
    pub conservative_update: bool,
}

/// Count Sketch - similar to Count-Min but uses signed counters for better accuracy
pub struct CountSketch {
    width: usize,
    depth: usize,
    table: Vec<Vec<i64>>,
}

impl CountSketch {
    /// Create a new Count sketch
    pub fn new(width: usize, depth: usize) -> Self {
        assert!(width > 0 && depth > 0, "Width and depth must be positive");

        CountSketch {
            width,
            depth,
            table: vec![vec![0i64; width]; depth],
        }
    }

    /// Update the count for an item
    pub fn update<T: Hashable + ?Sized>(&mut self, item: &T, count: i64) {
        for i in 0..self.depth {
            let hash_pos = self.hash_position(item, i);
            let sign = self.hash_sign(item, i);
            self.table[i][hash_pos] += count * sign;
        }
    }

    /// Estimate the frequency of an item
    pub fn estimate<T: Hashable + ?Sized>(&self, item: &T) -> i64 {
        debug_assert!(self.depth <= MAX_HASHES);
        let mut buf = [0i64; MAX_HASHES];
        let estimates = &mut buf[..self.depth];
        for (i, slot) in estimates.iter_mut().enumerate() {
            let hash_pos = self.hash_position(item, i);
            let sign = self.hash_sign(item, i);
            *slot = self.table[i][hash_pos] * sign;
        }

        // Return the median estimate
        estimates.sort_unstable();
        estimates[estimates.len() / 2]
    }

    /// Generate hash position for an item in a specific row.
    ///
    /// Each row uses an independent seed so positions across rows are
    /// pairwise independent, as required by Count Sketch's analysis.
    fn hash_position<T: Hashable + ?Sized>(&self, item: &T, row: usize) -> usize {
        let h = hash64_of(&Xxh3Hasher, item, row_seed(DEFAULT_SEED, row));
        (h % self.width as u64) as usize
    }

    /// Generate hash sign (+1 or -1) for an item in a specific row.
    ///
    /// Uses a separate base seed (0xC2B2AE3D27D4EB4F) so the sign hash is
    /// independent of the position hash. The low bit selects +1 or -1.
    fn hash_sign<T: Hashable + ?Sized>(&self, item: &T, row: usize) -> i64 {
        let h = hash64_of(&Xxh3Hasher, item, row_seed(0xC2B2AE3D27D4EB4F, row));
        if h.is_multiple_of(2) { 1 } else { -1 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_countmin_basic() {
        let mut cm = CountMinSketch::new(1000, 5, false);

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
        let mut cm = CountMinSketch::new(100, 5, true);

        // Add items multiple times
        for _ in 0..10 {
            cm.increment(&"item1");
        }

        // Conservative update should give exact count for single item
        assert_eq!(cm.estimate(&"item1"), 10);
    }

    #[test]
    fn test_countmin_error_bounds() {
        let cm = CountMinSketch::with_error_bounds(0.01, 0.01, false);

        // Check that dimensions are reasonable
        assert!(cm.width > 250); // Should be around e/0.01 ≈ 271
        assert!(cm.depth >= 5); // Should be around ln(1/0.01) ≈ 4.6
    }

    #[test]
    fn test_countmin_merge() {
        let mut cm1 = CountMinSketch::new(100, 5, false);
        let mut cm2 = CountMinSketch::new(100, 5, false);

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
        let mut cm = CountMinSketch::new(100, 5, false);

        for i in 0..50 {
            cm.update(&format!("item_{i}"), (i % 10) as u64 + 1);
        }

        let stats = cm.statistics();
        assert_eq!(stats.width, 100);
        assert_eq!(stats.depth, 5);
        assert!(stats.fill_ratio > 0.0);
        assert!(stats.total_count > 0);
    }

    #[test]
    fn countmin_overestimates_only_new_hash() {
        let mut c = CountMinSketch::new(2048, 5, false);
        for _ in 0..500 {
            c.increment(&"hot");
        }
        assert!(c.estimate(&"hot") >= 500);
    }

    #[test]
    fn murmur3_countmin_estimates() {
        use crate::hash::murmur3::Murmur3Hasher;
        let mut c = CountMinSketchGeneric::<Murmur3Hasher>::new(2048, 5, false);
        for _ in 0..100 {
            c.increment(&"hot");
        }
        assert!(c.estimate(&"hot") >= 100);
    }

    #[test]
    fn countmin_rows_are_independent() {
        // With a wide sketch (4096 columns) and independent per-row hashes,
        // a single item should map to different columns across almost all rows.
        let c = CountMinSketch::new(4096, 5, false);
        let mut cols = [0usize; 5];
        c.hash_positions_into(&"some_item", &mut cols);
        let distinct: std::collections::HashSet<usize> = cols.iter().copied().collect();
        assert!(
            distinct.len() >= 4,
            "rows collapse to same column: {cols:?}"
        );
    }
}
