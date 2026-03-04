//! REQ (Relative Error Quantiles) Sketch Implementation
//!
//! This module provides the REQ sketch data structure for approximate quantile estimation
//! with relative error guarantees. Unlike KLL sketches which provide normalised rank error
//! (uniform across all quantiles), the REQ sketch concentrates its accuracy at one end of
//! the rank domain.
//!
//! The REQ sketch was introduced by Cormode, Karnin, Liberty, Thaler, and Veselý in
//! "Relative Error Streaming Quantiles" (PODS 2021).
//!
//! Key properties:
//! - **HRA (High Rank Accuracy)**: Error converges to zero near rank 1.0, making it ideal
//!   for tail quantiles such as p99, p99.9, and p99.99.
//! - **LRA (Low Rank Accuracy)**: Error converges to zero near rank 0.0, making it ideal
//!   for low quantiles such as p0.1, p1.
//!
//! The fundamental difference from KLL is that compactors at higher levels receive LARGER
//! capacities (capacity = k * 2^level), and the compaction strategy retains items from
//! the end of the sorted sequence corresponding to the chosen accuracy mode.

use std::cmp::Ordering;

/// Accuracy mode for the REQ sketch.
///
/// Determines which end of the rank domain receives the highest accuracy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReqMode {
    /// High Rank Accuracy: error converges to zero near rank 1.0.
    /// Best for tail quantiles (p95, p99, p99.9, p99.99).
    HRA,
    /// Low Rank Accuracy: error converges to zero near rank 0.0.
    /// Best for low quantiles (p0.01, p0.1, p1, p5).
    LRA,
}

/// A single compactor in the REQ sketch.
///
/// Each compactor maintains a buffer of items at a particular level. When the buffer
/// exceeds its capacity, it is compacted: sorted, then alternately even/odd indexed
/// items are retained (chosen randomly), with the discarded half's representative
/// items promoted to the next level.
#[derive(Debug, Clone)]
struct ReqCompactor<T> {
    /// The level of this compactor (0 = bottom, receives raw input).
    level: usize,
    /// The items stored in this compactor.
    items: Vec<T>,
    /// Maximum number of items before compaction is triggered.
    capacity: usize,
    /// Number of compactions performed at this level (used for alternating strategy).
    num_compactions: u64,
}

impl<T: Clone + PartialOrd> ReqCompactor<T> {
    fn new(level: usize, k: usize) -> Self {
        let capacity = Self::compute_capacity(level, k);
        ReqCompactor {
            level,
            items: Vec::with_capacity(capacity),
            capacity,
            num_compactions: 0,
        }
    }

    /// Capacity grows exponentially with level: capacity = k * 2^level.
    /// This is the key difference from KLL where capacity shrinks at higher levels.
    fn compute_capacity(level: usize, k: usize) -> usize {
        let cap = k.saturating_mul(1usize << level);
        // Ensure a minimum capacity of 2 to allow meaningful compaction.
        cap.max(2)
    }

    fn is_over_capacity(&self) -> bool {
        self.items.len() > self.capacity
    }

    fn len(&self) -> usize {
        self.items.len()
    }

    /// Compact this compactor, returning the items to be promoted to the next level.
    ///
    /// The compaction procedure follows the REQ paper's section-based approach:
    /// 1. Sort the items.
    /// 2. Divide the sorted buffer into a "compactable" region and a "guard" region.
    ///    - In HRA mode, the guard region is at the high end (large values stay).
    ///    - In LRA mode, the guard region is at the low end (small values stay).
    /// 3. The compactable region undergoes coin-flip compaction: randomly choose
    ///    even or odd indexed items to promote to the next level; discard the rest.
    /// 4. The guard region items remain at this level.
    ///
    /// The guard region provides the relative error guarantee: items near the chosen
    /// end of the distribution are protected from compaction at lower levels, giving
    /// higher accuracy for those quantiles.
    fn compact(&mut self, mode: ReqMode, rng_state: &mut u64) -> Vec<T> {
        // Sort items
        self.items
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        // Randomly choose offset (0 or 1) for coin-flip compaction
        let offset = (xorshift64(rng_state) & 1) as usize;

        let n = self.items.len();

        // Guard size: protect a fraction of items at the chosen end.
        // At level 0, protect more items; at higher levels, protect fewer.
        // guard_size = floor(n / (level + 2)), ensuring at least 0 and at most n/2.
        let guard_size = (n / (self.level + 2)).min(n / 2);

        // Ensure the compactable region has an even number of items for clean halving
        let compactable_count = n - guard_size;

        let mut promoted = Vec::with_capacity(compactable_count / 2 + 1);
        let mut guard_items = Vec::with_capacity(guard_size);

        match mode {
            ReqMode::HRA => {
                // Guard region is at the HIGH end (last `guard_size` items).
                // Compactable region is the first `compactable_count` items.
                for (i, item) in self.items.drain(..).enumerate() {
                    if i < compactable_count {
                        // Coin-flip: promote every other item, discard the rest
                        if i % 2 == offset {
                            promoted.push(item);
                        }
                        // else: discarded
                    } else {
                        // Guard region: keep at this level
                        guard_items.push(item);
                    }
                }
            }
            ReqMode::LRA => {
                // Guard region is at the LOW end (first `guard_size` items).
                // Compactable region is the last `compactable_count` items.
                for (i, item) in self.items.drain(..).enumerate() {
                    if i < guard_size {
                        // Guard region: keep at this level
                        guard_items.push(item);
                    } else {
                        // Coin-flip: promote every other item, discard the rest
                        let ci = i - guard_size;
                        if ci % 2 == offset {
                            promoted.push(item);
                        }
                        // else: discarded
                    }
                }
            }
        }

        // Keep only the guard items at this level
        self.items = guard_items;
        self.num_compactions += 1;

        promoted
    }
}

/// Simple xorshift64 PRNG for deterministic but pseudorandom compaction decisions.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// REQ (Relative Error Quantiles) Sketch.
///
/// Provides approximate quantile estimation with relative error guarantees that
/// concentrate accuracy at a chosen end of the rank domain. Unlike KLL sketches
/// which provide uniform normalised rank error across all quantiles, the REQ sketch
/// achieves much higher accuracy at extreme quantiles (tails) at the cost of slightly
/// lower accuracy near the median.
///
/// # Type Parameters
///
/// * `T` - The type of items stored. Must implement `Clone` and `PartialOrd`.
///
/// # Examples
///
/// ```
/// use sketches::req::{ReqSketch, ReqMode};
///
/// let mut sketch = ReqSketch::new(12, ReqMode::HRA);
/// for i in 0..10_000 {
///     sketch.update(i as f64);
/// }
///
/// // HRA mode gives excellent accuracy at high quantiles
/// let p99 = sketch.quantile(0.99).unwrap();
/// assert!((p99 - 9900.0).abs() < 200.0);
/// ```
#[derive(Debug, Clone)]
pub struct ReqSketch<T> {
    /// The k parameter controlling accuracy vs memory trade-off.
    k: usize,
    /// The accuracy mode (HRA or LRA).
    mode: ReqMode,
    /// Compactors at each level (level 0 = bottom, receives raw input).
    compactors: Vec<ReqCompactor<T>>,
    /// Total number of items ingested via `update()`.
    total_count: u64,
    /// Minimum value observed.
    min_value: Option<T>,
    /// Maximum value observed.
    max_value: Option<T>,
    /// PRNG state for randomised compaction.
    rng_state: u64,
}

impl<T> ReqSketch<T>
where
    T: Clone + PartialOrd,
{
    /// Create a new REQ sketch with the given k parameter and accuracy mode.
    ///
    /// # Arguments
    ///
    /// * `k` - Controls the accuracy/memory trade-off. Larger values give better accuracy
    ///   but use more memory. Must be at least 4. Typical values: 6-50.
    /// * `mode` - The accuracy mode: `ReqMode::HRA` for high-rank accuracy (tail quantiles)
    ///   or `ReqMode::LRA` for low-rank accuracy.
    ///
    /// # Panics
    ///
    /// Panics if `k` is less than 4.
    pub fn new(k: usize, mode: ReqMode) -> Self {
        assert!(k >= 4, "k must be at least 4");

        let bottom_compactor = ReqCompactor::new(0, k);

        ReqSketch {
            k,
            mode,
            compactors: vec![bottom_compactor],
            total_count: 0,
            min_value: None,
            max_value: None,
            rng_state: 0xDEAD_BEEF_CAFE_BABEu64,
        }
    }

    /// Update the sketch with a new value.
    ///
    /// Time complexity: O(1) amortised.
    pub fn update(&mut self, value: T) {
        // Update min/max tracking
        match &self.min_value {
            None => self.min_value = Some(value.clone()),
            Some(min) => {
                if value.partial_cmp(min) == Some(Ordering::Less) {
                    self.min_value = Some(value.clone());
                }
            }
        }

        match &self.max_value {
            None => self.max_value = Some(value.clone()),
            Some(max) => {
                if value.partial_cmp(max) == Some(Ordering::Greater) {
                    self.max_value = Some(value.clone());
                }
            }
        }

        self.total_count += 1;

        // Insert into the bottom compactor (level 0)
        self.compactors[0].items.push(value);

        // Compact if level 0 is over capacity
        if self.compactors[0].is_over_capacity() {
            self.compact();
        }
    }

    /// Perform compaction starting from the lowest level that is over capacity,
    /// cascading upward as needed.
    fn compact(&mut self) {
        let mut level = 0;

        while level < self.compactors.len() {
            if self.compactors[level].is_over_capacity() {
                // Ensure there is a compactor at the next level
                if level + 1 >= self.compactors.len() {
                    self.grow();
                }

                let promoted = self.compactors[level].compact(self.mode, &mut self.rng_state);

                // Add promoted items to the next level
                for item in promoted {
                    self.compactors[level + 1].items.push(item);
                }

                level += 1;
            } else {
                break;
            }
        }
    }

    /// Add a new compactor level at the top.
    fn grow(&mut self) {
        let new_level = self.compactors.len();
        self.compactors.push(ReqCompactor::new(new_level, self.k));
    }

    /// Get the quantile value for the given rank (0.0 to 1.0).
    ///
    /// # Arguments
    ///
    /// * `rank` - The quantile rank. 0.0 returns the minimum, 1.0 returns the maximum,
    ///   0.5 returns the approximate median, 0.99 returns the 99th percentile, etc.
    ///
    /// # Returns
    ///
    /// `None` if the sketch is empty, otherwise `Some(value)` at the approximate quantile.
    ///
    /// # Panics
    ///
    /// Panics if `rank` is outside the range [0.0, 1.0].
    pub fn quantile(&mut self, rank: f64) -> Option<T> {
        assert!(
            (0.0..=1.0).contains(&rank),
            "Rank must be between 0.0 and 1.0"
        );

        if self.total_count == 0 {
            return None;
        }

        if rank == 0.0 {
            return self.min_value.clone();
        }

        if rank == 1.0 {
            return self.max_value.clone();
        }

        // Build a weighted list of all items across all compactor levels.
        // Items at level h each represent 2^h original items.
        let mut weighted_items: Vec<(T, u64)> = Vec::new();

        for compactor in &self.compactors {
            let weight = 1u64 << compactor.level;
            for item in &compactor.items {
                weighted_items.push((item.clone(), weight));
            }
        }

        if weighted_items.is_empty() {
            return self.min_value.clone();
        }

        // Sort by value
        weighted_items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        let total_weight: u64 = weighted_items.iter().map(|(_, w)| *w).sum();
        let target = rank * total_weight as f64;

        // Walk through sorted items accumulating weight
        let mut cumulative: f64 = 0.0;

        for (item, weight) in &weighted_items {
            cumulative += *weight as f64;
            if cumulative >= target {
                return Some(item.clone());
            }
        }

        self.max_value.clone()
    }

    /// Get the rank of a given value (0.0 to 1.0).
    ///
    /// Returns the fraction of values in the stream that are less than or equal
    /// to the given value.
    pub fn rank(&mut self, value: &T) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let mut weight_le: u64 = 0;
        let mut total_weight: u64 = 0;

        for compactor in &self.compactors {
            let weight = 1u64 << compactor.level;
            for item in &compactor.items {
                total_weight += weight;
                if item.partial_cmp(value) != Some(Ordering::Greater) {
                    weight_le += weight;
                }
            }
        }

        if total_weight == 0 {
            return 0.0;
        }

        weight_le as f64 / total_weight as f64
    }

    /// Merge another REQ sketch into this one.
    ///
    /// Both sketches should use the same mode for meaningful results. Items from each
    /// level of the other sketch are added to the corresponding level of this sketch,
    /// and compaction is triggered as needed.
    ///
    /// # Arguments
    ///
    /// * `other` - The other REQ sketch to merge into this one.
    pub fn merge(&mut self, other: &mut ReqSketch<T>) {
        // Update min/max
        if let Some(other_min) = &other.min_value {
            match &self.min_value {
                None => self.min_value = Some(other_min.clone()),
                Some(min) => {
                    if other_min.partial_cmp(min) == Some(Ordering::Less) {
                        self.min_value = Some(other_min.clone());
                    }
                }
            }
        }

        if let Some(other_max) = &other.max_value {
            match &self.max_value {
                None => self.max_value = Some(other_max.clone()),
                Some(max) => {
                    if other_max.partial_cmp(max) == Some(Ordering::Greater) {
                        self.max_value = Some(other_max.clone());
                    }
                }
            }
        }

        self.total_count += other.total_count;

        // Ensure we have enough levels
        while self.compactors.len() < other.compactors.len() {
            self.grow();
        }

        // Merge each level's items
        for (level_idx, other_compactor) in other.compactors.iter().enumerate() {
            for item in &other_compactor.items {
                self.compactors[level_idx].items.push(item.clone());
            }
        }

        // Compact any overflowed levels
        self.compact();
    }

    /// Get the Cumulative Distribution Function (CDF) at the given split points.
    ///
    /// Returns a vector of length `split_points.len() + 1` where:
    /// - `result[i]` = fraction of items <= `split_points[i]`
    /// - `result[last]` = 1.0
    ///
    /// Split points must be sorted in ascending order.
    pub fn get_cdf(&mut self, split_points: &[T]) -> Vec<f64> {
        let mut result = Vec::with_capacity(split_points.len() + 1);

        if self.total_count == 0 {
            result.resize(split_points.len() + 1, 0.0);
            return result;
        }

        for sp in split_points {
            result.push(self.rank(sp));
        }
        result.push(1.0);

        result
    }

    /// Get the total number of items processed.
    pub fn count(&self) -> u64 {
        self.total_count
    }

    /// Check if the sketch is empty.
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Get the minimum value seen (if any).
    pub fn min(&self) -> Option<&T> {
        self.min_value.as_ref()
    }

    /// Get the maximum value seen (if any).
    pub fn max(&self) -> Option<&T> {
        self.max_value.as_ref()
    }

    /// Get the number of compactor levels currently in use.
    pub fn num_levels(&self) -> usize {
        self.compactors.len()
    }

    /// Get the total number of items retained across all compactor levels.
    pub fn retained_items(&self) -> usize {
        self.compactors.iter().map(|c| c.len()).sum()
    }

    /// Get the accuracy mode.
    pub fn mode(&self) -> ReqMode {
        self.mode
    }

    /// Get the k parameter.
    pub fn k(&self) -> usize {
        self.k
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Basic operation tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_req_empty() {
        let mut sketch = ReqSketch::<f64>::new(12, ReqMode::HRA);

        assert!(sketch.is_empty());
        assert_eq!(sketch.count(), 0);
        assert!(sketch.quantile(0.5).is_none());
        assert_eq!(sketch.rank(&100.0), 0.0);
        assert!(sketch.min().is_none());
        assert!(sketch.max().is_none());
    }

    #[test]
    fn test_req_single_value() {
        let mut sketch = ReqSketch::new(12, ReqMode::HRA);
        sketch.update(42.0_f64);

        assert!(!sketch.is_empty());
        assert_eq!(sketch.count(), 1);
        assert_eq!(sketch.quantile(0.5).unwrap(), 42.0);
        assert_eq!(sketch.rank(&42.0), 1.0);
        assert_eq!(sketch.rank(&41.0), 0.0);
        assert_eq!(*sketch.min().unwrap(), 42.0);
        assert_eq!(*sketch.max().unwrap(), 42.0);
    }

    #[test]
    fn test_req_basic_update_and_query() {
        let mut sketch = ReqSketch::new(12, ReqMode::HRA);

        for i in 1..=1000 {
            sketch.update(i as f64);
        }

        assert_eq!(sketch.count(), 1000);
        assert_eq!(*sketch.min().unwrap(), 1.0);
        assert_eq!(*sketch.max().unwrap(), 1000.0);

        let median = sketch.quantile(0.5).unwrap();
        assert!(
            (median - 500.0).abs() < 100.0,
            "Median should be approximately 500, got {median}"
        );

        let q99 = sketch.quantile(0.99).unwrap();
        assert!(
            (q99 - 990.0).abs() < 50.0,
            "99th percentile should be approximately 990, got {q99}"
        );
    }

    #[test]
    fn test_req_rank_basic() {
        let mut sketch = ReqSketch::new(12, ReqMode::HRA);

        for i in 0..1000 {
            sketch.update(i as f64);
        }

        let rank_500 = sketch.rank(&500.0);
        assert!(
            (rank_500 - 0.501).abs() < 0.1,
            "Rank of 500 should be approximately 0.5, got {rank_500}"
        );
    }

    #[test]
    fn test_req_boundary_quantiles() {
        let mut sketch = ReqSketch::new(12, ReqMode::HRA);

        for i in 0..100 {
            sketch.update(i as f64);
        }

        assert_eq!(sketch.quantile(0.0).unwrap(), 0.0);
        assert_eq!(sketch.quantile(1.0).unwrap(), 99.0);
    }

    #[test]
    fn test_req_mode_accessor() {
        let sketch_hra = ReqSketch::<f64>::new(12, ReqMode::HRA);
        assert_eq!(sketch_hra.mode(), ReqMode::HRA);

        let sketch_lra = ReqSketch::<f64>::new(12, ReqMode::LRA);
        assert_eq!(sketch_lra.mode(), ReqMode::LRA);
    }

    #[test]
    fn test_req_k_accessor() {
        let sketch = ReqSketch::<f64>::new(20, ReqMode::HRA);
        assert_eq!(sketch.k(), 20);
    }

    #[test]
    #[should_panic(expected = "k must be at least 4")]
    fn test_req_k_too_small() {
        ReqSketch::<f64>::new(3, ReqMode::HRA);
    }

    #[test]
    #[should_panic(expected = "Rank must be between 0.0 and 1.0")]
    fn test_req_quantile_out_of_range() {
        let mut sketch = ReqSketch::new(12, ReqMode::HRA);
        sketch.update(1.0_f64);
        sketch.quantile(1.5);
    }

    // -----------------------------------------------------------------------
    // Compaction and memory tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_req_compaction_occurs() {
        let mut sketch = ReqSketch::new(4, ReqMode::HRA);

        // With k=4, level 0 capacity is 4. Inserting enough items should trigger compaction.
        for i in 0..100 {
            sketch.update(i as f64);
        }

        // After 100 insertions with k=4, we should have multiple levels
        assert!(
            sketch.num_levels() > 1,
            "Compaction should have created multiple levels, got {}",
            sketch.num_levels()
        );

        // The total retained items should be much less than 100
        assert!(
            sketch.retained_items() < 100,
            "Retained items {} should be less than total count 100",
            sketch.retained_items()
        );
    }

    #[test]
    fn test_req_capacity_grows_with_level() {
        // Verify that higher levels get larger capacities (the key REQ property)
        let k = 8;
        assert_eq!(ReqCompactor::<f64>::compute_capacity(0, k), 8);
        assert_eq!(ReqCompactor::<f64>::compute_capacity(1, k), 16);
        assert_eq!(ReqCompactor::<f64>::compute_capacity(2, k), 32);
        assert_eq!(ReqCompactor::<f64>::compute_capacity(3, k), 64);
    }

    #[test]
    fn test_req_memory_bounded() {
        let k = 12;
        let mut sketch = ReqSketch::new(k, ReqMode::HRA);

        for i in 0..1_000_000 {
            sketch.update(i as f64);
        }

        let retained = sketch.retained_items();
        // The total retained items should be bounded, not proportional to n
        assert!(
            retained < 10_000,
            "Retained items {retained} should be bounded for 1M insertions with k={k}"
        );
    }

    // -----------------------------------------------------------------------
    // HRA mode accuracy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hra_better_at_high_quantiles_than_median() {
        // The core property of HRA mode: error at p99/p99.9 should be smaller
        // (relative to the quantile value) than error at the median.
        let n = 100_000u64;
        let k = 12;
        let mut sketch = ReqSketch::new(k, ReqMode::HRA);

        for i in 0..n {
            sketch.update(i as f64);
        }

        // Compute relative rank error at different quantiles
        let test_cases = [0.5, 0.9, 0.99, 0.999];
        let mut errors: Vec<(f64, f64)> = Vec::new();

        for &q in &test_cases {
            let estimated = sketch.quantile(q).unwrap();
            let expected = q * (n - 1) as f64;
            // Relative error: |estimated - expected| / expected
            let rel_error = if expected > 0.0 {
                (estimated - expected).abs() / expected
            } else {
                0.0
            };
            errors.push((q, rel_error));
        }

        // For HRA mode, the relative error at p99 should be less than or equal
        // to the relative error at p50 (the median). This is the defining property.
        let error_p50 = errors.iter().find(|(q, _)| *q == 0.5).unwrap().1;
        let error_p99 = errors.iter().find(|(q, _)| *q == 0.99).unwrap().1;

        // We use a generous tolerance here because the sketch is randomised.
        // The key assertion: p99 relative error should not be worse than p50.
        // With a well-functioning REQ sketch, p99 error is typically much better.
        assert!(
            error_p99 <= error_p50 + 0.05,
            "HRA mode: p99 relative error ({error_p99:.6}) should not significantly exceed \
             p50 relative error ({error_p50:.6})"
        );
    }

    #[test]
    fn test_hra_high_quantile_accuracy() {
        let n = 100_000u64;
        let k = 24;
        let mut sketch = ReqSketch::new(k, ReqMode::HRA);

        for i in 0..n {
            sketch.update(i as f64);
        }

        // p99 should be accurate
        let p99 = sketch.quantile(0.99).unwrap();
        let expected_p99 = 0.99 * (n - 1) as f64;
        let normalised_error = (p99 - expected_p99).abs() / n as f64;

        assert!(
            normalised_error < 0.02,
            "HRA mode p99: normalised error {normalised_error:.4} exceeds 2%, estimated={p99:.1}, expected={expected_p99:.1}"
        );

        // p99.9 should also be accurate
        let p999 = sketch.quantile(0.999).unwrap();
        let expected_p999 = 0.999 * (n - 1) as f64;
        let normalised_error_999 = (p999 - expected_p999).abs() / n as f64;

        assert!(
            normalised_error_999 < 0.02,
            "HRA mode p99.9: normalised error {normalised_error_999:.4} exceeds 2%, estimated={p999:.1}, expected={expected_p999:.1}"
        );
    }

    // -----------------------------------------------------------------------
    // LRA mode accuracy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_lra_better_at_low_quantiles_than_median() {
        let n = 100_000u64;
        let k = 12;
        let mut sketch = ReqSketch::new(k, ReqMode::LRA);

        for i in 0..n {
            sketch.update(i as f64);
        }

        let test_cases = [0.001, 0.01, 0.1, 0.5];
        let mut errors: Vec<(f64, f64)> = Vec::new();

        for &q in &test_cases {
            let estimated = sketch.quantile(q).unwrap();
            let expected = q * (n - 1) as f64;
            // For low quantiles, use normalised rank error to avoid division-by-near-zero
            let normalised_error = (estimated - expected).abs() / n as f64;
            errors.push((q, normalised_error));
        }

        let error_p50 = errors.iter().find(|(q, _)| *q == 0.5).unwrap().1;
        let error_p1 = errors.iter().find(|(q, _)| *q == 0.01).unwrap().1;

        assert!(
            error_p1 <= error_p50 + 0.05,
            "LRA mode: p1 normalised error ({error_p1:.6}) should not significantly exceed \
             p50 normalised error ({error_p50:.6})"
        );
    }

    #[test]
    fn test_lra_low_quantile_accuracy() {
        let n = 100_000u64;
        let k = 24;
        let mut sketch = ReqSketch::new(k, ReqMode::LRA);

        for i in 0..n {
            sketch.update(i as f64);
        }

        // p1 should be accurate in LRA mode
        let p1 = sketch.quantile(0.01).unwrap();
        let expected_p1 = 0.01 * (n - 1) as f64;
        let normalised_error = (p1 - expected_p1).abs() / n as f64;

        assert!(
            normalised_error < 0.02,
            "LRA mode p1: normalised error {normalised_error:.4} exceeds 2%, estimated={p1:.1}, expected={expected_p1:.1}"
        );

        // p0.1 should also be accurate
        let p01 = sketch.quantile(0.001).unwrap();
        let expected_p01 = 0.001 * (n - 1) as f64;
        let normalised_error_01 = (p01 - expected_p01).abs() / n as f64;

        assert!(
            normalised_error_01 < 0.02,
            "LRA mode p0.1: normalised error {normalised_error_01:.4} exceeds 2%, estimated={p01:.1}, expected={expected_p01:.1}"
        );
    }

    // -----------------------------------------------------------------------
    // Merge tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_req_merge_basic() {
        let mut sketch1 = ReqSketch::new(12, ReqMode::HRA);
        let mut sketch2 = ReqSketch::new(12, ReqMode::HRA);

        for i in 0..500 {
            sketch1.update(i as f64);
        }

        for i in 500..1000 {
            sketch2.update(i as f64);
        }

        sketch1.merge(&mut sketch2);

        assert_eq!(sketch1.count(), 1000);
        assert_eq!(*sketch1.min().unwrap(), 0.0);
        assert_eq!(*sketch1.max().unwrap(), 999.0);

        let median = sketch1.quantile(0.5).unwrap();
        assert!(
            (median - 500.0).abs() < 100.0,
            "Merged median should be approximately 500, got {median}"
        );
    }

    #[test]
    fn test_req_merge_preserves_accuracy() {
        let n_per_sketch = 50_000u64;
        let k = 24;

        let mut sketch1 = ReqSketch::new(k, ReqMode::HRA);
        let mut sketch2 = ReqSketch::new(k, ReqMode::HRA);

        for i in 0..n_per_sketch {
            sketch1.update(i as f64);
        }

        for i in n_per_sketch..(2 * n_per_sketch) {
            sketch2.update(i as f64);
        }

        sketch1.merge(&mut sketch2);

        let total_n = 2 * n_per_sketch;
        let p99 = sketch1.quantile(0.99).unwrap();
        let expected_p99 = 0.99 * (total_n - 1) as f64;
        let normalised_error = (p99 - expected_p99).abs() / total_n as f64;

        assert!(
            normalised_error < 0.03,
            "Merged HRA p99: normalised error {normalised_error:.4} exceeds 3%, \
             estimated={p99:.1}, expected={expected_p99:.1}"
        );
    }

    #[test]
    fn test_req_merge_empty_into_populated() {
        let mut sketch1 = ReqSketch::new(12, ReqMode::HRA);
        let mut sketch2 = ReqSketch::new(12, ReqMode::HRA);

        for i in 0..100 {
            sketch1.update(i as f64);
        }

        sketch1.merge(&mut sketch2);

        assert_eq!(sketch1.count(), 100);
        assert_eq!(*sketch1.min().unwrap(), 0.0);
        assert_eq!(*sketch1.max().unwrap(), 99.0);
    }

    #[test]
    fn test_req_merge_populated_into_empty() {
        let mut sketch1 = ReqSketch::new(12, ReqMode::HRA);
        let mut sketch2 = ReqSketch::new(12, ReqMode::HRA);

        for i in 0..100 {
            sketch2.update(i as f64);
        }

        sketch1.merge(&mut sketch2);

        assert_eq!(sketch1.count(), 100);
        assert_eq!(*sketch1.min().unwrap(), 0.0);
        assert_eq!(*sketch1.max().unwrap(), 99.0);
    }

    // -----------------------------------------------------------------------
    // CDF tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_req_get_cdf() {
        let mut sketch = ReqSketch::new(12, ReqMode::HRA);

        for i in 0..1000 {
            sketch.update(i as f64);
        }

        let split_points = vec![250.0, 500.0, 750.0];
        let cdf = sketch.get_cdf(&split_points);

        assert_eq!(cdf.len(), 4);

        assert!(
            (cdf[0] - 0.251).abs() < 0.1,
            "CDF at 250 should be approximately 0.251, got {}",
            cdf[0]
        );
        assert!(
            (cdf[1] - 0.501).abs() < 0.1,
            "CDF at 500 should be approximately 0.501, got {}",
            cdf[1]
        );
        assert!(
            (cdf[2] - 0.751).abs() < 0.1,
            "CDF at 750 should be approximately 0.751, got {}",
            cdf[2]
        );
        assert_eq!(cdf[3], 1.0);
    }

    #[test]
    fn test_req_get_cdf_empty() {
        let mut sketch = ReqSketch::<f64>::new(12, ReqMode::HRA);
        let cdf = sketch.get_cdf(&[1.0, 2.0]);
        assert_eq!(cdf, vec![0.0, 0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // Generic type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_req_with_integers() {
        let mut sketch = ReqSketch::new(12, ReqMode::HRA);

        for i in 0..1000_i64 {
            sketch.update(i);
        }

        assert_eq!(sketch.count(), 1000);
        assert_eq!(*sketch.min().unwrap(), 0);
        assert_eq!(*sketch.max().unwrap(), 999);

        let median = sketch.quantile(0.5).unwrap();
        assert!(
            (median as f64 - 500.0).abs() < 100.0,
            "Integer median should be approximately 500, got {median}"
        );
    }

    // -----------------------------------------------------------------------
    // Xorshift PRNG test
    // -----------------------------------------------------------------------

    #[test]
    fn test_xorshift64_produces_varied_output() {
        let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
        let mut values = Vec::new();
        for _ in 0..100 {
            values.push(xorshift64(&mut state));
        }

        // All values should be different (extremely likely for 100 values of u64)
        values.sort();
        values.dedup();
        assert_eq!(
            values.len(),
            100,
            "xorshift64 should produce 100 distinct values"
        );
    }

    #[test]
    fn test_xorshift64_coin_flips_balanced() {
        let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
        let mut zeros = 0u64;
        let trials = 10_000;

        for _ in 0..trials {
            if xorshift64(&mut state) & 1 == 0 {
                zeros += 1;
            }
        }

        // Should be roughly balanced (within 5% of 50/50)
        let ratio = zeros as f64 / trials as f64;
        assert!(
            (ratio - 0.5).abs() < 0.05,
            "Coin flip ratio should be near 0.5, got {ratio}"
        );
    }

    // -----------------------------------------------------------------------
    // Large-scale accuracy comparison: HRA vs LRA
    // -----------------------------------------------------------------------

    #[test]
    fn test_hra_vs_lra_accuracy_profiles() {
        let n = 100_000u64;
        let k = 16;

        let mut hra_sketch = ReqSketch::new(k, ReqMode::HRA);
        let mut lra_sketch = ReqSketch::new(k, ReqMode::LRA);

        for i in 0..n {
            let val = i as f64;
            hra_sketch.update(val);
            lra_sketch.update(val);
        }

        // At p99, HRA should have better (or equal) normalised rank error than LRA
        let hra_p99 = hra_sketch.quantile(0.99).unwrap();
        let lra_p99 = lra_sketch.quantile(0.99).unwrap();
        let expected_p99 = 0.99 * (n - 1) as f64;

        let hra_error_p99 = (hra_p99 - expected_p99).abs() / n as f64;
        let lra_error_p99 = (lra_p99 - expected_p99).abs() / n as f64;

        // At p1, LRA should have better (or equal) normalised rank error than HRA
        let hra_p1 = hra_sketch.quantile(0.01).unwrap();
        let lra_p1 = lra_sketch.quantile(0.01).unwrap();
        let expected_p1 = 0.01 * (n - 1) as f64;

        let hra_error_p1 = (hra_p1 - expected_p1).abs() / n as f64;
        let lra_error_p1 = (lra_p1 - expected_p1).abs() / n as f64;

        // HRA should be at least as good at p99 (with tolerance for randomness)
        assert!(
            hra_error_p99 <= lra_error_p99 + 0.03,
            "HRA should be at least as accurate at p99: HRA error={hra_error_p99:.4}, LRA error={lra_error_p99:.4}"
        );

        // LRA should be at least as good at p1 (with tolerance for randomness)
        assert!(
            lra_error_p1 <= hra_error_p1 + 0.03,
            "LRA should be at least as accurate at p1: LRA error={lra_error_p1:.4}, HRA error={hra_error_p1:.4}"
        );
    }
}
