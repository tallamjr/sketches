//! KLL (Karnin-Lang-Liberty) Sketch for quantile estimation.
//!
//! The KLL sketch is a streaming quantile algorithm that maintains items across
//! multiple compaction levels. It is provably optimal among comparison-based
//! streaming quantile algorithms.
//!
//! # Error Bounds
//! - Normalised rank error: `~1.65 / sqrt(k)` at default k=200.
//! - Space complexity: `O(k * log(n/k))` items across all levels.
//!
//! # When to Use KLL vs T-Digest
//! - KLL has mathematically proven error bounds; T-Digest bounds are empirical.
//! - T-Digest excels at extreme quantiles (p99, p99.9); KLL is more uniform.
//! - KLL supports exact merging; T-Digest merge is approximate.
//!
//! # Common Uses
//! Database query planning, histogram construction, distribution monitoring.
//!
//! # References
//! - Karnin, Lang, Liberty. "Optimal Quantile Approximation in Streams."
//!   FOCS, 2016.

use std::cmp::Ordering;

/// KLL (Karnin-Lang-Liberty) Sketch for quantile estimation.
///
/// Implements the KLL algorithm from "Optimal Quantile Approximation in Streams"
/// by Karnin, Lang, and Liberty (FOCS 2016), following the capacity allocation
/// strategy used in Apache DataSketches.
///
/// The sketch maintains items across multiple levels. Level 0 receives raw input
/// items. When a level overflows its capacity, it is compacted: sorted, then every
/// other item (alternating between even and odd indexed, chosen pseudorandomly) is
/// promoted to the next level. Items at level h represent 2^h original items.
///
///
/// # Error Bounds
/// - Normalised rank error: `~1.65 / sqrt(k)` at default k=200.
/// - Space complexity: O(k * log(n/k)) items across all levels.
/// - Provably optimal among comparison-based streaming quantile algorithms.
///
/// # Common Uses
/// Database query planning, histogram construction, distribution monitoring.
///
/// # References
/// - Karnin, Lang, Liberty. "Optimal Quantile Approximation in Streams."
///   FOCS, 2016.
///
/// Capacities grow geometrically from bottom to top: higher levels have larger
/// capacities because their items are more valuable (each represents more original
/// items). This follows the DataSketches convention where the top level gets
/// capacity close to k, and lower levels get geometrically smaller capacities.
///
/// With k=200, this achieves approximately 1.65% normalised rank error.
#[derive(Debug)]
pub struct KllSketch<T> {
    k: usize,
    /// Items stored at each level. Level 0 receives raw input; higher levels
    /// hold compacted (promoted) items with weight 2^level each.
    levels: Vec<Vec<T>>,
    /// Maximum number of items allowed at each level before compaction.
    level_capacities: Vec<usize>,
    /// Total number of items ingested via `update()`.
    total_count: u64,
    min_value: Option<T>,
    max_value: Option<T>,
    /// Simple PRNG state for randomised compaction offset selection.
    rng_state: u64,
    /// Number of compaction levels (grows as needed).
    num_levels: usize,
}

/// Minimum capacity for any level.
const MIN_LEVEL_CAPACITY: usize = 2;

/// Statistics about a KLL sketch.
#[derive(Debug, Clone)]
pub struct KllStats {
    pub k: usize,
    pub levels: usize,
    pub total_items: usize,
    pub total_count: u64,
    pub memory_usage: usize,
    pub min_value_set: bool,
    pub max_value_set: bool,
}

/// Calculate capacities for all levels given k and number of levels.
///
/// Uses the DataSketches convention: higher levels (which hold more valuable items)
/// get larger capacities. Level h (0-indexed from bottom) gets:
///   capacity(h) = max(MIN_LEVEL_CAPACITY, floor(k * (2/3)^(num_levels - 1 - h)))
///
/// This means:
///   - Top level (num_levels - 1): capacity ~ k
///   - Bottom level (0): smallest capacity
fn compute_capacities(k: usize, num_levels: usize) -> Vec<usize> {
    let mut caps = Vec::with_capacity(num_levels);
    let ratio: f64 = 2.0 / 3.0;
    for h in 0..num_levels {
        let depth_from_top = (num_levels - 1 - h) as i32;
        let cap = (k as f64 * ratio.powi(depth_from_top)).floor() as usize;
        caps.push(cap.max(MIN_LEVEL_CAPACITY));
    }
    caps
}

/// Simple xorshift64 PRNG for deterministic but pseudorandom compaction.
fn xorshift64(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

impl<T> KllSketch<T>
where
    T: Clone + PartialOrd + std::fmt::Debug,
{
    /// Create a new KLL sketch with parameter k.
    ///
    /// Larger k values provide better accuracy but use more memory.
    /// The normalised rank error is approximately 1.65/sqrt(k) with high probability.
    /// At k=200, this gives roughly 1.65% error.
    pub fn new(k: usize) -> Self {
        assert!(k >= 8, "k must be at least 8");

        // Start with 2 levels so level 0 has meaningful capacity
        let num_levels = 2;
        let caps = compute_capacities(k, num_levels);

        let mut levels = Vec::with_capacity(num_levels);
        for &cap in &caps {
            levels.push(Vec::with_capacity(cap));
        }

        KllSketch {
            k,
            levels,
            level_capacities: caps,
            total_count: 0,
            min_value: None,
            max_value: None,
            rng_state: 0xDEAD_BEEF_CAFE_BABEu64,
            num_levels,
        }
    }

    /// Create a KLL sketch with specified accuracy target.
    ///
    /// # Arguments
    /// * `epsilon` - Target normalised rank error (e.g., 0.01 for 1% error)
    /// * `confidence` - Higher confidence levels increase k.
    ///
    /// The relationship is: epsilon ~= 1.65 / sqrt(k), so k ~= (1.65 / epsilon)^2.
    pub fn with_accuracy(epsilon: f64, confidence: f64) -> Self {
        assert!(
            epsilon > 0.0 && epsilon < 1.0,
            "Epsilon must be between 0 and 1"
        );
        assert!(
            confidence > 0.0 && confidence < 1.0,
            "Confidence must be between 0 and 1"
        );

        let confidence_factor = if confidence > 0.99 {
            2.0
        } else if confidence > 0.95 {
            1.5
        } else {
            1.0
        };
        let k = ((1.65 / epsilon).powi(2) * confidence_factor).ceil() as usize;
        let k = k.max(8);

        Self::new(k)
    }

    /// Grow the sketch by adding a new top level and recomputing all capacities.
    fn grow(&mut self) {
        self.num_levels += 1;
        self.level_capacities = compute_capacities(self.k, self.num_levels);

        // Add the new top level
        let top_cap = *self.level_capacities.last().unwrap();
        self.levels.push(Vec::with_capacity(top_cap));
    }

    /// Update the sketch with a new value.
    pub fn update(&mut self, value: T) {
        // Update min/max
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

        // Insert into level 0
        self.levels[0].push(value);

        // Lazy compaction: only compact if level 0 has exceeded its capacity
        if self.levels[0].len() > self.level_capacities[0] {
            self.compact();
        }
    }

    /// Find the lowest level that is over capacity and compact it.
    /// If compaction cascades upward and the top level overflows,
    /// grow the sketch by adding a new level and recomputing capacities.
    fn compact(&mut self) {
        // Find the lowest level that is over capacity
        let mut level = 0;
        while level < self.num_levels {
            if self.levels[level].len() > self.level_capacities[level] {
                // If this is the top level, we need to grow first
                if level + 1 >= self.num_levels {
                    self.grow();
                }

                self.compact_level(level);

                // Check if the next level now overflows
                level += 1;
            } else {
                break;
            }
        }
    }

    /// Compact a single level: sort it, select every other item (randomly
    /// choosing even or odd offset), and promote the selected items to the
    /// next level. The remaining items are discarded.
    fn compact_level(&mut self, level: usize) {
        // Sort the level
        self.levels[level].sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        // Randomly choose offset 0 or 1
        let offset = (xorshift64(&mut self.rng_state) & 1) as usize;

        // Collect promoted items (every other item from the sorted level)
        let promoted: Vec<T> = self.levels[level]
            .iter()
            .skip(offset)
            .step_by(2)
            .cloned()
            .collect();

        // Clear the compacted level
        self.levels[level].clear();

        // Add promoted items to the next level
        for item in promoted {
            self.levels[level + 1].push(item);
        }
    }

    /// Get quantile for the given rank (0.0 to 1.0).
    ///
    /// # Arguments
    /// * `rank` - The quantile rank (e.g., 0.5 for median, 0.95 for 95th percentile)
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

        // Build a weighted list of all items across all levels.
        let mut weighted_items: Vec<(T, u64)> = Vec::new();

        for (level_idx, level_items) in self.levels.iter().enumerate() {
            let weight = 1u64 << level_idx;
            for item in level_items {
                weighted_items.push((item.clone(), weight));
            }
        }

        if weighted_items.is_empty() {
            return self.min_value.clone();
        }

        // Sort by value
        weighted_items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Use the sum of retained weights for rank estimation.
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
    /// Returns the fraction of values in the stream that are less than
    /// or equal to the given value.
    pub fn rank(&mut self, value: &T) -> f64 {
        if self.total_count == 0 {
            return 0.0;
        }

        let mut weight_le: u64 = 0;
        let mut total_weight: u64 = 0;

        for (level_idx, level_items) in self.levels.iter().enumerate() {
            let weight = 1u64 << level_idx;
            for item in level_items {
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

    /// Merge another KLL sketch into this one.
    pub fn merge(&mut self, other: &mut KllSketch<T>) {
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
        while self.num_levels < other.num_levels {
            self.grow();
        }

        // Merge each level's items
        for (level_idx, other_level) in other.levels.iter().enumerate() {
            for item in other_level {
                self.levels[level_idx].push(item.clone());
            }
        }

        // Compact overflowed levels
        self.compact();
    }

    /// Get sketch statistics.
    pub fn statistics(&self) -> KllStats {
        let mut total_items = 0;
        let mut memory_usage = 0;

        for level_items in &self.levels {
            total_items += level_items.len();
            memory_usage += level_items.capacity() * std::mem::size_of::<T>();
        }

        KllStats {
            k: self.k,
            levels: self.num_levels,
            total_items,
            total_count: self.total_count,
            memory_usage,
            min_value_set: self.min_value.is_some(),
            max_value_set: self.max_value.is_some(),
        }
    }

    /// Get the total number of items processed.
    pub fn count(&self) -> u64 {
        self.total_count
    }

    /// Check if the sketch is empty.
    pub fn is_empty(&self) -> bool {
        self.total_count == 0
    }

    /// Get the k parameter.
    pub fn k_value(&self) -> usize {
        self.k
    }

    /// Get the number of levels.
    pub fn num_levels_value(&self) -> usize {
        self.num_levels
    }

    /// Get a reference to the levels.
    pub fn levels_ref(&self) -> &[Vec<T>] {
        &self.levels
    }

    /// Get the minimum value seen (if any).
    pub fn min(&self) -> Option<&T> {
        self.min_value.as_ref()
    }

    /// Get the maximum value seen (if any).
    pub fn max(&self) -> Option<&T> {
        self.max_value.as_ref()
    }

    /// Get the Cumulative Distribution Function (CDF) at the given split points.
    ///
    /// Returns a vector of length `split_points.len() + 1` where:
    /// - result[i] = fraction of items <= split_points[i]
    /// - result[last] = 1.0
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

    /// Get the Probability Mass Function (PMF) at the given split points.
    ///
    /// Returns a vector of length `split_points.len() + 1` where:
    /// - result[0] = fraction of items <= split_points[0]
    /// - result[i] = fraction of items in (split_points[i-1], split_points[i]]
    /// - result[last] = fraction of items > split_points[last]
    ///
    /// Split points must be sorted in ascending order.
    pub fn get_pmf(&mut self, split_points: &[T]) -> Vec<f64> {
        let cdf = self.get_cdf(split_points);
        let mut pmf = Vec::with_capacity(cdf.len());

        if cdf.is_empty() {
            return pmf;
        }

        pmf.push(cdf[0]);
        for i in 1..cdf.len() {
            pmf.push(cdf[i] - cdf[i - 1]);
        }

        pmf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kll_basic() {
        let mut sketch = KllSketch::new(200);

        for i in 1..=1000 {
            sketch.update(i);
        }

        let median = sketch.quantile(0.5).unwrap();
        assert!(
            (median as f64 - 500.0).abs() < 30.0,
            "Median should be around 500, got {median}"
        );

        let q95 = sketch.quantile(0.95).unwrap();
        assert!(
            (q95 as f64 - 950.0).abs() < 30.0,
            "95th percentile should be around 950, got {q95}"
        );

        let rank_500 = sketch.rank(&500);
        assert!(
            (rank_500 - 0.5).abs() < 0.03,
            "Rank of 500 should be around 0.5, got {rank_500}"
        );
    }

    #[test]
    fn test_kll_accuracy_k200_uniform_100k() {
        // Primary accuracy test: K=200 on 100,000 uniform elements
        // should achieve < 2% normalised rank error.
        let n: u64 = 100_000;
        let mut sketch = KllSketch::new(200);

        for i in 0..n {
            sketch.update(i as f64);
        }

        let test_quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];

        let tolerance = 0.02; // 2% normalised rank error

        for &q in &test_quantiles {
            let estimated = sketch.quantile(q).unwrap();
            let expected = q * (n - 1) as f64;
            let normalised_error = (estimated - expected).abs() / n as f64;

            assert!(
                normalised_error < tolerance,
                "Quantile {q} failed: estimated={estimated:.1}, expected={expected:.1}, normalised_error={normalised_error:.4} (limit={tolerance:.4})"
            );
        }
    }

    #[test]
    fn test_kll_rank_accuracy_k200() {
        let n: u64 = 100_000;
        let mut sketch = KllSketch::new(200);

        for i in 0..n {
            sketch.update(i as f64);
        }

        let tolerance = 0.02;

        let test_ranks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        for &target_rank in &test_ranks {
            let value = target_rank * (n - 1) as f64;
            let estimated_rank = sketch.rank(&value);
            let error = (estimated_rank - target_rank).abs();

            assert!(
                error < tolerance,
                "Rank of {value:.0} failed: estimated={estimated_rank:.4}, true={target_rank:.4}, error={error:.4} (limit={tolerance:.4})"
            );
        }
    }

    #[test]
    fn test_kll_merge() {
        let mut sketch1 = KllSketch::new(200);
        let mut sketch2 = KllSketch::new(200);

        for i in 0..500 {
            sketch1.update(i);
        }

        for i in 500..1000 {
            sketch2.update(i);
        }

        sketch1.merge(&mut sketch2);

        assert_eq!(sketch1.count(), 1000);
        assert_eq!(*sketch1.min().unwrap(), 0);
        assert_eq!(*sketch1.max().unwrap(), 999);

        let median = sketch1.quantile(0.5).unwrap();
        assert!(
            (median as f64 - 500.0).abs() < 30.0,
            "Merged median should be around 500, got {median}"
        );
    }

    #[test]
    fn test_kll_with_accuracy() {
        let sketch = KllSketch::<f64>::with_accuracy(0.01, 0.8);
        assert!(
            sketch.k >= 27225,
            "k should be large for 1% accuracy, got {}",
            sketch.k
        );

        let sketch2 = KllSketch::<f64>::with_accuracy(0.05, 0.8);
        assert!(
            sketch2.k >= 1089,
            "k should be >= 1089 for 5% accuracy, got {}",
            sketch2.k
        );
    }

    #[test]
    fn test_kll_empty() {
        let mut sketch = KllSketch::<i32>::new(100);

        assert!(sketch.is_empty());
        assert_eq!(sketch.count(), 0);
        assert!(sketch.quantile(0.5).is_none());
        assert_eq!(sketch.rank(&100), 0.0);
        assert!(sketch.min().is_none());
        assert!(sketch.max().is_none());
    }

    #[test]
    fn test_kll_single_value() {
        let mut sketch = KllSketch::new(100);
        sketch.update(42);

        assert!(!sketch.is_empty());
        assert_eq!(sketch.count(), 1);
        assert_eq!(sketch.quantile(0.5).unwrap(), 42);
        assert_eq!(sketch.rank(&42), 1.0);
        assert_eq!(sketch.rank(&41), 0.0);
        assert_eq!(*sketch.min().unwrap(), 42);
        assert_eq!(*sketch.max().unwrap(), 42);
    }

    #[test]
    fn test_kll_statistics() {
        let mut sketch = KllSketch::new(200);

        for i in 0..1000 {
            sketch.update(i);
        }

        let stats = sketch.statistics();
        assert_eq!(stats.k, 200);
        assert!(stats.levels > 0);
        assert!(stats.total_items > 0);
        assert_eq!(stats.total_count, 1000);
        assert!(stats.memory_usage > 0);
        assert!(stats.min_value_set);
        assert!(stats.max_value_set);
    }

    #[test]
    fn test_kll_level_capacities_decrease_toward_bottom() {
        // Verify that lower levels (closer to input) have smaller capacities
        // and the top level approaches k
        let k = 200;
        let caps = compute_capacities(k, 5);

        // Top level should be largest (close to k)
        assert_eq!(caps[4], k);

        // Each level below should be smaller
        assert!(
            caps[3] < caps[4],
            "Level 3 cap {} should be < level 4 cap {}",
            caps[3],
            caps[4]
        );
        assert!(
            caps[2] < caps[3],
            "Level 2 cap {} should be < level 3 cap {}",
            caps[2],
            caps[3]
        );
        assert!(
            caps[1] < caps[2],
            "Level 1 cap {} should be < level 2 cap {}",
            caps[1],
            caps[2]
        );
        assert!(
            caps[0] < caps[1],
            "Level 0 cap {} should be < level 1 cap {}",
            caps[0],
            caps[1]
        );
    }

    #[test]
    fn test_kll_get_cdf() {
        let mut sketch = KllSketch::new(200);

        for i in 0..1000 {
            sketch.update(i as f64);
        }

        let split_points = vec![250.0, 500.0, 750.0];
        let cdf = sketch.get_cdf(&split_points);

        assert_eq!(cdf.len(), 4);

        assert!(
            (cdf[0] - 0.251).abs() < 0.03,
            "CDF at 250 should be ~0.251, got {}",
            cdf[0]
        );
        assert!(
            (cdf[1] - 0.501).abs() < 0.03,
            "CDF at 500 should be ~0.501, got {}",
            cdf[1]
        );
        assert!(
            (cdf[2] - 0.751).abs() < 0.03,
            "CDF at 750 should be ~0.751, got {}",
            cdf[2]
        );
        assert_eq!(cdf[3], 1.0);
    }

    #[test]
    fn test_kll_get_pmf() {
        let mut sketch = KllSketch::new(200);

        for i in 0..1000 {
            sketch.update(i as f64);
        }

        let split_points = vec![250.0, 500.0, 750.0];
        let pmf = sketch.get_pmf(&split_points);

        assert_eq!(pmf.len(), 4);

        for (i, &p) in pmf.iter().enumerate() {
            assert!(
                (p - 0.25).abs() < 0.04,
                "PMF bucket {i} should be ~0.25, got {p}"
            );
        }

        let sum: f64 = pmf.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "PMF should sum to 1.0, got {sum}"
        );
    }

    #[test]
    fn test_kll_extreme_quantiles() {
        let n: u64 = 100_000;
        let mut sketch = KllSketch::new(200);

        for i in 0..n {
            sketch.update(i as f64);
        }

        let tolerance = 0.02;

        let q01 = sketch.quantile(0.01).unwrap();
        let true_q01 = 0.01 * (n - 1) as f64;
        let error_q01 = (q01 - true_q01).abs() / n as f64;
        assert!(
            error_q01 < tolerance,
            "1st percentile error {:.4} exceeds {}%: estimated={:.1}, expected={:.1}",
            error_q01,
            tolerance * 100.0,
            q01,
            true_q01
        );

        let q99 = sketch.quantile(0.99).unwrap();
        let true_q99 = 0.99 * (n - 1) as f64;
        let error_q99 = (q99 - true_q99).abs() / n as f64;
        assert!(
            error_q99 < tolerance,
            "99th percentile error {:.4} exceeds {}%: estimated={:.1}, expected={:.1}",
            error_q99,
            tolerance * 100.0,
            q99,
            true_q99
        );
    }

    #[test]
    fn test_kll_get_cdf_empty() {
        let mut sketch = KllSketch::<f64>::new(200);
        let cdf = sketch.get_cdf(&[1.0, 2.0]);
        assert_eq!(cdf, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_kll_get_pmf_empty() {
        let mut sketch = KllSketch::<f64>::new(200);
        let pmf = sketch.get_pmf(&[1.0, 2.0]);
        assert_eq!(pmf, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_kll_memory_bounded() {
        let k = 200;
        let mut sketch = KllSketch::new(k);

        for i in 0..1_000_000 {
            sketch.update(i);
        }

        let stats = sketch.statistics();
        assert!(
            stats.total_items < k * 10,
            "Total items {} should be bounded, k={}",
            stats.total_items,
            k
        );
    }
}
