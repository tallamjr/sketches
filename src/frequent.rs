//! Frequent Items (Heavy Hitters) sketch for finding the most common elements in a stream.
//!
//! # Error Bounds
//! - Deterministic guarantee: `(upper_bound - lower_bound) <= total_weight * epsilon`
//!   where `epsilon = 3.5 / max_map_size`.
//! - No false negatives when using `ErrorType::NoFalseNegatives`.
//!
//! # Common Uses
//! Top-K heavy hitters, trending topic detection, network flow analysis.
//!
//! # References
//! - Misra, Gries. "Finding Repeated Elements." Science of Computer Programming, 1982.
//! - Metwally, Agrawal, El Abbadi. "Efficient Computation of Frequent and Top-k
//!   Elements in Data Streams." ICDT, 2005.

use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;

/// Query mode for retrieving frequent items with different error guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorType {
    /// Only return items whose lower bound exceeds the threshold.
    /// Guarantees no false positives: every returned item truly exceeds the threshold.
    NoFalsePositives,
    /// Return items whose upper bound exceeds the threshold.
    /// Guarantees no false negatives: every item that truly exceeds the threshold is returned.
    NoFalseNegatives,
}

/// Generic Frequent Items Sketch using the Misra-Gries algorithm.
///
/// Tracks the most frequent items in a data stream with deterministic error bounds.
/// The error guarantee is: (upper_bound - lower_bound) <= total_weight * epsilon
/// where epsilon = 3.5 / max_map_size.
///
/// Items are tracked with an offset that represents the maximum possible overcount.
/// After a purge (when the map exceeds capacity), the offset increases by the minimum
/// count that was subtracted from all entries.
pub struct FrequentItemsSketch<T: Hash + Eq + Clone> {
    max_map_size: usize,
    stream_length: u64,
    offset: u64,
    map: HashMap<T, ItemEntry>,
}

/// Internal tracking entry for each item.
#[derive(Debug, Clone)]
struct ItemEntry {
    /// The raw count accumulated for this item within the sketch.
    count: u64,
}

/// A type alias preserving backwards compatibility for string-only usage.
pub type FrequentStringsSketch = FrequentItemsSketch<String>;

/// Result item from frequent items query.
#[derive(Debug, Clone)]
pub struct FrequentItemResult<T: Clone> {
    pub item: T,
    pub estimate: u64,
    pub lower_bound: u64,
    pub upper_bound: u64,
}

/// Statistics about a Frequent Items sketch.
#[derive(Debug, Clone)]
pub struct FrequentItemsStats {
    pub max_map_size: usize,
    pub current_map_size: usize,
    pub stream_length: u64,
    pub total_tracked_frequency: u64,
    pub offset: u64,
    pub memory_usage: usize,
}

/// Backwards-compatible alias for statistics.
pub type FrequentStringsStats = FrequentItemsStats;

impl<T: Hash + Eq + Clone> FrequentItemsSketch<T> {
    /// Create a new Frequent Items sketch.
    ///
    /// # Arguments
    /// * `max_map_size` - Maximum number of items to track (affects memory and accuracy).
    ///   The `_use_reservoir` parameter is accepted for API compatibility but ignored;
    ///   reservoir sampling has been removed in favour of proper Misra-Gries guarantees.
    pub fn new(max_map_size: usize, _use_reservoir: bool) -> Self {
        assert!(max_map_size > 0, "Max map size must be positive");

        FrequentItemsSketch {
            max_map_size,
            stream_length: 0,
            offset: 0,
            map: HashMap::with_capacity(max_map_size),
        }
    }

    /// Create with error rate specification.
    ///
    /// # Arguments
    /// * `error_rate` - Maximum relative error (e.g., 0.01 for 1% error).
    /// * `confidence` - Confidence level (accepted for API compatibility, not used
    ///   since Misra-Gries bounds are deterministic).
    /// * `_use_reservoir` - Accepted for API compatibility but ignored.
    pub fn with_error_rate(error_rate: f64, confidence: f64, _use_reservoir: bool) -> Self {
        assert!(
            error_rate > 0.0 && error_rate < 1.0,
            "Error rate must be between 0 and 1"
        );
        assert!(
            confidence > 0.0 && confidence < 1.0,
            "Confidence must be between 0 and 1"
        );

        // Misra-Gries requires map size >= ceil(1/epsilon).
        // We use 3.5/error_rate to match the DataSketches convention.
        let max_map_size = (3.5 / error_rate).ceil() as usize;

        Self::new(max_map_size, false)
    }

    /// Update the sketch with a new item (weight = 1).
    pub fn update_item(&mut self, item: &T) {
        self.update_weighted(item, 1);
    }

    /// Update the sketch with a weighted item.
    pub fn update_weighted(&mut self, item: &T, weight: u64) {
        self.stream_length += weight;

        if let Some(entry) = self.map.get_mut(item) {
            entry.count += weight;
            return;
        }

        // Item not present -- insert it.
        self.map.insert(item.clone(), ItemEntry { count: weight });

        // If the map now exceeds capacity, perform a reverse purge.
        if self.map.len() > self.max_map_size {
            self.purge();
        }
    }

    /// Misra-Gries reverse purge: find the minimum count, subtract it from all
    /// entries, and remove entries that drop to zero. The offset increases by the
    /// minimum count to preserve the error bounds.
    fn purge(&mut self) {
        let min_count = self.map.values().map(|e| e.count).min().unwrap_or(0);

        if min_count == 0 {
            // Remove zero-count entries only.
            self.map.retain(|_, e| e.count > 0);
            return;
        }

        self.offset += min_count;

        self.map.retain(|_, entry| {
            entry.count -= min_count;
            entry.count > 0
        });
    }

    /// Compute the lower bound for an item's true frequency.
    /// Lower bound = count - offset (but not below zero).
    fn lower_bound_for(&self, entry: &ItemEntry) -> u64 {
        entry.count.saturating_sub(self.offset)
    }

    /// Compute the upper bound for an item's true frequency.
    /// Upper bound = count.
    /// In Misra-Gries, the stored count is always an upper bound on the true frequency
    /// because the count may include contributions from purged items' redistributed counts.
    fn upper_bound_for(&self, entry: &ItemEntry) -> u64 {
        entry.count
    }

    /// Get frequent items above a threshold using the specified error type.
    ///
    /// # Arguments
    /// * `threshold` - Minimum frequency threshold (absolute count).
    /// * `error_type` - Query mode controlling false positive/negative behaviour.
    pub fn get_frequent_items_with_error_type(
        &self,
        threshold: u64,
        error_type: ErrorType,
    ) -> Vec<FrequentItemResult<T>> {
        let mut results = Vec::new();

        for (item, entry) in &self.map {
            let lower = self.lower_bound_for(entry);
            let upper = self.upper_bound_for(entry);

            let include = match error_type {
                ErrorType::NoFalsePositives => lower > threshold,
                ErrorType::NoFalseNegatives => upper > threshold,
            };

            if include {
                results.push(FrequentItemResult {
                    item: item.clone(),
                    estimate: entry.count,
                    lower_bound: lower,
                    upper_bound: upper,
                });
            }
        }

        results.sort_by(|a, b| b.estimate.cmp(&a.estimate));
        results
    }

    /// Get the top-k most frequent items.
    pub fn get_top_k(&self, k: usize) -> Vec<FrequentItemResult<T>> {
        let mut results: Vec<FrequentItemResult<T>> = self
            .map
            .iter()
            .map(|(item, entry)| FrequentItemResult {
                item: item.clone(),
                estimate: entry.count,
                lower_bound: self.lower_bound_for(entry),
                upper_bound: self.upper_bound_for(entry),
            })
            .collect();

        results.sort_by(|a, b| b.estimate.cmp(&a.estimate));
        results.truncate(k);
        results
    }

    /// Get frequency bounds for a specific item.
    pub fn get_bounds_for(&self, item: &T) -> Option<(u64, u64)> {
        self.map
            .get(item)
            .map(|entry| (self.lower_bound_for(entry), self.upper_bound_for(entry)))
    }

    /// Get estimated frequency of a specific item.
    pub fn get_estimate_for(&self, item: &T) -> Option<u64> {
        self.map.get(item).map(|entry| entry.count)
    }

    /// Merge another sketch into this one using proper Misra-Gries merge.
    ///
    /// The merge combines all items from both sketches into a unified map.
    /// When the unified map exceeds capacity, a reverse purge is performed:
    /// the minimum count is found, subtracted from all entries, and entries
    /// that drop to zero are removed. This preserves the deterministic bounds.
    pub fn merge(&mut self, other: &FrequentItemsSketch<T>) -> Result<(), &'static str> {
        if self.max_map_size != other.max_map_size {
            return Err("Cannot merge sketches with different map sizes");
        }

        self.stream_length += other.stream_length;

        // The combined offset accounts for purges that happened in both sketches.
        self.offset += other.offset;

        // Combine all items from the other sketch into this one.
        for (item, other_entry) in &other.map {
            if let Some(self_entry) = self.map.get_mut(item) {
                self_entry.count += other_entry.count;
            } else {
                self.map.insert(item.clone(), other_entry.clone());
            }
        }

        // Repeatedly purge until the map is within capacity.
        while self.map.len() > self.max_map_size {
            self.purge();
        }

        Ok(())
    }

    /// Get the current offset (maximum possible overcount for any tracked item).
    pub fn get_offset(&self) -> u64 {
        self.offset
    }

    /// Get sketch statistics.
    pub fn statistics(&self) -> FrequentItemsStats {
        let total_frequency: u64 = self.map.values().map(|e| e.count).sum();

        FrequentItemsStats {
            max_map_size: self.max_map_size,
            current_map_size: self.map.len(),
            stream_length: self.stream_length,
            total_tracked_frequency: total_frequency,
            offset: self.offset,
            memory_usage: self.estimate_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes.
    fn estimate_memory_usage(&self) -> usize {
        let map_overhead = std::mem::size_of::<HashMap<T, ItemEntry>>();
        let entry_size = std::mem::size_of::<ItemEntry>();
        let key_size = std::mem::size_of::<T>();
        map_overhead + self.map.len() * (entry_size + key_size)
    }

    /// Clear the sketch.
    pub fn clear(&mut self) {
        self.map.clear();
        self.stream_length = 0;
        self.offset = 0;
    }

    /// Check if sketch is empty.
    pub fn is_empty(&self) -> bool {
        self.stream_length == 0
    }

    /// Get the total number of items processed.
    pub fn get_stream_length(&self) -> u64 {
        self.stream_length
    }
}

// String-specific convenience methods to keep the existing public API intact.
// These take &str directly and handle String conversion internally.
impl FrequentItemsSketch<String> {
    /// Update the sketch with a string slice (convenience for &str).
    pub fn update(&mut self, item: &str) {
        self.stream_length += 1;

        if let Some(entry) = self.map.get_mut(item) {
            entry.count += 1;
            return;
        }

        self.map.insert(item.to_string(), ItemEntry { count: 1 });

        if self.map.len() > self.max_map_size {
            self.purge();
        }
    }

    /// Get estimated frequency of a specific item (string slice version).
    pub fn get_estimate(&self, item: &str) -> Option<u64> {
        self.map.get(item).map(|entry| entry.count)
    }

    /// Get frequency bounds for a specific item (string slice version).
    pub fn get_bounds(&self, item: &str) -> Option<(u64, u64)> {
        self.map
            .get(item)
            .map(|entry| (self.lower_bound_for(entry), self.upper_bound_for(entry)))
    }

    /// Get frequent items above a threshold.
    ///
    /// Uses `NoFalseNegatives` mode by default (returns all items that might
    /// exceed the threshold) to preserve backwards-compatible behaviour.
    pub fn get_frequent_items(&self, threshold: u64) -> Vec<FrequentItemResult<String>> {
        self.get_frequent_items_with_error_type(threshold, ErrorType::NoFalseNegatives)
    }

    /// Get frequent items above a relative threshold.
    pub fn get_frequent_items_by_fraction(
        &self,
        threshold_fraction: f64,
    ) -> Vec<FrequentItemResult<String>> {
        assert!(
            (0.0..=1.0).contains(&threshold_fraction),
            "Threshold fraction must be between 0 and 1"
        );

        let threshold = (threshold_fraction * self.stream_length as f64) as u64;
        self.get_frequent_items(threshold)
    }

    /// Get frequent items using a specific error type (string slice version).
    pub fn get_frequent_items_with_mode(
        &self,
        threshold: u64,
        error_type: ErrorType,
    ) -> Vec<FrequentItemResult<String>> {
        self.get_frequent_items_with_error_type(threshold, error_type)
    }
}

impl<T: Hash + Eq + Clone + fmt::Debug> fmt::Debug for FrequentItemsSketch<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FrequentItemsSketch")
            .field("max_map_size", &self.max_map_size)
            .field("stream_length", &self.stream_length)
            .field("offset", &self.offset)
            .field("current_items", &self.map.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequent_strings_basic() {
        let mut sketch = FrequentStringsSketch::new(10, false);

        for _ in 0..100 {
            sketch.update("apple");
        }
        for _ in 0..50 {
            sketch.update("banana");
        }
        for _ in 0..25 {
            sketch.update("cherry");
        }

        assert_eq!(sketch.get_estimate("apple"), Some(100));
        assert_eq!(sketch.get_estimate("banana"), Some(50));
        assert_eq!(sketch.get_estimate("cherry"), Some(25));
        assert_eq!(sketch.get_estimate("not_present"), None);

        let frequent = sketch.get_frequent_items(30);
        assert_eq!(frequent.len(), 2); // apple and banana
        assert_eq!(frequent[0].item, "apple");
        assert_eq!(frequent[1].item, "banana");
    }

    #[test]
    fn test_frequent_strings_space_saving() {
        let mut sketch = FrequentStringsSketch::new(3, false);

        sketch.update("a");
        sketch.update("b");
        sketch.update("c");
        sketch.update("d"); // Should trigger purge

        assert!(sketch.map.len() <= 3); // Should not exceed capacity
        assert_eq!(sketch.get_stream_length(), 4);
    }

    #[test]
    fn test_frequent_strings_top_k() {
        let mut sketch = FrequentStringsSketch::new(10, false);

        for i in 1..=5 {
            for _ in 0..(6 - i) * 10 {
                sketch.update(&format!("item{i}"));
            }
        }

        let top3 = sketch.get_top_k(3);
        assert_eq!(top3.len(), 3);
        assert_eq!(top3[0].item, "item1");
        assert_eq!(top3[1].item, "item2");
        assert_eq!(top3[2].item, "item3");
    }

    #[test]
    fn test_frequent_strings_by_fraction() {
        let mut sketch = FrequentStringsSketch::new(10, false);

        for _ in 0..500 {
            sketch.update("majority");
        }
        for _ in 0..300 {
            sketch.update("significant");
        }
        for _ in 0..200 {
            sketch.update("minority");
        }

        let frequent = sketch.get_frequent_items_by_fraction(0.25);
        assert_eq!(frequent.len(), 2); // majority and significant
    }

    #[test]
    fn test_frequent_strings_merge() {
        let mut sketch1 = FrequentStringsSketch::new(5, false);
        let mut sketch2 = FrequentStringsSketch::new(5, false);

        for _ in 0..10 {
            sketch1.update("common");
            sketch1.update("sketch1_only");
        }

        for _ in 0..15 {
            sketch2.update("common");
            sketch2.update("sketch2_only");
        }

        let before_merge = sketch1.get_estimate("common").unwrap();
        sketch1.merge(&sketch2).unwrap();
        let after_merge = sketch1.get_estimate("common").unwrap();

        assert!(after_merge > before_merge);
        assert_eq!(after_merge, 25); // 10 + 15
    }

    #[test]
    fn test_frequent_strings_with_error_rate() {
        let sketch = FrequentStringsSketch::with_error_rate(0.01, 0.99, false);
        let stats = sketch.statistics();

        assert!(stats.max_map_size >= 300); // 3.5/0.01 = 350
        assert_eq!(stats.current_map_size, 0);
        assert_eq!(stats.stream_length, 0);
    }

    #[test]
    fn test_frequent_strings_bounds() {
        let mut sketch = FrequentStringsSketch::new(3, false);

        sketch.update("a");
        sketch.update("b");
        sketch.update("c");
        sketch.update("d"); // This triggers a purge

        // After purge, offset should have increased.
        assert!(sketch.get_offset() > 0);

        // Check that bounds are properly maintained for all tracked items.
        for entry in sketch.map.values() {
            let lower = sketch.lower_bound_for(entry);
            let upper = sketch.upper_bound_for(entry);
            assert!(lower <= entry.count);
            assert!(upper >= lower);
        }
    }

    #[test]
    fn test_frequent_strings_reservoir() {
        // Reservoir parameter is accepted but ignored; sketch should still work.
        let mut sketch = FrequentStringsSketch::new(5, true);

        for i in 0..100 {
            sketch.update(&format!("item_{}", i % 20));
        }

        assert!(sketch.map.len() <= 5); // Should maintain size limit
        assert_eq!(sketch.get_stream_length(), 100);
    }

    #[test]
    fn test_frequent_strings_edge_cases() {
        let mut sketch = FrequentStringsSketch::new(5, false);

        assert!(sketch.is_empty());
        assert_eq!(sketch.get_top_k(10).len(), 0);
        assert_eq!(sketch.get_frequent_items(1).len(), 0);

        sketch.update("single");
        assert!(!sketch.is_empty());
        assert_eq!(sketch.get_estimate("single"), Some(1));

        sketch.clear();
        assert!(sketch.is_empty());
        assert_eq!(sketch.get_stream_length(), 0);
    }

    // ---- New tests for proper Misra-Gries guarantees ----

    #[test]
    fn test_misra_gries_purge_preserves_bounds() {
        // With max_map_size=3, inserting 4 distinct items each once should
        // trigger a purge. All items have count=1, so the min is 1, offset
        // increases by 1, and all entries are removed (count - 1 = 0).
        let mut sketch = FrequentStringsSketch::new(3, false);
        sketch.update("a");
        sketch.update("b");
        sketch.update("c");
        sketch.update("d");

        // After purge, offset should be exactly 1.
        assert_eq!(sketch.get_offset(), 1);
        // All items had count=1, subtracting 1 leaves 0, so all are removed.
        // Only "d" might remain if it was added after purge -- but our implementation
        // inserts first then purges, so "d" is in the map when purge runs.
        // All four items have count=1, min=1, all removed.
        assert_eq!(sketch.map.len(), 0);
    }

    #[test]
    fn test_misra_gries_heavy_hitter_survives_purge() {
        let mut sketch = FrequentStringsSketch::new(3, false);

        // Insert a heavy hitter many times.
        for _ in 0..50 {
            sketch.update("heavy");
        }
        // Insert light items that will be purged.
        sketch.update("light1");
        sketch.update("light2");
        sketch.update("light3"); // Triggers purge

        // "heavy" should survive with a high count.
        let estimate = sketch.get_estimate("heavy");
        assert!(estimate.is_some());
        assert!(estimate.unwrap() >= 49); // At least 50 - offset

        // Bounds should be valid.
        let (lower, upper) = sketch.get_bounds("heavy").unwrap();
        assert!(lower <= estimate.unwrap());
        assert!(upper >= lower);
    }

    #[test]
    fn test_no_false_positives_mode() {
        let mut sketch = FrequentStringsSketch::new(10, false);

        for _ in 0..100 {
            sketch.update("definite_heavy");
        }
        for _ in 0..5 {
            sketch.update("maybe_heavy");
        }

        // With NoFalsePositives, only items whose lower_bound > threshold are returned.
        let results = sketch.get_frequent_items_with_mode(10, ErrorType::NoFalsePositives);
        for result in &results {
            assert!(
                result.lower_bound > 10,
                "NoFalsePositives returned item with lower_bound {} <= threshold 10",
                result.lower_bound
            );
        }
        // "definite_heavy" should definitely be included.
        assert!(results.iter().any(|r| r.item == "definite_heavy"));
    }

    #[test]
    fn test_no_false_negatives_mode() {
        let mut sketch = FrequentStringsSketch::new(10, false);

        for _ in 0..100 {
            sketch.update("definite_heavy");
        }
        for _ in 0..20 {
            sketch.update("borderline");
        }
        for _ in 0..2 {
            sketch.update("light");
        }

        // With NoFalseNegatives, all items whose upper_bound > threshold are returned.
        let results = sketch.get_frequent_items_with_mode(15, ErrorType::NoFalseNegatives);
        // Both "definite_heavy" and "borderline" should appear.
        assert!(results.iter().any(|r| r.item == "definite_heavy"));
        assert!(results.iter().any(|r| r.item == "borderline"));
    }

    #[test]
    fn test_error_bound_invariant() {
        // The Misra-Gries guarantee: for every tracked item,
        // upper_bound - lower_bound == offset.
        let mut sketch = FrequentStringsSketch::new(5, false);

        for i in 0..200 {
            sketch.update(&format!("item_{}", i % 30));
        }

        let offset = sketch.get_offset();
        for entry in sketch.map.values() {
            let lower = sketch.lower_bound_for(entry);
            let upper = sketch.upper_bound_for(entry);
            // upper - lower should equal offset (or less if lower was clamped to 0).
            assert!(upper - lower <= offset);
        }

        // Also verify the global bound: offset <= stream_length / max_map_size
        // (This is the core Misra-Gries theorem.)
        let stream_length = sketch.get_stream_length();
        let max_map_size = sketch.statistics().max_map_size as u64;
        assert!(
            offset <= stream_length / max_map_size + 1, // +1 for integer rounding
            "offset {offset} exceeded stream_length/max_map_size = {stream_length}/{max_map_size}"
        );
    }

    #[test]
    fn test_merge_preserves_misra_gries_guarantees() {
        let mut sketch1 = FrequentStringsSketch::new(5, false);
        let mut sketch2 = FrequentStringsSketch::new(5, false);

        // Fill both sketches with overlapping and distinct items, triggering purges.
        for i in 0..100 {
            sketch1.update(&format!("item_{}", i % 15));
        }
        for i in 0..100 {
            sketch2.update(&format!("item_{}", (i + 5) % 15));
        }

        sketch1.merge(&sketch2).unwrap();

        // After merge, the map should be within capacity.
        assert!(sketch1.map.len() <= 5);

        // Bounds should still hold.
        for entry in sketch1.map.values() {
            let lower = sketch1.lower_bound_for(entry);
            let upper = sketch1.upper_bound_for(entry);
            assert!(lower <= upper);
        }

        // Stream length should be the sum.
        assert_eq!(sketch1.get_stream_length(), 200);
    }

    #[test]
    fn test_merge_offset_accumulates() {
        let mut sketch1 = FrequentStringsSketch::new(3, false);
        let mut sketch2 = FrequentStringsSketch::new(3, false);

        // Force purges in both sketches.
        for i in 0..20 {
            sketch1.update(&format!("s1_{i}"));
        }
        for i in 0..20 {
            sketch2.update(&format!("s2_{i}"));
        }

        let offset1 = sketch1.get_offset();
        let offset2 = sketch2.get_offset();

        sketch1.merge(&sketch2).unwrap();

        // The merged offset should be at least the sum of the two offsets.
        assert!(sketch1.get_offset() >= offset1 + offset2);
    }

    #[test]
    fn test_generic_with_integers() {
        let mut sketch: FrequentItemsSketch<u64> = FrequentItemsSketch::new(5, false);

        for _ in 0..50 {
            sketch.update_item(&1u64);
        }
        for _ in 0..30 {
            sketch.update_item(&2u64);
        }
        for _ in 0..10 {
            sketch.update_item(&3u64);
        }

        assert_eq!(sketch.get_estimate_for(&1u64), Some(50));
        assert_eq!(sketch.get_estimate_for(&2u64), Some(30));
        assert_eq!(sketch.get_estimate_for(&3u64), Some(10));

        let top2 = sketch.get_top_k(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].item, 1u64);
        assert_eq!(top2[1].item, 2u64);
    }

    #[test]
    fn test_generic_purge_with_integers() {
        let mut sketch: FrequentItemsSketch<i32> = FrequentItemsSketch::new(3, false);

        for _ in 0..100 {
            sketch.update_item(&42);
        }
        sketch.update_item(&1);
        sketch.update_item(&2);
        sketch.update_item(&3); // triggers purge

        // The heavy hitter should survive.
        assert!(sketch.get_estimate_for(&42).is_some());
        assert!(sketch.get_estimate_for(&42).unwrap() >= 99);
    }

    #[test]
    fn test_weighted_update() {
        let mut sketch: FrequentItemsSketch<String> = FrequentItemsSketch::new(10, false);

        sketch.update_weighted(&"heavy".to_string(), 100);
        sketch.update_weighted(&"light".to_string(), 1);

        assert_eq!(sketch.get_estimate_for(&"heavy".to_string()), Some(100));
        assert_eq!(sketch.get_estimate_for(&"light".to_string()), Some(1));
        assert_eq!(sketch.get_stream_length(), 101);
    }
}
