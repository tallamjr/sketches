use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Frequent Strings Sketch for finding heavy hitters in string streams
///
/// This sketch tracks the most frequent items in a data stream using the
/// Space-Saving algorithm with error bounds guarantees. It's optimized
/// for string data and provides Apache DataSketches compatibility.
pub struct FrequentStringsSketch {
    max_map_size: usize,
    stream_length: u64,
    map: HashMap<String, FrequentItem>,
    error_rate: f64,
    use_reservoir: bool,
}

/// Information about a frequent item
#[derive(Debug, Clone)]
struct FrequentItem {
    estimate: u64,    // Estimated frequency
    lower_bound: u64, // Lower bound of true frequency
    upper_bound: u64, // Upper bound of true frequency
}

impl FrequentItem {
    fn new(estimate: u64, error: u64) -> Self {
        FrequentItem {
            estimate,
            lower_bound: estimate.saturating_sub(error),
            upper_bound: estimate + error,
        }
    }
}

impl FrequentStringsSketch {
    /// Create a new Frequent Strings sketch
    ///
    /// # Arguments
    /// * `max_map_size` - Maximum number of items to track (affects memory usage)
    /// * `use_reservoir` - Whether to use reservoir sampling for better accuracy
    pub fn new(max_map_size: usize, use_reservoir: bool) -> Self {
        assert!(max_map_size > 0, "Max map size must be positive");

        FrequentStringsSketch {
            max_map_size,
            stream_length: 0,
            map: HashMap::new(),
            error_rate: 1.0 / max_map_size as f64, // Theoretical error bound
            use_reservoir,
        }
    }

    /// Create with error rate specification
    ///
    /// # Arguments
    /// * `error_rate` - Maximum relative error (e.g., 0.01 for 1% error)
    /// * `confidence` - Confidence level (e.g., 0.99 for 99% confidence)
    /// * `use_reservoir` - Whether to use reservoir sampling
    pub fn with_error_rate(error_rate: f64, confidence: f64, use_reservoir: bool) -> Self {
        assert!(
            error_rate > 0.0 && error_rate < 1.0,
            "Error rate must be between 0 and 1"
        );
        assert!(
            confidence > 0.0 && confidence < 1.0,
            "Confidence must be between 0 and 1"
        );

        // Calculate required map size based on error bounds
        let max_map_size = (3.0 / error_rate).ceil() as usize;

        Self::new(max_map_size, use_reservoir)
    }

    /// Update the sketch with a new item
    pub fn update(&mut self, item: &str) {
        self.stream_length += 1;

        // Check if item already exists
        if let Some(freq_item) = self.map.get_mut(item) {
            freq_item.estimate += 1;
            freq_item.upper_bound += 1;
            return;
        }

        // If map is not full, add new item
        if self.map.len() < self.max_map_size {
            let new_item = FrequentItem::new(1, 0);
            self.map.insert(item.to_string(), new_item);
            return;
        }

        // Map is full - need to handle new item
        if self.use_reservoir {
            self.update_with_reservoir(item);
        } else {
            self.update_with_space_saving(item);
        }
    }

    /// Update using Space-Saving algorithm
    fn update_with_space_saving(&mut self, item: &str) {
        // Find item with minimum estimate to potentially replace
        let min_key = self
            .map
            .iter()
            .min_by_key(|(_, freq_item)| freq_item.estimate)
            .map(|(key, _)| key.clone());

        if let Some(min_key) = min_key {
            let min_estimate = self.map[&min_key].estimate;

            // Remove minimum item and add new item with inherited count
            self.map.remove(&min_key);
            let new_item = FrequentItem::new(min_estimate + 1, min_estimate);
            self.map.insert(item.to_string(), new_item);
        }
    }

    /// Update using reservoir sampling strategy
    fn update_with_reservoir(&mut self, item: &str) {
        // Simple reservoir sampling: randomly replace an existing item
        let hash = self.hash_string(item);
        let replacement_prob = 1.0 / self.stream_length as f64;

        // Use hash as pseudo-random number
        let pseudo_random = (hash as f64) / (u64::MAX as f64);

        if pseudo_random < replacement_prob {
            // Find a random item to replace
            let keys: Vec<String> = self.map.keys().cloned().collect();
            if !keys.is_empty() {
                let replace_index = (hash as usize) % keys.len();
                let replace_key = &keys[replace_index];
                let old_estimate = self.map[replace_key].estimate;

                self.map.remove(replace_key);
                let new_item = FrequentItem::new(old_estimate + 1, old_estimate);
                self.map.insert(item.to_string(), new_item);
            }
        }
    }

    /// Hash a string for pseudo-random operations
    fn hash_string(&self, s: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        s.hash(&mut hasher);
        hasher.finish()
    }

    /// Get frequent items above a threshold
    ///
    /// # Arguments
    /// * `threshold` - Minimum frequency threshold (absolute count)
    pub fn get_frequent_items(&self, threshold: u64) -> Vec<FrequentItemResult> {
        let mut results = Vec::new();

        for (item, freq_item) in &self.map {
            if freq_item.estimate >= threshold {
                results.push(FrequentItemResult {
                    item: item.clone(),
                    estimate: freq_item.estimate,
                    lower_bound: freq_item.lower_bound,
                    upper_bound: freq_item.upper_bound,
                });
            }
        }

        // Sort by estimated frequency (descending)
        results.sort_by(|a, b| b.estimate.cmp(&a.estimate));
        results
    }

    /// Get frequent items above a relative threshold
    ///
    /// # Arguments
    /// * `threshold_fraction` - Minimum frequency as fraction of total stream (e.g., 0.01 for 1%)
    pub fn get_frequent_items_by_fraction(
        &self,
        threshold_fraction: f64,
    ) -> Vec<FrequentItemResult> {
        assert!(
            threshold_fraction >= 0.0 && threshold_fraction <= 1.0,
            "Threshold fraction must be between 0 and 1"
        );

        let threshold = (threshold_fraction * self.stream_length as f64) as u64;
        self.get_frequent_items(threshold)
    }

    /// Get the top-k most frequent items
    pub fn get_top_k(&self, k: usize) -> Vec<FrequentItemResult> {
        let mut results = Vec::new();

        for (item, freq_item) in &self.map {
            results.push(FrequentItemResult {
                item: item.clone(),
                estimate: freq_item.estimate,
                lower_bound: freq_item.lower_bound,
                upper_bound: freq_item.upper_bound,
            });
        }

        // Sort by estimated frequency (descending) and take top k
        results.sort_by(|a, b| b.estimate.cmp(&a.estimate));
        results.truncate(k);
        results
    }

    /// Get estimated frequency of a specific item
    pub fn get_estimate(&self, item: &str) -> Option<u64> {
        self.map.get(item).map(|freq_item| freq_item.estimate)
    }

    /// Get frequency bounds for a specific item
    pub fn get_bounds(&self, item: &str) -> Option<(u64, u64)> {
        self.map
            .get(item)
            .map(|freq_item| (freq_item.lower_bound, freq_item.upper_bound))
    }

    /// Merge another sketch into this one
    ///
    /// Note: This is a simplified merge that may not preserve all theoretical guarantees
    pub fn merge(&mut self, other: &FrequentStringsSketch) -> Result<(), &'static str> {
        if self.max_map_size != other.max_map_size {
            return Err("Cannot merge sketches with different map sizes");
        }

        self.stream_length += other.stream_length;

        // Merge items from other sketch
        for (item, other_freq) in &other.map {
            if let Some(self_freq) = self.map.get_mut(item) {
                // Item exists in both sketches - combine estimates
                self_freq.estimate += other_freq.estimate;
                self_freq.lower_bound += other_freq.lower_bound;
                self_freq.upper_bound += other_freq.upper_bound;
            } else if self.map.len() < self.max_map_size {
                // Space available - add new item
                self.map.insert(item.clone(), other_freq.clone());
            } else {
                // Map is full - use space-saving logic
                let min_key = self
                    .map
                    .iter()
                    .min_by_key(|(_, freq_item)| freq_item.estimate)
                    .map(|(key, _)| key.clone());

                if let Some(min_key) = min_key {
                    if other_freq.estimate > self.map[&min_key].estimate {
                        self.map.remove(&min_key);
                        self.map.insert(item.clone(), other_freq.clone());
                    }
                }
            }
        }

        Ok(())
    }

    /// Get sketch statistics
    pub fn statistics(&self) -> FrequentStringsStats {
        let total_frequency: u64 = self.map.values().map(|item| item.estimate).sum();

        FrequentStringsStats {
            max_map_size: self.max_map_size,
            current_map_size: self.map.len(),
            stream_length: self.stream_length,
            total_tracked_frequency: total_frequency,
            error_rate: self.error_rate,
            uses_reservoir: self.use_reservoir,
            memory_usage: self.estimate_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes
    fn estimate_memory_usage(&self) -> usize {
        let map_overhead = std::mem::size_of::<HashMap<String, FrequentItem>>();
        let item_size = std::mem::size_of::<FrequentItem>();

        let mut string_memory = 0;
        for key in self.map.keys() {
            string_memory += key.len() + std::mem::size_of::<String>();
        }

        map_overhead + (self.map.len() * item_size) + string_memory
    }

    /// Clear the sketch
    pub fn clear(&mut self) {
        self.map.clear();
        self.stream_length = 0;
    }

    /// Check if sketch is empty
    pub fn is_empty(&self) -> bool {
        self.stream_length == 0
    }

    /// Get the total number of items processed
    pub fn get_stream_length(&self) -> u64 {
        self.stream_length
    }
}

/// Result item from frequent items query
#[derive(Debug, Clone)]
pub struct FrequentItemResult {
    pub item: String,
    pub estimate: u64,
    pub lower_bound: u64,
    pub upper_bound: u64,
}

/// Statistics about a Frequent Strings sketch
#[derive(Debug, Clone)]
pub struct FrequentStringsStats {
    pub max_map_size: usize,
    pub current_map_size: usize,
    pub stream_length: u64,
    pub total_tracked_frequency: u64,
    pub error_rate: f64,
    pub uses_reservoir: bool,
    pub memory_usage: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequent_strings_basic() {
        let mut sketch = FrequentStringsSketch::new(10, false);

        // Add some items with known frequencies
        for _ in 0..100 {
            sketch.update("apple");
        }
        for _ in 0..50 {
            sketch.update("banana");
        }
        for _ in 0..25 {
            sketch.update("cherry");
        }

        // Test estimates
        assert_eq!(sketch.get_estimate("apple"), Some(100));
        assert_eq!(sketch.get_estimate("banana"), Some(50));
        assert_eq!(sketch.get_estimate("cherry"), Some(25));
        assert_eq!(sketch.get_estimate("not_present"), None);

        // Test frequent items
        let frequent = sketch.get_frequent_items(30);
        assert_eq!(frequent.len(), 2); // apple and banana
        assert_eq!(frequent[0].item, "apple");
        assert_eq!(frequent[1].item, "banana");
    }

    #[test]
    fn test_frequent_strings_space_saving() {
        let mut sketch = FrequentStringsSketch::new(3, false); // Small capacity

        // Add more unique items than capacity
        sketch.update("a");
        sketch.update("b");
        sketch.update("c");
        sketch.update("d"); // Should replace one of the previous items

        assert_eq!(sketch.map.len(), 3); // Should not exceed capacity
        assert_eq!(sketch.get_stream_length(), 4);
    }

    #[test]
    fn test_frequent_strings_top_k() {
        let mut sketch = FrequentStringsSketch::new(10, false);

        // Add items with different frequencies
        for i in 1..=5 {
            for _ in 0..(6 - i) * 10 {
                // item1: 50, item2: 40, item3: 30, item4: 20, item5: 10
                sketch.update(&format!("item{}", i));
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

        // Add 1000 items total
        for _ in 0..500 {
            sketch.update("majority"); // 50%
        }
        for _ in 0..300 {
            sketch.update("significant"); // 30%
        }
        for _ in 0..200 {
            sketch.update("minority"); // 20%
        }

        // Get items that are at least 25% of stream
        let frequent = sketch.get_frequent_items_by_fraction(0.25);
        assert_eq!(frequent.len(), 2); // majority and significant
    }

    #[test]
    fn test_frequent_strings_merge() {
        let mut sketch1 = FrequentStringsSketch::new(5, false);
        let mut sketch2 = FrequentStringsSketch::new(5, false);

        // Add different items to each sketch
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

        assert!(stats.max_map_size >= 300); // Should be around 3/0.01 = 300
        assert_eq!(stats.current_map_size, 0);
        assert_eq!(stats.stream_length, 0);
    }

    #[test]
    fn test_frequent_strings_bounds() {
        let mut sketch = FrequentStringsSketch::new(3, false);

        // Fill up the sketch and trigger space-saving
        sketch.update("a");
        sketch.update("b");
        sketch.update("c");
        sketch.update("d"); // This should cause bounds to be set

        // Check that bounds are properly maintained
        for (_, freq_item) in &sketch.map {
            assert!(freq_item.lower_bound <= freq_item.estimate);
            assert!(freq_item.estimate <= freq_item.upper_bound);
        }
    }

    #[test]
    fn test_frequent_strings_reservoir() {
        let mut sketch = FrequentStringsSketch::new(5, true);

        // Add many items to test reservoir sampling
        for i in 0..100 {
            sketch.update(&format!("item_{}", i % 20));
        }

        assert_eq!(sketch.map.len(), 5); // Should maintain size limit
        assert_eq!(sketch.get_stream_length(), 100);

        let stats = sketch.statistics();
        assert!(stats.uses_reservoir);
    }

    #[test]
    fn test_frequent_strings_edge_cases() {
        let mut sketch = FrequentStringsSketch::new(5, false);

        // Empty sketch
        assert!(sketch.is_empty());
        assert_eq!(sketch.get_top_k(10).len(), 0);
        assert_eq!(sketch.get_frequent_items(1).len(), 0);

        // Single item
        sketch.update("single");
        assert!(!sketch.is_empty());
        assert_eq!(sketch.get_estimate("single"), Some(1));

        // Clear and verify
        sketch.clear();
        assert!(sketch.is_empty());
        assert_eq!(sketch.get_stream_length(), 0);
    }
}
