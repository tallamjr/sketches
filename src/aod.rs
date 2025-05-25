//! Array of Doubles (AOD) Sketch Implementation
//!
//! This module implements the Array of Doubles sketch from Apache DataSketches.
//! AOD sketches extend Theta sketches by associating arrays of double values with each unique key,
//! enabling sophisticated analytics beyond simple cardinality estimation.
//!
//! # Algorithm Overview
//!
//! The AOD sketch uses probabilistic sampling to maintain a representative subset of input data.
//! Each entry consists of:
//! - A hash value (for unique key identification)
//! - An array of double values (summary statistics)
//!
//! The sketch maintains a sampling probability (theta) that determines which entries to retain.
//! When the sketch reaches capacity, it performs statistical sampling to stay within memory bounds.
//!
//! # Features
//!
//! - Cardinality estimation with error bounds
//! - Array-based summary statistics per unique key
//! - Set operations (union, intersection, difference)
//! - Configurable memory usage via capacity parameter
//! - Serialization support for distributed computing

use serde_json;
use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Configuration for AOD sketch creation
#[derive(Debug, Clone)]
pub struct AodConfig {
    /// Maximum number of entries before sampling
    pub capacity: usize,
    /// Number of double values per entry
    pub num_values: usize,
    /// Random seed for reproducible hashing
    pub seed: u64,
}

impl Default for AodConfig {
    fn default() -> Self {
        Self {
            capacity: 4096, // Default size similar to Apache DataSketches
            num_values: 1,
            seed: 0,
        }
    }
}

/// Entry in an AOD sketch containing a hash and array of doubles
#[derive(Debug, Clone, PartialEq)]
pub struct AodEntry {
    /// Hash value of the key
    pub hash: u64,
    /// Array of double values associated with this key
    pub values: Vec<f64>,
}

impl AodEntry {
    pub fn new(hash: u64, values: Vec<f64>) -> Self {
        Self { hash, values }
    }
}

/// Array of Doubles Sketch for cardinality estimation with summary statistics
#[derive(Debug, Clone)]
pub struct AodSketch {
    /// Configuration parameters
    pub config: AodConfig,
    /// Current sampling probability (1.0 = no sampling, < 1.0 = sampling active)
    theta: f64,
    /// Storage for sketch entries
    entries: HashMap<u64, Vec<f64>>,
    /// Whether sketch is empty
    is_empty: bool,
}

impl AodSketch {
    /// Create a new AOD sketch with default configuration
    pub fn new() -> Self {
        Self::with_config(AodConfig::default())
    }

    /// Create a new AOD sketch with custom configuration
    pub fn with_config(config: AodConfig) -> Self {
        Self {
            config,
            theta: 1.0,
            entries: HashMap::new(),
            is_empty: true,
        }
    }

    /// Create AOD sketch with specified capacity and number of values
    pub fn with_capacity_and_values(capacity: usize, num_values: usize) -> Self {
        Self::with_config(AodConfig {
            capacity,
            num_values,
            seed: 0,
        })
    }

    /// Update the sketch with a key and array of values
    pub fn update<T: Hash>(&mut self, key: &T, values: &[f64]) -> Result<(), String> {
        if values.len() != self.config.num_values {
            return Err(format!(
                "Expected {} values, got {}",
                self.config.num_values,
                values.len()
            ));
        }

        let hash = self.hash_key(key);

        // Check if hash should be included based on current theta
        if (hash as f64) / (u64::MAX as f64) <= self.theta {
            self.entries.insert(hash, values.to_vec());
            self.is_empty = false;

            // Resize if we exceed capacity
            if self.entries.len() > self.config.capacity {
                self.resize();
            }
        }

        Ok(())
    }

    /// Hash a key using the configured seed
    fn hash_key<T: Hash>(&self, key: &T) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.config.seed.hash(&mut hasher);
        key.hash(&mut hasher);
        hasher.finish()
    }

    /// Resize the sketch by reducing theta and removing entries
    fn resize(&mut self) {
        if self.entries.len() <= self.config.capacity {
            return;
        }

        // Calculate new theta to reduce sketch size
        let target_size = self.config.capacity * 3 / 4; // Reduce to 75% of capacity
        let mut hashes: Vec<u64> = self.entries.keys().cloned().collect();
        hashes.sort_unstable();

        if hashes.len() > target_size {
            // New theta is determined by the target_size-th smallest hash
            let threshold_hash = hashes[target_size - 1];
            self.theta = (threshold_hash as f64) / (u64::MAX as f64);

            // Remove entries above the new theta threshold
            self.entries
                .retain(|&hash, _| (hash as f64) / (u64::MAX as f64) <= self.theta);
        }
    }

    /// Get estimated number of unique keys
    pub fn estimate(&self) -> f64 {
        if self.is_empty {
            return 0.0;
        }

        if self.theta >= 1.0 {
            // No sampling - exact count
            self.entries.len() as f64
        } else {
            // Sampling active - scale by theta
            (self.entries.len() as f64) / self.theta
        }
    }

    /// Get upper bound estimate with given confidence
    pub fn upper_bound(&self, confidence: f64) -> f64 {
        if self.is_empty {
            return 0.0;
        }

        let estimate = self.estimate();

        if self.theta >= 1.0 {
            return estimate; // Exact count
        }

        // Calculate confidence interval based on binomial distribution
        let std_dev = (estimate * (1.0 - self.theta) / self.theta).sqrt();
        let z_score = match confidence {
            x if x >= 0.99 => 2.576,
            x if x >= 0.95 => 1.96,
            x if x >= 0.90 => 1.645,
            _ => 1.0,
        };

        estimate + z_score * std_dev
    }

    /// Get lower bound estimate with given confidence
    pub fn lower_bound(&self, confidence: f64) -> f64 {
        if self.is_empty {
            return 0.0;
        }

        let estimate = self.estimate();

        if self.theta >= 1.0 {
            return estimate; // Exact count
        }

        // Calculate confidence interval
        let std_dev = (estimate * (1.0 - self.theta) / self.theta).sqrt();
        let z_score = match confidence {
            x if x >= 0.99 => 2.576,
            x if x >= 0.95 => 1.96,
            x if x >= 0.90 => 1.645,
            _ => 1.0,
        };

        (estimate - z_score * std_dev).max(0.0)
    }

    /// Get current theta (sampling probability)
    pub fn theta(&self) -> f64 {
        self.theta
    }

    /// Check if sketch is empty
    pub fn is_empty(&self) -> bool {
        self.is_empty
    }

    /// Get number of entries currently stored
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Get number of values per entry
    pub fn num_values(&self) -> usize {
        self.config.num_values
    }

    /// Get iterator over sketch entries
    pub fn iter(&self) -> impl Iterator<Item = AodEntry> + '_ {
        self.entries.iter().map(|(&hash, values)| AodEntry {
            hash,
            values: values.clone(),
        })
    }

    /// Calculate sum of values for each column across all entries
    pub fn column_sums(&self) -> Vec<f64> {
        let mut sums = vec![0.0; self.config.num_values];

        for values in self.entries.values() {
            for (i, &value) in values.iter().enumerate() {
                sums[i] += value;
            }
        }

        // Scale by theta if sampling is active
        if self.theta < 1.0 {
            for sum in &mut sums {
                *sum /= self.theta;
            }
        }

        sums
    }

    /// Calculate mean of values for each column
    pub fn column_means(&self) -> Vec<f64> {
        if self.is_empty {
            return vec![0.0; self.config.num_values];
        }

        let sums = self.column_sums();
        let estimate = self.estimate();

        sums.into_iter().map(|sum| sum / estimate).collect()
    }

    /// Union this sketch with another AOD sketch
    pub fn union(&mut self, other: &AodSketch) -> Result<(), String> {
        if self.config.num_values != other.config.num_values {
            return Err("Cannot union sketches with different number of values".to_string());
        }

        // Use minimum theta
        let min_theta = self.theta.min(other.theta);

        // Add entries from other sketch that pass the theta test
        for (&hash, values) in &other.entries {
            if (hash as f64) / (u64::MAX as f64) <= min_theta {
                self.entries.insert(hash, values.clone());
            }
        }

        // Remove our entries that don't pass the new theta
        if min_theta < self.theta {
            self.entries
                .retain(|&hash, _| (hash as f64) / (u64::MAX as f64) <= min_theta);
        }

        self.theta = min_theta;
        self.is_empty = self.entries.is_empty();

        // Resize if necessary
        if self.entries.len() > self.config.capacity {
            self.resize();
        }

        Ok(())
    }

    /// Create a compact, immutable copy of this sketch
    pub fn compact(&self) -> AodSketch {
        self.clone()
    }

    /// Serialize sketch to bytes (simple implementation)
    pub fn to_bytes(&self) -> Vec<u8> {
        // This is a simplified serialization - in production you'd want a more efficient format
        let serialized = serde_json::json!({
            "config": {
                "capacity": self.config.capacity,
                "num_values": self.config.num_values,
                "seed": self.config.seed,
            },
            "theta": self.theta,
            "is_empty": self.is_empty,
            "entries": self.entries,
        });

        serde_json::to_vec(&serialized).unwrap_or_default()
    }

    /// Deserialize sketch from bytes (simple implementation)
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        let data: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("Failed to parse JSON: {}", e))?;

        let config = AodConfig {
            capacity: data["config"]["capacity"].as_u64().unwrap_or(4096) as usize,
            num_values: data["config"]["num_values"].as_u64().unwrap_or(1) as usize,
            seed: data["config"]["seed"].as_u64().unwrap_or(0),
        };

        let theta = data["theta"].as_f64().unwrap_or(1.0);
        let is_empty = data["is_empty"].as_bool().unwrap_or(true);

        let mut entries = HashMap::new();
        if let Some(entries_obj) = data["entries"].as_object() {
            for (key, value) in entries_obj {
                if let (Ok(hash), Some(values_array)) = (key.parse::<u64>(), value.as_array()) {
                    let values: Vec<f64> = values_array.iter().filter_map(|v| v.as_f64()).collect();
                    entries.insert(hash, values);
                }
            }
        }

        Ok(Self {
            config,
            theta,
            entries,
            is_empty,
        })
    }
}

impl Default for AodSketch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aod_basic() {
        let mut sketch = AodSketch::with_capacity_and_values(100, 2);

        sketch.update(&"key1", &[1.0, 2.0]).unwrap();
        sketch.update(&"key2", &[3.0, 4.0]).unwrap();
        sketch.update(&"key3", &[5.0, 6.0]).unwrap();

        assert_eq!(sketch.len(), 3);
        assert_eq!(sketch.estimate(), 3.0);
        assert!(!sketch.is_empty());
        assert_eq!(sketch.num_values(), 2);
    }

    #[test]
    fn test_aod_wrong_value_count() {
        let mut sketch = AodSketch::with_capacity_and_values(100, 2);
        let result = sketch.update(&"key1", &[1.0]); // Wrong number of values
        assert!(result.is_err());
    }

    #[test]
    fn test_aod_column_operations() {
        let mut sketch = AodSketch::with_capacity_and_values(100, 3);

        sketch.update(&"key1", &[1.0, 2.0, 3.0]).unwrap();
        sketch.update(&"key2", &[4.0, 5.0, 6.0]).unwrap();
        sketch.update(&"key3", &[7.0, 8.0, 9.0]).unwrap();

        let sums = sketch.column_sums();
        assert_eq!(sums, vec![12.0, 15.0, 18.0]);

        let means = sketch.column_means();
        assert_eq!(means, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_aod_union() {
        let mut sketch1 = AodSketch::with_capacity_and_values(100, 2);
        let mut sketch2 = AodSketch::with_capacity_and_values(100, 2);

        sketch1.update(&"key1", &[1.0, 2.0]).unwrap();
        sketch1.update(&"key2", &[3.0, 4.0]).unwrap();

        sketch2.update(&"key2", &[3.0, 4.0]).unwrap(); // Duplicate
        sketch2.update(&"key3", &[5.0, 6.0]).unwrap();

        sketch1.union(&sketch2).unwrap();

        assert_eq!(sketch1.len(), 3); // key1, key2, key3
        assert_eq!(sketch1.estimate(), 3.0);
    }

    #[test]
    fn test_aod_sampling() {
        let mut sketch = AodSketch::with_capacity_and_values(10, 1); // Small capacity to force sampling

        // Add many entries to trigger sampling
        for i in 0..100 {
            sketch.update(&format!("key{}", i), &[i as f64]).unwrap();
        }

        // Should have triggered sampling
        assert!(sketch.theta() < 1.0);
        assert!(sketch.len() <= 10);

        // Estimate should be higher than actual count due to sampling
        let estimate = sketch.estimate();
        assert!(estimate > sketch.len() as f64);
        assert!(estimate <= 100.0);
    }

    #[test]
    fn test_aod_bounds() {
        let mut sketch = AodSketch::with_capacity_and_values(100, 1);

        for i in 0..50 {
            sketch.update(&format!("key{}", i), &[i as f64]).unwrap();
        }

        let estimate = sketch.estimate();
        let lower = sketch.lower_bound(0.95);
        let upper = sketch.upper_bound(0.95);

        assert!(lower <= estimate);
        assert!(estimate <= upper);
        assert!(lower >= 0.0);
    }

    #[test]
    fn test_aod_serialization() {
        let mut sketch = AodSketch::with_capacity_and_values(100, 2);

        sketch.update(&"key1", &[1.0, 2.0]).unwrap();
        sketch.update(&"key2", &[3.0, 4.0]).unwrap();

        let bytes = sketch.to_bytes();
        let deserialized = AodSketch::from_bytes(&bytes).unwrap();

        assert_eq!(sketch.len(), deserialized.len());
        assert_eq!(sketch.estimate(), deserialized.estimate());
        assert_eq!(sketch.num_values(), deserialized.num_values());
        assert_eq!(sketch.column_sums(), deserialized.column_sums());
    }

    #[test]
    fn test_aod_iterator() {
        let mut sketch = AodSketch::with_capacity_and_values(100, 2);

        sketch.update(&"key1", &[1.0, 2.0]).unwrap();
        sketch.update(&"key2", &[3.0, 4.0]).unwrap();

        let entries: Vec<AodEntry> = sketch.iter().collect();
        assert_eq!(entries.len(), 2);

        for entry in entries {
            assert_eq!(entry.values.len(), 2);
        }
    }

    #[test]
    fn test_aod_empty_sketch() {
        let sketch = AodSketch::new();

        assert!(sketch.is_empty());
        assert_eq!(sketch.len(), 0);
        assert_eq!(sketch.estimate(), 0.0);
        assert_eq!(sketch.upper_bound(0.95), 0.0);
        assert_eq!(sketch.lower_bound(0.95), 0.0);
        assert_eq!(sketch.theta(), 1.0);
    }
}
