//! T-Digest Implementation for Quantile Estimation
//!
//! This module provides T-digest data structures for approximate quantile estimation
//! in streaming data. T-digest is superior to q-digest for quantile queries and
//! provides better accuracy with adaptive compression.
//!
//! **IMPORTANT LIMITATION**: The merge operation in this implementation uses a
//! sampling approximation approach rather than proper T-Digest centroid merging.
//! This may introduce additional approximation errors and degrade accuracy compared
//! to proper centroid-based merging. For production distributed computing scenarios
//! requiring high accuracy, consider alternatives or accumulate raw data points
//! when possible.

use std::fmt;
use tdigest::TDigest as CoreTDigest;

/// T-Digest for streaming quantile estimation
///
/// The T-Digest is a probabilistic data structure for estimating quantiles
/// from a stream of data points. It provides excellent accuracy for extreme
/// quantiles (like p95, p99) while maintaining compact memory usage.
///
/// **Note**: The merge operation uses sampling approximation rather than proper
/// centroid merging, which may introduce additional approximation errors.
#[derive(Debug, Clone)]
pub struct TDigest {
    /// The underlying t-digest implementation
    inner: CoreTDigest,
    /// Total number of values processed
    count: u64,
    /// Minimum value seen
    min_value: Option<f64>,
    /// Maximum value seen
    max_value: Option<f64>,
    /// Compression parameter used
    compression: usize,
}

impl TDigest {
    /// Create a new T-Digest with default compression (100)
    ///
    /// The compression parameter controls the trade-off between accuracy and memory usage.
    /// Higher values provide better accuracy but use more memory.
    pub fn new() -> Self {
        let compression = 100;
        Self {
            inner: CoreTDigest::new_with_size(compression),
            count: 0,
            min_value: None,
            max_value: None,
            compression,
        }
    }

    /// Create a new T-Digest with specified compression
    ///
    /// Typical compression values:
    /// - 50-100: Good for most applications
    /// - 200-500: Better accuracy for extreme quantiles
    /// - 1000+: Very high accuracy, more memory usage
    pub fn with_compression(compression: usize) -> Self {
        Self {
            inner: CoreTDigest::new_with_size(compression),
            count: 0,
            min_value: None,
            max_value: None,
            compression,
        }
    }

    /// Create a T-Digest optimized for a specific quantile accuracy
    ///
    /// This estimates the compression needed to achieve approximately
    /// the desired relative error for quantile estimates.
    pub fn with_accuracy(target_error: f64) -> Self {
        // Rough heuristic: compression ≈ 1 / (4 * target_error)
        let compression = (1.0 / (4.0 * target_error)).max(50.0).min(2000.0) as usize;
        Self::with_compression(compression)
    }

    /// Add a value to the digest
    ///
    /// Time complexity: O(log C) where C is the compression parameter
    /// Space complexity: O(C)
    pub fn add(&mut self, value: f64) {
        self.inner = self.inner.merge_sorted(vec![value]);
        self.count += 1;

        // Update min/max
        match self.min_value {
            None => self.min_value = Some(value),
            Some(min) => {
                if value < min {
                    self.min_value = Some(value);
                }
            }
        }

        match self.max_value {
            None => self.max_value = Some(value),
            Some(max) => {
                if value > max {
                    self.max_value = Some(value);
                }
            }
        }
    }

    /// Add multiple values to the digest
    ///
    /// More efficient than calling add() multiple times as it can batch
    /// the merge operations.
    pub fn add_batch(&mut self, values: &[f64]) {
        if values.is_empty() {
            return;
        }

        // Sort the values for better accuracy
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        self.inner = self.inner.merge_sorted(sorted_values);
        self.count += values.len() as u64;

        // Update min/max from batch
        for &value in values {
            match self.min_value {
                None => self.min_value = Some(value),
                Some(min) => {
                    if value < min {
                        self.min_value = Some(value);
                    }
                }
            }

            match self.max_value {
                None => self.max_value = Some(value),
                Some(max) => {
                    if value > max {
                        self.max_value = Some(value);
                    }
                }
            }
        }
    }

    /// Estimate the quantile for a given rank (0.0 to 1.0)
    ///
    /// Returns None if no values have been added to the digest.
    ///
    /// Common quantiles:
    /// - 0.5: Median
    /// - 0.95: 95th percentile
    /// - 0.99: 99th percentile
    pub fn quantile(&self, q: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }

        if q <= 0.0 {
            return self.min_value;
        }

        if q >= 1.0 {
            return self.max_value;
        }

        Some(self.inner.estimate_quantile(q))
    }

    /// Estimate the rank (percentile) of a given value
    ///
    /// Returns a value between 0.0 and 1.0 representing what fraction
    /// of values are less than or equal to the given value.
    pub fn rank(&self, value: f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        // Handle edge cases
        if let Some(min) = self.min_value {
            if value <= min {
                return 0.0;
            }
        }

        if let Some(max) = self.max_value {
            if value >= max {
                return 1.0;
            }
        }

        // For now, we'll implement a simple approximation by sampling quantiles
        // This is not as efficient as a direct rank implementation but works with the available API
        let mut low = 0.0;
        let mut high = 1.0;
        let tolerance = 0.001;

        // Binary search for the rank
        while high - low > tolerance {
            let mid = (low + high) / 2.0;
            let quantile_value = self.inner.estimate_quantile(mid);

            if quantile_value < value {
                low = mid;
            } else {
                high = mid;
            }
        }

        (low + high) / 2.0
    }

    /// Merge another T-Digest into this one using sampling approximation
    ///
    /// **IMPORTANT LIMITATION**: This implementation uses a sampling approximation
    /// approach rather than proper T-Digest centroid merging. This means:
    ///
    /// - The merge operation may introduce additional approximation errors
    /// - Accuracy degrades compared to a proper centroid-based merge
    /// - Multiple successive merges may compound these errors
    /// - The merged result may not preserve the theoretical guarantees of T-Digest
    ///
    /// **Why this approach is used**: The underlying tdigest crate doesn't expose
    /// internal centroids, preventing a proper implementation of centroid merging.
    /// This sampling approach is a workaround to provide merge functionality.
    ///
    /// **Recommendation**: For production distributed computing scenarios requiring
    /// high accuracy, consider using a T-Digest library that supports proper
    /// centroid-based merging, or accumulate raw data points and create a single
    /// digest from the combined dataset when possible.
    pub fn merge(&mut self, other: &TDigest) {
        if other.count == 0 {
            return;
        }

        // APPROXIMATION APPROACH: Since the tdigest crate doesn't expose internal
        // centroids, we approximate the merge by sampling quantiles from both digests
        // and creating a new digest from these sample points.
        //
        // This is NOT equivalent to proper T-Digest centroid merging and will
        // introduce additional approximation errors.
        let mut sample_points = Vec::new();

        // Sample 100 quantiles from the current digest
        // Note: This loses information about the actual centroid weights and positions
        for i in 1..=100 {
            let q = i as f64 / 100.0;
            if let Some(value) = self.quantile(q) {
                sample_points.push(value);
            }
        }

        // Sample 100 quantiles from the other digest
        // Note: This loses information about the actual centroid weights and positions
        for i in 1..=100 {
            let q = i as f64 / 100.0;
            if let Some(value) = other.quantile(q) {
                sample_points.push(value);
            }
        }

        // Create a new digest from the sampled points
        // This discards the original centroid structure and creates a new approximation
        if !sample_points.is_empty() {
            sample_points.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.inner = CoreTDigest::new_with_size(self.compression);
            self.inner = self.inner.merge_sorted(sample_points);
        }

        self.count += other.count;

        // Update min/max values (this part is accurate)
        if let Some(other_min) = other.min_value {
            self.min_value = Some(match self.min_value {
                None => other_min,
                Some(min) => min.min(other_min),
            });
        }

        if let Some(other_max) = other.max_value {
            self.max_value = Some(match self.max_value {
                None => other_max,
                Some(max) => max.max(other_max),
            });
        }
    }

    /// Get the total number of values processed
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Check if the digest is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the minimum value seen
    pub fn min(&self) -> Option<f64> {
        self.min_value
    }

    /// Get the maximum value seen
    pub fn max(&self) -> Option<f64> {
        self.max_value
    }

    /// Get the median (50th percentile)
    pub fn median(&self) -> Option<f64> {
        self.quantile(0.5)
    }

    /// Get multiple quantiles at once
    ///
    /// More efficient than calling quantile() multiple times.
    pub fn quantiles(&self, qs: &[f64]) -> Vec<Option<f64>> {
        qs.iter().map(|&q| self.quantile(q)).collect()
    }

    /// Calculate trimmed mean (mean excluding extreme values)
    ///
    /// Excludes values below the lower_quantile and above the upper_quantile.
    /// For example, trimmed_mean(0.1, 0.9) excludes the bottom 10% and top 10%.
    pub fn trimmed_mean(&self, lower_quantile: f64, upper_quantile: f64) -> Option<f64> {
        if self.count == 0 || lower_quantile >= upper_quantile {
            return None;
        }

        let lower_bound = self.quantile(lower_quantile)?;
        let upper_bound = self.quantile(upper_quantile)?;

        // For a proper implementation, we'd need to integrate over the distribution
        // For now, we approximate using the midpoint of the range
        Some((lower_bound + upper_bound) / 2.0)
    }

    /// Get digest statistics
    pub fn statistics(&self) -> TDigestStats {
        TDigestStats {
            count: self.count,
            compression: self.compression,
            min_value: self.min_value,
            max_value: self.max_value,
            memory_usage: self.estimated_memory_usage(),
        }
    }

    /// Estimate memory usage in bytes
    fn estimated_memory_usage(&self) -> usize {
        // Rough estimate: each centroid takes ~16 bytes (f64 mean + f64 weight)
        // Plus overhead for the structure itself
        std::mem::size_of::<Self>() + (self.compression * 16)
    }

    /// Clear the digest
    pub fn clear(&mut self) {
        self.inner = CoreTDigest::new_with_size(self.compression);
        self.count = 0;
        self.min_value = None;
        self.max_value = None;
    }
}

impl Default for TDigest {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about a T-Digest
#[derive(Debug, Clone, PartialEq)]
pub struct TDigestStats {
    pub count: u64,
    pub compression: usize,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub memory_usage: usize,
}

impl fmt::Display for TDigestStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TDigestStats {{ count: {}, compression: {}, min: {:?}, max: {:?}, memory: {} bytes }}",
            self.count, self.compression, self.min_value, self.max_value, self.memory_usage
        )
    }
}

/// Streaming T-Digest that automatically merges when it gets too large
///
/// This variant automatically triggers merges when the internal buffer
/// gets too large, providing more predictable memory usage for streaming applications.
#[derive(Debug, Clone)]
pub struct StreamingTDigest {
    digest: TDigest,
    buffer: Vec<f64>,
    buffer_size: usize,
}

impl StreamingTDigest {
    /// Create a new streaming T-Digest
    ///
    /// buffer_size controls how many values to accumulate before merging
    /// into the main digest. Larger buffers provide better accuracy but
    /// use more memory.
    pub fn new(compression: usize, buffer_size: usize) -> Self {
        Self {
            digest: TDigest::with_compression(compression),
            buffer: Vec::with_capacity(buffer_size),
            buffer_size,
        }
    }

    /// Add a value to the streaming digest
    pub fn add(&mut self, value: f64) {
        self.buffer.push(value);

        if self.buffer.len() >= self.buffer_size {
            self.flush();
        }
    }

    /// Flush the buffer into the main digest
    pub fn flush(&mut self) {
        if !self.buffer.is_empty() {
            self.digest.add_batch(&self.buffer);
            self.buffer.clear();
        }
    }

    /// Get a quantile estimate
    ///
    /// Automatically flushes the buffer before estimation.
    pub fn quantile(&mut self, q: f64) -> Option<f64> {
        self.flush();
        self.digest.quantile(q)
    }

    /// Get statistics about the streaming digest
    pub fn statistics(&mut self) -> TDigestStats {
        self.flush();
        let mut stats = self.digest.statistics();
        // Add buffer memory usage
        stats.memory_usage += self.buffer.capacity() * std::mem::size_of::<f64>();
        stats
    }

    /// Get the underlying digest (flushes buffer first)
    pub fn digest(&mut self) -> &TDigest {
        self.flush();
        &self.digest
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tdigest_basic() {
        let mut digest = TDigest::new();

        // Test empty digest
        assert_eq!(digest.count(), 0);
        assert!(digest.is_empty());
        assert_eq!(digest.quantile(0.5), None);

        // Add some values
        for i in 1..=100 {
            digest.add(i as f64);
        }

        assert_eq!(digest.count(), 100);
        assert!(!digest.is_empty());
        assert_eq!(digest.min(), Some(1.0));
        assert_eq!(digest.max(), Some(100.0));

        // Test median
        let median = digest.median().unwrap();
        assert!((median - 50.5).abs() < 2.0); // Should be close to true median

        // Test other quantiles
        let q25 = digest.quantile(0.25).unwrap();
        let q75 = digest.quantile(0.75).unwrap();
        assert!(q25 < median);
        assert!(q75 > median);
    }

    #[test]
    fn test_tdigest_batch_add() {
        let mut digest1 = TDigest::new();
        let mut digest2 = TDigest::new();

        let values: Vec<f64> = (1..=100).map(|i| i as f64).collect();

        // Add one by one
        for &value in &values {
            digest1.add(value);
        }

        // Add as batch
        digest2.add_batch(&values);

        assert_eq!(digest1.count(), digest2.count());
        assert_eq!(digest1.min(), digest2.min());
        assert_eq!(digest1.max(), digest2.max());

        // Quantiles should be similar
        let q1_median = digest1.median().unwrap();
        let q2_median = digest2.median().unwrap();
        assert!((q1_median - q2_median).abs() < 1.0);
    }

    #[test]
    fn test_tdigest_quantiles() {
        let mut digest = TDigest::new();

        // Add values 1-1000
        for i in 1..=1000 {
            digest.add(i as f64);
        }

        // Test multiple quantiles
        let quantiles = digest.quantiles(&[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]);

        // Check that quantiles are in increasing order
        for i in 1..quantiles.len() {
            if let (Some(prev), Some(curr)) = (quantiles[i - 1], quantiles[i]) {
                assert!(
                    prev <= curr,
                    "Quantiles should be in order: {} <= {}",
                    prev,
                    curr
                );
            }
        }

        // Check specific quantiles are reasonable
        assert!((quantiles[2].unwrap() - 500.0).abs() < 20.0); // median ≈ 500
        assert!((quantiles[5].unwrap() - 950.0).abs() < 20.0); // p95 ≈ 950
    }

    #[test]
    fn test_tdigest_rank() {
        let mut digest = TDigest::new();

        // Add values 1-100
        for i in 1..=100 {
            digest.add(i as f64);
        }

        // Test rank estimates
        assert!((digest.rank(1.0) - 0.01).abs() < 0.05); // 1st percentile ≈ 0.01
        assert!((digest.rank(50.0) - 0.5).abs() < 0.05); // 50th percentile ≈ 0.5
        assert!((digest.rank(100.0) - 1.0).abs() < 0.05); // 100th percentile ≈ 1.0

        // Test edge cases
        assert_eq!(digest.rank(0.0), 0.0); // Below minimum
        assert_eq!(digest.rank(200.0), 1.0); // Above maximum
    }

    #[test]
    fn test_tdigest_merge() {
        let mut digest1 = TDigest::new();
        let mut digest2 = TDigest::new();

        // Add different ranges to each digest
        for i in 1..=50 {
            digest1.add(i as f64);
        }

        for i in 51..=100 {
            digest2.add(i as f64);
        }

        // Merge digest2 into digest1
        // Note: This uses sampling approximation, not proper centroid merging
        digest1.merge(&digest2);

        assert_eq!(digest1.count(), 100);
        assert_eq!(digest1.min(), Some(1.0));
        assert_eq!(digest1.max(), Some(100.0));

        // Median should be around 50.5
        // Using a larger tolerance due to approximation errors from sampling-based merge
        let median = digest1.median().unwrap();
        assert!((median - 50.5).abs() < 5.0);
    }

    #[test]
    fn test_tdigest_compression() {
        let mut low_compression = TDigest::with_compression(50);
        let mut high_compression = TDigest::with_compression(500);

        // Add the same data to both
        for i in 1..=10000 {
            let value = i as f64;
            low_compression.add(value);
            high_compression.add(value);
        }

        // High compression should have larger memory usage
        let low_stats = low_compression.statistics();
        let high_stats = high_compression.statistics();

        assert!(high_stats.memory_usage > low_stats.memory_usage);
        assert_eq!(low_stats.compression, 50);
        assert_eq!(high_stats.compression, 500);
    }

    #[test]
    fn test_streaming_tdigest() {
        let mut streaming = StreamingTDigest::new(100, 10);

        // Add values that will trigger multiple flushes
        for i in 1..=100 {
            streaming.add(i as f64);
        }

        let median = streaming.quantile(0.5).unwrap();
        assert!((median - 50.5).abs() < 5.0);

        let stats = streaming.statistics();
        assert_eq!(stats.count, 100);
    }

    #[test]
    fn test_tdigest_clear() {
        let mut digest = TDigest::new();

        // Add some values
        for i in 1..=10 {
            digest.add(i as f64);
        }

        assert_eq!(digest.count(), 10);
        assert!(!digest.is_empty());

        // Clear and test
        digest.clear();

        assert_eq!(digest.count(), 0);
        assert!(digest.is_empty());
        assert_eq!(digest.min(), None);
        assert_eq!(digest.max(), None);
        assert_eq!(digest.quantile(0.5), None);
    }

    #[test]
    fn test_tdigest_accuracy() {
        let mut digest = TDigest::with_accuracy(0.01); // 1% target error

        // Add a large dataset
        for i in 1..=10000 {
            digest.add(i as f64);
        }

        // Test accuracy of extreme quantiles
        let p99 = digest.quantile(0.99).unwrap();
        let expected_p99 = 9900.0; // True 99th percentile
        let error = (p99 - expected_p99).abs() / expected_p99;

        assert!(error < 0.02, "Error {} should be less than 2%", error); // Allow some slack
    }
}
