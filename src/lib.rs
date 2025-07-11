//! Primary library module for sketches crate; provides Python bindings.

// Performance optimization setup
#[cfg(feature = "optimized")]
use jemallocator::Jemalloc;

#[cfg(feature = "optimized")]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

#[cfg(feature = "extension-module")]
use pyo3::PyObject;
#[cfg(feature = "extension-module")]
use pyo3::prelude::*;
#[cfg(feature = "extension-module")]
use pyo3::types::PyBytes;

pub mod aod;
pub mod bloom;
pub mod countmin;
pub mod cpc;
pub mod frequent;
pub mod hll;
pub mod linear;
pub mod quantiles;
pub mod sampling;
pub mod tdigest;
pub mod theta;

// Performance optimization modules
#[cfg(feature = "optimized")]
pub mod compact_memory;
#[cfg(feature = "optimized")]
pub mod fast_hash;
#[cfg(feature = "optimized")]
pub mod simd_ops;

/// Python binding for CPC sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "CpcSketch")]
pub struct CpcSketch {
    inner: cpc::CpcSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl CpcSketch {
    /// Create a new CPC sketch with log2(k) specified by `lg_k`. Defaults to 11.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        CpcSketch {
            inner: cpc::CpcSketch::new(lg_k.unwrap_or(11)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Merge another CPC sketch into this one (in-place union).
    pub fn merge(&mut self, other: &CpcSketch) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }

    /// Serialize the sketch to bytes.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyObject {
        PyBytes::new(py, &self.inner.to_bytes()).into()
    }
}

/// Python binding for HyperLogLog sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "HllSketch")]
pub struct HllSketch {
    inner: hll::HllSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl HllSketch {
    /// Create a new HLL sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        HllSketch {
            inner: hll::HllSketch::new(lg_k.unwrap_or(12)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Batch update with multiple items (optimized for Python).
    #[cfg(feature = "optimized")]
    pub fn update_batch(&mut self, py: Python, items: Vec<&str>) -> PyResult<()> {
        // Release GIL for CPU-intensive work
        py.allow_threads(|| {
            self.inner.update_batch(&items);
        })?;
        Ok(())
    }

    /// Update from numpy array or bytes buffer (zero-copy when possible).
    #[cfg(feature = "optimized")]
    pub fn update_from_buffer(&mut self, py: Python, buffer: &PyAny) -> PyResult<()> {
        // Try to get buffer interface for zero-copy access
        if let Ok(buffer_info) = buffer.extract::<pyo3::buffer::PyBuffer<u8>>() {
            let data = unsafe { buffer_info.as_slice(py)? };

            // Release GIL for processing
            py.allow_threads(|| {
                // Process data in chunks
                for chunk in data.chunks(8) {
                    self.inner.update(&chunk);
                }
            })?;
        } else {
            // Fallback to string conversion
            let items: Vec<String> = buffer.extract()?;
            let item_refs: Vec<&str> = items.iter().map(|s| s.as_str()).collect();

            py.allow_threads(|| {
                self.inner.update_batch(&item_refs);
            })?;
        }

        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Merge another HLL sketch into this one (in-place union).
    pub fn merge(&mut self, other: &HllSketch) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }

    /// Serialize the sketch to bytes.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyObject {
        PyBytes::new(py, &self.inner.to_bytes()).into()
    }
}

/// Python binding for HyperLogLog++ sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "HllPlusPlusSketch")]
pub struct HllPlusPlusSketch {
    inner: hll::HllPlusPlusSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl HllPlusPlusSketch {
    /// Create a new HLL++ sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        HllPlusPlusSketch {
            inner: hll::HllPlusPlusSketch::new(lg_k.unwrap_or(12)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Merge another HLL++ sketch into this one (in-place union).
    pub fn merge(&mut self, other: &HllPlusPlusSketch) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }

    /// Serialize the sketch to bytes.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyObject {
        PyBytes::new(py, &self.inner.to_bytes()).into()
    }
}

/// Python binding for sparse HyperLogLog++ sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "HllPlusPlusSparseSketch")]
pub struct HllPlusPlusSparseSketch {
    inner: hll::HllPlusPlusSparseSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl HllPlusPlusSparseSketch {
    /// Create a new sparse HLL++ sketch with precision `lg_k`. Defaults to 12.
    #[new]
    fn new(lg_k: Option<u8>) -> Self {
        HllPlusPlusSparseSketch {
            inner: hll::HllPlusPlusSparseSketch::new(lg_k.unwrap_or(12)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Merge another sparse sketch into this one (in-place union).
    pub fn merge(&mut self, other: &HllPlusPlusSparseSketch) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }

    /// Serialize the sketch to bytes.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyObject {
        PyBytes::new(py, &self.inner.to_bytes()).into()
    }
}

/// Python binding for Theta sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "ThetaSketch")]
pub struct ThetaSketch {
    inner: theta::ThetaSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl ThetaSketch {
    /// Create a new Theta sketch with sample size `k`. Defaults to 4096.
    #[new]
    fn new(k: Option<usize>) -> Self {
        ThetaSketch {
            inner: theta::ThetaSketch::new(k.unwrap_or(4096)),
        }
    }

    /// Update the sketch with a string item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Union of two sketches, returning a new sketch.
    pub fn union(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch {
            inner: theta::ThetaSketch::union(&self.inner, &other.inner),
        }
    }

    /// Intersection of two sketches, returning a new sketch.
    pub fn intersect(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch {
            inner: theta::ThetaSketch::intersect(&self.inner, &other.inner),
        }
    }

    /// Difference of two sketches (self \\ other), returning a new sketch.
    pub fn difference(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch {
            inner: theta::ThetaSketch::difference(&self.inner, &other.inner, self.inner.k),
        }
    }

    /// Return the sample capacity (heap size).
    pub fn sample_capacity(&self) -> usize {
        self.inner.sample_capacity()
    }
}



/// Python binding for Bloom Filter.
#[cfg(feature = "extension-module")]
#[pyclass(name = "BloomFilter")]
pub struct BloomFilter {
    inner: bloom::BloomFilter,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl BloomFilter {
    /// Create a new Bloom filter with specified capacity and error rate.
    #[new]
    fn new(capacity: usize, error_rate: Option<f64>, use_simd: Option<bool>) -> Self {
        BloomFilter {
            inner: bloom::BloomFilter::new(
                capacity,
                error_rate.unwrap_or(0.01),
                use_simd.unwrap_or(false),
            ),
        }
    }

    /// Add an element to the filter.
    pub fn add(&mut self, item: &str) -> PyResult<()> {
        self.inner.add(&item);
        Ok(())
    }

    /// Check if an element might be in the filter.
    pub fn contains(&self, item: &str) -> bool {
        self.inner.contains(&item)
    }

    /// Clear the filter.
    pub fn clear(&mut self) -> PyResult<()> {
        self.inner.clear();
        Ok(())
    }

    /// Get the current false positive probability.
    pub fn false_positive_probability(&self) -> f64 {
        self.inner.false_positive_probability()
    }

    /// Get filter statistics as a dictionary.
    pub fn statistics(&self, py: Python) -> PyObject {
        let stats = self.inner.statistics();
        let dict = py
            .import("builtins")
            .unwrap()
            .getattr("dict")
            .unwrap()
            .call0()
            .unwrap();

        dict.set_item("num_bits", stats.num_bits).unwrap();
        dict.set_item("num_hash_functions", stats.num_hash_functions)
            .unwrap();
        dict.set_item("bits_set", stats.bits_set).unwrap();
        dict.set_item("fill_ratio", stats.fill_ratio).unwrap();
        dict.set_item(
            "false_positive_probability",
            stats.false_positive_probability,
        )
        .unwrap();
        dict.set_item("uses_simd", stats.uses_simd).unwrap();

        dict.into()
    }
}

/// Python binding for Counting Bloom Filter.
#[cfg(feature = "extension-module")]
#[pyclass(name = "CountingBloomFilter")]
pub struct CountingBloomFilter {
    inner: bloom::CountingBloomFilter,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl CountingBloomFilter {
    /// Create a new counting Bloom filter.
    #[new]
    fn new(capacity: usize, error_rate: Option<f64>, max_count: Option<u8>) -> Self {
        CountingBloomFilter {
            inner: bloom::CountingBloomFilter::new(
                capacity,
                error_rate.unwrap_or(0.01),
                max_count.unwrap_or(255),
            ),
        }
    }

    /// Add an element to the filter.
    pub fn add(&mut self, item: &str) -> PyResult<()> {
        self.inner.add(&item);
        Ok(())
    }

    /// Remove an element from the filter.
    pub fn remove(&mut self, item: &str) -> bool {
        self.inner.remove(&item)
    }

    /// Check if an element might be in the filter.
    pub fn contains(&self, item: &str) -> bool {
        self.inner.contains(&item)
    }
}

/// Python binding for Count-Min Sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "CountMinSketch")]
pub struct CountMinSketch {
    inner: countmin::CountMinSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl CountMinSketch {
    /// Create a new Count-Min sketch with specified dimensions.
    #[new]
    fn new(
        width: usize,
        depth: usize,
        use_simd: Option<bool>,
        conservative_update: Option<bool>,
    ) -> Self {
        CountMinSketch {
            inner: countmin::CountMinSketch::new(
                width,
                depth,
                use_simd.unwrap_or(false),
                conservative_update.unwrap_or(false),
            ),
        }
    }

    /// Create a Count-Min sketch with error bounds.
    #[staticmethod]
    fn with_error_bounds(
        epsilon: f64,
        delta: f64,
        use_simd: Option<bool>,
        conservative_update: Option<bool>,
    ) -> Self {
        CountMinSketch {
            inner: countmin::CountMinSketch::with_error_bounds(
                epsilon,
                delta,
                use_simd.unwrap_or(false),
                conservative_update.unwrap_or(false),
            ),
        }
    }

    /// Update the count for an item.
    pub fn update(&mut self, item: &str, count: u64) -> PyResult<()> {
        self.inner.update(&item, count);
        Ok(())
    }

    /// Increment the count for an item by 1.
    pub fn increment(&mut self, item: &str) -> PyResult<()> {
        self.inner.increment(&item);
        Ok(())
    }

    /// Estimate the frequency of an item.
    pub fn estimate(&self, item: &str) -> u64 {
        self.inner.estimate(&item)
    }

    /// Merge another Count-Min sketch into this one.
    pub fn merge(&mut self, other: &CountMinSketch) -> PyResult<()> {
        self.inner
            .merge(&other.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Get the total count of all items.
    pub fn total_count(&self) -> u64 {
        self.inner.total_count()
    }

    /// Find heavy hitters above threshold.
    pub fn heavy_hitters(&self, threshold: u64) -> Vec<u64> {
        self.inner.heavy_hitters(threshold)
    }

    /// Clear the sketch.
    pub fn clear(&mut self) -> PyResult<()> {
        self.inner.clear();
        Ok(())
    }

    /// Get sketch statistics as a dictionary.
    pub fn statistics(&self, py: Python) -> PyObject {
        let stats = self.inner.statistics();
        let dict = py
            .import("builtins")
            .unwrap()
            .getattr("dict")
            .unwrap()
            .call0()
            .unwrap();

        dict.set_item("width", stats.width).unwrap();
        dict.set_item("depth", stats.depth).unwrap();
        dict.set_item("total_cells", stats.total_cells).unwrap();
        dict.set_item("non_zero_cells", stats.non_zero_cells)
            .unwrap();
        dict.set_item("fill_ratio", stats.fill_ratio).unwrap();
        dict.set_item("total_count", stats.total_count).unwrap();
        dict.set_item("max_count", stats.max_count).unwrap();
        dict.set_item("min_count", stats.min_count).unwrap();
        dict.set_item("uses_simd", stats.uses_simd).unwrap();
        dict.set_item("conservative_update", stats.conservative_update)
            .unwrap();

        dict.into()
    }
}

/// Python binding for Count Sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "CountSketch")]
pub struct CountSketch {
    inner: countmin::CountSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl CountSketch {
    /// Create a new Count sketch.
    #[new]
    fn new(width: usize, depth: usize) -> Self {
        CountSketch {
            inner: countmin::CountSketch::new(width, depth),
        }
    }

    /// Update the count for an item.
    pub fn update(&mut self, item: &str, count: i64) -> PyResult<()> {
        self.inner.update(&item, count);
        Ok(())
    }

    /// Estimate the frequency of an item.
    pub fn estimate(&self, item: &str) -> i64 {
        self.inner.estimate(&item)
    }
}

/// Python binding for KLL Sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "KllSketch")]
pub struct KllSketch {
    inner: quantiles::KllSketch<f64>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl KllSketch {
    /// Create a new KLL sketch with parameter k.
    #[new]
    fn new(k: Option<usize>) -> Self {
        KllSketch {
            inner: quantiles::KllSketch::new(k.unwrap_or(200)),
        }
    }

    /// Create a KLL sketch with specified accuracy.
    #[staticmethod]
    fn with_accuracy(epsilon: f64, confidence: f64) -> Self {
        KllSketch {
            inner: quantiles::KllSketch::with_accuracy(epsilon, confidence),
        }
    }

    /// Update the sketch with a new value.
    pub fn update(&mut self, value: f64) -> PyResult<()> {
        self.inner.update(value);
        Ok(())
    }

    /// Get quantile for the given rank (0.0 to 1.0).
    pub fn quantile(&mut self, rank: f64) -> Option<f64> {
        self.inner.quantile(rank)
    }

    /// Get the rank of a given value (0.0 to 1.0).
    pub fn rank(&mut self, value: f64) -> f64 {
        self.inner.rank(&value)
    }

    /// Merge another KLL sketch into this one.
    pub fn merge(&mut self, other: &mut KllSketch) -> PyResult<()> {
        self.inner.merge(&mut other.inner);
        Ok(())
    }

    /// Get the total number of items processed.
    pub fn count(&self) -> u64 {
        self.inner.count()
    }

    /// Check if the sketch is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the minimum value seen.
    pub fn min(&self) -> Option<f64> {
        self.inner.min().copied()
    }

    /// Get the maximum value seen.
    pub fn max(&self) -> Option<f64> {
        self.inner.max().copied()
    }

    /// Get sketch statistics as a dictionary.
    pub fn statistics(&self, py: Python) -> PyObject {
        let stats = self.inner.statistics();
        let dict = py
            .import("builtins")
            .unwrap()
            .getattr("dict")
            .unwrap()
            .call0()
            .unwrap();

        dict.set_item("k", stats.k).unwrap();
        dict.set_item("levels", stats.levels).unwrap();
        dict.set_item("total_items", stats.total_items).unwrap();
        dict.set_item("total_count", stats.total_count).unwrap();
        dict.set_item("memory_usage", stats.memory_usage).unwrap();
        dict.set_item("min_value_set", stats.min_value_set).unwrap();
        dict.set_item("max_value_set", stats.max_value_set).unwrap();

        dict.into()
    }

    /// Convenience methods for common quantiles
    pub fn median(&mut self) -> Option<f64> {
        self.quantile(0.5)
    }

    pub fn q95(&mut self) -> Option<f64> {
        self.quantile(0.95)
    }

    pub fn q99(&mut self) -> Option<f64> {
        self.quantile(0.99)
    }

    pub fn q25(&mut self) -> Option<f64> {
        self.quantile(0.25)
    }

    pub fn q75(&mut self) -> Option<f64> {
        self.quantile(0.75)
    }
}

/// Python binding for Linear Counter.
#[cfg(feature = "extension-module")]
#[pyclass(name = "LinearCounter")]
pub struct LinearCounter {
    inner: linear::LinearCounter,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl LinearCounter {
    /// Create a new Linear Counter.
    #[new]
    fn new(num_bits: usize, use_simd: Option<bool>) -> Self {
        LinearCounter {
            inner: linear::LinearCounter::new(num_bits, use_simd.unwrap_or(false)),
        }
    }

    /// Create a Linear Counter with optimal size for expected cardinality.
    #[staticmethod]
    fn with_expected_cardinality(
        expected_cardinality: usize,
        error_rate: f64,
        use_simd: Option<bool>,
    ) -> Self {
        LinearCounter {
            inner: linear::LinearCounter::with_expected_cardinality(
                expected_cardinality,
                error_rate,
                use_simd.unwrap_or(false),
            ),
        }
    }

    /// Update the counter with a new item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Check if should transition to HyperLogLog.
    pub fn should_transition_to_hll(&self) -> bool {
        self.inner.should_transition_to_hll()
    }

    /// Get the current fill ratio.
    pub fn fill_ratio(&self) -> f64 {
        self.inner.fill_ratio()
    }

    /// Merge another Linear Counter into this one.
    pub fn merge(&mut self, other: &LinearCounter) -> PyResult<()> {
        self.inner
            .merge(&other.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Clear the counter.
    pub fn clear(&mut self) -> PyResult<()> {
        self.inner.clear();
        Ok(())
    }

    /// Get counter statistics as a dictionary.
    pub fn statistics(&self, py: Python) -> PyObject {
        let stats = self.inner.statistics();
        let dict = py
            .import("builtins")
            .unwrap()
            .getattr("dict")
            .unwrap()
            .call0()
            .unwrap();

        dict.set_item("num_bits", stats.num_bits).unwrap();
        dict.set_item("bits_set", stats.bits_set).unwrap();
        dict.set_item("fill_ratio", stats.fill_ratio).unwrap();
        dict.set_item("estimated_cardinality", stats.estimated_cardinality)
            .unwrap();
        dict.set_item("should_transition", stats.should_transition)
            .unwrap();
        dict.set_item("memory_usage", stats.memory_usage).unwrap();
        dict.set_item("uses_simd", stats.uses_simd).unwrap();

        dict.into()
    }
}

/// Python binding for Hybrid Counter.
#[cfg(feature = "extension-module")]
#[pyclass(name = "HybridCounter")]
pub struct HybridCounter {
    inner: linear::HybridCounter,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl HybridCounter {
    /// Create a new Hybrid Counter.
    #[new]
    fn new(linear_bits: usize, lg_k: u8, transition_threshold: usize) -> Self {
        HybridCounter {
            inner: linear::HybridCounter::new(linear_bits, lg_k, transition_threshold),
        }
    }

    /// Create with optimal parameters for expected cardinality range.
    #[staticmethod]
    fn with_range(max_expected_cardinality: usize) -> Self {
        HybridCounter {
            inner: linear::HybridCounter::with_range(max_expected_cardinality),
        }
    }

    /// Update the counter with a new item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(&item);
        Ok(())
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Get current mode.
    pub fn mode(&self) -> String {
        self.inner.mode().to_string()
    }

    /// Get hybrid counter statistics as a dictionary.
    pub fn statistics(&self, py: Python) -> PyObject {
        let stats = self.inner.statistics();
        let dict = py
            .import("builtins")
            .unwrap()
            .getattr("dict")
            .unwrap()
            .call0()
            .unwrap();

        dict.set_item("mode", stats.mode).unwrap();
        dict.set_item("estimated_cardinality", stats.estimated_cardinality)
            .unwrap();
        dict.set_item("memory_usage", stats.memory_usage).unwrap();
        dict.set_item("transition_threshold", stats.transition_threshold)
            .unwrap();

        if let Some(fill_ratio) = stats.fill_ratio {
            dict.set_item("fill_ratio", fill_ratio).unwrap();
        }
        if let Some(bits_set) = stats.bits_set {
            dict.set_item("bits_set", bits_set).unwrap();
        }

        dict.into()
    }
}

/// Python binding for Frequent Strings Sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "FrequentStringsSketch")]
pub struct FrequentStringsSketch {
    inner: frequent::FrequentStringsSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl FrequentStringsSketch {
    /// Create a new Frequent Strings sketch.
    #[new]
    fn new(max_map_size: usize, use_reservoir: Option<bool>) -> Self {
        FrequentStringsSketch {
            inner: frequent::FrequentStringsSketch::new(
                max_map_size,
                use_reservoir.unwrap_or(false),
            ),
        }
    }

    /// Create with error rate specification.
    #[staticmethod]
    fn with_error_rate(error_rate: f64, confidence: f64, use_reservoir: Option<bool>) -> Self {
        FrequentStringsSketch {
            inner: frequent::FrequentStringsSketch::with_error_rate(
                error_rate,
                confidence,
                use_reservoir.unwrap_or(false),
            ),
        }
    }

    /// Update the sketch with a new item.
    pub fn update(&mut self, item: &str) -> PyResult<()> {
        self.inner.update(item);
        Ok(())
    }

    /// Get frequent items above a threshold.
    pub fn get_frequent_items(&self, threshold: u64) -> Vec<(String, u64, u64, u64)> {
        self.inner
            .get_frequent_items(threshold)
            .into_iter()
            .map(|item| (item.item, item.estimate, item.lower_bound, item.upper_bound))
            .collect()
    }

    /// Get frequent items above a relative threshold.
    pub fn get_frequent_items_by_fraction(
        &self,
        threshold_fraction: f64,
    ) -> Vec<(String, u64, u64, u64)> {
        self.inner
            .get_frequent_items_by_fraction(threshold_fraction)
            .into_iter()
            .map(|item| (item.item, item.estimate, item.lower_bound, item.upper_bound))
            .collect()
    }

    /// Get the top-k most frequent items.
    pub fn get_top_k(&self, k: usize) -> Vec<(String, u64, u64, u64)> {
        self.inner
            .get_top_k(k)
            .into_iter()
            .map(|item| (item.item, item.estimate, item.lower_bound, item.upper_bound))
            .collect()
    }

    /// Get estimated frequency of a specific item.
    pub fn get_estimate(&self, item: &str) -> Option<u64> {
        self.inner.get_estimate(item)
    }

    /// Get frequency bounds for a specific item.
    pub fn get_bounds(&self, item: &str) -> Option<(u64, u64)> {
        self.inner.get_bounds(item)
    }

    /// Merge another sketch into this one.
    pub fn merge(&mut self, other: &FrequentStringsSketch) -> PyResult<()> {
        self.inner
            .merge(&other.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Clear the sketch.
    pub fn clear(&mut self) -> PyResult<()> {
        self.inner.clear();
        Ok(())
    }

    /// Check if sketch is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the total number of items processed.
    pub fn get_stream_length(&self) -> u64 {
        self.inner.get_stream_length()
    }

    /// Get sketch statistics as a dictionary.
    pub fn statistics(&self, py: Python) -> PyObject {
        let stats = self.inner.statistics();
        let dict = py
            .import("builtins")
            .unwrap()
            .getattr("dict")
            .unwrap()
            .call0()
            .unwrap();

        dict.set_item("max_map_size", stats.max_map_size).unwrap();
        dict.set_item("current_map_size", stats.current_map_size)
            .unwrap();
        dict.set_item("stream_length", stats.stream_length).unwrap();
        dict.set_item("total_tracked_frequency", stats.total_tracked_frequency)
            .unwrap();
        dict.set_item("error_rate", stats.error_rate).unwrap();
        dict.set_item("uses_reservoir", stats.uses_reservoir)
            .unwrap();
        dict.set_item("memory_usage", stats.memory_usage).unwrap();

        dict.into()
    }
}

/// Python binding for Reservoir Sampler R.
#[cfg(feature = "extension-module")]
#[pyclass(name = "ReservoirSamplerR", unsendable)]
pub struct ReservoirSamplerR {
    inner: sampling::ReservoirSamplerR<String>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl ReservoirSamplerR {
    /// Create a new reservoir sampler with the given capacity.
    #[new]
    fn new(capacity: usize) -> Self {
        ReservoirSamplerR {
            inner: sampling::ReservoirSamplerR::new(capacity),
        }
    }

    /// Add an item to the reservoir.
    pub fn add(&mut self, item: &str) -> PyResult<()> {
        self.inner.add(item.to_string());
        Ok(())
    }

    /// Get the current sample.
    pub fn sample(&self) -> Vec<String> {
        self.inner.sample().to_vec()
    }

    /// Get the number of items seen so far.
    pub fn items_seen(&self) -> usize {
        self.inner.items_seen()
    }

    /// Get the capacity of the reservoir.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Check if the reservoir is full.
    pub fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    /// Clear the reservoir and reset counters.
    pub fn clear(&mut self) -> PyResult<()> {
        self.inner.clear();
        Ok(())
    }

    /// Merge another reservoir sampler into this one.
    pub fn merge(&mut self, other: &ReservoirSamplerR) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }
}

/// Python binding for Reservoir Sampler A.
#[cfg(feature = "extension-module")]
#[pyclass(name = "ReservoirSamplerA", unsendable)]
pub struct ReservoirSamplerA {
    inner: sampling::ReservoirSamplerA<String>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl ReservoirSamplerA {
    /// Create a new reservoir sampler with the given capacity.
    #[new]
    fn new(capacity: usize) -> Self {
        ReservoirSamplerA {
            inner: sampling::ReservoirSamplerA::new(capacity),
        }
    }

    /// Add an item to the reservoir.
    pub fn add(&mut self, item: &str) -> PyResult<()> {
        self.inner.add(item.to_string());
        Ok(())
    }

    /// Get the current sample.
    pub fn sample(&self) -> Vec<String> {
        self.inner.sample().to_vec()
    }

    /// Get the number of items seen so far.
    pub fn items_seen(&self) -> usize {
        self.inner.items_seen()
    }

    /// Get the capacity of the reservoir.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Check if the reservoir is full.
    pub fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    /// Clear the reservoir and reset counters.
    pub fn clear(&mut self) -> PyResult<()> {
        self.inner.clear();
        Ok(())
    }

    /// Merge another reservoir sampler into this one.
    pub fn merge(&mut self, other: &ReservoirSamplerA) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }
}

/// Python binding for Weighted Reservoir Sampler.
#[cfg(feature = "extension-module")]
#[pyclass(name = "WeightedReservoirSampler", unsendable)]
pub struct WeightedReservoirSampler {
    inner: sampling::WeightedReservoirSampler<String>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl WeightedReservoirSampler {
    /// Create a new weighted reservoir sampler.
    #[new]
    fn new(capacity: usize) -> Self {
        WeightedReservoirSampler {
            inner: sampling::WeightedReservoirSampler::new(capacity),
        }
    }

    /// Add an item with weight.
    pub fn add_weighted(&mut self, item: &str, weight: f64) -> PyResult<()> {
        self.inner.add_weighted(item.to_string(), weight);
        Ok(())
    }

    /// Add an item with weight 1.0.
    pub fn add(&mut self, item: &str) -> PyResult<()> {
        self.inner.add(item.to_string());
        Ok(())
    }

    /// Get the current sample (items only).
    pub fn sample(&self) -> Vec<String> {
        self.inner.sample()
    }

    /// Get the current sample with weights.
    pub fn sample_with_weights(&self) -> Vec<(String, f64)> {
        self.inner.sample_with_weights()
    }

    /// Get the total weight processed.
    pub fn total_weight(&self) -> f64 {
        self.inner.total_weight()
    }

    /// Get the capacity of the reservoir.
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Clear the reservoir.
    pub fn clear(&mut self) -> PyResult<()> {
        self.inner.clear();
        Ok(())
    }
}

/// Python binding for Stream Sampler.
#[cfg(feature = "extension-module")]
#[pyclass(name = "StreamSampler", unsendable)]
pub struct StreamSampler {
    inner: sampling::StreamSampler<String>,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl StreamSampler {
    /// Create a new stream sampler.
    #[new]
    fn new(capacity: usize, batch_size: usize) -> Self {
        StreamSampler {
            inner: sampling::StreamSampler::new(capacity, batch_size),
        }
    }

    /// Add items to the stream buffer.
    pub fn push_batch(&mut self, items: Vec<String>) -> PyResult<()> {
        self.inner.push_batch(items);
        Ok(())
    }

    /// Flush remaining items in buffer.
    pub fn flush(&mut self) -> PyResult<()> {
        self.inner.flush();
        Ok(())
    }

    /// Get the current sample.
    pub fn sample(&self) -> Vec<String> {
        self.inner.sample().to_vec()
    }

    /// Get statistics about the stream processing.
    pub fn stats(&self, py: Python) -> PyObject {
        let stats = self.inner.stats();
        let dict = py
            .import("builtins")
            .unwrap()
            .getattr("dict")
            .unwrap()
            .call0()
            .unwrap();

        dict.set_item("items_processed", stats.items_processed)
            .unwrap();
        dict.set_item("sample_size", stats.sample_size).unwrap();
        dict.set_item("capacity", stats.capacity).unwrap();
        dict.set_item("buffer_size", stats.buffer_size).unwrap();

        dict.into()
    }
}

/// Python binding for T-Digest.
#[cfg(feature = "extension-module")]
#[pyclass(name = "TDigest")]
pub struct TDigest {
    inner: tdigest::TDigest,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl TDigest {
    /// Create a new T-Digest with default compression.
    #[new]
    #[pyo3(signature = (compression=None))]
    fn new(compression: Option<usize>) -> Self {
        TDigest {
            inner: match compression {
                Some(c) => tdigest::TDigest::with_compression(c),
                None => tdigest::TDigest::new(),
            },
        }
    }

    /// Create a T-Digest optimized for specific accuracy.
    #[staticmethod]
    fn with_accuracy(target_error: f64) -> Self {
        TDigest {
            inner: tdigest::TDigest::with_accuracy(target_error),
        }
    }

    /// Add a value to the digest.
    pub fn add(&mut self, value: f64) -> PyResult<()> {
        self.inner.add(value);
        Ok(())
    }

    /// Add multiple values to the digest.
    pub fn add_batch(&mut self, values: Vec<f64>) -> PyResult<()> {
        self.inner.add_batch(&values);
        Ok(())
    }

    /// Estimate the quantile for a given rank (0.0 to 1.0).
    pub fn quantile(&self, q: f64) -> Option<f64> {
        self.inner.quantile(q)
    }

    /// Estimate the rank (percentile) of a given value.
    pub fn rank(&self, value: f64) -> f64 {
        self.inner.rank(value)
    }

    /// Merge another T-Digest into this one.
    pub fn merge(&mut self, other: &TDigest) -> PyResult<()> {
        self.inner.merge(&other.inner);
        Ok(())
    }

    /// Get the total number of values processed.
    pub fn count(&self) -> u64 {
        self.inner.count()
    }

    /// Check if the digest is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get the minimum value seen.
    pub fn min(&self) -> Option<f64> {
        self.inner.min()
    }

    /// Get the maximum value seen.
    pub fn max(&self) -> Option<f64> {
        self.inner.max()
    }

    /// Get the median (50th percentile).
    pub fn median(&self) -> Option<f64> {
        self.inner.median()
    }

    /// Get multiple quantiles at once.
    pub fn quantiles(&self, qs: Vec<f64>) -> Vec<Option<f64>> {
        self.inner.quantiles(&qs)
    }

    /// Calculate trimmed mean (mean excluding extreme values).
    pub fn trimmed_mean(&self, lower_quantile: f64, upper_quantile: f64) -> Option<f64> {
        self.inner.trimmed_mean(lower_quantile, upper_quantile)
    }

    /// Clear the digest.
    pub fn clear(&mut self) -> PyResult<()> {
        self.inner.clear();
        Ok(())
    }

    /// Get digest statistics as a dictionary.
    pub fn statistics(&self, py: Python) -> PyObject {
        let stats = self.inner.statistics();
        let dict = py
            .import("builtins")
            .unwrap()
            .getattr("dict")
            .unwrap()
            .call0()
            .unwrap();

        dict.set_item("count", stats.count).unwrap();
        dict.set_item("compression", stats.compression).unwrap();
        dict.set_item("min_value", stats.min_value).unwrap();
        dict.set_item("max_value", stats.max_value).unwrap();
        dict.set_item("memory_usage", stats.memory_usage).unwrap();

        dict.into()
    }

    /// Convenience methods for common quantiles
    pub fn p25(&self) -> Option<f64> {
        self.quantile(0.25)
    }

    pub fn p75(&self) -> Option<f64> {
        self.quantile(0.75)
    }

    pub fn p90(&self) -> Option<f64> {
        self.quantile(0.90)
    }

    pub fn p95(&self) -> Option<f64> {
        self.quantile(0.95)
    }

    pub fn p99(&self) -> Option<f64> {
        self.quantile(0.99)
    }

    pub fn p999(&self) -> Option<f64> {
        self.quantile(0.999)
    }
}

/// Python binding for Streaming T-Digest.
#[cfg(feature = "extension-module")]
#[pyclass(name = "StreamingTDigest")]
pub struct StreamingTDigest {
    inner: tdigest::StreamingTDigest,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl StreamingTDigest {
    /// Create a new streaming T-Digest.
    #[new]
    #[pyo3(signature = (compression=None, buffer_size=None))]
    fn new(compression: Option<usize>, buffer_size: Option<usize>) -> Self {
        StreamingTDigest {
            inner: tdigest::StreamingTDigest::new(
                compression.unwrap_or(100),
                buffer_size.unwrap_or(1000),
            ),
        }
    }

    /// Add a value to the streaming digest.
    pub fn add(&mut self, value: f64) -> PyResult<()> {
        self.inner.add(value);
        Ok(())
    }

    /// Flush the buffer into the main digest.
    pub fn flush(&mut self) -> PyResult<()> {
        self.inner.flush();
        Ok(())
    }

    /// Get a quantile estimate (automatically flushes buffer).
    pub fn quantile(&mut self, q: f64) -> Option<f64> {
        self.inner.quantile(q)
    }

    /// Get statistics about the streaming digest.
    pub fn statistics(&mut self, py: Python) -> PyObject {
        let stats = self.inner.statistics();
        let dict = py
            .import("builtins")
            .unwrap()
            .getattr("dict")
            .unwrap()
            .call0()
            .unwrap();

        dict.set_item("count", stats.count).unwrap();
        dict.set_item("compression", stats.compression).unwrap();
        dict.set_item("min_value", stats.min_value).unwrap();
        dict.set_item("max_value", stats.max_value).unwrap();
        dict.set_item("memory_usage", stats.memory_usage).unwrap();

        dict.into()
    }

    /// Get common quantiles with automatic flushing
    pub fn median(&mut self) -> Option<f64> {
        self.quantile(0.5)
    }

    pub fn p95(&mut self) -> Option<f64> {
        self.quantile(0.95)
    }

    pub fn p99(&mut self) -> Option<f64> {
        self.quantile(0.99)
    }
}

/// Python binding for Array of Doubles Sketch.
#[cfg(feature = "extension-module")]
#[pyclass(name = "AodSketch")]
pub struct AodSketch {
    inner: aod::AodSketch,
}

#[cfg(feature = "extension-module")]
#[pymethods]
impl AodSketch {
    /// Create a new AOD sketch with default parameters.
    #[new]
    #[pyo3(signature = (capacity=None, num_values=None))]
    fn new(capacity: Option<usize>, num_values: Option<usize>) -> Self {
        AodSketch {
            inner: aod::AodSketch::with_capacity_and_values(
                capacity.unwrap_or(4096),
                num_values.unwrap_or(1),
            ),
        }
    }

    /// Update the sketch with a key and array of values.
    pub fn update(&mut self, key: &str, values: Vec<f64>) -> PyResult<()> {
        self.inner
            .update(&key, &values)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Get estimated number of unique keys.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Get upper bound estimate with given confidence.
    pub fn upper_bound(&self, confidence: Option<f64>) -> f64 {
        self.inner.upper_bound(confidence.unwrap_or(0.95))
    }

    /// Get lower bound estimate with given confidence.
    pub fn lower_bound(&self, confidence: Option<f64>) -> f64 {
        self.inner.lower_bound(confidence.unwrap_or(0.95))
    }

    /// Get current theta (sampling probability).
    pub fn theta(&self) -> f64 {
        self.inner.theta()
    }

    /// Check if sketch is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get number of entries currently stored.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Get number of values per entry.
    pub fn num_values(&self) -> usize {
        self.inner.num_values()
    }

    /// Calculate sum of values for each column across all entries.
    pub fn column_sums(&self) -> Vec<f64> {
        self.inner.column_sums()
    }

    /// Calculate mean of values for each column.
    pub fn column_means(&self) -> Vec<f64> {
        self.inner.column_means()
    }

    /// Union this sketch with another AOD sketch.
    pub fn union(&mut self, other: &AodSketch) -> PyResult<()> {
        self.inner
            .union(&other.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))
    }

    /// Create a compact, immutable copy of this sketch.
    pub fn compact(&self) -> AodSketch {
        AodSketch {
            inner: self.inner.compact(),
        }
    }

    /// Serialize sketch to bytes.
    pub fn to_bytes<'py>(&self, py: Python<'py>) -> PyObject {
        PyBytes::new(py, &self.inner.to_bytes()).into()
    }

    /// Deserialize sketch from bytes.
    #[staticmethod]
    pub fn from_bytes(bytes: &[u8]) -> PyResult<AodSketch> {
        let inner = aod::AodSketch::from_bytes(bytes)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
        Ok(AodSketch { inner })
    }

    /// Get entries as list of (hash, values) tuples.
    pub fn get_entries(&self) -> Vec<(u64, Vec<f64>)> {
        self.inner
            .iter()
            .map(|entry| (entry.hash, entry.values))
            .collect()
    }

    /// Clear the sketch.
    pub fn clear(&mut self) -> PyResult<()> {
        let capacity = self.inner.config.capacity;
        let num_values = self.inner.config.num_values;
        self.inner = aod::AodSketch::with_capacity_and_values(capacity, num_values);
        Ok(())
    }

    /// Get sketch statistics as a dictionary.
    pub fn statistics(&self, py: Python) -> PyObject {
        let dict = py
            .import("builtins")
            .unwrap()
            .getattr("dict")
            .unwrap()
            .call0()
            .unwrap();

        dict.set_item("capacity", self.inner.config.capacity)
            .unwrap();
        dict.set_item("num_values", self.inner.config.num_values)
            .unwrap();
        dict.set_item("theta", self.inner.theta()).unwrap();
        dict.set_item("is_empty", self.inner.is_empty()).unwrap();
        dict.set_item("current_size", self.inner.len()).unwrap();
        dict.set_item("estimated_cardinality", self.inner.estimate())
            .unwrap();
        dict.set_item(
            "memory_usage",
            self.inner.len() * (8 + 8 * self.inner.num_values()),
        )
        .unwrap();

        dict.into()
    }
}

/// Python module definition
#[cfg(feature = "extension-module")]
#[pymodule]
fn sketches(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes to module
    m.add_class::<BloomFilter>()?;
    m.add_class::<CountingBloomFilter>()?;
    m.add_class::<CountMinSketch>()?;
    m.add_class::<CountSketch>()?;
    m.add_class::<CpcSketch>()?;
    m.add_class::<FrequentStringsSketch>()?;
    m.add_class::<HllSketch>()?;
    m.add_class::<HllPlusPlusSketch>()?;
    m.add_class::<HllPlusPlusSparseSketch>()?;
    m.add_class::<KllSketch>()?;
    m.add_class::<LinearCounter>()?;
    m.add_class::<HybridCounter>()?;
    m.add_class::<ReservoirSamplerR>()?;
    m.add_class::<ReservoirSamplerA>()?;
    m.add_class::<WeightedReservoirSampler>()?;
    m.add_class::<StreamSampler>()?;
    m.add_class::<TDigest>()?;
    m.add_class::<StreamingTDigest>()?;
    m.add_class::<ThetaSketch>()?;
    m.add_class::<AodSketch>()?;

    // Build info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(())
}
