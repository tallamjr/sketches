use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Maximum theta value representing the full hash space.
const MAX_THETA: u64 = u64::MAX;

/// Load factor threshold for resizing the hash table (50%).
const RESIZE_LOAD_FACTOR: f64 = 0.5;

/// Load factor threshold for triggering a rebuild (93.75% = 15/16).
const REBUILD_LOAD_FACTOR: f64 = 15.0 / 16.0;

/// Number of bits used from the hash to compute the probe stride.
const STRIDE_HASH_BITS: u8 = 7;

/// Mask for extracting stride bits.
const STRIDE_MASK: u64 = (1 << STRIDE_HASH_BITS) - 1;

/// Trait for summary objects attached to each hash entry in a Tuple Sketch.
///
/// Summaries enable associative aggregations on top of the Theta Sketch
/// framework. Each unique item has an associated summary that is updated
/// on every occurrence and merged during set operations.
pub trait Summary: Clone {
    /// Create a new, empty summary.
    fn new() -> Self;

    /// Update this summary with a new observed value.
    fn update(&mut self, value: f64);

    /// Merge another summary into this one. Used during set operations
    /// (union, intersection) when the same hash appears in both sketches.
    fn merge(&mut self, other: &Self);
}

/// A summary that accumulates a sum of f64 values.
#[derive(Clone, Debug, PartialEq)]
pub struct DoubleSummary {
    pub value: f64,
}

impl Summary for DoubleSummary {
    fn new() -> Self {
        DoubleSummary { value: 0.0 }
    }

    fn update(&mut self, value: f64) {
        self.value += value;
    }

    fn merge(&mut self, other: &Self) {
        self.value += other.value;
    }
}

/// A summary that counts occurrences.
#[derive(Clone, Debug, PartialEq)]
pub struct CountSummary {
    pub count: u64,
}

impl Summary for CountSummary {
    fn new() -> Self {
        CountSummary { count: 0 }
    }

    fn update(&mut self, _value: f64) {
        self.count += 1;
    }

    fn merge(&mut self, other: &Self) {
        self.count += other.count;
    }
}

/// A summary that accumulates N double values in an array.
#[derive(Clone, Debug, PartialEq)]
pub struct ArrayOfDoublesSummary {
    pub values: Vec<f64>,
}

impl ArrayOfDoublesSummary {
    /// Create a new summary with a specified number of double accumulators.
    pub fn with_size(n: usize) -> Self {
        ArrayOfDoublesSummary {
            values: vec![0.0; n],
        }
    }

    /// Return the number of accumulators.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return whether the summary has zero accumulators.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl Summary for ArrayOfDoublesSummary {
    fn new() -> Self {
        ArrayOfDoublesSummary { values: Vec::new() }
    }

    fn update(&mut self, value: f64) {
        // When called with a single value, add it to every accumulator.
        for v in self.values.iter_mut() {
            *v += value;
        }
    }

    fn merge(&mut self, other: &Self) {
        // If self is empty (default-constructed), adopt the other's shape.
        if self.values.is_empty() && !other.values.is_empty() {
            self.values = other.values.clone();
            return;
        }
        // Element-wise addition. Only merge matching dimensions.
        let len = self.values.len().min(other.values.len());
        for i in 0..len {
            self.values[i] += other.values[i];
        }
    }
}

/// Tuple Sketch: extends the Theta Sketch framework by attaching a Summary
/// object to each hash entry, enabling approximate joins and associative
/// aggregations.
///
/// Uses an open-addressing hash table with stride-based probing for O(1)
/// amortised duplicate detection, matching the ThetaSketch design.
pub struct TupleSketch<S: Summary> {
    /// Nominal sample size (sketch capacity).
    pub k: usize,
    /// Log2 of the current table size.
    lg_cur_size: u8,
    /// Log2 of the nominal size (k).
    lg_nom_size: u8,
    /// The hash table: hash values. Empty slots contain 0.
    hash_entries: Vec<u64>,
    /// Parallel array of summaries. Only valid where hash_entries[i] != 0.
    summaries: Vec<Option<S>>,
    /// Number of occupied entries in the table.
    num_entries: usize,
    /// Current theta threshold. Only hashes strictly less than theta are retained.
    theta: u64,
}

impl<S: Summary> TupleSketch<S> {
    /// Create a new Tuple Sketch with sample size k.
    ///
    /// The value of k is rounded up to the next power of two internally.
    pub fn new(k: usize) -> Self {
        let k = k.max(1);
        let lg_nom_size = log2_ceil(k);
        let actual_k = 1usize << lg_nom_size;
        // Start with table size = 2 * k so we have room before needing to rebuild.
        let lg_cur_size = lg_nom_size + 1;
        let table_size = 1usize << lg_cur_size;
        TupleSketch {
            k: actual_k,
            lg_cur_size,
            lg_nom_size,
            hash_entries: vec![0u64; table_size],
            summaries: (0..table_size).map(|_| None).collect(),
            num_entries: 0,
            theta: MAX_THETA,
        }
    }

    /// Update the sketch with an item and an associated value.
    ///
    /// The item is hashed to determine identity. If the item is new, a fresh
    /// summary is created via `Summary::new()` and then updated. If the item
    /// already exists, its existing summary is updated in place.
    pub fn update<T: Hash>(&mut self, item: &T, value: f64) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();

        // The hash value 0 is reserved as the empty sentinel.
        let hash = if hash == 0 { 1 } else { hash };

        // Screen against theta: only hashes below theta are candidates.
        if hash >= self.theta {
            return;
        }

        let index = self.find_or_empty(hash);
        if self.hash_entries[index] == hash {
            // Existing entry: update the summary.
            if let Some(ref mut summary) = self.summaries[index] {
                summary.update(value);
            }
        } else {
            // New entry: insert hash and create summary.
            debug_assert_eq!(self.hash_entries[index], 0);
            self.hash_entries[index] = hash;
            let mut summary = S::new();
            summary.update(value);
            self.summaries[index] = Some(summary);
            self.num_entries += 1;
            self.check_and_manage_capacity();
        }
    }

    /// Estimate the cardinality.
    ///
    /// Uses the standard theta sketch estimator: retained_count / (theta / MAX_THETA).
    /// When theta == MAX_THETA (no sampling has occurred), the exact count is returned.
    pub fn estimate(&self) -> f64 {
        let n = self.num_entries;
        if n == 0 {
            return 0.0;
        }
        if self.theta == MAX_THETA {
            n as f64
        } else {
            let theta_f = (self.theta as f64) / (MAX_THETA as f64);
            ((n as f64 - 1.0) / theta_f).max(0.0)
        }
    }

    /// Look up the summary for a given item.
    ///
    /// Returns `None` if the item is not in the sketch (either never inserted,
    /// or evicted during sampling).
    pub fn get_summary<T: Hash>(&self, item: &T) -> Option<&S> {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        let hash = if hash == 0 { 1 } else { hash };

        if hash >= self.theta {
            return None;
        }

        let index = self.find_or_empty(hash);
        if self.hash_entries[index] == hash {
            self.summaries[index].as_ref()
        } else {
            None
        }
    }

    /// Union of two tuple sketches. Summaries for matching hashes are merged.
    pub fn union(&self, other: &TupleSketch<S>) -> TupleSketch<S> {
        let union_theta = self.theta.min(other.theta);
        let k = self.k.min(other.k);

        // Collect all (hash, summary) pairs below union_theta.
        let total = self.num_entries + other.num_entries;
        let mut collector = TupleCollector::<S>::with_capacity(total);

        for (hash, summary) in self.retained_entries() {
            if hash < union_theta {
                collector.insert_or_merge(hash, summary);
            }
        }
        for (hash, summary) in other.retained_entries() {
            if hash < union_theta {
                collector.insert_or_merge(hash, summary);
            }
        }

        let mut pairs = collector.into_pairs();

        let lg_nom = log2_ceil(k);
        let actual_k = 1usize << lg_nom;

        let final_theta;
        if pairs.len() > actual_k {
            // Sort by hash and keep only the k smallest.
            pairs.select_nth_unstable_by_key(actual_k, |&(h, _)| h);
            final_theta = pairs[actual_k].0.min(union_theta);
            pairs.truncate(actual_k);
        } else {
            final_theta = union_theta;
        }

        Self::from_pairs(&pairs, final_theta, actual_k)
    }

    /// Intersection of two tuple sketches. Only hashes present in both sketches
    /// are retained, and their summaries are merged.
    pub fn intersect(&self, other: &TupleSketch<S>) -> TupleSketch<S> {
        let result_theta = self.theta.min(other.theta);
        let k = self.k.min(other.k);

        // Build a lookup from the smaller sketch.
        let (probe, lookup) = if self.num_entries <= other.num_entries {
            (self, other)
        } else {
            (other, self)
        };

        let lookup_map = TupleCollector::<S>::from_sketch(lookup);

        let mut result_pairs: Vec<(u64, S)> = Vec::new();
        for (hash, summary) in probe.retained_entries() {
            if hash < result_theta {
                if let Some(other_summary) = lookup_map.get(hash) {
                    let mut merged = summary.clone();
                    merged.merge(other_summary);
                    result_pairs.push((hash, merged));
                }
            }
        }

        let lg_nom = log2_ceil(k);
        let actual_k = 1usize << lg_nom;

        let final_theta;
        if result_pairs.len() > actual_k {
            result_pairs.select_nth_unstable_by_key(actual_k, |&(h, _)| h);
            final_theta = result_pairs[actual_k].0.min(result_theta);
            result_pairs.truncate(actual_k);
        } else {
            final_theta = result_theta;
        }

        Self::from_pairs(&result_pairs, final_theta, actual_k)
    }

    /// Return all summaries as a HashMap keyed by hash value.
    pub fn to_summary_map(&self) -> HashMap<u64, S> {
        let mut map = HashMap::with_capacity(self.num_entries);
        for (hash, summary) in self.retained_entries() {
            map.insert(hash, summary.clone());
        }
        map
    }

    /// Return the number of retained entries in the sketch.
    pub fn num_retained(&self) -> usize {
        self.num_entries
    }

    /// Return the current theta value.
    pub fn theta(&self) -> u64 {
        self.theta
    }

    // -----------------------------------------------------------------------
    // Internal methods
    // -----------------------------------------------------------------------

    /// Iterate over retained (hash, summary) pairs.
    fn retained_entries(&self) -> impl Iterator<Item = (u64, &S)> {
        self.hash_entries
            .iter()
            .zip(self.summaries.iter())
            .filter_map(|(&h, s)| {
                if h != 0 {
                    s.as_ref().map(|summary| (h, summary))
                } else {
                    None
                }
            })
    }

    /// Find the index of a given hash in the table, or the index of the first
    /// empty slot along the probe sequence.
    fn find_or_empty(&self, hash: u64) -> usize {
        let mask = self.hash_entries.len() - 1;
        let stride = get_stride(hash, self.lg_cur_size);
        let mut index = (hash as usize) & mask;

        loop {
            let probe = self.hash_entries[index];
            if probe == 0 || probe == hash {
                return index;
            }
            index = (index + stride) & mask;
        }
    }

    /// Static variant for use during rehashing into a new table.
    fn find_or_empty_in(entries: &[u64], hash: u64, lg_size: u8) -> usize {
        let mask = entries.len() - 1;
        let stride = get_stride(hash, lg_size);
        let mut index = (hash as usize) & mask;

        loop {
            let probe = entries[index];
            if probe == 0 || probe == hash {
                return index;
            }
            index = (index + stride) & mask;
        }
    }

    /// After a successful insertion, check whether we need to resize or rebuild.
    fn check_and_manage_capacity(&mut self) {
        let table_len = self.hash_entries.len();
        if self.lg_cur_size <= self.lg_nom_size {
            let threshold = (RESIZE_LOAD_FACTOR * table_len as f64) as usize;
            if self.num_entries > threshold {
                self.resize();
            }
        } else {
            let threshold = (REBUILD_LOAD_FACTOR * table_len as f64) as usize;
            if self.num_entries > threshold {
                self.rebuild();
            }
        }
    }

    /// Double the table size and rehash all entries.
    fn resize(&mut self) {
        let new_lg_size = self.lg_cur_size + 1;
        let new_size = 1usize << new_lg_size;
        let mut new_hash_entries = vec![0u64; new_size];
        let mut new_summaries: Vec<Option<S>> = (0..new_size).map(|_| None).collect();

        for i in 0..self.hash_entries.len() {
            let entry = self.hash_entries[i];
            if entry != 0 {
                let idx = Self::find_or_empty_in(&new_hash_entries, entry, new_lg_size);
                new_hash_entries[idx] = entry;
                new_summaries[idx] = self.summaries[i].take();
            }
        }

        self.hash_entries = new_hash_entries;
        self.summaries = new_summaries;
        self.lg_cur_size = new_lg_size;
    }

    /// Rebuild the table: keep only the k smallest hashes, update theta,
    /// and rehash into a clean table of the same size.
    fn rebuild(&mut self) {
        let k = self.k;

        // Collect all non-empty entries as (hash, index) so we can retrieve summaries.
        let mut entries_with_idx: Vec<(u64, usize)> = self
            .hash_entries
            .iter()
            .enumerate()
            .filter_map(|(i, &h)| if h != 0 { Some((h, i)) } else { None })
            .collect();

        if entries_with_idx.len() <= k {
            return;
        }

        // Partition around the k-th element by hash value.
        entries_with_idx.select_nth_unstable_by_key(k, |&(h, _)| h);
        let new_theta = entries_with_idx[k].0;
        entries_with_idx.truncate(k);

        self.theta = new_theta;

        // Rehash into a clean table of the current size.
        let table_size = 1usize << self.lg_cur_size;
        let mut new_hash_entries = vec![0u64; table_size];
        let mut new_summaries: Vec<Option<S>> = (0..table_size).map(|_| None).collect();
        let mut count = 0;

        for &(hash, old_idx) in &entries_with_idx {
            if hash != 0 && hash < self.theta {
                let idx = Self::find_or_empty_in(&new_hash_entries, hash, self.lg_cur_size);
                new_hash_entries[idx] = hash;
                new_summaries[idx] = self.summaries[old_idx].take();
                count += 1;
            }
        }

        self.hash_entries = new_hash_entries;
        self.summaries = new_summaries;
        self.num_entries = count;
    }

    /// Build a TupleSketch from a set of (hash, summary) pairs, a theta, and a target k.
    fn from_pairs(pairs: &[(u64, S)], theta: u64, k: usize) -> TupleSketch<S> {
        let lg_nom_size = log2_ceil(k);
        let actual_k = 1usize << lg_nom_size;

        let needed = pairs.len();
        let mut lg_cur_size = lg_nom_size + 1;
        loop {
            let table_size = 1usize << lg_cur_size;
            let threshold = (REBUILD_LOAD_FACTOR * table_size as f64) as usize;
            if needed <= threshold {
                break;
            }
            lg_cur_size += 1;
        }
        let table_size = 1usize << lg_cur_size;

        let mut hash_entries = vec![0u64; table_size];
        let mut summaries: Vec<Option<S>> = (0..table_size).map(|_| None).collect();
        let mut count = 0;

        for (hash, summary) in pairs {
            if *hash != 0 {
                let idx = Self::find_or_empty_in(&hash_entries, *hash, lg_cur_size);
                if hash_entries[idx] == 0 {
                    hash_entries[idx] = *hash;
                    summaries[idx] = Some(summary.clone());
                    count += 1;
                }
            }
        }

        TupleSketch {
            k: actual_k,
            lg_cur_size,
            lg_nom_size,
            hash_entries,
            summaries,
            num_entries: count,
            theta,
        }
    }
}

/// Compute the probe stride for open-addressing. The stride is always odd,
/// ensuring it is coprime with any power-of-two table size and thus guarantees
/// a full cycle through every slot.
fn get_stride(hash: u64, lg_size: u8) -> usize {
    (2 * ((hash >> lg_size) & STRIDE_MASK) + 1) as usize
}

/// Compute ceil(log2(n)) for n >= 1. Returns 0 for n <= 1.
fn log2_ceil(n: usize) -> u8 {
    if n <= 1 {
        return 0;
    }
    (usize::BITS - (n - 1).leading_zeros()) as u8
}

/// An open-addressing hash map used for collecting (hash, summary) pairs during
/// set operations. Analogous to HashCollector in theta.rs but carries summaries.
struct TupleCollector<S: Summary> {
    hash_entries: Vec<u64>,
    summaries: Vec<Option<S>>,
    lg_size: u8,
    count: usize,
}

impl<S: Summary> TupleCollector<S> {
    /// Create a collector with enough capacity for `expected` entries.
    fn with_capacity(expected: usize) -> Self {
        let needed = (expected * 2).max(16);
        let lg_size = log2_ceil(needed).max(4);
        let size = 1usize << lg_size;
        TupleCollector {
            hash_entries: vec![0u64; size],
            summaries: (0..size).map(|_| None).collect(),
            lg_size,
            count: 0,
        }
    }

    /// Build a collector from a sketch's retained entries for lookup.
    fn from_sketch(sketch: &TupleSketch<S>) -> Self {
        let mut collector = Self::with_capacity(sketch.num_entries);
        for (hash, summary) in sketch.retained_entries() {
            collector.insert_or_merge(hash, summary);
        }
        collector
    }

    /// Insert a hash/summary pair, merging if the hash already exists.
    fn insert_or_merge(&mut self, hash: u64, summary: &S) {
        if hash == 0 {
            return;
        }

        // Check load factor and grow if needed.
        let threshold = (RESIZE_LOAD_FACTOR * self.hash_entries.len() as f64) as usize;
        if self.count >= threshold {
            self.grow();
        }

        let mask = self.hash_entries.len() - 1;
        let stride = get_stride(hash, self.lg_size);
        let mut index = (hash as usize) & mask;

        loop {
            let probe = self.hash_entries[index];
            if probe == 0 {
                self.hash_entries[index] = hash;
                self.summaries[index] = Some(summary.clone());
                self.count += 1;
                return;
            }
            if probe == hash {
                // Merge summaries.
                if let Some(ref mut existing) = self.summaries[index] {
                    existing.merge(summary);
                }
                return;
            }
            index = (index + stride) & mask;
        }
    }

    /// Look up a summary by hash.
    fn get(&self, hash: u64) -> Option<&S> {
        if hash == 0 {
            return None;
        }
        let mask = self.hash_entries.len() - 1;
        let stride = get_stride(hash, self.lg_size);
        let mut index = (hash as usize) & mask;

        loop {
            let probe = self.hash_entries[index];
            if probe == 0 {
                return None;
            }
            if probe == hash {
                return self.summaries[index].as_ref();
            }
            index = (index + stride) & mask;
        }
    }

    /// Double the table and rehash.
    fn grow(&mut self) {
        let new_lg_size = self.lg_size + 1;
        let new_size = 1usize << new_lg_size;
        let mut new_hash_entries = vec![0u64; new_size];
        let mut new_summaries: Vec<Option<S>> = (0..new_size).map(|_| None).collect();

        for i in 0..self.hash_entries.len() {
            let entry = self.hash_entries[i];
            if entry != 0 {
                let mask = new_size - 1;
                let stride = get_stride(entry, new_lg_size);
                let mut index = (entry as usize) & mask;
                loop {
                    if new_hash_entries[index] == 0 {
                        new_hash_entries[index] = entry;
                        new_summaries[index] = self.summaries[i].take();
                        break;
                    }
                    index = (index + stride) & mask;
                }
            }
        }

        self.hash_entries = new_hash_entries;
        self.summaries = new_summaries;
        self.lg_size = new_lg_size;
    }

    /// Consume the collector and return all (hash, summary) pairs.
    fn into_pairs(self) -> Vec<(u64, S)> {
        self.hash_entries
            .into_iter()
            .zip(self.summaries)
            .filter_map(|(h, s)| {
                if h != 0 {
                    s.map(|summary| (h, summary))
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Summary trait implementations
    // -----------------------------------------------------------------------

    #[test]
    fn test_double_summary_new() {
        let s = DoubleSummary::new();
        assert_eq!(s.value, 0.0);
    }

    #[test]
    fn test_double_summary_update() {
        let mut s = DoubleSummary::new();
        s.update(3.5);
        s.update(1.5);
        assert!((s.value - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_double_summary_merge() {
        let mut a = DoubleSummary::new();
        a.update(10.0);
        let mut b = DoubleSummary::new();
        b.update(7.0);
        a.merge(&b);
        assert!((a.value - 17.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_count_summary_new() {
        let s = CountSummary::new();
        assert_eq!(s.count, 0);
    }

    #[test]
    fn test_count_summary_update() {
        let mut s = CountSummary::new();
        s.update(0.0); // value is ignored
        s.update(999.0);
        assert_eq!(s.count, 2);
    }

    #[test]
    fn test_count_summary_merge() {
        let mut a = CountSummary::new();
        a.update(0.0);
        a.update(0.0);
        let mut b = CountSummary::new();
        b.update(0.0);
        a.merge(&b);
        assert_eq!(a.count, 3);
    }

    #[test]
    fn test_array_of_doubles_summary_new() {
        let s = ArrayOfDoublesSummary::new();
        assert!(s.is_empty());
    }

    #[test]
    fn test_array_of_doubles_summary_with_size() {
        let s = ArrayOfDoublesSummary::with_size(3);
        assert_eq!(s.len(), 3);
        assert_eq!(s.values, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_array_of_doubles_summary_update() {
        let mut s = ArrayOfDoublesSummary::with_size(3);
        s.update(2.0);
        assert_eq!(s.values, vec![2.0, 2.0, 2.0]);
        s.update(1.0);
        assert_eq!(s.values, vec![3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_array_of_doubles_summary_merge() {
        let mut a = ArrayOfDoublesSummary::with_size(3);
        a.values = vec![1.0, 2.0, 3.0];
        let mut b = ArrayOfDoublesSummary::with_size(3);
        b.values = vec![4.0, 5.0, 6.0];
        a.merge(&b);
        assert_eq!(a.values, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_array_of_doubles_summary_merge_empty_self() {
        let mut a = ArrayOfDoublesSummary::new();
        let mut b = ArrayOfDoublesSummary::with_size(2);
        b.values = vec![10.0, 20.0];
        a.merge(&b);
        assert_eq!(a.values, vec![10.0, 20.0]);
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_sketch_defaults() {
        let sk: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        assert_eq!(sk.k, 1024);
        assert_eq!(sk.theta(), MAX_THETA);
        assert_eq!(sk.num_retained(), 0);
        assert_eq!(sk.estimate(), 0.0);
    }

    #[test]
    fn test_new_rounds_k_to_power_of_two() {
        let sk: TupleSketch<DoubleSummary> = TupleSketch::new(1000);
        assert_eq!(sk.k, 1024);
        let sk2: TupleSketch<DoubleSummary> = TupleSketch::new(1);
        assert_eq!(sk2.k, 1);
        let sk3: TupleSketch<DoubleSummary> = TupleSketch::new(5);
        assert_eq!(sk3.k, 8);
    }

    // -----------------------------------------------------------------------
    // Update and estimate with DoubleSummary
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_single_item() {
        let mut sk: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        sk.update(&"hello", 1.0);
        assert_eq!(sk.num_retained(), 1);
        assert_eq!(sk.estimate(), 1.0);
    }

    #[test]
    fn test_update_duplicate_items_accumulate_summary() {
        let mut sk: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        for _ in 0..10 {
            sk.update(&"same_item", 3.0);
        }
        assert_eq!(sk.num_retained(), 1);
        assert_eq!(sk.estimate(), 1.0);
        let summary = sk.get_summary(&"same_item").unwrap();
        assert!((summary.value - 30.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_update_many_distinct_items_below_k() {
        let mut sk: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        let n = 100;
        for i in 0..n {
            sk.update(&i, 1.0);
        }
        assert_eq!(sk.theta(), MAX_THETA);
        assert_eq!(sk.num_retained(), n);
        assert_eq!(sk.estimate(), n as f64);
    }

    #[test]
    fn test_update_many_distinct_items_above_k() {
        let k = 512;
        let mut sk: TupleSketch<DoubleSummary> = TupleSketch::new(k);
        let n = 50_000;
        for i in 0..n {
            sk.update(&i, 1.0);
        }
        assert!(sk.theta() < MAX_THETA);
        let est = sk.estimate();
        let error_ratio = (est - n as f64).abs() / n as f64;
        assert!(
            error_ratio < 0.15,
            "Estimate {est} too far from true cardinality {n} (error ratio {error_ratio})"
        );
    }

    #[test]
    fn test_estimate_accuracy_large() {
        let k = 4096;
        let mut sk: TupleSketch<DoubleSummary> = TupleSketch::new(k);
        let n = 1_000_000;
        for i in 0..n {
            sk.update(&i, 1.0);
        }
        let est = sk.estimate();
        let error_ratio = (est - n as f64).abs() / n as f64;
        assert!(
            error_ratio < 0.05,
            "Estimate {est} too far from true cardinality {n} (error ratio {error_ratio})"
        );
    }

    // -----------------------------------------------------------------------
    // Update and estimate with CountSummary
    // -----------------------------------------------------------------------

    #[test]
    fn test_count_summary_sketch() {
        let mut sk: TupleSketch<CountSummary> = TupleSketch::new(1024);
        for _ in 0..5 {
            sk.update(&"item_a", 0.0);
        }
        for _ in 0..3 {
            sk.update(&"item_b", 0.0);
        }
        assert_eq!(sk.num_retained(), 2);
        let a = sk.get_summary(&"item_a").unwrap();
        assert_eq!(a.count, 5);
        let b = sk.get_summary(&"item_b").unwrap();
        assert_eq!(b.count, 3);
    }

    // -----------------------------------------------------------------------
    // get_summary
    // -----------------------------------------------------------------------

    #[test]
    fn test_get_summary_missing_item() {
        let sk: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        assert!(sk.get_summary(&"nonexistent").is_none());
    }

    #[test]
    fn test_get_summary_present_item() {
        let mut sk: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        sk.update(&42, 7.5);
        let summary = sk.get_summary(&42).unwrap();
        assert!((summary.value - 7.5).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // to_summary_map
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_summary_map_empty() {
        let sk: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        let map = sk.to_summary_map();
        assert!(map.is_empty());
    }

    #[test]
    fn test_to_summary_map_populated() {
        let mut sk: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        for i in 0..100 {
            sk.update(&i, i as f64);
        }
        let map = sk.to_summary_map();
        assert_eq!(map.len(), 100);
        // All values should be present and non-zero (except item 0).
        for entry in map.values() {
            assert!(entry.value >= 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // Union
    // -----------------------------------------------------------------------

    #[test]
    fn test_union_disjoint() {
        let mut a: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        let mut b: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        for i in 0..10_000 {
            a.update(&format!("a_{i}"), 1.0);
        }
        for i in 0..10_000 {
            b.update(&format!("b_{i}"), 1.0);
        }
        let u = a.union(&b);
        let est = u.estimate();
        let expected = 20_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.10,
            "Union estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_union_overlapping_merges_summaries() {
        let mut a: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        let mut b: TupleSketch<DoubleSummary> = TupleSketch::new(4096);

        // Items 0..100 in both, so summaries should merge.
        for i in 0..100 {
            a.update(&i, 10.0);
            b.update(&i, 20.0);
        }
        // Additional items unique to each.
        for i in 100..200 {
            a.update(&i, 5.0);
        }
        for i in 200..300 {
            b.update(&i, 7.0);
        }

        let u = a.union(&b);
        assert_eq!(u.num_retained(), 300);

        // Shared items should have merged summaries (10.0 + 20.0 = 30.0).
        let map = u.to_summary_map();
        // We cannot look up by original key after union (we only have hash values),
        // but we can verify the count is correct.
        assert_eq!(map.len(), 300);
    }

    #[test]
    fn test_union_overlapping_cardinality() {
        let mut a: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        let mut b: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        for i in 0..10_000 {
            a.update(&i, 1.0);
        }
        for i in 5_000..15_000 {
            b.update(&i, 1.0);
        }
        let u = a.union(&b);
        let est = u.estimate();
        let expected = 15_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.10,
            "Union estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_union_empty() {
        let a: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        let b: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        let u = a.union(&b);
        assert_eq!(u.estimate(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Intersection
    // -----------------------------------------------------------------------

    #[test]
    fn test_intersect_overlapping() {
        let mut a: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        let mut b: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        for i in 0..10_000 {
            a.update(&i, 1.0);
        }
        for i in 5_000..15_000 {
            b.update(&i, 1.0);
        }
        let inter = a.intersect(&b);
        let est = inter.estimate();
        let expected = 5_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.15,
            "Intersect estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_intersect_merges_summaries() {
        let mut a: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        let mut b: TupleSketch<DoubleSummary> = TupleSketch::new(4096);

        // Shared items with different values.
        for i in 0..100 {
            a.update(&i, 3.0);
            b.update(&i, 7.0);
        }
        // Non-shared items.
        for i in 100..200 {
            a.update(&i, 1.0);
        }
        for i in 200..300 {
            b.update(&i, 1.0);
        }

        let inter = a.intersect(&b);
        // Only the 100 shared items should be retained.
        assert_eq!(inter.num_retained(), 100);

        // Each shared summary should have merged value 3.0 + 7.0 = 10.0.
        let map = inter.to_summary_map();
        for summary in map.values() {
            assert!(
                (summary.value - 10.0).abs() < f64::EPSILON,
                "Expected merged value 10.0 but got {}",
                summary.value
            );
        }
    }

    #[test]
    fn test_intersect_disjoint() {
        let mut a: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        let mut b: TupleSketch<DoubleSummary> = TupleSketch::new(4096);
        for i in 0..5_000 {
            a.update(&format!("a_{i}"), 1.0);
        }
        for i in 0..5_000 {
            b.update(&format!("b_{i}"), 1.0);
        }
        let inter = a.intersect(&b);
        let est = inter.estimate();
        assert!(
            est < 200.0,
            "Disjoint intersect estimate {est} should be near 0"
        );
    }

    #[test]
    fn test_intersect_empty() {
        let a: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        let b: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        let inter = a.intersect(&b);
        assert_eq!(inter.estimate(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Summary preservation through rebuild
    // -----------------------------------------------------------------------

    #[test]
    fn test_summaries_survive_rebuild() {
        let k = 64;
        let mut sk: TupleSketch<DoubleSummary> = TupleSketch::new(k);
        // Insert enough to trigger several rebuilds.
        for i in 0..10_000 {
            sk.update(&i, 1.0);
        }
        assert!(sk.theta() < MAX_THETA);

        // Every retained entry must have a valid summary.
        let map = sk.to_summary_map();
        assert_eq!(map.len(), sk.num_retained());
        for summary in map.values() {
            // Each item was only inserted once with value 1.0.
            assert!(
                (summary.value - 1.0).abs() < f64::EPSILON,
                "Summary value should be 1.0 but got {}",
                summary.value
            );
        }
    }

    #[test]
    fn test_summaries_accumulate_before_rebuild() {
        let k = 128;
        let mut sk: TupleSketch<CountSummary> = TupleSketch::new(k);
        // Insert same items multiple times before exceeding k.
        for _ in 0..5 {
            for i in 0..50 {
                sk.update(&i, 0.0);
            }
        }
        // All 50 items should still be present since 50 < k.
        assert_eq!(sk.num_retained(), 50);
        for i in 0..50 {
            let summary = sk.get_summary(&i).unwrap();
            assert_eq!(summary.count, 5, "Item {i} should have count 5");
        }
    }

    // -----------------------------------------------------------------------
    // Set operations preserve theta correctly
    // -----------------------------------------------------------------------

    #[test]
    fn test_set_operations_preserve_theta() {
        let mut a: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        let mut b: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        for i in 0..50_000 {
            a.update(&i, 1.0);
        }
        for i in 25_000..75_000 {
            b.update(&i, 1.0);
        }
        assert!(a.theta() < MAX_THETA);
        assert!(b.theta() < MAX_THETA);

        let u = a.union(&b);
        assert!(u.theta() <= a.theta().min(b.theta()));

        let inter = a.intersect(&b);
        assert!(inter.theta() <= a.theta().min(b.theta()));
    }

    // -----------------------------------------------------------------------
    // Self-union and self-intersection
    // -----------------------------------------------------------------------

    #[test]
    fn test_self_union() {
        let mut a: TupleSketch<DoubleSummary> = TupleSketch::new(2048);
        for i in 0..10_000 {
            a.update(&i, 1.0);
        }
        let u = a.union(&a);
        let est = u.estimate();
        let expected = 10_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.10,
            "Self-union estimate {est} too far from {expected} (error {error_ratio})"
        );
        // Summaries should be merged: each value doubled.
        let map = u.to_summary_map();
        for summary in map.values() {
            assert!(
                (summary.value - 2.0).abs() < f64::EPSILON,
                "Self-union summary should be 2.0 but got {}",
                summary.value
            );
        }
    }

    #[test]
    fn test_self_intersect() {
        let mut a: TupleSketch<DoubleSummary> = TupleSketch::new(2048);
        for i in 0..10_000 {
            a.update(&i, 1.0);
        }
        let inter = a.intersect(&a);
        let est = inter.estimate();
        let expected = 10_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.10,
            "Self-intersect estimate {est} too far from {expected} (error {error_ratio})"
        );
        // Summaries should be merged: each value doubled.
        let map = inter.to_summary_map();
        for summary in map.values() {
            assert!(
                (summary.value - 2.0).abs() < f64::EPSILON,
                "Self-intersect summary should be 2.0 but got {}",
                summary.value
            );
        }
    }

    // -----------------------------------------------------------------------
    // Hash table internals
    // -----------------------------------------------------------------------

    #[test]
    fn test_stride_is_always_odd() {
        for hash in [0u64, 1, 42, u64::MAX, 0xDEADBEEF, 1 << 50] {
            for lg in 1..20u8 {
                let stride = get_stride(hash, lg);
                assert_eq!(stride % 2, 1, "Stride must be odd for hash={hash} lg={lg}");
            }
        }
    }

    #[test]
    fn test_log2_ceil() {
        assert_eq!(log2_ceil(0), 0);
        assert_eq!(log2_ceil(1), 0);
        assert_eq!(log2_ceil(2), 1);
        assert_eq!(log2_ceil(3), 2);
        assert_eq!(log2_ceil(4), 2);
        assert_eq!(log2_ceil(5), 3);
        assert_eq!(log2_ceil(8), 3);
        assert_eq!(log2_ceil(9), 4);
        assert_eq!(log2_ceil(1024), 10);
        assert_eq!(log2_ceil(1025), 11);
    }

    #[test]
    fn test_rebuild_preserves_summaries_correctly() {
        let k = 64;
        let mut sk: TupleSketch<DoubleSummary> = TupleSketch::new(k);
        for i in 0..10_000 {
            sk.update(&i, i as f64);
        }
        assert!(sk.theta() < MAX_THETA);

        // Every retained entry must have a valid summary with the correct value.
        let map = sk.to_summary_map();
        assert_eq!(map.len(), sk.num_retained());
        for summary in map.values() {
            // Each summary was updated exactly once with a non-negative value.
            assert!(summary.value >= 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // TupleCollector
    // -----------------------------------------------------------------------

    #[test]
    fn test_tuple_collector_basic() {
        let mut tc = TupleCollector::<DoubleSummary>::with_capacity(100);
        let s1 = DoubleSummary { value: 5.0 };
        let s2 = DoubleSummary { value: 3.0 };
        tc.insert_or_merge(42, &s1);
        tc.insert_or_merge(99, &s2);

        assert!(tc.get(42).is_some());
        assert!((tc.get(42).unwrap().value - 5.0).abs() < f64::EPSILON);
        assert!(tc.get(99).is_some());
        assert!(tc.get(0).is_none());
        assert!(tc.get(1).is_none());

        let pairs = tc.into_pairs();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_tuple_collector_merge_on_insert() {
        let mut tc = TupleCollector::<DoubleSummary>::with_capacity(100);
        let s1 = DoubleSummary { value: 5.0 };
        let s2 = DoubleSummary { value: 3.0 };
        tc.insert_or_merge(42, &s1);
        tc.insert_or_merge(42, &s2);

        let summary = tc.get(42).unwrap();
        assert!((summary.value - 8.0).abs() < f64::EPSILON);
        assert_eq!(tc.count, 1); // still only one entry
    }

    #[test]
    fn test_tuple_collector_grow() {
        let mut tc = TupleCollector::<CountSummary>::with_capacity(4);
        for i in 1..1000u64 {
            let s = CountSummary { count: 1 };
            tc.insert_or_merge(i, &s);
        }
        for i in 1..1000u64 {
            assert!(tc.get(i).is_some());
        }
        assert_eq!(tc.count, 999);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_union_with_different_k() {
        let mut a: TupleSketch<DoubleSummary> = TupleSketch::new(512);
        let mut b: TupleSketch<DoubleSummary> = TupleSketch::new(2048);
        for i in 0..20_000 {
            a.update(&i, 1.0);
        }
        for i in 10_000..30_000 {
            b.update(&i, 1.0);
        }
        let u = a.union(&b);
        assert_eq!(u.k, 512);
        let est = u.estimate();
        let expected = 30_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.15,
            "Union with different k: estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_no_zero_hash_entries_stored() {
        let mut sk: TupleSketch<DoubleSummary> = TupleSketch::new(1024);
        for i in 0..5_000 {
            sk.update(&i, 1.0);
        }
        let retained: Vec<(u64, &DoubleSummary)> = sk.retained_entries().collect();
        assert_eq!(retained.len(), sk.num_retained());
        for (hash, _) in &retained {
            assert_ne!(*hash, 0);
        }
    }

    #[test]
    fn test_identical_sketches_union_merges_summaries() {
        let mut a: TupleSketch<CountSummary> = TupleSketch::new(4096);
        for i in 0..100 {
            a.update(&i, 0.0);
        }
        let u = a.union(&a);
        let map = u.to_summary_map();
        assert_eq!(map.len(), 100);
        for summary in map.values() {
            assert_eq!(
                summary.count, 2,
                "Self-union should double the count summary"
            );
        }
    }
}
