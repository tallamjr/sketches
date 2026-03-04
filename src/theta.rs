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

/// Theta Sketch for approximate set cardinality and set operations.
///
/// Uses an open-addressing hash table with stride-based probing for O(1)
/// amortised duplicate detection, replacing the previous BinaryHeap + linear
/// scan approach.
#[derive(Debug)]
pub struct ThetaSketch {
    /// Nominal sample size (sketch capacity).
    pub k: usize,
    /// Log2 of the current table size.
    lg_cur_size: u8,
    /// Log2 of the nominal size (k).
    lg_nom_size: u8,
    /// The hash table entries. Empty slots contain 0.
    entries: Vec<u64>,
    /// Number of occupied entries in the table.
    num_entries: usize,
    /// Current theta threshold. Only hashes strictly less than theta are retained.
    theta: u64,
}

impl ThetaSketch {
    /// Create a new Theta sketch with sample size k.
    ///
    /// The value of k is rounded up to the next power of two internally.
    pub fn new(k: usize) -> Self {
        let k = k.max(1);
        let lg_nom_size = log2_ceil(k);
        let actual_k = 1usize << lg_nom_size;
        // Start with table size = 2 * k so we have room before needing to rebuild.
        let lg_cur_size = lg_nom_size + 1;
        let table_size = 1usize << lg_cur_size;
        ThetaSketch {
            k: actual_k,
            lg_cur_size,
            lg_nom_size,
            entries: vec![0u64; table_size],
            num_entries: 0,
            theta: MAX_THETA,
        }
    }

    /// Update the sketch with an item implementing Hash.
    pub fn update<T: Hash>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();

        // The hash value 0 is reserved as the empty sentinel.
        // Remap it to 1 so we never store a zero.
        let hash = if hash == 0 { 1 } else { hash };

        // Screen against theta: only hashes below theta are candidates.
        if hash >= self.theta {
            return;
        }

        // Attempt to insert into the open-addressing table.
        if self.insert_hash(hash) {
            // Successfully inserted a new entry. Check load factor thresholds.
            self.check_and_manage_capacity();
        }
    }

    /// Insert a hash value into the table. Returns true if newly inserted,
    /// false if it was already present or is zero.
    fn insert_hash(&mut self, hash: u64) -> bool {
        if hash == 0 {
            return false;
        }
        let index = self.find_or_empty(hash);
        if self.entries[index] == hash {
            return false; // duplicate
        }
        debug_assert_eq!(self.entries[index], 0);
        self.entries[index] = hash;
        self.num_entries += 1;
        true
    }

    /// Find the index of a given hash in the table, or the index of the first
    /// empty slot along the probe sequence. Uses stride-based probing.
    fn find_or_empty(&self, hash: u64) -> usize {
        let mask = self.entries.len() - 1;
        let stride = get_stride(hash, self.lg_cur_size);
        let mut index = (hash as usize) & mask;

        loop {
            let probe = self.entries[index];
            if probe == 0 || probe == hash {
                return index;
            }
            index = (index + stride) & mask;
        }
    }

    /// Find the index of a given hash in the provided entries array.
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
        let table_len = self.entries.len();
        if self.lg_cur_size <= self.lg_nom_size {
            // Table is at or below nominal size: use resize threshold (50%).
            let threshold = (RESIZE_LOAD_FACTOR * table_len as f64) as usize;
            if self.num_entries > threshold {
                self.resize();
            }
        } else {
            // Table is above nominal size: use rebuild threshold (93.75%).
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
        let mut new_entries = vec![0u64; new_size];

        for &entry in &self.entries {
            if entry != 0 {
                let idx = Self::find_or_empty_in(&new_entries, entry, new_lg_size);
                new_entries[idx] = entry;
            }
        }

        self.entries = new_entries;
        self.lg_cur_size = new_lg_size;
    }

    /// Rebuild the table: keep only the k smallest hashes, update theta to the
    /// k-th smallest, and rehash into a clean table of the same size.
    fn rebuild(&mut self) {
        let k = self.k;
        // Collect all non-zero entries.
        let mut values: Vec<u64> = self.entries.iter().copied().filter(|&e| e != 0).collect();

        if values.len() <= k {
            // Nothing to prune.
            return;
        }

        // Partition around the k-th element (0-indexed: element at index k is the (k+1)-th smallest).
        values.select_nth_unstable(k);
        let new_theta = values[k];

        // Keep only the first k elements (those smaller than new_theta).
        values.truncate(k);

        // Update theta.
        self.theta = new_theta;

        // Rehash into a clean table of the current size.
        let table_size = 1usize << self.lg_cur_size;
        let mut new_entries = vec![0u64; table_size];
        let mut count = 0;
        for &v in &values {
            if v != 0 && v < self.theta {
                let idx = Self::find_or_empty_in(&new_entries, v, self.lg_cur_size);
                new_entries[idx] = v;
                count += 1;
            }
        }

        self.entries = new_entries;
        self.num_entries = count;
    }

    /// Estimate the cardinality.
    ///
    /// Uses the standard theta sketch estimator: retained_count / (theta / MAX_THETA).
    /// When theta == MAX_THETA (no sampling has occurred), the exact count is returned.
    /// A bias correction of (n-1) is applied per the standard formulation.
    pub fn estimate(&self) -> f64 {
        let n = self.num_entries;
        if n == 0 {
            return 0.0;
        }
        if self.theta == MAX_THETA {
            // No sampling has occurred: return exact count.
            n as f64
        } else {
            let theta_f = (self.theta as f64) / (MAX_THETA as f64);
            // Bias-corrected estimator: (n-1) / (theta / MAX_THETA).
            // This works correctly regardless of whether n < k, n == k, or n > k
            // (the latter can occur between rebuilds).
            ((n as f64 - 1.0) / theta_f).max(0.0)
        }
    }

    /// Return an iterator over the retained hash values.
    fn retained_values(&self) -> impl Iterator<Item = u64> + '_ {
        self.entries.iter().copied().filter(|&e| e != 0)
    }

    /// Build a ThetaSketch from a set of hash values, a theta, and a target k.
    fn from_values(values: &[u64], theta: u64, k: usize) -> ThetaSketch {
        let lg_nom_size = log2_ceil(k);
        let actual_k = 1usize << lg_nom_size;

        // Determine an appropriate table size. We need enough room for the values
        // without exceeding the rebuild threshold.
        let needed = values.len();
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

        let mut entries = vec![0u64; table_size];
        let mut count = 0;
        for &v in values {
            if v != 0 {
                let idx = Self::find_or_empty_in(&entries, v, lg_cur_size);
                if entries[idx] == 0 {
                    entries[idx] = v;
                    count += 1;
                }
            }
        }

        ThetaSketch {
            k: actual_k,
            lg_cur_size,
            lg_nom_size,
            entries,
            num_entries: count,
            theta,
        }
    }

    /// Union of multiple sketches, returning a new sketch with capacity k.
    pub fn union_many(sketches: &[&ThetaSketch], k: usize) -> ThetaSketch {
        if sketches.is_empty() {
            return ThetaSketch::new(k);
        }

        // The union theta is the minimum theta across all input sketches.
        let union_theta = sketches.iter().map(|s| s.theta).min().unwrap_or(MAX_THETA);

        // Collect all unique hash values below union_theta using a temporary hash set
        // built as an open-addressing table.
        let total_values: usize = sketches.iter().map(|s| s.num_entries).sum();
        let mut collector = HashCollector::with_capacity(total_values);

        for sk in sketches {
            for v in sk.retained_values() {
                if v < union_theta {
                    collector.insert(v);
                }
            }
        }

        let mut all_values = collector.into_values();

        let lg_nom = log2_ceil(k);
        let actual_k = 1usize << lg_nom;

        // If we have more than k values, select the k smallest and update theta.
        let final_theta;
        if all_values.len() > actual_k {
            all_values.select_nth_unstable(actual_k);
            final_theta = all_values[actual_k].min(union_theta);
            all_values.truncate(actual_k);
        } else {
            final_theta = union_theta;
        }

        Self::from_values(&all_values, final_theta, actual_k)
    }

    /// Union of two sketches with the same capacity.
    pub fn union(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch::union_many(&[self, other], self.k)
    }

    /// Intersection of two sketches.
    ///
    /// The `intersect_many` name is kept for API compatibility. This performs a
    /// pairwise intersection of sketches `a` and `b`.
    pub fn intersect_many(a: &ThetaSketch, b: &ThetaSketch, k: usize) -> ThetaSketch {
        let result_theta = a.theta.min(b.theta);

        // Build a lookup set from the smaller sketch for O(1) membership testing.
        let (probe, lookup) = if a.num_entries <= b.num_entries {
            (a, b)
        } else {
            (b, a)
        };

        let lookup_set = HashCollector::from_sketch(lookup);

        let mut intersection_values = Vec::new();
        for v in probe.retained_values() {
            if v < result_theta && lookup_set.contains(v) {
                intersection_values.push(v);
            }
        }

        let lg_nom = log2_ceil(k);
        let actual_k = 1usize << lg_nom;

        let final_theta;
        if intersection_values.len() > actual_k {
            intersection_values.select_nth_unstable(actual_k);
            final_theta = intersection_values[actual_k].min(result_theta);
            intersection_values.truncate(actual_k);
        } else {
            final_theta = result_theta;
        }

        Self::from_values(&intersection_values, final_theta, actual_k)
    }

    /// Intersection of two sketches with the same capacity.
    /// Returns a new sketch containing elements present in both sketches.
    pub fn intersect(&self, other: &ThetaSketch) -> ThetaSketch {
        let result_k = self.k.min(other.k);
        ThetaSketch::intersect_many(self, other, result_k)
    }

    /// Difference A \ B: items in A not in B.
    pub fn difference(a: &ThetaSketch, b: &ThetaSketch, k: usize) -> ThetaSketch {
        let result_theta = a.theta.min(b.theta);

        let b_set = HashCollector::from_sketch(b);

        let mut diff_values = Vec::new();
        for v in a.retained_values() {
            if v < result_theta && !b_set.contains(v) {
                diff_values.push(v);
            }
        }

        let lg_nom = log2_ceil(k);
        let actual_k = 1usize << lg_nom;

        let final_theta;
        if diff_values.len() > actual_k {
            diff_values.select_nth_unstable(actual_k);
            final_theta = diff_values[actual_k].min(result_theta);
            diff_values.truncate(actual_k);
        } else {
            final_theta = result_theta;
        }

        Self::from_values(&diff_values, final_theta, actual_k)
    }

    /// Return capacity of the underlying sample data vector.
    pub fn sample_capacity(&self) -> usize {
        self.entries.capacity()
    }

    /// Get the log2 of the nominal size (k).
    pub fn lg_nom_size(&self) -> u8 {
        self.lg_nom_size
    }

    /// Get the current theta value.
    pub fn theta_value(&self) -> u64 {
        self.theta
    }

    /// Get the number of retained entries.
    pub fn num_retained(&self) -> usize {
        self.num_entries
    }

    /// Get the nominal size k.
    pub fn nominal_size(&self) -> usize {
        self.k
    }

    /// Collect all retained hash values (non-zero entries) in sorted order.
    pub fn retained_hashes(&self) -> Vec<u64> {
        let mut values: Vec<u64> = self.retained_values().collect();
        values.sort_unstable();
        values
    }

    /// Serialize the sketch to bytes for storage or transmission.
    ///
    /// Wire format (little-endian):
    ///   8 bytes: k (usize)
    ///   8 bytes: theta (u64)
    ///   8 bytes: number of retained values (usize)
    ///   N * 8 bytes: retained hash values (u64 each), sorted for determinism
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        bytes.extend_from_slice(&self.k.to_le_bytes());
        bytes.extend_from_slice(&self.theta.to_le_bytes());

        let mut values: Vec<u64> = self.retained_values().collect();
        values.sort_unstable();

        bytes.extend_from_slice(&values.len().to_le_bytes());

        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        bytes
    }

    /// Deserialize a sketch from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 24 {
            return Err("Insufficient bytes for theta sketch header");
        }

        let mut offset = 0;

        let k = usize::from_le_bytes(
            bytes[offset..offset + 8]
                .try_into()
                .map_err(|_| "Invalid k bytes")?,
        );
        offset += 8;

        let theta = u64::from_le_bytes(
            bytes[offset..offset + 8]
                .try_into()
                .map_err(|_| "Invalid theta bytes")?,
        );
        offset += 8;

        let value_count = usize::from_le_bytes(
            bytes[offset..offset + 8]
                .try_into()
                .map_err(|_| "Invalid value count bytes")?,
        );
        offset += 8;

        if bytes.len() < offset + value_count * 8 {
            return Err("Insufficient bytes for hash values");
        }

        let mut values = Vec::with_capacity(value_count);
        for _ in 0..value_count {
            let value = u64::from_le_bytes(
                bytes[offset..offset + 8]
                    .try_into()
                    .map_err(|_| "Invalid hash value bytes")?,
            );
            values.push(value);
            offset += 8;
        }

        Ok(Self::from_values(&values, theta, k))
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
    // For powers of two, (n - 1).leading_zeros() gives us what we need.
    (usize::BITS - (n - 1).leading_zeros()) as u8
}

/// A simple open-addressing hash set used for collecting values during
/// set operations. Avoids pulling in HashSet from std to keep consistent
/// hashing semantics (identity hash on u64 values that are already hashed).
struct HashCollector {
    entries: Vec<u64>,
    lg_size: u8,
    count: usize,
}

impl HashCollector {
    /// Create a collector with enough capacity for `expected` entries.
    fn with_capacity(expected: usize) -> Self {
        // Size the table so the load factor stays below 50%.
        let needed = (expected * 2).max(16);
        let lg_size = log2_ceil(needed).max(4);
        let size = 1usize << lg_size;
        HashCollector {
            entries: vec![0u64; size],
            lg_size,
            count: 0,
        }
    }

    /// Build a collector from a sketch's retained values for membership testing.
    fn from_sketch(sketch: &ThetaSketch) -> Self {
        let mut collector = Self::with_capacity(sketch.num_entries);
        for v in sketch.retained_values() {
            collector.insert(v);
        }
        collector
    }

    /// Insert a value. Returns true if newly inserted.
    fn insert(&mut self, hash: u64) -> bool {
        if hash == 0 {
            return false;
        }
        // Check load factor and resize if needed before probing.
        let threshold = (RESIZE_LOAD_FACTOR * self.entries.len() as f64) as usize;
        if self.count >= threshold {
            self.grow();
        }

        let mask = self.entries.len() - 1;
        let stride = get_stride(hash, self.lg_size);
        let mut index = (hash as usize) & mask;

        loop {
            let probe = self.entries[index];
            if probe == 0 {
                self.entries[index] = hash;
                self.count += 1;
                return true;
            }
            if probe == hash {
                return false; // already present
            }
            index = (index + stride) & mask;
        }
    }

    /// Check whether a value is present.
    fn contains(&self, hash: u64) -> bool {
        if hash == 0 {
            return false;
        }
        let mask = self.entries.len() - 1;
        let stride = get_stride(hash, self.lg_size);
        let mut index = (hash as usize) & mask;

        loop {
            let probe = self.entries[index];
            if probe == 0 {
                return false;
            }
            if probe == hash {
                return true;
            }
            index = (index + stride) & mask;
        }
    }

    /// Double the table and rehash.
    fn grow(&mut self) {
        let new_lg_size = self.lg_size + 1;
        let new_size = 1usize << new_lg_size;
        let mut new_entries = vec![0u64; new_size];

        for &entry in &self.entries {
            if entry != 0 {
                let mask = new_size - 1;
                let stride = get_stride(entry, new_lg_size);
                let mut index = (entry as usize) & mask;
                loop {
                    if new_entries[index] == 0 {
                        new_entries[index] = entry;
                        break;
                    }
                    index = (index + stride) & mask;
                }
            }
        }

        self.entries = new_entries;
        self.lg_size = new_lg_size;
    }

    /// Consume the collector and return all stored values.
    fn into_values(self) -> Vec<u64> {
        self.entries.into_iter().filter(|&e| e != 0).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_sketch_defaults() {
        let sk = ThetaSketch::new(1024);
        assert_eq!(sk.k, 1024);
        assert_eq!(sk.theta, MAX_THETA);
        assert_eq!(sk.num_entries, 0);
        assert_eq!(sk.estimate(), 0.0);
    }

    #[test]
    fn test_new_rounds_k_to_power_of_two() {
        let sk = ThetaSketch::new(1000);
        assert_eq!(sk.k, 1024); // next power of two
        let sk2 = ThetaSketch::new(1);
        assert_eq!(sk2.k, 1);
        let sk3 = ThetaSketch::new(5);
        assert_eq!(sk3.k, 8);
    }

    // -----------------------------------------------------------------------
    // Update and estimate
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_single_item() {
        let mut sk = ThetaSketch::new(1024);
        sk.update(&"hello");
        assert_eq!(sk.num_entries, 1);
        assert_eq!(sk.estimate(), 1.0);
    }

    #[test]
    fn test_update_duplicate_items() {
        let mut sk = ThetaSketch::new(1024);
        for _ in 0..100 {
            sk.update(&"same_item");
        }
        assert_eq!(sk.num_entries, 1);
        assert_eq!(sk.estimate(), 1.0);
    }

    #[test]
    fn test_update_many_distinct_items_below_k() {
        let mut sk = ThetaSketch::new(4096);
        let n = 100;
        for i in 0..n {
            sk.update(&i);
        }
        // Below k, theta should still be MAX and count should be exact.
        assert_eq!(sk.theta, MAX_THETA);
        assert_eq!(sk.num_entries, n);
        assert_eq!(sk.estimate(), n as f64);
    }

    #[test]
    fn test_update_many_distinct_items_above_k() {
        let k = 512;
        let mut sk = ThetaSketch::new(k);
        let n = 50_000;
        for i in 0..n {
            sk.update(&i);
        }
        // After exceeding k, theta should have decreased.
        assert!(sk.theta < MAX_THETA);
        // The estimate should be in the right ballpark.
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
        let mut sk = ThetaSketch::new(k);
        let n = 1_000_000;
        for i in 0..n {
            sk.update(&i);
        }
        let est = sk.estimate();
        let error_ratio = (est - n as f64).abs() / n as f64;
        assert!(
            error_ratio < 0.05,
            "Estimate {est} too far from true cardinality {n} (error ratio {error_ratio})"
        );
    }

    // -----------------------------------------------------------------------
    // Union
    // -----------------------------------------------------------------------

    #[test]
    fn test_union_disjoint() {
        let mut a = ThetaSketch::new(4096);
        let mut b = ThetaSketch::new(4096);
        for i in 0..10_000 {
            a.update(&format!("a_{i}"));
        }
        for i in 0..10_000 {
            b.update(&format!("b_{i}"));
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
    fn test_union_overlapping() {
        let mut a = ThetaSketch::new(4096);
        let mut b = ThetaSketch::new(4096);
        // Shared items 0..5000, unique to a: 5000..10000, unique to b: 10000..15000
        for i in 0..10_000 {
            a.update(&i);
        }
        for i in 5_000..15_000 {
            b.update(&i);
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
    fn test_union_many() {
        let mut sketches_owned: Vec<ThetaSketch> = Vec::new();
        for s in 0..5 {
            let mut sk = ThetaSketch::new(2048);
            for i in 0..5_000 {
                sk.update(&format!("s{s}_{i}"));
            }
            sketches_owned.push(sk);
        }
        let refs: Vec<&ThetaSketch> = sketches_owned.iter().collect();
        let u = ThetaSketch::union_many(&refs, 2048);
        let est = u.estimate();
        let expected = 25_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.10,
            "Union many estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_union_empty() {
        let a = ThetaSketch::new(1024);
        let b = ThetaSketch::new(1024);
        let u = a.union(&b);
        assert_eq!(u.estimate(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Intersection
    // -----------------------------------------------------------------------

    #[test]
    fn test_intersect_overlapping() {
        let mut a = ThetaSketch::new(4096);
        let mut b = ThetaSketch::new(4096);
        // Items 0..10000 in a, items 5000..15000 in b. Intersection: 5000..10000 = 5000 items.
        for i in 0..10_000 {
            a.update(&i);
        }
        for i in 5_000..15_000 {
            b.update(&i);
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
    fn test_intersect_disjoint() {
        let mut a = ThetaSketch::new(4096);
        let mut b = ThetaSketch::new(4096);
        for i in 0..5_000 {
            a.update(&format!("a_{i}"));
        }
        for i in 0..5_000 {
            b.update(&format!("b_{i}"));
        }
        let inter = a.intersect(&b);
        let est = inter.estimate();
        // Disjoint sets should have near-zero intersection.
        assert!(
            est < 200.0,
            "Disjoint intersect estimate {est} should be near 0"
        );
    }

    #[test]
    fn test_intersect_many() {
        let mut a = ThetaSketch::new(4096);
        let mut b = ThetaSketch::new(4096);
        for i in 0..10_000 {
            a.update(&i);
        }
        for i in 5_000..15_000 {
            b.update(&i);
        }
        let inter = ThetaSketch::intersect_many(&a, &b, 4096);
        let est = inter.estimate();
        let expected = 5_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.15,
            "Intersect many estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_intersect_empty() {
        let a = ThetaSketch::new(1024);
        let b = ThetaSketch::new(1024);
        let inter = a.intersect(&b);
        assert_eq!(inter.estimate(), 0.0);
    }

    // -----------------------------------------------------------------------
    // Difference
    // -----------------------------------------------------------------------

    #[test]
    fn test_difference() {
        let mut a = ThetaSketch::new(4096);
        let mut b = ThetaSketch::new(4096);
        // A has 0..10000, B has 5000..15000. A \ B = 0..5000 = 5000 items.
        for i in 0..10_000 {
            a.update(&i);
        }
        for i in 5_000..15_000 {
            b.update(&i);
        }
        let diff = ThetaSketch::difference(&a, &b, 4096);
        let est = diff.estimate();
        let expected = 5_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.15,
            "Difference estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_difference_disjoint() {
        let mut a = ThetaSketch::new(4096);
        let mut b = ThetaSketch::new(4096);
        for i in 0..5_000 {
            a.update(&format!("a_{i}"));
        }
        for i in 0..5_000 {
            b.update(&format!("b_{i}"));
        }
        let diff = ThetaSketch::difference(&a, &b, 4096);
        let est = diff.estimate();
        let expected = 5_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.15,
            "Disjoint difference estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_difference_subset() {
        // B is a superset of A, so A \ B should be empty.
        let mut a = ThetaSketch::new(4096);
        let mut b = ThetaSketch::new(4096);
        for i in 0..3_000 {
            a.update(&i);
            b.update(&i);
        }
        for i in 3_000..10_000 {
            b.update(&i);
        }
        let diff = ThetaSketch::difference(&a, &b, 4096);
        let est = diff.estimate();
        assert!(
            est < 200.0,
            "Subset difference estimate {est} should be near 0"
        );
    }

    // -----------------------------------------------------------------------
    // Serialisation round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialisation_round_trip_empty() {
        let sk = ThetaSketch::new(1024);
        let bytes = sk.to_bytes();
        let sk2 = ThetaSketch::from_bytes(&bytes).unwrap();
        assert_eq!(sk2.k, sk.k);
        assert_eq!(sk2.theta, sk.theta);
        assert_eq!(sk2.num_entries, sk.num_entries);
        assert_eq!(sk2.estimate(), sk.estimate());
    }

    #[test]
    fn test_serialisation_round_trip_populated() {
        let mut sk = ThetaSketch::new(1024);
        for i in 0..5_000 {
            sk.update(&i);
        }
        let bytes = sk.to_bytes();
        let sk2 = ThetaSketch::from_bytes(&bytes).unwrap();
        assert_eq!(sk2.k, sk.k);
        assert_eq!(sk2.theta, sk.theta);
        assert_eq!(sk2.num_entries, sk.num_entries);
        assert!((sk2.estimate() - sk.estimate()).abs() < 1e-10);
    }

    #[test]
    fn test_serialisation_round_trip_below_k() {
        let mut sk = ThetaSketch::new(4096);
        for i in 0..100 {
            sk.update(&i);
        }
        let bytes = sk.to_bytes();
        let sk2 = ThetaSketch::from_bytes(&bytes).unwrap();
        assert_eq!(sk2.k, sk.k);
        assert_eq!(sk2.theta, MAX_THETA);
        assert_eq!(sk2.num_entries, 100);
        assert_eq!(sk2.estimate(), 100.0);
    }

    #[test]
    fn test_from_bytes_insufficient_header() {
        let bytes = vec![0u8; 10];
        assert!(ThetaSketch::from_bytes(&bytes).is_err());
    }

    #[test]
    fn test_from_bytes_insufficient_values() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1024usize.to_le_bytes());
        bytes.extend_from_slice(&MAX_THETA.to_le_bytes());
        bytes.extend_from_slice(&10usize.to_le_bytes()); // claims 10 values
        // but no value bytes
        assert!(ThetaSketch::from_bytes(&bytes).is_err());
    }

    // -----------------------------------------------------------------------
    // Sample capacity
    // -----------------------------------------------------------------------

    #[test]
    fn test_sample_capacity() {
        let sk = ThetaSketch::new(1024);
        // The internal table should have been allocated.
        assert!(sk.sample_capacity() >= sk.entries.len());
    }

    // -----------------------------------------------------------------------
    // Hash table internal mechanics
    // -----------------------------------------------------------------------

    #[test]
    fn test_stride_is_always_odd() {
        // Odd stride guarantees full cycle in a power-of-two table.
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
    fn test_rebuild_keeps_values_below_theta() {
        let k = 64;
        let mut sk = ThetaSketch::new(k);
        // Insert enough items to trigger several rebuilds.
        for i in 0..10_000 {
            sk.update(&i);
        }
        assert!(sk.theta < MAX_THETA);
        // Between rebuilds, num_entries can exceed k (up to rebuild threshold).
        // But it should be bounded by the rebuild threshold: table_size * 15/16.
        let table_size = 1usize << sk.lg_cur_size;
        let rebuild_threshold = (REBUILD_LOAD_FACTOR * table_size as f64) as usize;
        assert!(
            sk.num_entries <= rebuild_threshold,
            "num_entries {} should not exceed rebuild threshold {}",
            sk.num_entries,
            rebuild_threshold
        );
        // Every retained value must be below theta.
        for v in sk.retained_values() {
            assert!(
                v < sk.theta,
                "Retained value {} must be below theta {}",
                v,
                sk.theta
            );
        }
    }

    #[test]
    fn test_no_zero_entries_stored() {
        let mut sk = ThetaSketch::new(1024);
        for i in 0..5_000 {
            sk.update(&i);
        }
        // The retained_values iterator filters zeros; verify the count matches.
        let retained: Vec<u64> = sk.retained_values().collect();
        assert_eq!(retained.len(), sk.num_entries);
        assert!(!retained.contains(&0));
    }

    // -----------------------------------------------------------------------
    // Hash collector
    // -----------------------------------------------------------------------

    #[test]
    fn test_hash_collector_basic() {
        let mut hc = HashCollector::with_capacity(100);
        assert!(hc.insert(42));
        assert!(!hc.insert(42)); // duplicate
        assert!(hc.insert(99));
        assert!(hc.contains(42));
        assert!(hc.contains(99));
        assert!(!hc.contains(0));
        assert!(!hc.contains(1));

        let vals = hc.into_values();
        assert_eq!(vals.len(), 2);
        assert!(vals.contains(&42));
        assert!(vals.contains(&99));
    }

    #[test]
    fn test_hash_collector_grow() {
        let mut hc = HashCollector::with_capacity(4);
        for i in 1..1000u64 {
            hc.insert(i);
        }
        for i in 1..1000u64 {
            assert!(hc.contains(i));
        }
        assert_eq!(hc.count, 999);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_k_equals_one() {
        let mut sk = ThetaSketch::new(1);
        assert_eq!(sk.k, 1);
        for i in 0..1_000 {
            sk.update(&i);
        }
        assert!(sk.num_entries <= 1);
        // With k=1, the estimate uses (k-1)/theta = 0/theta = 0 when fully sampled.
        // This is a known limitation of the (k-1) estimator at k=1.
        // The sketch is still functional; it just cannot produce a useful estimate.
    }

    #[test]
    fn test_union_with_different_k() {
        let mut a = ThetaSketch::new(512);
        let mut b = ThetaSketch::new(2048);
        for i in 0..20_000 {
            a.update(&i);
        }
        for i in 10_000..30_000 {
            b.update(&i);
        }
        // Union using the smaller k (from a.union which uses self.k).
        let u = a.union(&b);
        assert!(u.k == 512);
        let est = u.estimate();
        let expected = 30_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.15,
            "Union with different k: estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_update_after_serialisation() {
        let mut sk = ThetaSketch::new(1024);
        for i in 0..5_000 {
            sk.update(&i);
        }
        let bytes = sk.to_bytes();
        let mut sk2 = ThetaSketch::from_bytes(&bytes).unwrap();
        // Continue updating the deserialized sketch.
        for i in 5_000..10_000 {
            sk2.update(&i);
        }
        let est = sk2.estimate();
        let expected = 10_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.10,
            "Post-deserialisation update: estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_set_operations_preserve_theta() {
        let mut a = ThetaSketch::new(1024);
        let mut b = ThetaSketch::new(1024);
        for i in 0..50_000 {
            a.update(&i);
        }
        for i in 25_000..75_000 {
            b.update(&i);
        }
        // Both sketches are fully sampled (theta < MAX).
        assert!(a.theta < MAX_THETA);
        assert!(b.theta < MAX_THETA);

        let u = a.union(&b);
        // Union theta should not exceed the minimum of input thetas.
        assert!(u.theta <= a.theta.min(b.theta));

        let inter = a.intersect(&b);
        assert!(inter.theta <= a.theta.min(b.theta));

        let diff = ThetaSketch::difference(&a, &b, 1024);
        assert!(diff.theta <= a.theta.min(b.theta));
    }

    #[test]
    fn test_identical_sketches_union() {
        let mut a = ThetaSketch::new(2048);
        for i in 0..10_000 {
            a.update(&i);
        }
        let u = a.union(&a);
        let est = u.estimate();
        let expected = 10_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.10,
            "Self-union estimate {est} too far from {expected} (error {error_ratio})"
        );
    }

    #[test]
    fn test_identical_sketches_intersect() {
        let mut a = ThetaSketch::new(2048);
        for i in 0..10_000 {
            a.update(&i);
        }
        let inter = a.intersect(&a);
        let est = inter.estimate();
        let expected = 10_000.0;
        let error_ratio = (est - expected).abs() / expected;
        assert!(
            error_ratio < 0.10,
            "Self-intersect estimate {est} too far from {expected} (error {error_ratio})"
        );
    }
}
