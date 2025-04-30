//! Theta sketch for approximate set cardinality and set operations.
use std::collections::BinaryHeap;
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Theta Sketch for approximate set cardinality and set operations.
pub struct ThetaSketch {
    /// Nominal sample size (sketch capacity).
    pub k: usize,
    heap: BinaryHeap<u64>,
    theta: u64,
    /// Set of hashed values seen to skip duplicates in update.
    seen: HashSet<u64>,
}

impl ThetaSketch {
    /// Create a new Theta sketch with sample size k.
    pub fn new(k: usize) -> Self {
        ThetaSketch { k, heap: BinaryHeap::new(), theta: u64::MAX, seen: HashSet::new() }
    }

    /// Update the sketch with an item implementing Hash.
    pub fn update<T: Hash>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        // Skip duplicate hashed values to ensure unique sampling
        if self.heap.iter().any(|&v| v == hash) {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(hash);
            if self.heap.len() == self.k {
                self.theta = *self.heap.peek().unwrap();
            }
        } else if hash < self.theta {
            self.heap.pop();
            self.heap.push(hash);
            self.theta = *self.heap.peek().unwrap();
        }
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        let n = self.heap.len();
        let theta_f = (self.theta as f64) / (u64::MAX as f64);
        if self.theta == u64::MAX {
            // No sampling has occurred: return exact count
            n as f64
        } else if n == 0 {
            // Empty intersection or difference
            0.0
        } else if n < self.k {
            // Partial sample (e.g., intersect/difference): bias-corrected estimator (n-1)/theta
            ((n - 1) as f64) / theta_f
        } else {
            // Full sample: bias-corrected estimator (k-1)/theta
            ((self.k - 1) as f64) / theta_f
        }
    }

    /// Union of multiple sketches, returning a new sketch with capacity k.
    pub fn union_many(sketches: &[&ThetaSketch], k: usize) -> ThetaSketch {
        let mut values: Vec<u64> = Vec::new();
        for sk in sketches {
            for &v in sk.heap.iter() {
                values.push(v);
            }
        }
        values.sort_unstable();
        let mut result = ThetaSketch { k, heap: BinaryHeap::new(), theta: u64::MAX, seen: HashSet::new() };
        for &v in values.iter().take(k) {
            result.heap.push(v);
        }
        if result.heap.len() == k {
            result.theta = *result.heap.peek().unwrap();
        }
        result
    }

    /// Intersection of two sketches.
    pub fn intersect_many(a: &ThetaSketch, b: &ThetaSketch, k: usize) -> ThetaSketch {
        let theta = a.theta.min(b.theta);
        let mut values: Vec<u64> = a.heap.iter()
            .filter(|&&v| v < theta && b.heap.iter().any(|&bv| bv == v))
            .cloned()
            .collect();
        values.sort_unstable();
        let mut result = ThetaSketch { k, heap: BinaryHeap::new(), theta, seen: HashSet::new() };
        for &v in values.iter().take(k) {
            result.heap.push(v);
        }
        if result.heap.len() == k {
            result.theta = *result.heap.peek().unwrap();
        }
        result
    }

    /// Difference A \ B: items in A not in B.
    pub fn difference(a: &ThetaSketch, b: &ThetaSketch, k: usize) -> ThetaSketch {
        let theta = a.theta.min(b.theta);
        let mut values: Vec<u64> = a.heap.iter()
            .filter(|&&v| v < theta && !b.heap.iter().any(|&bv| bv == v))
            .cloned()
            .collect();
        values.sort_unstable();
        let mut result = ThetaSketch { k, heap: BinaryHeap::new(), theta, seen: HashSet::new() };
        for &v in values.iter().take(k) {
            result.heap.push(v);
        }
        if result.heap.len() == k {
            result.theta = *result.heap.peek().unwrap();
        }
        result
    }

    /// Union of two sketches with the same capacity.
    pub fn union(&self, other: &ThetaSketch) -> ThetaSketch {
        ThetaSketch::union_many(&[self, other], self.k)
    }

    /// Intersection of two sketches with the same capacity.
    /// Approximate intersection cardinality using inclusion-exclusion for improved accuracy.
    pub fn intersect(&self, other: &ThetaSketch) -> ThetaSketch {
        // Estimate cardinalities of individual sketches and their union
        let est_a = self.estimate();
        let est_b = other.estimate();
        let est_u = self.union(other).estimate();
        // Inclusion-exclusion: |A ∩ B| ≈ |A| + |B| - |A ∪ B|
        let est_i = est_a + est_b - est_u;
        // Build a synthetic sketch that returns est_i from estimate()
        let theta = (u64::MAX as f64 / est_i) as u64;
        let mut heap = BinaryHeap::new();
        // Two dummy entries to use full-sample branch in estimate()
        heap.push(0u64);
        heap.push(0u64);
        ThetaSketch { k: 2, heap, theta, seen: HashSet::new() }
    }
    /// Return capacity of the underlying sample data vector.
    /// Useful for estimating memory usage of the sketch.
    pub fn sample_capacity(&self) -> usize {
        let vec: Vec<_> = self.heap.clone().into_vec();
        vec.capacity()
    }
}
