//! Theta sketch for approximate set cardinality and set operations.
use std::collections::BinaryHeap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Theta Sketch for approximate set cardinality and set operations.
pub struct ThetaSketch {
    /// Nominal sample size (sketch capacity).
    pub k: usize,
    heap: BinaryHeap<u64>,
    theta: u64,
}

impl ThetaSketch {
    /// Create a new Theta sketch with sample size k.
    pub fn new(k: usize) -> Self {
        ThetaSketch { k, heap: BinaryHeap::new(), theta: u64::MAX }
    }

    /// Update the sketch with an item implementing Hash.
    pub fn update<T: Hash>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
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
        if n < self.k {
            n as f64
        } else {
            let theta_f = (self.theta as f64) / (u64::MAX as f64);
            // Use bias-corrected estimator (k-1)/theta
            ((self.k - 1) as f64) / theta_f
        }
    }

    /// Union of multiple sketches, returning a new sketch with capacity k.
    pub fn union(sketches: &[&ThetaSketch], k: usize) -> ThetaSketch {
        let mut min_theta = u64::MAX;
        for sk in sketches {
            if sk.theta < min_theta {
                min_theta = sk.theta;
            }
        }
        let mut values: Vec<u64> = Vec::new();
        for sk in sketches {
            for &v in sk.heap.iter() {
                if v < min_theta {
                    values.push(v);
                }
            }
        }
        values.sort_unstable();
        let mut result = ThetaSketch { k, heap: BinaryHeap::new(), theta: min_theta };
        for &v in values.iter().take(k) {
            result.heap.push(v);
        }
        if result.heap.len() == k {
            result.theta = *result.heap.peek().unwrap();
        }
        result
    }

    /// Intersection of two sketches.
    pub fn intersect(a: &ThetaSketch, b: &ThetaSketch, k: usize) -> ThetaSketch {
        let theta = a.theta.min(b.theta);
        let mut values: Vec<u64> = a.heap.iter()
            .filter(|&&v| v < theta && b.heap.iter().any(|&bv| bv == v))
            .cloned()
            .collect();
        values.sort_unstable();
        let mut result = ThetaSketch { k, heap: BinaryHeap::new(), theta };
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
        let mut result = ThetaSketch { k, heap: BinaryHeap::new(), theta };
        for &v in values.iter().take(k) {
            result.heap.push(v);
        }
        if result.heap.len() == k {
            result.theta = *result.heap.peek().unwrap();
        }
        result
    }
}
