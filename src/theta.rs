use std::collections::BinaryHeap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Theta Sketch for approximate set cardinality and set operations.
pub struct ThetaSketch {
    /// Nominal sample size (sketch capacity).
    pub k: usize,
    heap: BinaryHeap<u64>,
    theta: u64,
    // Note: duplicates are skipped by checking the heap, so no separate seen set is stored.
}

impl ThetaSketch {
    /// Create a new Theta sketch with sample size k.
    pub fn new(k: usize) -> Self {
        ThetaSketch {
            k,
            heap: BinaryHeap::new(),
            theta: u64::MAX,
        }
    }

    /// Update the sketch with an item implementing Hash.
    pub fn update<T: Hash>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        // Skip duplicate hashed values to ensure unique sampling (heap scan)
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
        let mut result = ThetaSketch {
            k,
            heap: BinaryHeap::new(),
            theta: u64::MAX,
        };
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
        let mut values: Vec<u64> = a
            .heap
            .iter()
            .filter(|&&v| v < theta && b.heap.iter().any(|&bv| bv == v))
            .cloned()
            .collect();
        values.sort_unstable();
        let mut result = ThetaSketch {
            k,
            heap: BinaryHeap::new(),
            theta,
        };
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
        let mut values: Vec<u64> = a
            .heap
            .iter()
            .filter(|&&v| v < theta && !b.heap.iter().any(|&bv| bv == v))
            .cloned()
            .collect();
        values.sort_unstable();
        let mut result = ThetaSketch {
            k,
            heap: BinaryHeap::new(),
            theta,
        };
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
    /// Returns a new sketch containing elements present in both sketches.
    pub fn intersect(&self, other: &ThetaSketch) -> ThetaSketch {
        // Use the more conservative capacity (smaller k)
        let result_k = self.k.min(other.k);
        
        // Use minimum theta from both sketches
        let result_theta = self.theta.min(other.theta);
        
        // Find intersection of hash values that are less than result_theta
        let mut intersection_values = Vec::new();
        
        for &value in &self.heap {
            if value < result_theta && other.heap.iter().any(|&other_val| other_val == value) {
                intersection_values.push(value);
            }
        }
        
        // Sort and take up to k values
        intersection_values.sort_unstable();
        intersection_values.truncate(result_k);
        
        // Create result sketch
        let mut result_heap = BinaryHeap::new();
        for value in intersection_values {
            result_heap.push(value);
        }
        
        // Adjust theta if we have exactly k elements
        let final_theta = if result_heap.len() == result_k {
            result_heap.peek().copied().unwrap_or(result_theta)
        } else {
            result_theta
        };
        
        ThetaSketch {
            k: result_k,
            heap: result_heap,
            theta: final_theta,
        }
    }

    /// Return capacity of the underlying sample data vector.
    pub fn sample_capacity(&self) -> usize {
        let vec: Vec<_> = self.heap.clone().into_vec();
        vec.capacity()
    }

    /// Serialize the sketch to bytes for storage or transmission.
    pub fn to_bytes(&self) -> Vec<u8> {
        // Simple serialization format:
        // 8 bytes: k (usize)
        // 8 bytes: theta (u64)
        // 8 bytes: heap length (usize)
        // N * 8 bytes: heap values (u64 each)
        
        let mut bytes = Vec::new();
        
        // Serialize k
        bytes.extend_from_slice(&self.k.to_le_bytes());
        
        // Serialize theta
        bytes.extend_from_slice(&self.theta.to_le_bytes());
        
        // Convert heap to sorted vector for consistent serialization
        let mut heap_values: Vec<u64> = self.heap.iter().copied().collect();
        heap_values.sort_unstable();
        
        // Serialize heap length
        bytes.extend_from_slice(&heap_values.len().to_le_bytes());
        
        // Serialize heap values
        for value in heap_values {
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
        
        // Deserialize k
        let k = usize::from_le_bytes(
            bytes[offset..offset + 8].try_into().map_err(|_| "Invalid k bytes")?
        );
        offset += 8;
        
        // Deserialize theta
        let theta = u64::from_le_bytes(
            bytes[offset..offset + 8].try_into().map_err(|_| "Invalid theta bytes")?
        );
        offset += 8;
        
        // Deserialize heap length
        let heap_len = usize::from_le_bytes(
            bytes[offset..offset + 8].try_into().map_err(|_| "Invalid heap length bytes")?
        );
        offset += 8;
        
        // Check remaining bytes for heap values
        if bytes.len() < offset + heap_len * 8 {
            return Err("Insufficient bytes for heap values");
        }
        
        // Deserialize heap values
        let mut heap = BinaryHeap::new();
        for _i in 0..heap_len {
            let value = u64::from_le_bytes(
                bytes[offset..offset + 8].try_into().map_err(|_| "Invalid heap value bytes")?
            );
            heap.push(value);
            offset += 8;
        }
        
        Ok(ThetaSketch { k, heap, theta })
    }
}
