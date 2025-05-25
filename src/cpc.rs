use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// Simplified CPC Sketch implementation that provides good accuracy
pub struct CpcSketch {
    lg_k: u8,
    k: usize,
    sparse_mode: bool,
    sparse_set: HashSet<u64>,
    table: Vec<u8>,
}

impl CpcSketch {
    /// Create a new CPC sketch
    pub fn new(lg_k: u8) -> Self {
        assert!(lg_k >= 4 && lg_k <= 26, "lg_k must be between 4 and 26");
        
        let k = 1 << lg_k;
        
        CpcSketch {
            lg_k,
            k,
            sparse_mode: true,
            sparse_set: HashSet::new(),
            table: vec![0; k],
        }
    }

    /// Update the sketch with an item
    pub fn update<T: Hash>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        
        if self.sparse_mode {
            self.sparse_set.insert(hash);
            
            // Switch to table mode if sparse set gets too large
            if self.sparse_set.len() > self.k / 4 {
                self.switch_to_table_mode();
            }
        } else {
            // Table mode - similar to HLL
            let index = (hash & ((1 << self.lg_k) - 1)) as usize;
            let w = hash >> self.lg_k;
            let rho = if w == 0 {
                64 - self.lg_k + 1
            } else {
                w.leading_zeros() as u8 + 1
            };
            
            if rho > self.table[index] {
                self.table[index] = rho;
            }
        }
    }

    /// Switch from sparse to table mode
    fn switch_to_table_mode(&mut self) {
        self.sparse_mode = false;
        
        // Re-hash all items into the table
        for &hash in &self.sparse_set {
            let index = (hash & ((1 << self.lg_k) - 1)) as usize;
            let w = hash >> self.lg_k;
            let rho = if w == 0 {
                64 - self.lg_k + 1
            } else {
                w.leading_zeros() as u8 + 1
            };
            
            if rho > self.table[index] {
                self.table[index] = rho;
            }
        }
        
        self.sparse_set.clear();
    }

    /// Estimate cardinality
    pub fn estimate(&self) -> f64 {
        if self.sparse_mode {
            // In sparse mode, return exact count for small values
            let n = self.sparse_set.len() as f64;
            
            // For CPC sparse mode, we can just return the exact count
            // since we're storing actual hashes
            n
        } else {
            // Table mode - use HLL estimator
            let m = self.k as f64;
            let mut raw_estimate = 0.0;
            let mut zeros = 0;
            
            for &val in &self.table {
                if val == 0 {
                    zeros += 1;
                }
                raw_estimate += 1.0 / (1u64 << val) as f64;
            }
            
            // Alpha constant
            let alpha = if self.lg_k == 4 {
                0.673
            } else if self.lg_k == 5 {
                0.697
            } else if self.lg_k == 6 {
                0.709
            } else {
                0.7213 / (1.0 + 1.079 / m)
            };
            
            let mut estimate = alpha * m * m / raw_estimate;
            
            // Small range correction
            if estimate <= 2.5 * m && zeros > 0 {
                estimate = m * (m / zeros as f64).ln();
            }
            
            estimate
        }
    }

    /// Merge another sketch
    pub fn merge(&mut self, other: &CpcSketch) {
        assert_eq!(self.lg_k, other.lg_k, "Cannot merge sketches with different lg_k");
        
        // If either is in table mode, switch both to table mode
        if !self.sparse_mode || !other.sparse_mode {
            if self.sparse_mode {
                self.switch_to_table_mode();
            }
            
            // Merge tables
            if other.sparse_mode {
                // Add other's sparse items
                for &hash in &other.sparse_set {
                    let index = (hash & ((1 << self.lg_k) - 1)) as usize;
                    let w = hash >> self.lg_k;
                    let rho = if w == 0 {
                        64 - self.lg_k + 1
                    } else {
                        w.leading_zeros() as u8 + 1
                    };
                    
                    if rho > self.table[index] {
                        self.table[index] = rho;
                    }
                }
            } else {
                // Merge other's table
                for i in 0..self.k {
                    if other.table[i] > self.table[i] {
                        self.table[i] = other.table[i];
                    }
                }
            }
        } else {
            // Both in sparse mode
            for &hash in &other.sparse_set {
                self.sparse_set.insert(hash);
                
                // Check if we need to switch
                if self.sparse_set.len() > self.k / 4 {
                    self.switch_to_table_mode();
                    break;
                }
            }
        }
    }

    /// Serialize to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        
        // Header
        bytes.push(self.lg_k);
        bytes.push(if self.sparse_mode { 0 } else { 1 });
        
        if self.sparse_mode {
            // Sparse mode
            bytes.extend_from_slice(&(self.sparse_set.len() as u32).to_le_bytes());
            for &hash in &self.sparse_set {
                bytes.extend_from_slice(&hash.to_le_bytes());
            }
        } else {
            // Table mode - simple RLE compression
            let mut i = 0;
            while i < self.table.len() {
                let val = self.table[i];
                let mut run = 1;
                
                while i + run < self.table.len() && self.table[i + run] == val && run < 255 {
                    run += 1;
                }
                
                if val == 0 && run > 2 {
                    bytes.push(0);
                    bytes.push(run as u8);
                } else {
                    for _ in 0..run {
                        bytes.push(val);
                    }
                }
                
                i += run;
            }
        }
        
        bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpc_basic() {
        let mut sketch = CpcSketch::new(10);
        
        for i in 0..1000 {
            sketch.update(&i);
        }
        
        let estimate = sketch.estimate();
        let error = (estimate - 1000.0).abs() / 1000.0;
        
        assert!(error < 0.05, "Error {} is too high", error);
    }

    #[test]
    fn test_cpc_merge() {
        let mut sketch1 = CpcSketch::new(10);
        let mut sketch2 = CpcSketch::new(10);
        
        for i in 0..500 {
            sketch1.update(&i);
        }
        
        for i in 500..1000 {
            sketch2.update(&i);
        }
        
        sketch1.merge(&sketch2);
        
        let estimate = sketch1.estimate();
        let error = (estimate - 1000.0).abs() / 1000.0;
        
        assert!(error < 0.05, "Merge error {} is too high", error);
    }

    #[test]
    fn test_cpc_small() {
        let mut sketch = CpcSketch::new(8);
        
        for i in 0..50 {
            sketch.update(&format!("item_{}", i));
        }
        
        let estimate = sketch.estimate();
        let error = (estimate - 50.0).abs() / 50.0;
        
        assert!(error < 0.1, "Small cardinality error {} is too high", error);
    }
}