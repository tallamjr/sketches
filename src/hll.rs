// HyperLogLog (HLL) and HyperLogLog++ implementations.

use std::collections::{BTreeMap, hash_map::DefaultHasher};
use std::hash::{Hash, Hasher};

#[cfg(feature = "optimized")]
use crate::compact_memory::PackedRegisters;
#[cfg(feature = "optimized")]
use crate::fast_hash;
#[cfg(feature = "optimized")]
use crate::simd_ops::hyperloglog;

/// HyperLogLog Sketch for approximate distinct counting.
pub struct HllSketch {
    p: u8,
    m: usize,
    #[cfg(feature = "optimized")]
    registers: PackedRegisters,
    #[cfg(not(feature = "optimized"))]
    registers: Vec<u8>,
}

impl HllSketch {
    /// Create a new HLL sketch with precision p (number of register index bits).
    pub fn new(p: u8) -> Self {
        let m = 1 << p;
        HllSketch {
            p,
            m,
            #[cfg(feature = "optimized")]
            registers: PackedRegisters::new(p, 6), // 6 bits per register for HLL
            #[cfg(not(feature = "optimized"))]
            registers: vec![0; m],
        }
    }

    /// Update the sketch with an item implementing Hash.
    pub fn update<T: Hash>(&mut self, item: &T) {
        #[cfg(feature = "optimized")]
        {
            let hash = fast_hash::fast_hash(item);
            let idx = (hash >> (64 - self.p)) as usize;
            let w = hash << self.p;
            let leading = w.leading_zeros().saturating_add(1);
            let rank = leading.min(64) as u8;
            self.registers.update_max(idx, rank);
        }

        #[cfg(not(feature = "optimized"))]
        {
            let mut hasher = DefaultHasher::new();
            item.hash(&mut hasher);
            let hash = hasher.finish();
            let idx = (hash >> (64 - self.p)) as usize;
            let w = hash << self.p;
            let leading = w.leading_zeros().saturating_add(1);
            let rank = leading.min(64) as u8;
            if self.registers[idx] < rank {
                self.registers[idx] = rank;
            }
        }
    }

    /// Batch update the sketch with multiple items for better performance.
    #[cfg(feature = "optimized")]
    pub fn update_batch<T: Hash + Sync>(&mut self, items: &[T]) {
        use rayon::prelude::*;

        // Parallel hashing with rayon for large batches
        let hashes: Vec<u64> = if items.len() > 10000 {
            items
                .par_iter()
                .map(|item| fast_hash::fast_hash(item))
                .collect()
        } else {
            items
                .iter()
                .map(|item| fast_hash::fast_hash(item))
                .collect()
        };

        // Use SIMD for leading zeros computation when available
        let w_values: Vec<u64> = hashes.iter().map(|&hash| hash << self.p).collect();
        let leading_zeros = hyperloglog::leading_zeros_batch(&w_values);

        // Parallel computation of positions and ranks
        let updates: Vec<(usize, u8)> = hashes
            .par_iter()
            .zip(leading_zeros.par_iter())
            .map(|(&hash, &lz)| {
                let idx = (hash >> (64 - self.p)) as usize;
                let rank = (lz.saturating_add(1)).min(64) as u8;
                (idx, rank)
            })
            .collect();

        // Apply updates with branch-free operations
        for (pos, rank) in updates {
            self.registers.update_max(pos, rank);
        }
    }

    /// Batch update fallback for non-optimized builds.
    #[cfg(not(feature = "optimized"))]
    pub fn update_batch<T: Hash>(&mut self, items: &[T]) {
        for item in items {
            self.update(item);
        }
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        let m = self.m as f64;
        let mut sum = 0f64;
        let mut zeros = 0usize;

        #[cfg(feature = "optimized")]
        {
            for reg in self.registers.iter() {
                sum += 2f64.powf(-(reg as f64));
                if reg == 0 {
                    zeros += 1;
                }
            }
        }

        #[cfg(not(feature = "optimized"))]
        {
            for &reg in &self.registers {
                sum += 2f64.powf(-(reg as f64));
                if reg == 0 {
                    zeros += 1;
                }
            }
        }

        let alpha = match self.m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let estimate = alpha * m * m / sum;
        // small range correction
        if estimate <= 2.5 * m && zeros > 0 {
            m * (m / zeros as f64).ln()
        } else {
            estimate
        }
    }

    /// Merge another sketch into this one (in-place union).
    pub fn merge(&mut self, other: &HllSketch) {
        if self.p != other.p {
            panic!(
                "Cannot merge HLL sketches with different precision: {} vs {}",
                self.p, other.p
            );
        }

        #[cfg(feature = "optimized")]
        {
            for i in 0..self.m {
                let self_val = self.registers.get(i);
                let other_val = other.registers.get(i);
                if self_val < other_val {
                    self.registers.set(i, other_val);
                }
            }
        }

        #[cfg(not(feature = "optimized"))]
        {
            for i in 0..self.m {
                if self.registers[i] < other.registers[i] {
                    self.registers[i] = other.registers[i];
                }
            }
        }
    }

    /// Serialize registers to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        #[cfg(feature = "optimized")]
        {
            self.registers.iter().collect()
        }

        #[cfg(not(feature = "optimized"))]
        {
            self.registers.clone()
        }
    }
}

/// HyperLogLog++ Sketch for approximate distinct counting with improved small-range estimator.
pub struct HllPlusPlusSketch {
    p: u8,
    m: usize,
    registers: Vec<u8>,
}

impl HllPlusPlusSketch {
    /// Create a new HLL++ sketch with precision p (number of index bits).
    /// Supported p values are in [4, 18].
    pub fn new(p: u8) -> Self {
        assert!(
            (4..=18).contains(&p),
            "Precision p must be between 4 and 18"
        );
        let m = 1usize << p;
        HllPlusPlusSketch {
            p,
            m,
            registers: vec![0; m],
        }
    }

    /// Update the sketch with an item implementing Hash.
    pub fn update<T: Hash>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        let idx = (hash >> (64 - self.p)) as usize;
        let w = hash << self.p;
        // number of leading zeros in w plus 1
        let leading = w.leading_zeros().saturating_add(1);
        let rank = leading.min(64) as u8;
        let reg = &mut self.registers[idx];
        if *reg < rank {
            *reg = rank;
        }
    }

    /// Estimate the cardinality using HLL++ algorithm.
    /// Uses linear counting for very small cardinalities based on a threshold table,
    /// otherwise returns the HyperLogLog raw estimate.
    pub fn estimate(&self) -> f64 {
        let m_f = self.m as f64;
        // compute raw HyperLogLog estimate
        let mut sum = 0f64;
        let mut zeros = 0usize;
        for &reg in &self.registers {
            sum += 2f64.powf(-(reg as f64));
            if reg == 0 {
                zeros += 1;
            }
        }
        let alpha = match self.m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m_f),
        };
        let raw_est = alpha * m_f * m_f / sum;
        // linear counting estimate for small cardinalities
        if zeros > 0 {
            let lin = m_f * (m_f / (zeros as f64)).ln();
            // threshold table for p = 4..18
            const THRESH: [f64; 15] = [
                10.0,     // p=4
                20.0,     // p=5
                40.0,     // p=6
                80.0,     // p=7
                220.0,    // p=8
                400.0,    // p=9
                900.0,    // p=10
                1800.0,   // p=11
                3100.0,   // p=12
                6500.0,   // p=13
                11500.0,  // p=14
                20000.0,  // p=15
                50000.0,  // p=16
                120000.0, // p=17
                350000.0, // p=18
            ];
            let idx = (self.p as usize).saturating_sub(4);
            if idx < THRESH.len() && lin <= THRESH[idx] {
                return lin;
            }
        }
        raw_est
    }

    /// Merge another HLL++ sketch into this one (in-place union).
    pub fn merge(&mut self, other: &HllPlusPlusSketch) {
        assert_eq!(
            self.p, other.p,
            "Cannot merge sketches with different precision"
        );
        for (r, o) in self.registers.iter_mut().zip(other.registers.iter()) {
            if *r < *o {
                *r = *o;
            }
        }
    }

    /// Serialize registers to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.registers.clone()
    }
}

/// Sparse HyperLogLog++ sketch storing only non-zero registers.
pub struct HllPlusPlusSparseSketch {
    p: u8,
    m: usize,
    map: BTreeMap<usize, u8>,
}

impl HllPlusPlusSparseSketch {
    /// Create a new sparse HLL++ sketch with precision p in [4,18].
    pub fn new(p: u8) -> Self {
        assert!(
            (4..=18).contains(&p),
            "Precision p must be between 4 and 18"
        );
        let m = 1usize << p;
        HllPlusPlusSparseSketch {
            p,
            m,
            map: BTreeMap::new(),
        }
    }

    /// Update the sketch with an item implementing Hash.
    pub fn update<T: Hash>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        let idx = (hash >> (64 - self.p)) as usize;
        let w = hash << self.p;
        let rank = w.leading_zeros().saturating_add(1).min(64) as u8;
        let entry = self.map.entry(idx).or_insert(0);
        if *entry < rank {
            *entry = rank;
        }
    }

    /// Estimate the cardinality using sparse HLL++.
    pub fn estimate(&self) -> f64 {
        let m_f = self.m as f64;
        let zeros = self.m - self.map.len();
        let sum_nonzero: f64 = self.map.values().map(|&r| 2f64.powf(-(r as f64))).sum();
        let sum = sum_nonzero + zeros as f64 * 1.0;
        let alpha = match self.m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m_f),
        };
        let raw_est = alpha * m_f * m_f / sum;
        if zeros > 0 {
            let lin = m_f * (m_f / zeros as f64).ln();
            const THRESH: [f64; 15] = [
                10.0, 20.0, 40.0, 80.0, 220.0, 400.0, 900.0, 1800.0, 3100.0, 6500.0, 11500.0,
                20000.0, 50000.0, 120000.0, 350000.0,
            ];
            let idx_th = (self.p as usize).saturating_sub(4);
            if idx_th < THRESH.len() && lin <= THRESH[idx_th] {
                return lin;
            }
        }
        raw_est
    }

    /// Merge another sparse sketch into this one.
    pub fn merge(&mut self, other: &HllPlusPlusSparseSketch) {
        assert_eq!(
            self.p, other.p,
            "Cannot merge sketches with different precision"
        );
        for (&idx, &r) in other.map.iter() {
            let entry = self.map.entry(idx).or_insert(0);
            if *entry < r {
                *entry = r;
            }
        }
    }

    /// Serialize to dense byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut regs = vec![0; self.m];
        for (&idx, &r) in self.map.iter() {
            regs[idx] = r;
        }
        regs
    }
}

#[cfg(test)]
mod tests {
    use super::{HllPlusPlusSketch, HllPlusPlusSparseSketch, HllSketch};

    #[test]
    fn estimate_empty() {
        let sk = HllSketch::new(4);
        assert_eq!(sk.estimate(), 0.0);
    }

    #[test]
    fn estimate_empty_plus() {
        let sk = HllPlusPlusSketch::new(4);
        assert_eq!(sk.estimate(), 0.0);
    }

    #[test]
    fn estimate_empty_sparse() {
        let sk = HllPlusPlusSparseSketch::new(4);
        assert_eq!(sk.estimate(), 0.0);
    }

    #[test]
    fn sparse_matches_dense() {
        let mut sd = HllPlusPlusSketch::new(10);
        let mut ss = HllPlusPlusSparseSketch::new(10);
        let values = vec!["a", "b", "c", "d", "a", "c"];
        for &v in &values {
            sd.update(&v);
            ss.update(&v);
        }
        assert_eq!(sd.to_bytes(), ss.to_bytes());
    }
}
