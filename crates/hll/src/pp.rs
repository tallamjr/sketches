use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::collections::BTreeMap;

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
        assert!((4..=18).contains(&p), "Precision p must be between 4 and 18");
        let m = 1usize << p;
        HllPlusPlusSketch { p, m, registers: vec![0; m] }
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
                10.0,    // p=4
                20.0,    // p=5
                40.0,    // p=6
                80.0,    // p=7
                220.0,   // p=8
                400.0,   // p=9
                900.0,   // p=10
                1800.0,  // p=11
                3100.0,  // p=12
                6500.0,  // p=13
                11500.0, // p=14
                20000.0, // p=15
                50000.0, // p=16
                120000.0,// p=17
                350000.0,// p=18
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
        assert_eq!(self.p, other.p, "Cannot merge sketches with different precision");
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
        assert!((4..=18).contains(&p), "Precision p must be between 4 and 18");
        let m = 1usize << p;
        HllPlusPlusSparseSketch { p, m, map: BTreeMap::new() }
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
                10.0, 20.0, 40.0, 80.0, 220.0, 400.0, 900.0, 1800.0,
                3100.0, 6500.0, 11500.0, 20000.0, 50000.0, 120000.0, 350000.0,
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
        assert_eq!(self.p, other.p, "Cannot merge sketches with different precision");
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
    use super::{HllPlusPlusSketch, HllPlusPlusSparseSketch};

    #[test]
    fn estimate_empty() {
        let sk = HllPlusPlusSketch::new(4);
        // no items -> estimate should be 0 via linear counting
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
        for i in 0..1000 {
            sd.update(&i);
            ss.update(&i);
        }
        let ed = sd.estimate();
        let es = ss.estimate();
        let err = (ed - es).abs();
        // Allow tiny difference due to floating rounding
        assert!(err < 1e-6, "Sparse estimate {} differs from dense {}", es, ed);
    }
}