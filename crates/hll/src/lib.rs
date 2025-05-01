// crates/hll/src/lib.rs
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// HyperLogLog Sketch for approximate distinct counting.
pub struct HllSketch {
    p: u8,
    m: usize,
    registers: Vec<u8>,
}

impl HllSketch {
    /// Create a new HLL sketch with precision p (number of register index bits).
    pub fn new(p: u8) -> Self {
        let m = 1 << p;
        HllSketch { p, m, registers: vec![0; m] }
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
        if self.registers[idx] < rank {
            self.registers[idx] = rank;
        }
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        let m = self.m as f64;
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
        for i in 0..self.m {
            if self.registers[i] < other.registers[i] {
                self.registers[i] = other.registers[i];
            }
        }
    }

    /// Serialize registers to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.registers.clone()
    }
}

mod pp;
pub use pp::HllPlusPlusSketch;
pub use pp::HllPlusPlusSparseSketch;
