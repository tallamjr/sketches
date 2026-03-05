//! HyperLogLog (HLL) and HyperLogLog++ (HLL++) for approximate distinct counting.
//!
//! HLL uses a fixed array of `m = 2^p` registers to estimate stream cardinality
//! by tracking the maximum observed rank (leading zeros + 1) per register bucket.
//! The estimate uses a bias-corrected harmonic mean across all registers.
//!
//! # Error Bounds
//! - Standard error: `delta ~= 1.04 / sqrt(m)` where `m = 2^p`
//! - p=12 (default): 4096 registers, ~3 KB, ~1.6% error
//! - p=14: 16384 registers, ~16 KB, ~0.8% error
//!
//! # HLL++ Enhancements
//! - 64-bit hashing to reduce large-cardinality collisions
//! - Sparse representation for memory efficiency at low cardinalities
//! - Empirical bias correction from Heule et al.
//!
//! # References
//! - Flajolet, Fusy, Gandouet, Meunier. "HyperLogLog: the analysis of a near-optimal
//!   cardinality estimation algorithm." DMTCS, 2007.
//! - Heule, Nunkesser, Hall. "HyperLogLog in Practice: Algorithmic Engineering of a
//!   State of the Art Cardinality Estimation Algorithm." EDBT, 2013.

#[cfg(not(feature = "optimized"))]
use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[cfg(feature = "optimized")]
use crate::compact_memory::{CompactHashTable, PackedRegisters};
#[cfg(feature = "optimized")]
use crate::fast_hash;
#[cfg(feature = "optimized")]
use crate::simd_ops::hyperloglog;

/// HyperLogLog Sketch for approximate distinct counting.
#[derive(Debug)]
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

    /// Get the precision parameter p.
    pub fn precision(&self) -> u8 {
        self.p
    }

    /// Get the number of registers m.
    pub fn num_registers(&self) -> usize {
        self.m
    }

    /// Get the register value at the given index.
    pub fn register_value(&self, idx: usize) -> u8 {
        #[cfg(feature = "optimized")]
        {
            self.registers.get(idx)
        }

        #[cfg(not(feature = "optimized"))]
        {
            self.registers[idx]
        }
    }

    /// Create an HllSketch from raw register bytes and precision p.
    pub fn from_registers(p: u8, registers: Vec<u8>) -> Self {
        let m = 1 << p;
        assert_eq!(registers.len(), m, "Register count must equal 2^p");
        HllSketch {
            p,
            m,
            #[cfg(feature = "optimized")]
            registers: {
                let mut packed = PackedRegisters::new(p, 6);
                for (i, &val) in registers.iter().enumerate() {
                    packed.set(i, val);
                }
                packed
            },
            #[cfg(not(feature = "optimized"))]
            registers,
        }
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        #[cfg(feature = "optimized")]
        {
            self.registers.memory_usage() + std::mem::size_of::<Self>()
        }

        #[cfg(not(feature = "optimized"))]
        {
            self.registers.len() + std::mem::size_of::<Self>()
        }
    }
}

/// HyperLogLog++ Sketch for approximate distinct counting with improved small-range estimator.
#[derive(Debug)]
pub struct HllPlusPlusSketch {
    p: u8,
    m: usize,
    #[cfg(feature = "optimized")]
    registers: PackedRegisters,
    #[cfg(not(feature = "optimized"))]
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
            #[cfg(feature = "optimized")]
            registers: PackedRegisters::new(p, 6), // 6 bits per register for HLL++
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
            let reg = &mut self.registers[idx];
            if *reg < rank {
                *reg = rank;
            }
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
            for (r, o) in self.registers.iter_mut().zip(other.registers.iter()) {
                if *r < *o {
                    *r = *o;
                }
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

    /// Get the precision parameter p.
    pub fn precision(&self) -> u8 {
        self.p
    }

    /// Get the number of registers m.
    pub fn num_registers(&self) -> usize {
        self.m
    }

    /// Get the register value at the given index.
    pub fn register_value(&self, idx: usize) -> u8 {
        #[cfg(feature = "optimized")]
        {
            self.registers.get(idx)
        }

        #[cfg(not(feature = "optimized"))]
        {
            self.registers[idx]
        }
    }

    /// Create an HllPlusPlusSketch from raw register bytes and precision p.
    pub fn from_registers(p: u8, registers: Vec<u8>) -> Self {
        assert!(
            (4..=18).contains(&p),
            "Precision p must be between 4 and 18"
        );
        let m = 1 << p;
        assert_eq!(registers.len(), m, "Register count must equal 2^p");
        HllPlusPlusSketch {
            p,
            m,
            #[cfg(feature = "optimized")]
            registers: {
                let mut packed = PackedRegisters::new(p, 6);
                for (i, &val) in registers.iter().enumerate() {
                    packed.set(i, val);
                }
                packed
            },
            #[cfg(not(feature = "optimized"))]
            registers,
        }
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        #[cfg(feature = "optimized")]
        {
            self.registers.memory_usage() + std::mem::size_of::<Self>()
        }

        #[cfg(not(feature = "optimized"))]
        {
            self.registers.len() + std::mem::size_of::<Self>()
        }
    }
}

/// Sparse HyperLogLog++ sketch storing only non-zero registers.
pub struct HllPlusPlusSparseSketch {
    p: u8,
    m: usize,
    #[cfg(feature = "optimized")]
    map: CompactHashTable<u8>,
    #[cfg(not(feature = "optimized"))]
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
            #[cfg(feature = "optimized")]
            map: CompactHashTable::new(64), // Start with small capacity
            #[cfg(not(feature = "optimized"))]
            map: BTreeMap::new(),
        }
    }

    /// Update the sketch with an item implementing Hash.
    pub fn update<T: Hash>(&mut self, item: &T) {
        #[cfg(feature = "optimized")]
        {
            let hash = fast_hash::fast_hash(item);
            let idx = (hash >> (64 - self.p)) as usize;
            let w = hash << self.p;
            let rank = w.leading_zeros().saturating_add(1).min(64) as u8;

            if let Some(current) = self.map.get_mut(idx as u64) {
                if *current < rank {
                    *current = rank;
                }
            } else {
                self.map.insert(idx as u64, rank);
            }
        }

        #[cfg(not(feature = "optimized"))]
        {
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
    }

    /// Estimate the cardinality using sparse HLL++.
    pub fn estimate(&self) -> f64 {
        let m_f = self.m as f64;
        let zeros = self.m - self.map.len();

        #[cfg(feature = "optimized")]
        let sum_nonzero: f64 = self.map.iter().map(|(_, r)| 2f64.powf(-(*r as f64))).sum();

        #[cfg(not(feature = "optimized"))]
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

        #[cfg(feature = "optimized")]
        {
            for (idx, &r) in other.map.iter() {
                if let Some(current) = self.map.get_mut(idx) {
                    if *current < r {
                        *current = r;
                    }
                } else {
                    self.map.insert(idx, r);
                }
            }
        }

        #[cfg(not(feature = "optimized"))]
        {
            for (&idx, &r) in other.map.iter() {
                let entry = self.map.entry(idx).or_insert(0);
                if *entry < r {
                    *entry = r;
                }
            }
        }
    }

    /// Serialize to dense byte vector.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut regs = vec![0; self.m];

        #[cfg(feature = "optimized")]
        {
            for (idx, &r) in self.map.iter() {
                if (idx as usize) < self.m {
                    regs[idx as usize] = r;
                }
            }
        }

        #[cfg(not(feature = "optimized"))]
        {
            for (&idx, &r) in self.map.iter() {
                regs[idx] = r;
            }
        }

        regs
    }

    /// Get memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        #[cfg(feature = "optimized")]
        {
            self.map.memory_usage() + std::mem::size_of::<Self>()
        }

        #[cfg(not(feature = "optimized"))]
        {
            // BTreeMap overhead: ~24-32 bytes per entry + data
            self.map.len() * (std::mem::size_of::<usize>() + std::mem::size_of::<u8>() + 24)
                + std::mem::size_of::<Self>()
        }
    }
}

/// Adaptive HyperLogLog++ that automatically transitions between sparse and dense modes
/// for optimal memory efficiency.
pub struct AdaptiveHllPlusPlus {
    p: u8,
    m: usize,
    sparse_threshold_ratio: f64, // Transition when sparse memory > ratio * dense memory
    representation: HllRepresentation,
}

enum HllRepresentation {
    Sparse(HllPlusPlusSparseSketch),
    Dense(HllPlusPlusSketch),
}

impl AdaptiveHllPlusPlus {
    /// Create a new adaptive HLL++ sketch with precision p.
    /// Starts in sparse mode and transitions to dense when memory efficient.
    pub fn new(p: u8) -> Self {
        Self::with_threshold_ratio(p, 0.7) // Transition when sparse > 70% of dense size
    }

    /// Create with custom sparse-to-dense transition threshold.
    /// threshold_ratio: transition when sparse_memory > threshold_ratio * dense_memory
    pub fn with_threshold_ratio(p: u8, threshold_ratio: f64) -> Self {
        assert!(
            (4..=18).contains(&p),
            "Precision p must be between 4 and 18"
        );
        assert!(
            threshold_ratio > 0.0 && threshold_ratio <= 1.0,
            "Threshold ratio must be between 0 and 1"
        );

        let m = 1usize << p;
        AdaptiveHllPlusPlus {
            p,
            m,
            sparse_threshold_ratio: threshold_ratio,
            representation: HllRepresentation::Sparse(HllPlusPlusSparseSketch::new(p)),
        }
    }

    /// Update the sketch, potentially triggering sparse-to-dense transition.
    pub fn update<T: Hash>(&mut self, item: &T) {
        match &mut self.representation {
            HllRepresentation::Sparse(sparse) => {
                sparse.update(item);
                // Check if we should transition to dense mode
                self.check_transition_to_dense();
            }
            HllRepresentation::Dense(dense) => {
                dense.update(item);
            }
        }
    }

    /// Batch update with automatic transition checking.
    pub fn update_batch<T: Hash + Sync>(&mut self, items: &[T]) {
        match &mut self.representation {
            HllRepresentation::Sparse(sparse) => {
                for item in items {
                    sparse.update(item);
                }
                // Check transition less frequently for batch updates
                if items.len() > 100 {
                    self.check_transition_to_dense();
                }
            }
            HllRepresentation::Dense(dense) => {
                #[cfg(feature = "optimized")]
                dense.update_batch(items);
                #[cfg(not(feature = "optimized"))]
                for item in items {
                    dense.update(item);
                }
            }
        }
    }

    /// Check if we should transition from sparse to dense representation.
    fn check_transition_to_dense(&mut self) {
        if let HllRepresentation::Sparse(sparse) = &self.representation {
            let sparse_memory = sparse.memory_usage();

            // Estimate dense memory usage (approximate)
            #[cfg(feature = "optimized")]
            let dense_memory = (self.m * 6) / 8 + std::mem::size_of::<HllPlusPlusSketch>(); // 6-bit packed

            #[cfg(not(feature = "optimized"))]
            let dense_memory = self.m + std::mem::size_of::<HllPlusPlusSketch>(); // 8-bit unpacked

            if sparse_memory as f64 > self.sparse_threshold_ratio * dense_memory as f64 {
                self.transition_to_dense();
            }
        }
    }

    /// Transition from sparse to dense representation.
    fn transition_to_dense(&mut self) {
        if let HllRepresentation::Sparse(sparse) = &self.representation {
            let mut dense = HllPlusPlusSketch::new(self.p);

            // Copy all register values from sparse to dense
            let sparse_bytes = sparse.to_bytes();
            for (idx, &value) in sparse_bytes.iter().enumerate() {
                if value > 0 {
                    #[cfg(feature = "optimized")]
                    dense.registers.set(idx, value);
                    #[cfg(not(feature = "optimized"))]
                    {
                        dense.registers[idx] = value;
                    }
                }
            }

            self.representation = HllRepresentation::Dense(dense);
        }
    }

    /// Get cardinality estimate.
    pub fn estimate(&self) -> f64 {
        match &self.representation {
            HllRepresentation::Sparse(sparse) => sparse.estimate(),
            HllRepresentation::Dense(dense) => dense.estimate(),
        }
    }

    /// Merge another adaptive sketch into this one.
    pub fn merge(&mut self, other: &AdaptiveHllPlusPlus) {
        assert_eq!(
            self.p, other.p,
            "Cannot merge sketches with different precision"
        );

        // Force both to dense mode for merging if needed
        match (&self.representation, &other.representation) {
            (HllRepresentation::Sparse(_), _) => self.transition_to_dense(),
            (_, HllRepresentation::Sparse(_)) => {
                // Create a temporary dense version of other for merging
                let mut other_dense = HllPlusPlusSketch::new(other.p);
                let other_bytes = other.to_bytes();

                for (idx, &value) in other_bytes.iter().enumerate() {
                    if value > 0 {
                        #[cfg(feature = "optimized")]
                        other_dense.registers.set(idx, value);
                        #[cfg(not(feature = "optimized"))]
                        {
                            other_dense.registers[idx] = value;
                        }
                    }
                }

                if let HllRepresentation::Dense(dense) = &mut self.representation {
                    dense.merge(&other_dense);
                }
                return;
            }
            _ => {} // Both are dense, proceed normally
        }

        // Both are dense now
        if let (HllRepresentation::Dense(self_dense), HllRepresentation::Dense(other_dense)) =
            (&mut self.representation, &other.representation)
        {
            self_dense.merge(other_dense);
        }
    }

    /// Serialize to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        match &self.representation {
            HllRepresentation::Sparse(sparse) => sparse.to_bytes(),
            HllRepresentation::Dense(dense) => dense.to_bytes(),
        }
    }

    /// Get current memory usage.
    pub fn memory_usage(&self) -> usize {
        (match &self.representation {
            HllRepresentation::Sparse(sparse) => sparse.memory_usage(),
            HllRepresentation::Dense(dense) => dense.memory_usage(),
        }) + std::mem::size_of::<Self>()
    }

    /// Check if currently using sparse representation.
    pub fn is_sparse(&self) -> bool {
        matches!(self.representation, HllRepresentation::Sparse(_))
    }

    /// Get the number of non-zero registers (sparse mode only).
    pub fn sparse_size(&self) -> Option<usize> {
        match &self.representation {
            HllRepresentation::Sparse(sparse) => Some(sparse.map.len()),
            HllRepresentation::Dense(_) => None,
        }
    }

    /// Force transition to dense mode (useful for benchmarking).
    pub fn force_dense(&mut self) {
        if self.is_sparse() {
            self.transition_to_dense();
        }
    }
}

// ---------------------------------------------------------------------------
// Apache DataSketches-inspired HLL with List -> Set -> HLL mode transitions
// ---------------------------------------------------------------------------

/// Pack a (slot, value) pair into a single coupon u32.
/// Upper 6 bits = value, lower 26 bits = slot.
pub fn make_coupon(slot: usize, value: u8) -> u32 {
    ((value as u32) << 26) | (slot as u32 & 0x03FF_FFFF)
}

/// Extract the slot index from a coupon.
pub fn coupon_slot(coupon: u32) -> usize {
    (coupon & 0x03FF_FFFF) as usize
}

/// Extract the register value from a coupon.
pub fn coupon_value(coupon: u32) -> u8 {
    (coupon >> 26) as u8
}

/// Compute a coupon from a 64-bit hash for a given lg_k.
fn hash_to_coupon(hash: u64, lg_k: u8) -> u32 {
    let slot = (hash >> (64 - lg_k)) as usize;
    let remaining = hash << lg_k;
    let value = remaining.leading_zeros().saturating_add(1).min(63) as u8;
    make_coupon(slot, value)
}

/// Compute a 64-bit hash for an item using DefaultHasher.
fn compute_hash<T: Hash>(item: &T) -> u64 {
    let mut hasher = DefaultHasher::new();
    item.hash(&mut hasher);
    hasher.finish()
}

// -- List mode ---------------------------------------------------------------

/// LIST mode capacity: 2^3 = 8 coupons before transitioning.
const LIST_CAPACITY: usize = 8;

struct ListMode {
    coupons: Vec<u32>,
}

impl ListMode {
    fn new() -> Self {
        ListMode {
            coupons: Vec::with_capacity(LIST_CAPACITY),
        }
    }

    /// Insert a coupon. If the same slot already exists, keep the max value.
    /// Returns true if the coupon was new or updated the value.
    fn insert(&mut self, coupon: u32) -> bool {
        for existing in self.coupons.iter_mut() {
            if coupon_slot(*existing) == coupon_slot(coupon) {
                if coupon_value(coupon) > coupon_value(*existing) {
                    *existing = coupon;
                    return true;
                }
                return false;
            }
        }
        self.coupons.push(coupon);
        true
    }

    fn is_full(&self) -> bool {
        self.coupons.len() >= LIST_CAPACITY
    }

    fn len(&self) -> usize {
        self.coupons.len()
    }

    fn iter(&self) -> impl Iterator<Item = &u32> {
        self.coupons.iter()
    }

    fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.coupons.capacity() * std::mem::size_of::<u32>()
    }
}

// -- Set mode ----------------------------------------------------------------

/// SET mode initial capacity: 2^5 = 32 slots.
const SET_INITIAL_LG_SIZE: u8 = 5;

struct SetMode {
    lg_size: u8,
    slots: Vec<u32>,
    count: usize,
}

impl SetMode {
    fn new() -> Self {
        let size = 1usize << SET_INITIAL_LG_SIZE;
        SetMode {
            lg_size: SET_INITIAL_LG_SIZE,
            slots: vec![0; size],
            count: 0,
        }
    }

    fn capacity(&self) -> usize {
        1usize << self.lg_size
    }

    /// Compute the stride for linear probing: must be odd to ensure full coverage.
    fn stride(coupon: u32, lg_size: u8) -> usize {
        (coupon as usize >> lg_size) | 1
    }

    /// Find the slot index for a coupon.
    /// Returns (index, true) if an existing entry with the same HLL slot was found,
    /// or (index, false) if an empty position was found.
    fn find_slot(&self, coupon: u32) -> (usize, bool) {
        let mask = self.capacity() - 1;
        let mut idx = (coupon as usize) & mask;
        let stride = Self::stride(coupon, self.lg_size);
        loop {
            let existing = self.slots[idx];
            if existing == 0 {
                return (idx, false);
            }
            if coupon_slot(existing) == coupon_slot(coupon) {
                return (idx, true);
            }
            idx = (idx + stride) & mask;
        }
    }

    /// Insert a coupon. Returns true if the value was new or updated.
    fn insert(&mut self, coupon: u32) -> bool {
        let (idx, found) = self.find_slot(coupon);
        if found {
            let existing = self.slots[idx];
            if coupon_value(coupon) > coupon_value(existing) {
                self.slots[idx] = coupon;
                return true;
            }
            return false;
        }
        self.slots[idx] = coupon;
        self.count += 1;
        // Resize at 75% load factor
        if self.count * 4 > self.capacity() * 3 {
            self.resize();
        }
        true
    }

    fn resize(&mut self) {
        let old_slots = std::mem::take(&mut self.slots);
        self.lg_size += 1;
        let new_size = 1usize << self.lg_size;
        self.slots = vec![0; new_size];
        self.count = 0;
        for coupon in old_slots {
            if coupon != 0 {
                let (idx, _) = self.find_slot(coupon);
                self.slots[idx] = coupon;
                self.count += 1;
            }
        }
    }

    /// Transition to HLL mode when lg_size reaches lg_k - 3.
    fn should_transition_to_hll(&self, lg_k: u8) -> bool {
        if lg_k <= 3 {
            return true;
        }
        self.lg_size >= lg_k.saturating_sub(3)
    }

    fn iter_coupons(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots.iter().copied().filter(|&c| c != 0)
    }

    fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.slots.len() * std::mem::size_of::<u32>()
    }
}

// -- HIP Estimator -----------------------------------------------------------

/// Historic Inverse Probability estimator for improved single-sketch accuracy.
struct HipEstimator {
    hip_accum: f64,
    kxq0: f64,
    kxq1: f64,
    out_of_order: bool,
}

impl HipEstimator {
    fn new(k: usize) -> Self {
        HipEstimator {
            hip_accum: 0.0,
            kxq0: k as f64, // k * 2^(-0) = k
            kxq1: 0.0,
            out_of_order: false,
        }
    }

    /// Called when a register is updated from old_value to new_value.
    /// kxq tracks sum of invPow2(reg[i]) across all registers (split at threshold 32).
    /// The HIP formula is: hip_accum += configK / (kxq0 + kxq1).
    fn register_updated(&mut self, k: usize, old_value: u8, new_value: u8) {
        if old_value >= new_value {
            return;
        }

        let k_f = k as f64;

        // Update HIP accumulator BEFORE changing kxq
        if !self.out_of_order {
            let inv_p = self.kxq0 + self.kxq1;
            if inv_p > 0.0 {
                self.hip_accum += k_f / inv_p;
            }
        }

        // Remove old contribution (no k factor -- kxq is sum of invPow2(reg[i]))
        if old_value < 32 {
            self.kxq0 -= 2f64.powi(-(old_value as i32));
        } else {
            self.kxq1 -= 2f64.powi(-(old_value as i32));
        }

        // Add new contribution
        if new_value < 32 {
            self.kxq0 += 2f64.powi(-(new_value as i32));
        } else {
            self.kxq1 += 2f64.powi(-(new_value as i32));
        }
    }

    fn mark_out_of_order(&mut self) {
        self.out_of_order = true;
    }

    /// Returns Some(hip_estimate) if valid, None if out_of_order.
    fn estimate(&self) -> Option<f64> {
        if self.out_of_order {
            None
        } else {
            Some(self.hip_accum)
        }
    }
}

// -- HLL mode ----------------------------------------------------------------

struct HllMode {
    lg_k: u8,
    registers: Vec<u8>,
    hip: HipEstimator,
}

impl HllMode {
    fn new(lg_k: u8) -> Self {
        let k = 1usize << lg_k;
        HllMode {
            lg_k,
            registers: vec![0; k],
            hip: HipEstimator::new(k),
        }
    }

    fn k(&self) -> usize {
        1usize << self.lg_k
    }

    /// Update a single register from a coupon.
    fn update_coupon(&mut self, coupon: u32) {
        let slot = coupon_slot(coupon);
        let value = coupon_value(coupon);
        if slot >= self.registers.len() {
            return;
        }
        let old = self.registers[slot];
        if value > old {
            self.registers[slot] = value;
            self.hip.register_updated(self.k(), old, value);
        }
    }

    /// Standard HLL raw estimate with alpha correction and linear counting.
    fn raw_estimate(&self) -> f64 {
        let k = self.k();
        let k_f = k as f64;
        let mut sum = 0f64;
        let mut zeros = 0usize;
        for &reg in &self.registers {
            sum += 2f64.powi(-(reg as i32));
            if reg == 0 {
                zeros += 1;
            }
        }
        let alpha = match k {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / k_f),
        };
        let est = alpha * k_f * k_f / sum;
        if est <= 2.5 * k_f && zeros > 0 {
            k_f * (k_f / zeros as f64).ln()
        } else {
            est
        }
    }

    fn estimate(&self) -> f64 {
        match self.hip.estimate() {
            Some(hip_est) => hip_est,
            None => self.raw_estimate(),
        }
    }

    fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>() + self.registers.len()
    }
}

// -- HllSketchMode -----------------------------------------------------------

/// Internal mode of an HllSketchMode sketch.
enum SketchMode {
    List(ListMode),
    Set(SetMode),
    Hll(HllMode),
}

/// HLL sketch with proper List -> Set -> HLL mode transitions,
/// following the Apache DataSketches design.
pub struct HllSketchMode {
    lg_k: u8,
    mode: SketchMode,
}

impl HllSketchMode {
    /// Create a new HLL sketch with the given log2(k) precision.
    /// Supported lg_k values are in [4, 21].
    pub fn new(lg_k: u8) -> Self {
        assert!(
            (4..=21).contains(&lg_k),
            "lg_k must be between 4 and 21, got {lg_k}"
        );
        HllSketchMode {
            lg_k,
            mode: SketchMode::List(ListMode::new()),
        }
    }

    /// Update the sketch with a hashable item.
    pub fn update<T: Hash>(&mut self, item: &T) {
        let hash = compute_hash(item);
        let coupon = hash_to_coupon(hash, self.lg_k);
        if coupon_value(coupon) == 0 {
            return;
        }
        self.update_with_coupon(coupon);
    }

    fn update_with_coupon(&mut self, coupon: u32) {
        match &mut self.mode {
            SketchMode::List(list) => {
                list.insert(coupon);
                if list.is_full() {
                    self.promote_list();
                }
            }
            SketchMode::Set(set) => {
                set.insert(coupon);
                if set.should_transition_to_hll(self.lg_k) {
                    self.promote_set();
                }
            }
            SketchMode::Hll(hll) => {
                hll.update_coupon(coupon);
            }
        }
    }

    /// Promote from List to Set or directly to HLL (if lg_k < 8).
    fn promote_list(&mut self) {
        let coupons: Vec<u32> = match &self.mode {
            SketchMode::List(l) => l.iter().copied().collect(),
            _ => return,
        };

        if self.lg_k < 8 {
            // For small lg_k, skip Set mode and go directly to HLL
            let mut hll = HllMode::new(self.lg_k);
            for coupon in coupons {
                hll.update_coupon(coupon);
            }
            self.mode = SketchMode::Hll(hll);
        } else {
            let mut set = SetMode::new();
            for coupon in coupons {
                set.insert(coupon);
            }
            // Check if Set should immediately transition to HLL
            if set.should_transition_to_hll(self.lg_k) {
                let set_coupons: Vec<u32> = set.iter_coupons().collect();
                let mut hll = HllMode::new(self.lg_k);
                for coupon in set_coupons {
                    hll.update_coupon(coupon);
                }
                self.mode = SketchMode::Hll(hll);
            } else {
                self.mode = SketchMode::Set(set);
            }
        }
    }

    /// Promote from Set to HLL mode.
    fn promote_set(&mut self) {
        let coupons: Vec<u32> = match &self.mode {
            SketchMode::Set(s) => s.iter_coupons().collect(),
            _ => return,
        };
        let mut hll = HllMode::new(self.lg_k);
        for coupon in coupons {
            hll.update_coupon(coupon);
        }
        self.mode = SketchMode::Hll(hll);
    }

    /// Estimate the number of distinct items.
    pub fn estimate(&self) -> f64 {
        match &self.mode {
            SketchMode::List(list) => list.len() as f64,
            SketchMode::Set(set) => set.count as f64,
            SketchMode::Hll(hll) => hll.estimate(),
        }
    }

    /// Get the current mode as a string (for diagnostics).
    pub fn mode_name(&self) -> &'static str {
        match &self.mode {
            SketchMode::List(_) => "LIST",
            SketchMode::Set(_) => "SET",
            SketchMode::Hll(_) => "HLL",
        }
    }

    /// Get the lg_k precision.
    pub fn lg_k(&self) -> u8 {
        self.lg_k
    }

    /// Return memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        std::mem::size_of::<Self>()
            + match &self.mode {
                SketchMode::List(list) => list.memory_usage(),
                SketchMode::Set(set) => set.memory_usage(),
                SketchMode::Hll(hll) => hll.memory_usage(),
            }
    }

    /// Access the HLL registers (only available in HLL mode).
    #[allow(dead_code)]
    fn registers(&self) -> Option<&[u8]> {
        match &self.mode {
            SketchMode::Hll(hll) => Some(&hll.registers),
            _ => None,
        }
    }

    /// Iterate over all coupons (for List and Set modes).
    fn iter_coupons(&self) -> Vec<u32> {
        match &self.mode {
            SketchMode::List(list) => list.iter().copied().collect(),
            SketchMode::Set(set) => set.iter_coupons().collect(),
            SketchMode::Hll(_) => Vec::new(),
        }
    }

    /// Check if the HIP estimator is out-of-order (invalidated by merge).
    pub fn is_out_of_order(&self) -> bool {
        match &self.mode {
            SketchMode::Hll(hll) => hll.hip.out_of_order,
            _ => false,
        }
    }
}

// -- HllUnion ----------------------------------------------------------------

/// Downsample a coupon from a higher precision to a lower precision.
fn downsample_coupon(coupon: u32, src_lg_k: u8, dst_lg_k: u8) -> u32 {
    let slot = coupon_slot(coupon);
    let value = coupon_value(coupon);
    let new_slot = slot >> (src_lg_k - dst_lg_k);
    make_coupon(new_slot, value)
}

/// Union operation for HllSketchMode sketches.
/// Maintains an internal gadget sketch at the maximum precision.
pub struct HllUnion {
    lg_max_k: u8,
    gadget: HllSketchMode,
}

impl HllUnion {
    /// Create a new union with the specified maximum precision.
    pub fn new(lg_max_k: u8) -> Self {
        HllUnion {
            lg_max_k,
            gadget: HllSketchMode::new(lg_max_k),
        }
    }

    /// Update the union directly with a hashable item.
    pub fn update<T: Hash>(&mut self, item: &T) {
        self.gadget.update(item);
    }

    /// Merge a sketch into this union.
    pub fn update_with_sketch(&mut self, sketch: &HllSketchMode) {
        match &sketch.mode {
            SketchMode::List(_) | SketchMode::Set(_) => {
                let coupons = sketch.iter_coupons();
                for coupon in coupons {
                    let final_coupon = if sketch.lg_k > self.lg_max_k {
                        downsample_coupon(coupon, sketch.lg_k, self.lg_max_k)
                    } else {
                        coupon
                    };
                    self.gadget.update_with_coupon(final_coupon);
                }
            }
            SketchMode::Hll(src_hll) => {
                // For HLL mode, merge registers directly.
                self.ensure_hll_mode();

                if let SketchMode::Hll(ref mut dst_hll) = self.gadget.mode {
                    dst_hll.hip.mark_out_of_order();

                    if sketch.lg_k == self.lg_max_k {
                        for (i, &src_val) in src_hll.registers.iter().enumerate() {
                            let old = dst_hll.registers[i];
                            if src_val > old {
                                dst_hll.registers[i] = src_val;
                                dst_hll.hip.register_updated(dst_hll.k(), old, src_val);
                            }
                        }
                    } else if sketch.lg_k > self.lg_max_k {
                        // Source has higher precision: downsample
                        let src_k = 1usize << sketch.lg_k;
                        let lg_ratio = sketch.lg_k - self.lg_max_k;
                        for src_slot in 0..src_k {
                            let src_val = src_hll.registers[src_slot];
                            if src_val > 0 {
                                let dst_slot = src_slot >> lg_ratio;
                                let old = dst_hll.registers[dst_slot];
                                if src_val > old {
                                    dst_hll.registers[dst_slot] = src_val;
                                    dst_hll.hip.register_updated(dst_hll.k(), old, src_val);
                                }
                            }
                        }
                    } else {
                        // Source has lower precision: each source register covers
                        // multiple destination registers
                        let src_k = 1usize << sketch.lg_k;
                        let dst_k = 1usize << self.lg_max_k;
                        let ratio = dst_k / src_k;
                        for src_slot in 0..src_k {
                            let src_val = src_hll.registers[src_slot];
                            if src_val > 0 {
                                for offset in 0..ratio {
                                    let dst_slot = src_slot * ratio + offset;
                                    let old = dst_hll.registers[dst_slot];
                                    if src_val > old {
                                        dst_hll.registers[dst_slot] = src_val;
                                        dst_hll.hip.register_updated(dst_hll.k(), old, src_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Force the gadget into HLL mode if it is not already.
    fn ensure_hll_mode(&mut self) {
        match &self.gadget.mode {
            SketchMode::Hll(_) => {}
            _ => {
                let coupons = self.gadget.iter_coupons();
                let mut hll = HllMode::new(self.lg_max_k);
                for coupon in coupons {
                    hll.update_coupon(coupon);
                }
                self.gadget.mode = SketchMode::Hll(hll);
            }
        }
    }

    /// Get the cardinality estimate from the union.
    pub fn estimate(&self) -> f64 {
        self.gadget.estimate()
    }

    /// Return a copy of the internal sketch state as a new HllSketchMode.
    pub fn to_sketch(&self) -> HllSketchMode {
        let mut result = HllSketchMode::new(self.lg_max_k);
        match &self.gadget.mode {
            SketchMode::List(_) | SketchMode::Set(_) => {
                let coupons = self.gadget.iter_coupons();
                for coupon in coupons {
                    result.update_with_coupon(coupon);
                }
            }
            SketchMode::Hll(src_hll) => {
                let mut hll = HllMode::new(self.lg_max_k);
                for (i, &val) in src_hll.registers.iter().enumerate() {
                    if val > 0 {
                        hll.update_coupon(make_coupon(i, val));
                    }
                }
                hll.hip.mark_out_of_order();
                result.mode = SketchMode::Hll(hll);
            }
        }
        result
    }

    /// Get the maximum precision of this union.
    pub fn lg_max_k(&self) -> u8 {
        self.lg_max_k
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        AdaptiveHllPlusPlus, HllPlusPlusSketch, HllPlusPlusSparseSketch, HllSketch, HllSketchMode,
        HllUnion, coupon_slot, coupon_value, make_coupon,
    };

    // ---- Existing tests (kept intact) ----

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

    #[test]
    fn adaptive_starts_sparse() {
        let sketch = AdaptiveHllPlusPlus::new(10);
        assert!(sketch.is_sparse());
        assert_eq!(sketch.sparse_size(), Some(0));
    }

    #[test]
    fn adaptive_transitions_to_dense() {
        let mut sketch = AdaptiveHllPlusPlus::with_threshold_ratio(8, 0.5);
        for i in 0..500 {
            sketch.update(&format!("element_{i}"));
        }
        assert!(!sketch.is_sparse());
    }

    #[test]
    fn adaptive_memory_efficiency() {
        let mut adaptive = AdaptiveHllPlusPlus::with_threshold_ratio(12, 0.9);
        let mut regular_sparse = HllPlusPlusSparseSketch::new(12);
        let mut regular_dense = HllPlusPlusSketch::new(12);

        let small_data: Vec<String> = (0..20).map(|i| format!("item_{i}")).collect();
        for item in &small_data {
            adaptive.update(item);
            regular_sparse.update(item);
            regular_dense.update(item);
        }
        assert!(adaptive.memory_usage() <= regular_dense.memory_usage());

        let large_data: Vec<String> = (0..2000).map(|i| format!("large_item_{i}")).collect();
        for item in &large_data {
            adaptive.update(item);
        }
        assert!(!adaptive.is_sparse());
    }

    #[test]
    fn adaptive_estimates_accuracy() {
        let mut adaptive = AdaptiveHllPlusPlus::new(12);
        let mut regular = HllPlusPlusSketch::new(12);
        let test_data: Vec<String> = (0..1000).map(|i| format!("test_{i}")).collect();
        for item in &test_data {
            adaptive.update(item);
            regular.update(item);
        }
        let adaptive_est = adaptive.estimate();
        let regular_est = regular.estimate();
        let diff_ratio = (adaptive_est - regular_est).abs() / regular_est;
        assert!(
            diff_ratio < 0.05,
            "Estimates differ by more than 5%: adaptive={adaptive_est}, regular={regular_est}"
        );
    }

    // ---- New HllSketchMode tests ----

    #[test]
    fn test_coupon_encoding() {
        // Verify coupon pack/unpack roundtrips correctly for a range of values
        for slot in [0usize, 1, 255, 1023, 0x03FF_FFFF] {
            for value in [1u8, 2, 31, 63] {
                let coupon = make_coupon(slot, value);
                assert_eq!(
                    coupon_slot(coupon),
                    slot,
                    "Slot roundtrip failed for slot={slot}, value={value}"
                );
                assert_eq!(
                    coupon_value(coupon),
                    value,
                    "Value roundtrip failed for slot={slot}, value={value}"
                );
            }
        }
        // Verify slot is masked to 26 bits
        let coupon = make_coupon(0x0FFF_FFFF, 5);
        assert_eq!(coupon_slot(coupon), 0x03FF_FFFF);
        assert_eq!(coupon_value(coupon), 5);
    }

    #[test]
    fn test_list_mode_small_cardinality() {
        // Insert fewer than 8 distinct items: should stay in LIST mode and give exact count
        let mut sketch = HllSketchMode::new(12);
        let items = ["alpha", "bravo", "charlie", "delta", "echo"];
        for item in &items {
            sketch.update(item);
        }
        assert_eq!(sketch.mode_name(), "LIST");
        let est = sketch.estimate();
        assert_eq!(
            est, 5.0,
            "LIST mode should give exact count for 5 items, got {est}"
        );
    }

    #[test]
    fn test_small_cardinality_exact() {
        // Items fewer than LIST_CAPACITY should be counted exactly
        let mut sketch = HllSketchMode::new(14);
        for i in 0..7 {
            sketch.update(&format!("unique_item_{i}"));
        }
        assert_eq!(sketch.mode_name(), "LIST");
        assert_eq!(sketch.estimate(), 7.0);
    }

    #[test]
    fn test_list_to_set_transition() {
        // For lg_k >= 8, inserting LIST_CAPACITY distinct items triggers List -> Set
        let mut sketch = HllSketchMode::new(12);

        // First 7 items: should remain in LIST
        for i in 0..7 {
            sketch.update(&format!("item_{i}"));
        }
        assert_eq!(sketch.mode_name(), "LIST");

        // The 8th distinct item fills the list, triggering promotion
        sketch.update(&"item_7");
        // After promotion, should be in SET mode (lg_k=12 >= 8)
        assert_eq!(
            sketch.mode_name(),
            "SET",
            "Expected SET mode after 8 distinct items with lg_k=12"
        );

        // Estimate should be close to 8
        let est = sketch.estimate();
        assert!(
            (est - 8.0).abs() < 1.0,
            "Expected estimate near 8, got {est}"
        );
    }

    #[test]
    fn test_list_to_hll_for_small_lgk() {
        // For lg_k < 8, List should transition directly to HLL (skip SET)
        let mut sketch = HllSketchMode::new(4);
        // With lg_k=4 (16 slots), use enough items to guarantee 8+ unique slots
        for i in 0..30 {
            sketch.update(&format!("elem_{i}"));
        }
        // Should have gone from LIST directly to HLL (not SET)
        assert_eq!(
            sketch.mode_name(),
            "HLL",
            "Expected HLL mode for lg_k=4 after transition"
        );
        // Estimate should be reasonable (4-bit HLL has high variance, allow wide range)
        let est = sketch.estimate();
        assert!(
            est > 10.0 && est < 60.0,
            "Estimate for 30 items with lg_k=4 should be in [10, 60], got {est}"
        );
    }

    #[test]
    fn test_set_to_hll_transition() {
        // Insert enough items to transition from SET to HLL
        let mut sketch = HllSketchMode::new(12);
        // We need enough items to grow the set's lg_size to lg_k - 3 = 9
        // That means 2^9 = 512 capacity, at 75% that is 384 items to trigger resize.
        // Conservatively insert many items.
        for i in 0..600 {
            sketch.update(&format!("transition_item_{i}"));
        }
        assert_eq!(
            sketch.mode_name(),
            "HLL",
            "Expected HLL mode after many inserts"
        );
        let est = sketch.estimate();
        let error_pct = (est - 600.0).abs() / 600.0;
        assert!(
            error_pct < 0.05,
            "Expected estimate near 600, got {} (error {:.1}%)",
            est,
            error_pct * 100.0
        );
    }

    #[test]
    fn test_mode_transitions_preserve_accuracy() {
        // Verify that the estimate does not jump wildly at mode transitions
        let mut sketch = HllSketchMode::new(12);
        let mut prev_est = 0.0f64;

        for i in 0..700 {
            sketch.update(&format!("acc_item_{i}"));
            let est = sketch.estimate();
            let actual = (i + 1) as f64;

            // The estimate should be monotonically non-decreasing (we only add new items)
            assert!(
                est >= prev_est - 1.0,
                "Estimate decreased from {prev_est} to {est} at item {i}"
            );

            // After enough items to have stable estimates, check accuracy
            if i > 20 {
                let error_pct = (est - actual).abs() / actual;
                assert!(
                    error_pct < 0.15,
                    "Estimate {} too far from actual {} at item {} (error {:.1}%)",
                    est,
                    actual,
                    i,
                    error_pct * 100.0
                );
            }
            prev_est = est;
        }
    }

    #[test]
    fn test_hip_estimator_accuracy() {
        // Single sketch: HIP should provide a good estimate
        let mut sketch = HllSketchMode::new(12);
        let n = 10_000;
        for i in 0..n {
            sketch.update(&format!("hip_item_{i}"));
        }
        assert_eq!(sketch.mode_name(), "HLL");
        assert!(
            !sketch.is_out_of_order(),
            "HIP should be valid for single sketch"
        );

        let est = sketch.estimate();
        let error_pct = (est - n as f64).abs() / n as f64;
        assert!(
            error_pct < 0.03,
            "HIP estimate {} too far from actual {} (error {:.2}%)",
            est,
            n,
            error_pct * 100.0
        );
    }

    #[test]
    fn test_hip_invalidated_after_merge() {
        let mut s1 = HllSketchMode::new(12);
        let mut s2 = HllSketchMode::new(12);

        // Insert enough to get both into HLL mode
        for i in 0..1000 {
            s1.update(&format!("set_a_{i}"));
        }
        for i in 0..1000 {
            s2.update(&format!("set_b_{i}"));
        }

        assert!(!s1.is_out_of_order());
        assert!(!s2.is_out_of_order());

        let mut union = HllUnion::new(12);
        union.update_with_sketch(&s1);
        union.update_with_sketch(&s2);

        let result = union.to_sketch();
        assert!(
            result.is_out_of_order(),
            "HIP should be invalidated after merge"
        );

        // Estimate should still be reasonable (falls back to raw HLL)
        let est = result.estimate();
        assert!(
            est > 1500.0 && est < 2500.0,
            "Union estimate of two 1000-item sets should be near 2000, got {est}"
        );
    }

    #[test]
    fn test_union_basic() {
        let mut s1 = HllSketchMode::new(12);
        let mut s2 = HllSketchMode::new(12);

        for i in 0..5000 {
            s1.update(&format!("union_a_{i}"));
        }
        for i in 0..5000 {
            s2.update(&format!("union_b_{i}"));
        }

        let mut union = HllUnion::new(12);
        union.update_with_sketch(&s1);
        union.update_with_sketch(&s2);

        let est = union.estimate();
        // 10000 distinct items total (no overlap)
        // Union uses raw HLL estimate (HIP invalidated), standard error ~1.6% for lg_k=12
        // Allow up to 5% for merged sketch estimation
        let error_pct = (est - 10000.0).abs() / 10000.0;
        assert!(
            error_pct < 0.05,
            "Union estimate {} too far from 10000 (error {:.2}%)",
            est,
            error_pct * 100.0
        );
    }

    #[test]
    fn test_union_with_different_lg_k() {
        // Test union handles downsampling when sketches have different precisions
        let mut s_high = HllSketchMode::new(14); // higher precision
        let mut s_low = HllSketchMode::new(10); // lower precision

        for i in 0..5000 {
            s_high.update(&format!("high_{i}"));
        }
        for i in 5000..10000 {
            s_low.update(&format!("low_{i}"));
        }

        // Union at the lower precision
        let mut union = HllUnion::new(10);
        union.update_with_sketch(&s_high);
        union.update_with_sketch(&s_low);

        let est = union.estimate();
        // At lg_k=10 the standard error is ~3.25%, allow up to 10% for combined error
        let error_pct = (est - 10000.0).abs() / 10000.0;
        assert!(
            error_pct < 0.10,
            "Union estimate {} too far from 10000 with different lg_k (error {:.2}%)",
            est,
            error_pct * 100.0
        );
    }

    #[test]
    fn test_union_preserves_accuracy() {
        // Multiple sketches with overlapping data
        let mut sketches: Vec<HllSketchMode> = Vec::new();
        for batch in 0..4 {
            let mut s = HllSketchMode::new(12);
            // Each batch has 2500 items, with 50% overlap with adjacent batches
            let start = batch * 1250;
            for i in start..(start + 2500) {
                s.update(&format!("overlap_{i}"));
            }
            sketches.push(s);
        }

        let mut union = HllUnion::new(12);
        for s in &sketches {
            union.update_with_sketch(s);
        }

        // True distinct count: items 0..6250 (0 to 3*1250 + 2500 = 6250)
        let true_count = 6250.0;
        let est = union.estimate();
        let error_pct = (est - true_count).abs() / true_count;
        assert!(
            error_pct < 0.05,
            "Union estimate {} too far from {} (error {:.2}%)",
            est,
            true_count,
            error_pct * 100.0
        );
    }

    #[test]
    fn test_large_cardinality_accuracy() {
        let mut sketch = HllSketchMode::new(12);
        let n = 100_000;
        for i in 0..n {
            sketch.update(&i);
        }
        assert_eq!(sketch.mode_name(), "HLL");
        let est = sketch.estimate();
        let error_pct = (est - n as f64).abs() / n as f64;
        assert!(
            error_pct < 0.03,
            "Estimate {} too far from {} (error {:.2}%)",
            est,
            n,
            error_pct * 100.0
        );
    }

    #[test]
    fn test_merge_two_sketches() {
        // Direct union via HllUnion, verify combined cardinality
        let mut s1 = HllSketchMode::new(12);
        let mut s2 = HllSketchMode::new(12);

        // Non-overlapping sets of 3000 each
        for i in 0..3000 {
            s1.update(&format!("merge_a_{i}"));
        }
        for i in 3000..6000 {
            s2.update(&format!("merge_b_{i}"));
        }

        let mut union = HllUnion::new(12);
        union.update_with_sketch(&s1);
        union.update_with_sketch(&s2);

        let est = union.estimate();
        let error_pct = (est - 6000.0).abs() / 6000.0;
        assert!(
            error_pct < 0.03,
            "Merge estimate {} too far from 6000 (error {:.2}%)",
            est,
            error_pct * 100.0
        );
    }

    #[test]
    fn test_memory_efficiency_modes() {
        // List mode should use far less memory than full HLL array
        let list_sketch = HllSketchMode::new(12);
        let list_mem = list_sketch.memory_usage();

        // A full HLL-mode sketch at lg_k=12 would use at least 2^12 = 4096 bytes for registers
        let full_hll_min_mem = 4096usize;

        assert!(
            list_mem < full_hll_min_mem,
            "List mode memory {list_mem} should be much less than full HLL {full_hll_min_mem}"
        );
        assert!(list_mem > 0, "Memory usage should not be zero");

        // Now transition to HLL and check memory is realistic
        let mut hll_sketch = HllSketchMode::new(12);
        for i in 0..10000 {
            hll_sketch.update(&i);
        }
        let hll_mem = hll_sketch.memory_usage();
        assert!(
            hll_mem >= full_hll_min_mem,
            "HLL mode memory {hll_mem} should be at least {full_hll_min_mem} for 2^12 registers"
        );
        assert!(
            hll_mem < 100_000,
            "HLL mode memory {hll_mem} should not be astronomical"
        );
    }

    #[test]
    fn test_set_mode_no_duplicates() {
        // Insert the same items multiple times and verify count does not inflate
        let mut sketch = HllSketchMode::new(12);

        // Insert 6 distinct items (stays in LIST, under threshold)
        let items: Vec<String> = (0..6).map(|i| format!("dup_item_{i}")).collect();
        for item in &items {
            sketch.update(item);
        }
        assert_eq!(sketch.mode_name(), "LIST");
        assert_eq!(sketch.estimate(), 6.0);

        // Insert the same 6 items again -- should not change count
        for item in &items {
            sketch.update(item);
        }
        assert_eq!(
            sketch.estimate(),
            6.0,
            "Duplicates should not inflate LIST count"
        );

        // Now push into SET mode with more unique items
        for i in 6..20 {
            sketch.update(&format!("dup_item_{i}"));
        }
        // May be in SET or HLL depending on collisions, but estimate should be near 20
        let est = sketch.estimate();
        assert!(
            (est - 20.0).abs() < 3.0,
            "After 20 distinct items, estimate should be near 20, got {est}"
        );

        // Insert all 20 items again -- estimate should not change significantly
        for i in 0..20 {
            sketch.update(&format!("dup_item_{i}"));
        }
        let est2 = sketch.estimate();
        assert!(
            (est2 - est).abs() < 1.0,
            "Re-inserting same items should not change estimate: before={est}, after={est2}"
        );
    }
}
