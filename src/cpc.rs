//! CPC (Compressed Probabilistic Counting) Sketch - FM85 Algorithm
//!
//! Implementation of Kevin Lang's FM85 algorithm for cardinality estimation.
//! CPC achieves ~40% less space than HLL for comparable accuracy by using
//! compressed probabilistic counting with five flavour states.
//!
//! Reference: Kevin Lang, "Back to the Future: an Even More Nearly Optimal
//! Cardinality Estimation Algorithm" (2017)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// The five flavour states of a CPC sketch, representing different
/// compression modes as cardinality grows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Flavour {
    /// No data has been added
    Empty,
    /// Small cardinality: store individual (column, row) pairs
    Sparse,
    /// Transition: some columns fully populated, sparse pairs for remainder
    Hybrid,
    /// Medium cardinality: sliding window over bit matrix columns
    Pinned,
    /// Large cardinality: sliding window advances as cardinality grows
    Sliding,
}

/// A (column, row) coupon representing a single hash observation.
/// Column = register index, Row = leading zeros value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Coupon {
    column: u32,
    row: u8,
}

impl Coupon {
    fn from_hash(hash: u64, lg_k: u8) -> Self {
        let column = (hash & ((1u64 << lg_k) - 1)) as u32;
        let w = hash >> lg_k;
        let row = if w == 0 {
            64 - lg_k
        } else {
            w.trailing_zeros() as u8
        };
        Coupon { column, row }
    }

    fn as_u32(self) -> u32 {
        (self.column << 6) | (self.row as u32)
    }

    fn from_u32(val: u32, _lg_k: u8) -> Self {
        Coupon {
            column: val >> 6,
            row: (val & 0x3F) as u8,
        }
    }
}

/// Sparse pair table using open-addressing hash table for storing coupons.
#[derive(Debug, Clone)]
struct PairTable {
    slots: Vec<u32>,
    count: usize,
    lg_size: u8,
}

impl PairTable {
    fn new(lg_size: u8) -> Self {
        let size = 1usize << lg_size;
        PairTable {
            slots: vec![u32::MAX; size],
            count: 0,
            lg_size,
        }
    }

    fn capacity(&self) -> usize {
        1 << self.lg_size
    }

    fn load_factor(&self) -> f64 {
        self.count as f64 / self.capacity() as f64
    }

    /// Insert a coupon. Returns true if the coupon was new (not a duplicate).
    fn insert(&mut self, coupon_val: u32) -> bool {
        if self.load_factor() > 0.6 {
            self.resize();
        }

        let mask = self.capacity() - 1;
        let mut idx = (coupon_val as usize) & mask;

        loop {
            if self.slots[idx] == u32::MAX {
                // Empty slot -- insert
                self.slots[idx] = coupon_val;
                self.count += 1;
                return true;
            }
            if self.slots[idx] == coupon_val {
                // Duplicate
                return false;
            }
            idx = (idx + 1) & mask;
        }
    }

    #[allow(dead_code)]
    fn contains(&self, coupon_val: u32) -> bool {
        let mask = self.capacity() - 1;
        let mut idx = (coupon_val as usize) & mask;

        loop {
            if self.slots[idx] == u32::MAX {
                return false;
            }
            if self.slots[idx] == coupon_val {
                return true;
            }
            idx = (idx + 1) & mask;
        }
    }

    fn resize(&mut self) {
        let old_slots = std::mem::take(&mut self.slots);
        self.lg_size += 1;
        let new_size = 1usize << self.lg_size;
        self.slots = vec![u32::MAX; new_size];
        self.count = 0;

        for &val in &old_slots {
            if val != u32::MAX {
                self.insert(val);
            }
        }
    }

    fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots.iter().copied().filter(|&v| v != u32::MAX)
    }

    fn clear(&mut self) {
        self.slots.fill(u32::MAX);
        self.count = 0;
    }
}

/// CPC Sketch implementing the FM85 algorithm.
///
/// CPC provides cardinality estimation with superior space efficiency
/// compared to HLL. It automatically transitions through five flavour
/// states as cardinality grows, optimising compression at each stage.
#[derive(Debug)]
pub struct CpcSketch {
    lg_k: u8,
    k: usize,
    num_coupons: u64,
    flavour: Flavour,

    // Sparse mode storage
    pair_table: PairTable,

    // Dense mode storage: bit matrix stored as sliding window
    // window[column] stores a byte of row bits for that column
    window: Vec<u8>,

    // The offset tracks how far the sliding window has advanced
    window_offset: u8,

    // Surprising values above the window (for Hybrid/Pinned/Sliding)
    surprising_values: PairTable,

    // HIP (Historical Inverse Probability) estimator state
    hip_estimate: f64,
    kxp: f64,

    // Tracking for flavour transitions
    #[allow(dead_code)]
    first_interesting_column: u8,

    // Whether this sketch has been merged (affects estimator choice)
    was_merged: bool,
}

impl CpcSketch {
    /// Default lg_k value
    pub const DEFAULT_LG_K: u8 = 11;
    /// Minimum allowed lg_k
    pub const MIN_LG_K: u8 = 4;
    /// Maximum allowed lg_k
    pub const MAX_LG_K: u8 = 26;

    /// Create a new CPC sketch with the given log2(k) parameter.
    ///
    /// Higher lg_k values provide better accuracy but use more memory.
    /// Default is 11 (k=2048).
    pub fn new(lg_k: u8) -> Self {
        assert!(
            (Self::MIN_LG_K..=Self::MAX_LG_K).contains(&lg_k),
            "lg_k must be between {} and {}",
            Self::MIN_LG_K,
            Self::MAX_LG_K
        );

        let k = 1usize << lg_k;

        CpcSketch {
            lg_k,
            k,
            num_coupons: 0,
            flavour: Flavour::Empty,
            pair_table: PairTable::new(4), // Start small
            window: Vec::new(),
            window_offset: 0,
            surprising_values: PairTable::new(2),
            hip_estimate: 0.0,
            kxp: k as f64, // k * 2^0 = k (all registers at 0)
            first_interesting_column: 0,
            was_merged: false,
        }
    }

    /// Update the sketch with a hashable item.
    pub fn update<T: Hash>(&mut self, item: &T) {
        let mut hasher = DefaultHasher::new();
        item.hash(&mut hasher);
        let hash = hasher.finish();
        self.update_with_hash(hash);
    }

    /// Internal update using a pre-computed hash value.
    fn update_with_hash(&mut self, hash: u64) {
        let coupon = Coupon::from_hash(hash, self.lg_k);
        let coupon_val = coupon.as_u32();

        match self.flavour {
            Flavour::Empty => {
                // First coupon transitions to Sparse
                self.pair_table.insert(coupon_val);
                self.num_coupons = 1;
                self.flavour = Flavour::Sparse;
                self.update_hip(coupon.row);
            }
            Flavour::Sparse => {
                if self.pair_table.insert(coupon_val) {
                    self.num_coupons += 1;
                    self.update_hip(coupon.row);

                    // Check if we should transition to Hybrid
                    // Transition when num_coupons > k/4 * 3
                    if self.num_coupons as usize > (self.k * 3) / 4 {
                        self.transition_to_hybrid();
                    }
                }
            }
            Flavour::Hybrid | Flavour::Pinned | Flavour::Sliding => {
                self.update_dense(coupon);
            }
        }
    }

    /// Update HIP estimator with a new coupon at the given row.
    fn update_hip(&mut self, row: u8) {
        let p = 1.0_f64 / (1u64 << row.min(63)) as f64;
        self.kxp -= p;
        self.hip_estimate += 1.0 / (self.kxp.max(f64::MIN_POSITIVE));
    }

    /// Update in dense mode (Hybrid, Pinned, or Sliding).
    fn update_dense(&mut self, coupon: Coupon) {
        let row = coupon.row;
        let col = coupon.column as usize;

        // Check if coupon falls within the window
        if row >= self.window_offset && row < self.window_offset + 8 {
            let window_row = row - self.window_offset;
            let bit = 1u8 << window_row;
            if self.window[col] & bit != 0 {
                return; // Already seen
            }
            self.window[col] |= bit;
            self.num_coupons += 1;
            self.update_hip(row);
        } else if row < self.window_offset {
            // Below window -- already counted (implicit)
            return;
        } else {
            // Above window -- store as surprising value
            let coupon_val = coupon.as_u32();
            if self.surprising_values.insert(coupon_val) {
                self.num_coupons += 1;
                self.update_hip(row);
            } else {
                return;
            }
        }

        // Check if we need to advance the sliding window
        self.maybe_advance_window();
    }

    /// Transition from Sparse to Hybrid mode.
    fn transition_to_hybrid(&mut self) {
        // Allocate the window (one byte per column, 8 rows of bits)
        self.window = vec![0u8; self.k];
        self.window_offset = 0;
        self.surprising_values.clear();

        // Move pair table coupons into window or surprising values
        let old_coupons: Vec<u32> = self.pair_table.iter().collect();
        self.pair_table.clear();

        for coupon_val in old_coupons {
            let coupon = Coupon::from_u32(coupon_val, self.lg_k);
            let row = coupon.row;
            let col = coupon.column as usize;

            if row < 8 {
                let bit = 1u8 << row;
                self.window[col] |= bit;
            } else {
                self.surprising_values.insert(coupon_val);
            }
        }

        self.flavour = if self.surprising_values.count > 0 {
            Flavour::Hybrid
        } else {
            Flavour::Pinned
        };
    }

    /// Check if the sliding window needs to advance.
    fn maybe_advance_window(&mut self) {
        // Count how many columns have the bottom bit set
        let bottom_row_full = self.window.iter().filter(|&&w| w & 1 != 0).count();

        // If most columns have the bottom row filled, advance
        if bottom_row_full as f64 > self.k as f64 * 0.9 {
            self.advance_window();
        }
    }

    /// Advance the sliding window by one row.
    fn advance_window(&mut self) {
        // Shift all window bytes right by one (dropping the bottom bit)
        for w in self.window.iter_mut() {
            *w >>= 1;
        }
        self.window_offset += 1;

        // Move surprising values that now fall within the window
        let sv_coupons: Vec<u32> = self.surprising_values.iter().collect();
        self.surprising_values.clear();

        for coupon_val in sv_coupons {
            let coupon = Coupon::from_u32(coupon_val, self.lg_k);
            let row = coupon.row;
            let col = coupon.column as usize;

            if row >= self.window_offset && row < self.window_offset + 8 {
                let window_row = row - self.window_offset;
                let bit = 1u8 << window_row;
                self.window[col] |= bit;
            } else if row >= self.window_offset + 8 {
                // Still above window -- keep as surprising
                self.surprising_values.insert(coupon_val);
            }
            // Below window: already counted, discard
        }

        self.flavour = if self.surprising_values.count > 0 {
            Flavour::Sliding
        } else {
            Flavour::Pinned
        };
    }

    /// Estimate the cardinality.
    ///
    /// Uses a coupon-based estimator that accounts for the multi-row
    /// structure of CPC sketches.
    pub fn estimate(&self) -> f64 {
        match self.flavour {
            Flavour::Empty => 0.0,
            Flavour::Sparse => {
                // In sparse mode, we have exact unique coupon count
                self.num_coupons as f64
            }
            _ => self.dense_estimate(),
        }
    }

    /// Estimate cardinality in dense mode using per-column max row values.
    ///
    /// Uses an HLL-style harmonic mean estimator over the maximum row
    /// value observed in each column. This naturally handles window
    /// advancement and large cardinalities.
    fn dense_estimate(&self) -> f64 {
        let k = self.k as f64;

        // For each column, compute the maximum row that has been filled.
        // Rows below window_offset are implicitly all filled.
        // The window stores 8 rows of bits per column.
        // Surprising values store rows above the window.
        let mut max_rows = vec![0u8; self.k];

        // All rows below window_offset are filled
        for mr in max_rows.iter_mut() {
            *mr = self.window_offset;
        }

        // Check window bits
        for (col, max_row) in max_rows.iter_mut().enumerate().take(self.k) {
            let w = self.window[col];
            if w != 0 {
                // Find highest set bit in window
                for bit_pos in (0..8u8).rev() {
                    if w & (1 << bit_pos) != 0 {
                        let row = self.window_offset + bit_pos + 1;
                        if row > *max_row {
                            *max_row = row;
                        }
                        break;
                    }
                }
            }
        }

        // Check surprising values (above window)
        for coupon_val in self.surprising_values.iter() {
            let coupon = Coupon::from_u32(coupon_val, self.lg_k);
            let col = coupon.column as usize;
            let row = coupon.row + 1; // +1 because row value represents leading zeros
            if col < self.k && row > max_rows[col] {
                max_rows[col] = row;
            }
        }

        // HLL-style harmonic mean estimator
        let mut sum = 0.0f64;
        let mut zeros = 0usize;
        for &mr in &max_rows {
            sum += 2f64.powi(-(mr as i32));
            if mr == 0 {
                zeros += 1;
            }
        }

        let alpha = match self.k {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / k),
        };

        let raw_estimate = alpha * k * k / sum;

        // Small range correction
        if raw_estimate <= 2.5 * k && zeros > 0 {
            k * (k / zeros as f64).ln()
        } else {
            raw_estimate
        }
    }

    /// Get the number of coupons (hash observations) stored.
    pub fn num_coupons(&self) -> u64 {
        self.num_coupons
    }

    /// Get the current flavour state.
    pub fn flavour(&self) -> &str {
        match self.flavour {
            Flavour::Empty => "Empty",
            Flavour::Sparse => "Sparse",
            Flavour::Hybrid => "Hybrid",
            Flavour::Pinned => "Pinned",
            Flavour::Sliding => "Sliding",
        }
    }

    /// Get the lg_k parameter.
    pub fn lg_k(&self) -> u8 {
        self.lg_k
    }

    /// Check if the sketch is empty.
    pub fn is_empty(&self) -> bool {
        self.flavour == Flavour::Empty
    }

    /// Get the first interesting column value.
    pub fn first_interesting_col(&self) -> u8 {
        self.first_interesting_column
    }

    /// Check whether this sketch has a sliding window allocated.
    pub fn has_window(&self) -> bool {
        !self.window.is_empty()
    }

    /// Check whether this sketch has a pair table with entries.
    pub fn has_table(&self) -> bool {
        self.pair_table.count > 0
    }

    /// Get the window offset.
    pub fn window_offset_value(&self) -> u8 {
        self.window_offset
    }

    /// Get reference to the window bytes.
    pub fn window_bytes(&self) -> &[u8] {
        &self.window
    }

    /// Get the surprising values as u32 coupons.
    pub fn surprising_values_iter(&self) -> Vec<u32> {
        self.surprising_values.iter().collect()
    }

    /// Get the pair table entries as u32 coupons.
    pub fn pair_table_iter(&self) -> Vec<u32> {
        self.pair_table.iter().collect()
    }

    /// Get whether this sketch was created by merging.
    pub fn was_merged_flag(&self) -> bool {
        self.was_merged
    }

    /// Get the HIP estimate value.
    pub fn hip_estimate_value(&self) -> f64 {
        self.hip_estimate
    }

    /// Get the kxp value.
    pub fn kxp_value(&self) -> f64 {
        self.kxp
    }

    /// Merge another CPC sketch into this one.
    ///
    /// Both sketches must have the same lg_k parameter.
    pub fn merge(&mut self, other: &CpcSketch) {
        assert_eq!(
            self.lg_k, other.lg_k,
            "Cannot merge sketches with different lg_k"
        );

        self.was_merged = true;

        if other.flavour == Flavour::Empty {
            return;
        }

        if self.flavour == Flavour::Empty {
            // Copy other into self
            self.num_coupons = other.num_coupons;
            self.flavour = other.flavour;
            self.pair_table = other.pair_table.clone();
            self.window = other.window.clone();
            self.window_offset = other.window_offset;
            self.surprising_values = other.surprising_values.clone();
            self.hip_estimate = other.hip_estimate;
            self.kxp = other.kxp;
            self.was_merged = true;
            return;
        }

        // Ensure both are in dense mode
        if self.flavour == Flavour::Sparse {
            self.transition_to_hybrid();
        }

        // Get all coupons from other
        let other_coupons = Self::collect_all_coupons(other);

        // Re-insert all coupons from other
        // This is correct but not the most efficient approach for very large sketches
        for coupon_val in other_coupons {
            let coupon = Coupon::from_u32(coupon_val, self.lg_k);
            self.update_dense(coupon);
        }
    }

    /// Collect all coupons from a sketch (for merge operations).
    fn collect_all_coupons(sketch: &CpcSketch) -> Vec<u32> {
        let mut coupons = Vec::new();

        match sketch.flavour {
            Flavour::Empty => {}
            Flavour::Sparse => {
                coupons.extend(sketch.pair_table.iter());
            }
            Flavour::Hybrid | Flavour::Pinned | Flavour::Sliding => {
                // Coupons from the window
                for col in 0..sketch.k {
                    let w = sketch.window[col];
                    for bit_pos in 0..8u8 {
                        if w & (1 << bit_pos) != 0 {
                            let row = sketch.window_offset + bit_pos;
                            let coupon = Coupon {
                                column: col as u32,
                                row,
                            };
                            coupons.push(coupon.as_u32());
                        }
                    }
                }

                // Implicit coupons below the window
                for col in 0..sketch.k {
                    for row in 0..sketch.window_offset {
                        let coupon = Coupon {
                            column: col as u32,
                            row,
                        };
                        coupons.push(coupon.as_u32());
                    }
                }

                // Surprising values above the window
                coupons.extend(sketch.surprising_values.iter());
            }
        }

        coupons
    }

    /// Serialize the sketch to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();

        // Header: lg_k, flavour, window_offset
        bytes.push(self.lg_k);
        bytes.push(self.flavour as u8);
        bytes.push(self.window_offset);
        bytes.push(if self.was_merged { 1 } else { 0 });

        // num_coupons
        bytes.extend_from_slice(&self.num_coupons.to_le_bytes());

        // HIP state
        bytes.extend_from_slice(&self.hip_estimate.to_le_bytes());
        bytes.extend_from_slice(&self.kxp.to_le_bytes());

        match self.flavour {
            Flavour::Empty => {}
            Flavour::Sparse => {
                // Pair table count + values
                let count = self.pair_table.count as u32;
                bytes.extend_from_slice(&count.to_le_bytes());
                for val in self.pair_table.iter() {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
            }
            Flavour::Hybrid | Flavour::Pinned | Flavour::Sliding => {
                // Window data
                bytes.extend_from_slice(&self.window);

                // Surprising values count + values
                let sv_count = self.surprising_values.count as u32;
                bytes.extend_from_slice(&sv_count.to_le_bytes());
                for val in self.surprising_values.iter() {
                    bytes.extend_from_slice(&val.to_le_bytes());
                }
            }
        }

        bytes
    }

    /// Deserialise a CPC sketch from its native byte format.
    ///
    /// This is the inverse of `to_bytes()`. The format is:
    ///   byte 0: lg_k
    ///   byte 1: flavour
    ///   byte 2: window_offset
    ///   byte 3: was_merged flag
    ///   bytes 4-11: num_coupons (u64 LE)
    ///   bytes 12-19: hip_estimate (f64 LE)
    ///   bytes 20-27: kxp (f64 LE)
    ///   Then flavour-specific data.
    pub fn from_native_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 28 {
            return Err("Insufficient bytes for CPC header");
        }

        let lg_k = bytes[0];
        let flavour_byte = bytes[1];
        let window_offset = bytes[2];
        let was_merged = bytes[3] != 0;

        if !(Self::MIN_LG_K..=Self::MAX_LG_K).contains(&lg_k) {
            return Err("lg_k out of valid range");
        }

        let k = 1usize << lg_k;

        let num_coupons =
            u64::from_le_bytes(bytes[4..12].try_into().map_err(|_| "invalid num_coupons")?);

        let hip_estimate = f64::from_le_bytes(
            bytes[12..20]
                .try_into()
                .map_err(|_| "invalid hip_estimate")?,
        );

        let kxp = f64::from_le_bytes(bytes[20..28].try_into().map_err(|_| "invalid kxp")?);

        let flavour = match flavour_byte {
            0 => Flavour::Empty,
            1 => Flavour::Sparse,
            2 => Flavour::Hybrid,
            3 => Flavour::Pinned,
            4 => Flavour::Sliding,
            _ => return Err("invalid flavour byte"),
        };

        let mut offset = 28usize;

        let mut pair_table = PairTable::new(4);
        let mut window = Vec::new();
        let mut surprising_values = PairTable::new(2);

        match flavour {
            Flavour::Empty => {}
            Flavour::Sparse => {
                if bytes.len() < offset + 4 {
                    return Err("Insufficient bytes for pair table count");
                }
                let count = u32::from_le_bytes(
                    bytes[offset..offset + 4]
                        .try_into()
                        .map_err(|_| "invalid pair count")?,
                ) as usize;
                offset += 4;

                if bytes.len() < offset + count * 4 {
                    return Err("Insufficient bytes for pair table data");
                }
                for i in 0..count {
                    let o = offset + i * 4;
                    let val = u32::from_le_bytes(
                        bytes[o..o + 4]
                            .try_into()
                            .map_err(|_| "invalid pair value")?,
                    );
                    pair_table.insert(val);
                }
            }
            Flavour::Hybrid | Flavour::Pinned | Flavour::Sliding => {
                // Window data: k bytes
                if bytes.len() < offset + k {
                    return Err("Insufficient bytes for window data");
                }
                window = bytes[offset..offset + k].to_vec();
                offset += k;

                // Surprising values count + values
                if bytes.len() < offset + 4 {
                    return Err("Insufficient bytes for surprising values count");
                }
                let sv_count = u32::from_le_bytes(
                    bytes[offset..offset + 4]
                        .try_into()
                        .map_err(|_| "invalid sv count")?,
                ) as usize;
                offset += 4;

                if bytes.len() < offset + sv_count * 4 {
                    return Err("Insufficient bytes for surprising values data");
                }
                for i in 0..sv_count {
                    let o = offset + i * 4;
                    let val = u32::from_le_bytes(
                        bytes[o..o + 4].try_into().map_err(|_| "invalid sv value")?,
                    );
                    surprising_values.insert(val);
                }
            }
        }

        Ok(CpcSketch {
            lg_k,
            k,
            num_coupons,
            flavour,
            pair_table,
            window,
            window_offset,
            surprising_values,
            hip_estimate,
            kxp,
            first_interesting_column: 0,
            was_merged,
        })
    }

    /// Get the estimated memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let pair_table = self.pair_table.slots.len() * 4;
        let window = self.window.len();
        let surprising = self.surprising_values.slots.len() * 4;
        base + pair_table + window + surprising
    }
}

/// CPC Union for merging multiple CPC sketches efficiently.
pub struct CpcUnion {
    #[allow(dead_code)]
    lg_k: u8,
    accumulator: CpcSketch,
}

impl CpcUnion {
    /// Create a new CPC union with the given lg_k.
    pub fn new(lg_k: u8) -> Self {
        CpcUnion {
            lg_k,
            accumulator: CpcSketch::new(lg_k),
        }
    }

    /// Add a sketch to the union.
    pub fn update(&mut self, sketch: &CpcSketch) {
        self.accumulator.merge(sketch);
    }

    /// Get the result sketch.
    pub fn result(&self) -> &CpcSketch {
        &self.accumulator
    }

    /// Get the cardinality estimate from the union.
    pub fn estimate(&self) -> f64 {
        self.accumulator.estimate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpc_empty() {
        let sketch = CpcSketch::new(11);
        assert!(sketch.is_empty());
        assert_eq!(sketch.estimate(), 0.0);
        assert_eq!(sketch.flavour(), "Empty");
    }

    #[test]
    fn test_cpc_basic() {
        let mut sketch = CpcSketch::new(11);

        for i in 0..1000 {
            sketch.update(&i);
        }

        let estimate = sketch.estimate();
        let error = (estimate - 1000.0).abs() / 1000.0;

        assert!(
            error < 0.10,
            "CPC error {:.1}% exceeds 10% tolerance (estimate={:.1}, expected=1000)",
            error * 100.0,
            estimate
        );
    }

    #[test]
    fn test_cpc_flavour_transitions() {
        let mut sketch = CpcSketch::new(8); // k=256, smaller for testing
        assert_eq!(sketch.flavour(), "Empty");

        sketch.update(&1);
        assert_eq!(sketch.flavour(), "Sparse");

        // Add enough to trigger transition to Hybrid/Pinned
        for i in 0..500 {
            sketch.update(&i);
        }

        let flav = sketch.flavour();
        assert!(
            flav == "Hybrid" || flav == "Pinned" || flav == "Sliding",
            "Expected dense flavour, got {flav}"
        );
    }

    #[test]
    fn test_cpc_merge() {
        let mut sketch1 = CpcSketch::new(11);
        let mut sketch2 = CpcSketch::new(11);

        for i in 0..500 {
            sketch1.update(&i);
        }

        for i in 500..1000 {
            sketch2.update(&i);
        }

        sketch1.merge(&sketch2);

        let estimate = sketch1.estimate();
        let error = (estimate - 1000.0).abs() / 1000.0;

        assert!(
            error < 0.15,
            "CPC merge error {:.1}% exceeds 15% tolerance (estimate={:.1})",
            error * 100.0,
            estimate
        );
    }

    #[test]
    fn test_cpc_union() {
        let mut union = CpcUnion::new(11);

        let mut s1 = CpcSketch::new(11);
        let mut s2 = CpcSketch::new(11);
        let mut s3 = CpcSketch::new(11);

        for i in 0..1000 {
            s1.update(&i);
        }
        for i in 500..1500 {
            s2.update(&i);
        }
        for i in 1000..2000 {
            s3.update(&i);
        }

        union.update(&s1);
        union.update(&s2);
        union.update(&s3);

        let estimate = union.estimate();
        let error = (estimate - 2000.0).abs() / 2000.0;

        assert!(
            error < 0.15,
            "CPC union error {:.1}% exceeds 15% tolerance (estimate={:.1})",
            error * 100.0,
            estimate
        );
    }

    #[test]
    fn test_cpc_small() {
        let mut sketch = CpcSketch::new(8);

        for i in 0..50 {
            sketch.update(&format!("item_{i}"));
        }

        let estimate = sketch.estimate();
        let error = (estimate - 50.0).abs() / 50.0;

        assert!(
            error < 0.15,
            "Small cardinality error {:.1}% too high (estimate={:.1})",
            error * 100.0,
            estimate
        );
    }

    #[test]
    fn test_cpc_large() {
        let mut sketch = CpcSketch::new(12);

        let n = 100_000;
        for i in 0..n {
            sketch.update(&i);
        }

        let estimate = sketch.estimate();
        let error = (estimate - n as f64).abs() / n as f64;

        assert!(
            error < 0.10,
            "Large cardinality error {:.1}% exceeds 10% tolerance (estimate={:.1}, expected={})",
            error * 100.0,
            estimate,
            n
        );
    }

    #[test]
    fn test_cpc_serialisation() {
        let mut sketch = CpcSketch::new(10);
        for i in 0..100 {
            sketch.update(&i);
        }

        let bytes = sketch.to_bytes();
        assert!(!bytes.is_empty());
        assert!(bytes.len() > 4); // At least header + some data
    }

    #[test]
    fn test_cpc_pair_table() {
        let mut table = PairTable::new(4);

        // Insert unique values
        assert!(table.insert(100));
        assert!(table.insert(200));
        assert!(table.insert(300));

        // Duplicate
        assert!(!table.insert(100));

        assert_eq!(table.count, 3);
        assert!(table.contains(100));
        assert!(table.contains(200));
        assert!(!table.contains(999));
    }

    #[test]
    fn test_cpc_memory_usage() {
        let sketch = CpcSketch::new(11);
        let mem = sketch.memory_usage();
        assert!(mem > 0);

        let mut sketch2 = CpcSketch::new(11);
        for i in 0..10000 {
            sketch2.update(&i);
        }
        // After adding data, memory should increase
        assert!(sketch2.memory_usage() >= mem);
    }

    #[test]
    fn test_cpc_duplicates() {
        let mut sketch = CpcSketch::new(11);

        // Add same item many times
        for _ in 0..1000 {
            sketch.update(&"same_item");
        }

        let estimate = sketch.estimate();
        // Should estimate approximately 1
        assert!(
            estimate < 5.0,
            "Duplicate handling failed: estimate={estimate:.1}, expected ~1"
        );
    }
}
