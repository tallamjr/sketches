//! CPC (Compressed Probabilistic Counting) Sketch - FM85 Algorithm
//!
//! Implementation of Kevin Lang's FM85 algorithm for cardinality estimation.
//! CPC achieves ~40% less space than HLL for comparable accuracy by collecting
//! "coupons" (row, column) pairs into a sliding window plus a surprising-value
//! table, and estimating cardinality with the HIP (Historical Inverse
//! Probability) estimator (or the ICON estimator after a merge).
//!
//! Reference: Kevin Lang, "Back to the Future: an Even More Nearly Optimal
//! Cardinality Estimation Algorithm" (2017), <https://arxiv.org/abs/1708.06839>.
//!
//! The core update path, flavour machinery, sliding window, surprising-value
//! `PairTable`, and HIP/kxp accumulation are ported faithfully from the Apache
//! DataSketches Rust reference (`cpc/{sketch.rs,mod.rs,pair_table.rs}`). The
//! only adaptation is the coupon derivation, which uses this crate's xxh3
//! 128-bit hash split into two 64-bit halves in place of the reference's
//! MurmurHash3 128-bit output.
//!
//! # Error Bounds
//! - Comparable accuracy to HLL but ~40% smaller serialised size.
//! - Standard error: approximately `1.0 / sqrt(2^lg_k)`.
//!
//! # When to Use CPC vs HLL
//! - Use CPC when network transfer cost or storage size is the primary concern.
//! - Use HLL when simplicity and merge performance matter more.

use crate::cpc_tables::{INVERSE_POWERS_OF_2, KXP_BYTE_TABLE};
use crate::hash::xxh3::Xxh3Hasher;
use crate::hash::{DEFAULT_SEED, Hashable, hash128_of};

/// Default log2 of K.
const DEFAULT_LG_K: u8 = 11;
/// Minimum allowed lg_k.
const MIN_LG_K: u8 = 4;
/// Maximum allowed lg_k.
const MAX_LG_K: u8 = 26;

/// The five flavour states of a CPC sketch, representing different
/// compression modes as cardinality grows. Boundaries are expressed in terms
/// of the coupon count C relative to K = 2^lg_k.
///
/// The update path does not branch on this enum directly (it branches on
/// whether the sliding window is allocated), so it is only used to expose and
/// assert flavour transitions in tests.
#[cfg(test)]
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
enum Flavor {
    Empty,   //    0  == C <    1
    Sparse,  //    1  <= C <   3K/32
    Hybrid,  // 3K/32 <= C <   K/2
    Pinned,  //   K/2 <= C < 27K/8  [NB: 27/8 = 3 + 3/8]
    Sliding, // 27K/8 <= C
}

/// Determine the flavour for a given `lg_k` and coupon count.
///
/// Ported from `cpc/mod.rs::determine_flavor`. Used only by the test-only
/// [`CpcSketch::flavor`] accessor.
#[cfg(test)]
fn determine_flavor(lg_k: u8, num_coupons: u32) -> Flavor {
    let k = 1u32 << lg_k;
    let c2 = num_coupons << 1;
    let c8 = num_coupons << 3;
    let c32 = num_coupons << 5;
    if num_coupons == 0 {
        Flavor::Empty
    } else if c32 < (3 * k) {
        Flavor::Sparse
    } else if c2 < k {
        Flavor::Hybrid
    } else if c8 < (27 * k) {
        Flavor::Pinned
    } else {
        Flavor::Sliding
    }
}

/// Determine the correct sliding-window offset for a given `lg_k` and coupon
/// count.
///
/// Ported from `cpc/mod.rs::determine_correct_offset`.
fn determine_correct_offset(lg_k: u8, num_coupons: u32) -> u8 {
    let k = 1i64 << lg_k;
    let tmp = ((num_coupons as i64) << 3) - (19 * k); // 8C - 19K
    if tmp < 0 {
        0
    } else {
        (tmp >> (lg_k + 3)) as u8 // tmp / 8K
    }
}

/// Count the number of set bits across a bit matrix (one u64 row per register).
fn count_bits_set_in_matrix(matrix: &[u64]) -> u32 {
    let mut count = 0;
    for word in matrix {
        count += word.count_ones();
    }
    count
}

const UPSIZE_NUMERATOR: u32 = 3;
const UPSIZE_DENOMINATOR: u32 = 4;
const DOWNSIZE_NUMERATOR: u32 = 1;
const DOWNSIZE_DENOMINATOR: u32 = 4;

/// A highly specialised hash table used for sparse data.
///
/// Stores `(row, col)` pairs packed as `row_col = (row << 6) | col` and uses
/// linear probing for collision resolution. Ported faithfully from
/// `cpc/pair_table.rs`.
#[derive(Debug, Clone)]
struct PairTable {
    /// log2 of number of slots.
    lg_size: u8,
    num_valid_bits: u8,
    num_items: u32,
    slots: Vec<u32>,
}

impl PairTable {
    fn new(lg_size: u8, num_valid_bits: u8) -> Self {
        assert!(
            (2..=26).contains(&lg_size),
            "lg_size must be in [2, 26], got {lg_size}"
        );
        assert!(
            ((lg_size + 1)..=32).contains(&num_valid_bits),
            "num_valid_bits must be in [lg_size + 1, 32], got {num_valid_bits} where lg_size = {lg_size}"
        );
        Self {
            lg_size,
            num_valid_bits,
            num_items: 0,
            slots: vec![u32::MAX; 1 << lg_size],
        }
    }

    fn slots(&self) -> &[u32] {
        &self.slots
    }

    fn clear(&mut self) {
        self.slots.fill(u32::MAX);
        self.num_items = 0;
    }

    fn maybe_delete(&mut self, item: u32) -> bool {
        let index = self.lookup(item) as usize;
        if self.slots[index] == u32::MAX {
            return false;
        }
        assert_eq!(
            self.slots[index], item,
            "item {item} not found at index {index}"
        );
        assert!(self.num_items > 0, "no items to delete");

        // delete the item
        self.slots[index] = u32::MAX;
        self.num_items -= 1;

        // re-insert all items between the freed slot and the next empty slot
        let mask = (1usize << self.lg_size) - 1;
        let mut probe = (index + 1) & mask;
        let mut fetched = self.slots[probe];
        while fetched != u32::MAX {
            self.slots[probe] = u32::MAX;
            self.must_insert(fetched);
            probe = (probe + 1) & mask;
            fetched = self.slots[probe];
        }

        // shrink if necessary
        while ((DOWNSIZE_DENOMINATOR * self.num_items) < (DOWNSIZE_NUMERATOR * (1 << self.lg_size)))
            && (self.lg_size > 2)
        {
            self.rebuild(self.lg_size - 1);
        }

        true
    }

    fn maybe_insert(&mut self, item: u32) -> bool {
        let index = self.lookup(item) as usize;
        if self.slots[index] == item {
            return false;
        }
        assert_eq!(
            self.slots[index],
            u32::MAX,
            "no empty slot found for item {item} at index {index}"
        );
        self.slots[index] = item;
        self.num_items += 1;
        while (UPSIZE_DENOMINATOR * self.num_items) > (UPSIZE_NUMERATOR * (1 << self.lg_size)) {
            self.rebuild(self.lg_size + 1);
        }
        true
    }

    fn must_insert(&mut self, item: u32) {
        let index = self.lookup(item) as usize;
        assert_ne!(
            self.slots[index], item,
            "item {item} already present in table"
        );
        assert_eq!(
            self.slots[index],
            u32::MAX,
            "no empty slot found for item {item} at index {index}"
        );
        self.slots[index] = item;
        // counts and resizing must be handled by the caller.
    }

    fn lookup(&self, item: u32) -> u32 {
        let size = 1u32 << self.lg_size;
        let mask = size - 1;

        let shift = self.num_valid_bits - self.lg_size;

        // extract high table size bits
        let mut probe = item >> shift;
        assert!(probe <= mask, "probe = {probe}, mask = {mask}");

        loop {
            let slot = self.slots[probe as usize];
            if slot != item && slot != u32::MAX {
                probe = (probe + 1) & mask;
            } else {
                break;
            }
        }

        probe
    }

    /// Rebuilds to a different size. `num_items` and `num_valid_bits` remain
    /// unchanged.
    fn rebuild(&mut self, lg_size: u8) {
        assert!(
            (2..=26).contains(&lg_size),
            "lg_size must be in [2, 26], got {lg_size}"
        );
        assert!(
            ((lg_size + 1)..=32).contains(&self.num_valid_bits),
            "num_valid_bits must be in [lg_size + 1, 32], got {} where lg_size = {lg_size}",
            self.num_valid_bits
        );

        let new_size = 1u32 << lg_size;
        assert!(
            new_size > self.num_items,
            "new table size ({new_size}) must be larger than number of items {}",
            self.num_items
        );

        let slots = std::mem::replace(&mut self.slots, vec![u32::MAX; new_size as usize]);
        self.lg_size = lg_size;
        for slot in slots {
            if slot != u32::MAX {
                self.must_insert(slot);
            }
        }
    }
}

/// CPC Sketch implementing the FM85 algorithm.
///
/// CPC provides cardinality estimation with superior space efficiency compared
/// to HLL. It automatically transitions through five flavour states as
/// cardinality grows, optimising compression at each stage.
#[derive(Debug, Clone)]
pub struct CpcSketch {
    // immutable config
    lg_k: u8,

    // sketch state
    /// Part of a speed optimisation: the lowest column index still considered
    /// surprising.
    first_interesting_column: u8,
    /// The number of coupons collected so far.
    num_coupons: u32,
    /// Sparse and surprising values. `None` only while EMPTY.
    surprising_value_table: Option<PairTable>,
    /// Derivable from num_coupons, but made explicit for speed.
    window_offset: u8,
    /// Size K bytes in dense mode (flavor >= HYBRID); empty in sparse mode.
    sliding_window: Vec<u8>,

    // estimator state
    /// Whether the sketch is the result of merging. If `false`, the HIP
    /// estimator is used; if `true`, ICON is used.
    merge_flag: bool,
    /// A pre-calculated probability factor (k * p) used to compute the HIP
    /// increment delta. Only valid in the HIP estimator.
    kxp: f64,
    /// The accumulated HIP cardinality estimate.
    hip_est_accum: f64,
}

impl Default for CpcSketch {
    fn default() -> Self {
        Self::new(DEFAULT_LG_K)
    }
}

impl CpcSketch {
    /// Default lg_k value.
    pub const DEFAULT_LG_K: u8 = DEFAULT_LG_K;
    /// Minimum allowed lg_k.
    pub const MIN_LG_K: u8 = MIN_LG_K;
    /// Maximum allowed lg_k.
    pub const MAX_LG_K: u8 = MAX_LG_K;

    /// Create a new CPC sketch with the given log2(k) parameter.
    ///
    /// Higher lg_k values provide better accuracy but use more memory.
    ///
    /// # Panics
    ///
    /// Panics if `lg_k` is not in the range `[4, 26]`.
    pub fn new(lg_k: u8) -> Self {
        assert!(
            (MIN_LG_K..=MAX_LG_K).contains(&lg_k),
            "lg_k out of range; got {lg_k}",
        );

        CpcSketch {
            lg_k,
            first_interesting_column: 0,
            num_coupons: 0,
            surprising_value_table: None,
            window_offset: 0,
            sliding_window: Vec::new(),
            merge_flag: false,
            kxp: (1u64 << lg_k) as f64,
            hip_est_accum: 0.0,
        }
    }

    /// Update the sketch with a hashable item.
    ///
    /// The 128-bit xxh3 hash is split into two 64-bit halves: the low half
    /// supplies the row (low `lg_k` bits), the high half supplies the column
    /// via its leading-zero count. This matches the reference's use of two
    /// 64-bit hash outputs (sketch.rs:182-189), which is essential for the
    /// HIP/ICON estimator's distributional assumptions to hold.
    pub fn update<T: Hashable + ?Sized>(&mut self, item: &T) {
        let h = hash128_of(&Xxh3Hasher, item, DEFAULT_SEED);
        let h1 = h as u64; // low 64 bits  -> row
        let h2 = (h >> 64) as u64; // high 64 bits -> column via leading zeros

        let k = 1u64 << self.lg_k;
        let col = h2.leading_zeros(); // 0 <= col <= 64
        let col = if col > 63 { 63 } else { col }; // clip so that 0 <= col <= 63
        let row = (h1 & (k - 1)) as u32;
        let mut row_col = (row << 6) | col;
        // To avoid the hash table's "empty" value (u32::MAX), we change the row
        // of the following pair. Extremely unlikely, but handled.
        if row_col == u32::MAX {
            row_col ^= 1 << 6;
        }
        self.row_col_update(row_col);
    }

    #[cfg(test)]
    fn flavor(&self) -> Flavor {
        determine_flavor(self.lg_k, self.num_coupons)
    }

    fn row_col_update(&mut self, row_col: u32) {
        let col = (row_col & 63) as u8;
        if col < self.first_interesting_column {
            // important speed optimisation
            return;
        }

        if self.num_coupons == 0 {
            // promote EMPTY to SPARSE
            self.surprising_value_table = Some(PairTable::new(2, 6 + self.lg_k));
        }

        if self.sliding_window.is_empty() {
            self.update_sparse(row_col);
        } else {
            self.update_windowed(row_col);
        }
    }

    fn surprising_value_table(&self) -> &PairTable {
        self.surprising_value_table
            .as_ref()
            .expect("surprising value table must be initialised")
    }

    fn mut_surprising_value_table(&mut self) -> &mut PairTable {
        self.surprising_value_table
            .as_mut()
            .expect("surprising value table must be initialised")
    }

    fn update_hip(&mut self, row_col: u32) {
        let k = 1u64 << self.lg_k;
        let col = (row_col & 63) as usize;
        let one_over_p = (k as f64) / self.kxp;
        self.hip_est_accum += one_over_p;
        self.kxp -= INVERSE_POWERS_OF_2[col + 1]; // notice the "+1"
    }

    fn update_sparse(&mut self, row_col: u32) {
        let k = 1u64 << self.lg_k;
        let c32pre = (self.num_coupons as u64) << 5;
        debug_assert!(c32pre < 3 * k); // C < 3K/32, in other words, flavor == SPARSE
        let is_novel = self.mut_surprising_value_table().maybe_insert(row_col);
        if is_novel {
            self.num_coupons += 1;
            self.update_hip(row_col);
            let c32post = (self.num_coupons as u64) << 5;
            if c32post >= 3 * k {
                self.promote_sparse_to_windowed();
            }
        }
    }

    fn promote_sparse_to_windowed(&mut self) {
        debug_assert_eq!(self.window_offset, 0);

        let k = 1u64 << self.lg_k;
        let c32 = (self.num_coupons as u64) << 5;
        debug_assert!((c32 == (3 * k)) || ((self.lg_k == 4) && (c32 > (3 * k))));

        self.sliding_window.resize(k as usize, 0);

        let old_table = self
            .surprising_value_table
            .replace(PairTable::new(2, 6 + self.lg_k))
            .expect("surprising value table must be initialised");
        let old_slots = old_table.slots();
        for &row_col in old_slots {
            if row_col != u32::MAX {
                let col = (row_col & 63) as u8;
                if col < 8 {
                    let row = (row_col >> 6) as usize;
                    self.sliding_window[row] |= 1 << col;
                } else {
                    // cannot use must_insert(), because it doesn't provide for growth
                    let is_novel = self.mut_surprising_value_table().maybe_insert(row_col);
                    debug_assert!(is_novel);
                }
            }
        }
    }

    fn update_windowed(&mut self, row_col: u32) {
        debug_assert!(self.window_offset <= 56);
        let k = 1u64 << self.lg_k;
        let c32pre = (self.num_coupons as u64) << 5;
        debug_assert!(c32pre >= 3 * k); // C >= 3K/32, in other words flavor >= HYBRID
        let c8pre = (self.num_coupons as u64) << 3;
        let w8pre = (self.window_offset as u64) << 3;
        debug_assert!(c8pre < (27 + w8pre) * k); // C < (K * 27/8) + (K * windowOffset)

        let mut is_novel = false; // novel if new coupon
        let col = (row_col & 63) as u8;
        if col < self.window_offset {
            // track the surprising 0's "before" the window
            is_novel = self.mut_surprising_value_table().maybe_delete(row_col); // inverted logic
        } else if col < self.window_offset + 8 {
            // track the 8 bits inside the window
            let row = (row_col >> 6) as usize;
            let old_bits = self.sliding_window[row];
            let new_bits = old_bits | (1 << (col - self.window_offset));
            if old_bits != new_bits {
                self.sliding_window[row] = new_bits;
                is_novel = true;
            }
        } else {
            // track the surprising 1's "after" the window
            is_novel = self.mut_surprising_value_table().maybe_insert(row_col); // normal logic
        }

        if is_novel {
            self.num_coupons += 1;
            self.update_hip(row_col);
            let c8post = (self.num_coupons as u64) << 3;
            if c8post >= (27 + w8pre) * k {
                self.move_window();
                debug_assert!((1..=56).contains(&self.window_offset));
                let w8post = (self.window_offset as u64) << 3;
                debug_assert!(c8post < ((27 + w8post) * k)); // C < (K * 27/8) + (K * windowOffset)
            }
        }
    }

    fn move_window(&mut self) {
        let new_offset = self.window_offset + 1;
        debug_assert!(new_offset <= 56);
        debug_assert_eq!(
            new_offset,
            determine_correct_offset(self.lg_k, self.num_coupons)
        );

        let k = 1usize << self.lg_k;

        // Construct the full-sized bit matrix that corresponds to the sketch.
        let bit_matrix = self.build_bit_matrix();

        // refresh the KXP register on every 8th window shift.
        if (new_offset & 0x7) == 0 {
            self.refresh_kxp(&bit_matrix);
        }

        self.mut_surprising_value_table().clear(); // the new number of surprises will be about the same

        let mask_for_clearing_window = (0xFFu64 << new_offset) ^ u64::MAX;
        let mask_for_flipping_early_zone = (1u64 << new_offset) - 1;

        let mut all_surprises_ored = 0u64;
        for (i, &matrix_row) in bit_matrix.iter().enumerate().take(k) {
            let mut pattern = matrix_row;
            self.sliding_window[i] = ((pattern >> new_offset) & 0xff) as u8;
            pattern &= mask_for_clearing_window;
            // The following line converts surprising 0's to 1's in the "early
            // zone" (and vice versa, which is essential for this procedure's
            // O(k) time cost).
            pattern ^= mask_for_flipping_early_zone;
            all_surprises_ored |= pattern; // a cheap way to recalculate first_interesting_column
            while pattern != 0 {
                let col = pattern.trailing_zeros();
                pattern ^= 1 << col; // erase the 1
                let row_col = ((i as u32) << 6) | col;
                let is_novel = self.mut_surprising_value_table().maybe_insert(row_col);
                debug_assert!(is_novel);
            }
        }

        self.window_offset = new_offset;
        self.first_interesting_column = all_surprises_ored.trailing_zeros() as u8;
        if self.first_interesting_column > new_offset {
            self.first_interesting_column = new_offset; // corner case
        }
    }

    /// The KXP register is a double with roughly 50 bits of precision, but it
    /// might need roughly 90 bits to track the value with perfect accuracy.
    ///
    /// Therefore, we recalculate KXP occasionally from the sketch's full
    /// bit_matrix so that it will reflect changes that were previously outside
    /// the mantissa.
    fn refresh_kxp(&mut self, bit_matrix: &[u64]) {
        // for improved numerical accuracy, we separately sum the bytes of the u64's
        let mut byte_sums = [0.0; 8];
        for &bits in bit_matrix {
            let mut word = bits;
            for sum in byte_sums.iter_mut() {
                let byte = (word & 0xFF) as usize;
                *sum += KXP_BYTE_TABLE[byte];
                word >>= 8;
            }
        }

        let mut total = 0.0;
        for i in (0..8).rev() {
            // the reverse order is important
            let factor = INVERSE_POWERS_OF_2[i * 8]; // pow(256.0, -j)
            total += factor * byte_sums[i];
        }

        self.kxp = total;
    }

    /// Construct the full-sized bit matrix (one u64 row per register) that
    /// corresponds to the current sketch state.
    fn build_bit_matrix(&self) -> Vec<u64> {
        let k = 1usize << self.lg_k;
        let offset = self.window_offset;
        debug_assert!(offset <= 56);

        // Fill the matrix with default rows in which the "early zone" is filled
        // with ones. This is essential for the routine's O(k) time cost (as
        // opposed to O(C)).
        let default_row = (1u64 << offset) - 1;

        let mut matrix = vec![default_row; k];
        if self.num_coupons == 0 {
            return matrix;
        }

        if !self.sliding_window.is_empty() {
            // In other words, we are in window mode, not sparse mode.
            for (i, slot) in matrix.iter_mut().enumerate().take(k) {
                // set the window bits, trusting the sketch's current offset
                *slot |= (self.sliding_window[i] as u64) << offset;
            }
        }

        for &row_col in self.surprising_value_table().slots() {
            if row_col != u32::MAX {
                let col = (row_col & 63) as u8;
                let row = (row_col >> 6) as usize;
                // Flip the specified matrix bit from its default value. In the
                // "early" zone the bit changes from 1 to 0; in the "late" zone
                // the bit changes from 0 to 1.
                matrix[row] ^= 1 << col;
            }
        }

        matrix
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        estimator::cpc_estimate(
            self.merge_flag,
            self.hip_est_accum,
            self.lg_k,
            self.num_coupons,
        )
    }

    /// Get the number of coupons (hash observations) collected.
    pub fn num_coupons(&self) -> u32 {
        self.num_coupons
    }

    /// Get the lg_k parameter.
    pub fn lg_k(&self) -> u8 {
        self.lg_k
    }

    /// Check if the sketch is empty.
    pub fn is_empty(&self) -> bool {
        self.num_coupons == 0
    }

    /// Get the first interesting column value (a speed-optimisation marker).
    pub fn first_interesting_col(&self) -> u8 {
        self.first_interesting_column
    }

    /// Validate the sketch by reconstructing its bit matrix and confirming the
    /// number of set bits matches `num_coupons`. Primarily for testing.
    pub fn validate(&self) -> bool {
        let bit_matrix = self.build_bit_matrix();
        count_bits_set_in_matrix(&bit_matrix) == self.num_coupons
    }

    /// Estimate of memory usage in bytes (heap allocations plus the struct).
    pub fn memory_usage(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let table = self
            .surprising_value_table
            .as_ref()
            .map(|t| t.slots.len() * 4)
            .unwrap_or(0);
        let window = self.sliding_window.len();
        base + table + window
    }

    /// Merge another CPC sketch into this one (in-place union).
    ///
    /// NOTE: Correct merge semantics are implemented in Task 4 (`CpcUnion`
    /// rewrite). This is a compiling placeholder that preserves the public API;
    /// it must not be relied upon for accuracy yet.
    pub fn merge(&mut self, _other: &CpcSketch) {
        // Task 4 will rewrite the union path. Intentionally left unimplemented
        // here so the public API keeps compiling for the PyO3 wrapper.
        self.merge_flag = true;
    }

    /// Serialise the sketch to bytes.
    ///
    /// NOTE: Faithful (de)serialisation is implemented in Task 5. This is a
    /// compiling placeholder used only to keep the existing `Serializable`
    /// impl and PyO3 wrapper building; it is not the final wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(self.lg_k);
        bytes.push(self.first_interesting_column);
        bytes.push(self.window_offset);
        bytes.push(if self.merge_flag { 1 } else { 0 });
        bytes.extend_from_slice(&self.num_coupons.to_le_bytes());
        bytes.extend_from_slice(&self.kxp.to_le_bytes());
        bytes.extend_from_slice(&self.hip_est_accum.to_le_bytes());
        bytes
    }

    /// Reconstruct a CPC sketch from the placeholder byte format produced by
    /// [`to_bytes`](Self::to_bytes).
    ///
    /// NOTE: Faithful deserialisation is implemented in Task 5. This decodes
    /// only the header written by the placeholder `to_bytes` (it does not
    /// restore the window or surprising-value table) and exists solely to keep
    /// the existing `Serializable` impl compiling.
    pub fn from_native_bytes(bytes: &[u8]) -> Result<Self, &'static str> {
        if bytes.len() < 24 {
            return Err("Insufficient bytes for CPC header");
        }
        let lg_k = bytes[0];
        if !(MIN_LG_K..=MAX_LG_K).contains(&lg_k) {
            return Err("lg_k out of valid range");
        }
        let first_interesting_column = bytes[1];
        let window_offset = bytes[2];
        let merge_flag = bytes[3] != 0;
        let num_coupons =
            u32::from_le_bytes(bytes[4..8].try_into().map_err(|_| "invalid num_coupons")?);
        let kxp = f64::from_le_bytes(bytes[8..16].try_into().map_err(|_| "invalid kxp")?);
        let hip_est_accum = f64::from_le_bytes(
            bytes[16..24]
                .try_into()
                .map_err(|_| "invalid hip_est_accum")?,
        );

        Ok(CpcSketch {
            lg_k,
            first_interesting_column,
            num_coupons,
            surprising_value_table: if num_coupons == 0 {
                None
            } else {
                Some(PairTable::new(2, 6 + lg_k))
            },
            window_offset,
            sliding_window: Vec::new(),
            merge_flag,
            kxp,
            hip_est_accum,
        })
    }
}

/// CPC Union for merging multiple CPC sketches.
///
/// NOTE: The correct union implementation lands in Task 4. This is a compiling
/// placeholder preserving the public API.
pub struct CpcUnion {
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

    /// The lg_k parameter of this union.
    pub fn lg_k(&self) -> u8 {
        self.lg_k
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

/// ICON + HIP cardinality estimator for CPC sketches.
///
/// Ported from the Apache DataSketches Rust reference (`cpc/estimator.rs`,
/// Task 2). The ICON estimator is defined by Kevin Lang's FM85 arXiv paper.
mod estimator {
    use crate::cpc_tables::{
        ICON_MAX_LOG_K, ICON_MIN_LOG_K, ICON_POLYNOMIAL_COEFFICIENTS,
        ICON_POLYNOMIAL_NUM_COEFFICIENTS,
    };

    /// Evaluate a polynomial via Horner's method over `coefficients[start..start+num]`.
    fn evaluate_polynomial(coefficients: &[f64], start: usize, num: usize, x: f64) -> f64 {
        let end = start + num - 1;
        let mut total = coefficients[end];
        for i in (start..end).rev() {
            total *= x;
            total += coefficients[i];
        }
        total
    }

    /// Exponential approximation of the ICON estimator for large coupon counts.
    fn icon_exponential_approximation(k: f64, c: f64) -> f64 {
        0.7940236163830469 * k * 2f64.powf(c / k)
    }

    /// ICON cardinality estimate for a sketch with the given `lg_k` and coupon count.
    pub(super) fn icon_estimate(lg_k: u8, num_coupons: u32) -> f64 {
        let lg_k = lg_k as usize;
        assert!(
            (ICON_MIN_LOG_K..=ICON_MAX_LOG_K).contains(&lg_k),
            "lg_k out of range; got {lg_k}",
        );

        match num_coupons {
            0 => return 0.0,
            1 => return 1.0,
            _ => {}
        }

        let k = (1 << lg_k) as f64;
        let c = num_coupons as f64;

        // Differing thresholds ensure that the approximated estimator is monotonically increasing.
        let threshold_factor = if lg_k < 14 { 5.7 } else { 5.6 };
        if c > threshold_factor * k {
            return icon_exponential_approximation(k, c);
        }

        let factor = evaluate_polynomial(
            &ICON_POLYNOMIAL_COEFFICIENTS,
            ICON_POLYNOMIAL_NUM_COEFFICIENTS * (lg_k - ICON_MIN_LOG_K),
            ICON_POLYNOMIAL_NUM_COEFFICIENTS,
            // The constant 2.0 is baked into the table ICON_POLYNOMIAL_COEFFICIENTS.
            c / (2.0 * k),
        );
        let ratio = c / k;
        // The constant 66.774757 is baked into the table ICON_POLYNOMIAL_COEFFICIENTS.
        let term = 1.0 + (ratio * ratio * ratio / 66.774757);
        let result = c * factor * term;
        if result >= c { result } else { c }
    }

    /// Combined CPC estimate: HIP accumulator when available, ICON otherwise.
    pub(super) fn cpc_estimate(
        merge_flag: bool,
        hip_est_accum: f64,
        lg_k: u8,
        num_coupons: u32,
    ) -> f64 {
        if !merge_flag {
            hip_est_accum
        } else {
            icon_estimate(lg_k, num_coupons)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn icon_estimate_is_reasonable_and_monotonic() {
        let e_small = estimator::icon_estimate(12, 100);
        let e_big = estimator::icon_estimate(12, 3000);
        assert!(e_big > e_small);
        assert!(e_small > 90.0 && e_small < 130.0, "icon(12,100)={e_small}");
    }

    #[test]
    fn test_cpc_empty() {
        let sketch = CpcSketch::new(11);
        assert!(sketch.is_empty());
        assert_eq!(sketch.estimate(), 0.0);
        assert_eq!(sketch.flavor(), Flavor::Empty);
    }

    #[test]
    fn test_cpc_basic() {
        let mut sketch = CpcSketch::new(11);
        for i in 0..1000u64 {
            sketch.update(&i);
        }
        let estimate = sketch.estimate();
        let error = (estimate - 1000.0).abs() / 1000.0;
        assert!(
            error < 0.05,
            "CPC error {:.1}% exceeds 5% tolerance (estimate={:.1})",
            error * 100.0,
            estimate
        );
    }

    #[test]
    fn test_cpc_flavour_transitions() {
        let mut sketch = CpcSketch::new(8); // k=256
        assert_eq!(sketch.flavor(), Flavor::Empty);

        sketch.update(&1u64);
        assert_eq!(sketch.flavor(), Flavor::Sparse);

        for i in 0..2000u64 {
            sketch.update(&i);
        }
        let flav = sketch.flavor();
        assert!(
            matches!(flav, Flavor::Hybrid | Flavor::Pinned | Flavor::Sliding),
            "Expected dense flavour, got {flav:?}"
        );
        // The reconstructed bit matrix must contain exactly num_coupons bits.
        assert!(sketch.validate(), "bit matrix bit count != num_coupons");
    }

    #[test]
    fn test_cpc_large() {
        let mut sketch = CpcSketch::new(12);
        let n = 100_000u64;
        for i in 0..n {
            sketch.update(&i);
        }
        let estimate = sketch.estimate();
        let error = (estimate - n as f64).abs() / n as f64;
        assert!(
            error < 0.03,
            "Large cardinality error {:.1}% exceeds 3% tolerance (estimate={:.1})",
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
    fn test_cpc_duplicates() {
        let mut sketch = CpcSketch::new(11);
        for _ in 0..1000 {
            sketch.update(&"same_item");
        }
        let estimate = sketch.estimate();
        assert!(
            estimate < 5.0,
            "Duplicate handling failed: estimate={estimate:.1}, expected ~1"
        );
    }

    #[test]
    fn test_cpc_validate_across_flavours() {
        // Validate the bit-matrix invariant holds as we cross flavour
        // boundaries (sparse -> hybrid -> pinned -> sliding).
        let mut sketch = CpcSketch::new(10);
        for i in 0..200_000u64 {
            sketch.update(&i);
            if i % 5000 == 0 {
                assert!(sketch.validate(), "validate failed at i={i}");
            }
        }
        assert!(sketch.validate());
    }

    #[test]
    fn test_cpc_pair_table() {
        let mut table = PairTable::new(2, 8);
        assert!(table.maybe_insert(100));
        assert!(table.maybe_insert(200));
        assert!(table.maybe_insert(255));
        assert!(!table.maybe_insert(100));
        assert_eq!(table.num_items, 3);
    }
}
