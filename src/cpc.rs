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
//! `PairTable`, and HIP/kxp accumulation are ported from the Apache
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

use crate::codec::{CodecError, Family, SketchHeader, SketchReader, SketchWriter};
use crate::cpc_compression::CompressedState;
use crate::cpc_tables::{INVERSE_POWERS_OF_2, KXP_BYTE_TABLE};
use crate::hash::xxh3::Xxh3Hasher;
use crate::hash::{DEFAULT_SEED, Hashable, SketchHasher, hash128_of};
use core::marker::PhantomData;

/// Default log2 of K.
const DEFAULT_LG_K: u8 = 11;
/// Minimum allowed lg_k.
const MIN_LG_K: u8 = 4;
/// Maximum allowed lg_k.
const MAX_LG_K: u8 = 26;
/// Serial format version for the codec serialisation of CPC.
const CPC_SERIAL_VERSION: u8 = 2;

/// The five flavour states of a CPC sketch, representing different
/// compression modes as cardinality grows. Boundaries are expressed in terms
/// of the coupon count C relative to K = 2^lg_k.
///
/// The update path does not branch on this enum directly (it branches on
/// whether the sliding window is allocated), so it is used to expose and
/// assert flavour transitions in tests and to drive the union merge logic.
#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
pub(crate) enum Flavor {
    Empty,   //    0  == C <    1
    Sparse,  //    1  <= C <   3K/32
    Hybrid,  // 3K/32 <= C <   K/2
    Pinned,  //   K/2 <= C < 27K/8  [NB: 27/8 = 3 + 3/8]
    Sliding, // 27K/8 <= C
}

/// Determine the flavour for a given `lg_k` and coupon count.
///
/// Ported from `cpc/mod.rs::determine_flavor`. Used by the [`CpcSketch::flavor`]
/// accessor and the union merge logic.
pub(crate) fn determine_flavor(lg_k: u8, num_coupons: u32) -> Flavor {
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
pub(crate) fn determine_correct_offset(lg_k: u8, num_coupons: u32) -> u8 {
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
/// linear probing for collision resolution. Ported from
/// `cpc/pair_table.rs`.
#[derive(Debug, Clone)]
pub(crate) struct PairTable {
    /// log2 of number of slots.
    lg_size: u8,
    num_valid_bits: u8,
    num_items: u32,
    slots: Vec<u32>,
}

impl PairTable {
    pub(crate) fn new(lg_size: u8, num_valid_bits: u8) -> Self {
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

    /// Return the occupied (non-empty) `row_col` pairs, sorted ascending.
    pub(crate) fn occupied_pairs(&self) -> Vec<u32> {
        let mut pairs: Vec<u32> = self
            .slots
            .iter()
            .copied()
            .filter(|&s| s != u32::MAX)
            .collect();
        pairs.sort_unstable();
        pairs
    }

    /// Build a `PairTable` from a slice of `row_col` pairs by inserting each.
    ///
    /// The table starts at the same lg-size as a fresh sparse sketch and grows
    /// via `maybe_insert` as the pairs are added.
    pub(crate) fn from_pairs(lg_k: u8, pairs: &[u32]) -> PairTable {
        let mut table = PairTable::new(2, 6 + lg_k);
        for &row_col in pairs {
            table.maybe_insert(row_col);
        }
        table
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
pub struct CpcSketchGeneric<H: SketchHasher> {
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

    /// Zero-size marker binding the hasher type. The hasher itself is
    /// `Default`, so no runtime state is carried.
    _hasher: PhantomData<H>,
}

/// CPC sketch using the crate default xxh3 hasher.
pub type CpcSketch = CpcSketchGeneric<Xxh3Hasher>;

impl<H: SketchHasher> Default for CpcSketchGeneric<H> {
    fn default() -> Self {
        Self::new(DEFAULT_LG_K)
    }
}

impl<H: SketchHasher> CpcSketchGeneric<H> {
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

        CpcSketchGeneric {
            lg_k,
            first_interesting_column: 0,
            num_coupons: 0,
            surprising_value_table: None,
            window_offset: 0,
            sliding_window: Vec::new(),
            merge_flag: false,
            kxp: (1u64 << lg_k) as f64,
            hip_est_accum: 0.0,
            _hasher: PhantomData,
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
        let h = hash128_of(&H::default(), item, DEFAULT_SEED);
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

    pub(crate) fn flavor(&self) -> Flavor {
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

    /// The occupied surprising-value pairs, sorted ascending (empty if EMPTY).
    ///
    /// Used by the compression round-trip tests and the byte-level
    /// serialisation in a later task of this sub-project.
    #[allow(dead_code)]
    pub(crate) fn surprising_pairs(&self) -> Vec<u32> {
        match &self.surprising_value_table {
            Some(table) => table.occupied_pairs(),
            None => Vec::new(),
        }
    }

    /// The raw sliding-window bytes (empty in sparse/empty flavours).
    pub(crate) fn sliding_window(&self) -> &[u8] {
        &self.sliding_window
    }

    /// The sliding-window offset.
    pub(crate) fn window_offset(&self) -> u8 {
        self.window_offset
    }

    /// Reference to the surprising-value table, panicking if uninitialised.
    pub(crate) fn surprising_value_table_ref(&self) -> &PairTable {
        self.surprising_value_table()
    }

    /// Rebuild a sketch from uncompressed flavour state plus persisted scalars.
    ///
    /// Mirrors how the reference feeds an [`UncompressedState`] back into a
    /// sketch: the window and surprising-value table are installed verbatim and
    /// every scalar (including the HIP/kxp estimator state) is restored exactly.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_uncompressed(
        lg_k: u8,
        num_coupons: u32,
        first_interesting_column: u8,
        window_offset: u8,
        merge_flag: bool,
        kxp: f64,
        hip_est_accum: f64,
        table: Option<PairTable>,
        window: Vec<u8>,
    ) -> CpcSketchGeneric<H> {
        CpcSketchGeneric {
            lg_k,
            first_interesting_column,
            num_coupons,
            surprising_value_table: table,
            window_offset,
            sliding_window: window,
            merge_flag,
            kxp,
            hip_est_accum,
            _hasher: PhantomData,
        }
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
    /// Implemented via a [`CpcUnion`]: both this sketch and `other` are folded
    /// into a fresh union accumulator and the reduced result replaces `self`.
    /// The result carries `merge_flag = true`, so [`estimate`](Self::estimate)
    /// uses the ICON estimator (HIP is invalid after a merge).
    pub fn merge(&mut self, other: &CpcSketchGeneric<H>) {
        let mut union = CpcUnionGeneric::<H>::new(self.lg_k);
        union.update(self);
        union.update(other);
        *self = union.to_sketch();
    }

    /// Serialise the sketch to bytes using the shared codec format with the
    /// surprising-value table and sliding window entropy-coded.
    ///
    /// Writes a [`SketchHeader`] tagged [`Family::Cpc`] (serial version 2)
    /// followed by the persisted scalars (`lg_k`, `num_coupons`,
    /// `first_interesting_column`, `window_offset`, `merge_flag`, `kxp`,
    /// `hip_est_accum`) and the compressed payload produced by
    /// [`CompressedState::compress`]. The compressed buffers are written
    /// length-prefixed together with the word/entry counts the decompressor
    /// needs. The resulting bytes reconstruct an identical sketch via
    /// [`from_bytes`](Self::from_bytes); HIP state is stored directly rather
    /// than recomputed by replay.
    pub fn to_bytes(&self) -> Vec<u8> {
        let compressed = CompressedState::compress(self);

        let mut w = SketchWriter::new();
        SketchHeader {
            family: Family::Cpc,
            version: CPC_SERIAL_VERSION,
            flags: 0,
        }
        .write(&mut w);

        w.put_u8(self.lg_k);
        w.put_u32_le(self.num_coupons);
        w.put_u8(self.first_interesting_column);
        w.put_u8(self.window_offset);
        w.put_u8(u8::from(self.merge_flag));
        w.put_f64_le(self.kxp);
        w.put_f64_le(self.hip_est_accum);

        // Compressed payload. Each entropy-coded buffer is stored truncated to
        // the number of words the coder actually used, length-prefixed, so the
        // decompressor can drive the inverse coders with the exact counts.
        w.put_u32_le(compressed.table_data_words as u32);
        w.put_u32_le(compressed.table_num_entries);
        for &word in &compressed.table_data[..compressed.table_data_words] {
            w.put_u32_le(word);
        }
        w.put_u32_le(compressed.window_data_words as u32);
        for &word in &compressed.window_data[..compressed.window_data_words] {
            w.put_u32_le(word);
        }

        w.into_vec()
    }

    /// Reconstruct a CPC sketch from the codec bytes produced by
    /// [`to_bytes`](Self::to_bytes).
    ///
    /// Restores the scalars and HIP state verbatim, rebuilds the
    /// [`CompressedState`] from the stored buffers, decompresses it back into
    /// the flavour state (surprising-value table plus sliding window), and
    /// installs that state via [`from_uncompressed`](Self::from_uncompressed).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CodecError> {
        let mut r = SketchReader::new(bytes);
        SketchHeader::read_expecting(&mut r, Family::Cpc)?;

        let lg_k = r.get_u8()?;
        let num_coupons = r.get_u32_le()?;
        let first_interesting_column = r.get_u8()?;
        let window_offset = r.get_u8()?;
        let merge_flag = r.get_u8()? != 0;
        let kxp = r.get_f64_le()?;
        let hip_est_accum = r.get_f64_le()?;

        let table_data_words = r.get_u32_le()? as usize;
        let table_num_entries = r.get_u32_le()?;
        let mut table_data = Vec::with_capacity(table_data_words);
        for _ in 0..table_data_words {
            table_data.push(r.get_u32_le()?);
        }
        let window_data_words = r.get_u32_le()? as usize;
        let mut window_data = Vec::with_capacity(window_data_words);
        for _ in 0..window_data_words {
            window_data.push(r.get_u32_le()?);
        }

        let compressed = CompressedState {
            table_data,
            table_data_words,
            table_num_entries,
            window_data,
            window_data_words,
        };

        let un = compressed.uncompress(lg_k, num_coupons);
        Ok(CpcSketchGeneric::from_uncompressed(
            lg_k,
            num_coupons,
            first_interesting_column,
            window_offset,
            merge_flag,
            kxp,
            hip_est_accum,
            un.table,
            un.window,
        ))
    }
}

/// The internal state of the union operation.
///
/// At most one of `Accumulator` and `BitMatrix` is held at any moment. The
/// accumulator is a sketch object that is employed until it graduates out of
/// Sparse mode, at which point it is converted into a full-sized bit matrix.
/// The bit matrix is mathematically a sketch but does not maintain any of the
/// "extra" fields of our sketch objects.
#[derive(Debug, Clone)]
enum UnionState<H: SketchHasher> {
    Accumulator(CpcSketchGeneric<H>),
    BitMatrix(Vec<u64>),
}

/// CPC Union (merge) operation for combining multiple CPC sketches.
///
/// Ported from the Apache DataSketches Rust reference
/// (`cpc/union.rs`). The accumulating union ORs the window and surprising-value
/// table of each source sketch into a bit matrix, downsampling lg_k where a
/// source has a smaller lg_k, and reduces the matrix back to a sketch with
/// `merge_flag = true` so [`estimate`](Self::estimate) uses ICON.
#[derive(Debug, Clone)]
pub struct CpcUnionGeneric<H: SketchHasher> {
    /// Immutable target lg_k. Due to merging with sources of lower lg_k, this
    /// can be reduced below the configured value.
    lg_k: u8,
    state: UnionState<H>,
}

/// CPC union using the crate default xxh3 hasher.
pub type CpcUnion = CpcUnionGeneric<Xxh3Hasher>;

impl<H: SketchHasher> Default for CpcUnionGeneric<H> {
    fn default() -> Self {
        Self::new(DEFAULT_LG_K)
    }
}

impl<H: SketchHasher> CpcUnionGeneric<H> {
    /// Create a new CPC union with the given lg_k.
    ///
    /// # Panics
    ///
    /// Panics if `lg_k` is not in the range `[4, 26]`.
    pub fn new(lg_k: u8) -> Self {
        // We begin with the accumulator holding an empty merged sketch object.
        let sketch = CpcSketchGeneric::<H>::new(lg_k);
        CpcUnionGeneric {
            lg_k,
            state: UnionState::Accumulator(sketch),
        }
    }

    /// The lg_k parameter of this union.
    ///
    /// Note that due to merging with source sketches that may have a lower
    /// value of lg_k, this value can be less than what the union object was
    /// configured with.
    pub fn lg_k(&self) -> u8 {
        self.lg_k
    }

    /// Add a sketch to the union.
    pub fn update(&mut self, sketch: &CpcSketchGeneric<H>) {
        let flavor = sketch.flavor();
        if flavor == Flavor::Empty {
            return;
        }

        if sketch.lg_k() < self.lg_k {
            self.reduce_k(sketch.lg_k());
        }

        // if the source is past SPARSE mode, make sure the union is a bit matrix.
        if flavor > Flavor::Sparse
            && let UnionState::Accumulator(old_sketch) = &self.state
        {
            let bit_matrix = old_sketch.build_bit_matrix();
            self.state = UnionState::BitMatrix(bit_matrix);
        }

        match &mut self.state {
            UnionState::Accumulator(old_sketch) => {
                // [Case A] source Sparse, union is the sketch accumulator.
                if flavor == Flavor::Sparse {
                    let old_flavor = old_sketch.flavor();
                    if old_flavor != Flavor::Sparse && old_flavor != Flavor::Empty {
                        unreachable!("unexpected old flavor in union accumulator: {old_flavor:?}");
                    }

                    // The following partially fixes the snowplow problem provided
                    // that the K's are equal.
                    if old_flavor == Flavor::Empty && self.lg_k == sketch.lg_k() {
                        *old_sketch = sketch.clone();
                        return;
                    }

                    walk_table_updating_sketch(old_sketch, sketch.surprising_value_table());
                    let final_flavor = old_sketch.flavor();

                    // if the accumulator graduated beyond sparse, switch to a bit
                    // matrix representation.
                    if final_flavor > Flavor::Sparse {
                        let bit_matrix = old_sketch.build_bit_matrix();
                        self.state = UnionState::BitMatrix(bit_matrix);
                    }

                    return;
                }

                // If the flavor is past SPARSE mode, the state must have been
                // converted to a bit matrix above. Empty was handled at the start
                // and Sparse was handled above.
                unreachable!("unexpected flavor in union accumulator: {flavor:?}");
            }
            UnionState::BitMatrix(old_matrix) => {
                if flavor == Flavor::Sparse {
                    // [Case B] source Sparse, union is a bit matrix.
                    or_table_into_matrix(old_matrix, self.lg_k, sketch.surprising_value_table());
                    return;
                }

                if matches!(flavor, Flavor::Hybrid | Flavor::Pinned) {
                    // [Case C] source Hybrid or Pinned, union is a bit matrix.
                    or_window_into_matrix(
                        old_matrix,
                        self.lg_k,
                        &sketch.sliding_window,
                        sketch.window_offset,
                        sketch.lg_k(),
                    );
                    or_table_into_matrix(old_matrix, self.lg_k, sketch.surprising_value_table());
                    return;
                }

                // [Case D] source Sliding, union is a bit matrix.
                // SLIDING mode uses inverted logic, so we cannot just walk the
                // source. Instead we convert it to a bit matrix and OR it in.
                assert_eq!(flavor, Flavor::Sliding);
                let src_matrix = sketch.build_bit_matrix();
                or_matrix_into_matrix(old_matrix, self.lg_k, &src_matrix, sketch.lg_k());
            }
        }
    }

    /// Reduce the union's lg_k to `new_lg_k`, downsampling the internal state.
    fn reduce_k(&mut self, new_lg_k: u8) {
        match &mut self.state {
            UnionState::Accumulator(sketch) => {
                if sketch.is_empty() {
                    self.lg_k = new_lg_k;
                    self.state = UnionState::Accumulator(CpcSketchGeneric::<H>::new(new_lg_k));
                    return;
                }

                let mut new_sketch = CpcSketchGeneric::<H>::new(new_lg_k);
                walk_table_updating_sketch(&mut new_sketch, sketch.surprising_value_table());

                let final_new_flavor = new_sketch.flavor();
                // the SV table had to have something in it.
                assert_ne!(final_new_flavor, Flavor::Empty);
                if final_new_flavor == Flavor::Sparse {
                    self.lg_k = new_lg_k;
                    self.state = UnionState::Accumulator(new_sketch);
                    return;
                }

                // the new sketch graduated beyond sparse, so convert to a bit matrix.
                self.lg_k = new_lg_k;
                self.state = UnionState::BitMatrix(new_sketch.build_bit_matrix());
            }
            UnionState::BitMatrix(matrix) => {
                let new_k = 1usize << new_lg_k;
                let mut new_matrix = vec![0u64; new_k];
                or_matrix_into_matrix(&mut new_matrix, new_lg_k, matrix, self.lg_k);
                self.lg_k = new_lg_k;
                self.state = UnionState::BitMatrix(new_matrix);
            }
        }
    }

    /// Get the union result as a new sketch.
    ///
    /// If the union holds an accumulator, a copy of that sketch is returned with
    /// `merge_flag = true`. If the union holds a bit matrix, the matrix is
    /// reduced back to a sketch, recomputing `num_coupons` from the popcount and
    /// rebuilding the window, surprising-value table, offset, and
    /// first-interesting-column.
    pub fn to_sketch(&self) -> CpcSketchGeneric<H> {
        match &self.state {
            UnionState::Accumulator(sketch) => {
                if sketch.is_empty() {
                    CpcSketchGeneric::<H>::new(self.lg_k)
                } else {
                    let mut sketch = sketch.clone();
                    assert_eq!(sketch.flavor(), Flavor::Sparse);
                    sketch.merge_flag = true;
                    sketch
                }
            }
            UnionState::BitMatrix(matrix) => {
                let lg_k = self.lg_k;

                let mut sketch = CpcSketchGeneric::<H>::new(lg_k);
                let num_coupons = count_bits_set_in_matrix(matrix);
                sketch.num_coupons = num_coupons;
                let offset = determine_correct_offset(lg_k, num_coupons);
                sketch.window_offset = offset;

                let k = 1usize << lg_k;
                let mut sliding_window = vec![0u8; k];

                // LgSize = K/16; in some cases this will end up being oversized.
                let new_table_lg_size = (lg_k - 4).max(2);
                let mut table = PairTable::new(new_table_lg_size, 6 + lg_k);

                // the following works even when the offset is zero.
                let mask_for_clearing_window = (0xFFu64 << offset) ^ u64::MAX;
                let mask_for_flipping_early_zone = (1u64 << offset) - 1;
                let mut all_surprises_ored = 0u64;

                // The snowplow effect was caused by processing the rows in order,
                // but it is fixed by using a sufficiently large hash table.
                for (i, &matrix_row) in matrix.iter().enumerate().take(k) {
                    let mut pattern = matrix_row;
                    sliding_window[i] = ((pattern >> offset) & 0xFF) as u8;
                    pattern &= mask_for_clearing_window;
                    // this flipping converts surprising 0's to 1's.
                    pattern ^= mask_for_flipping_early_zone;
                    all_surprises_ored |= pattern;
                    while pattern != 0 {
                        let col = pattern.trailing_zeros();
                        pattern ^= 1u64 << col; // erase the 1
                        let row_col = ((i as u32) << 6) | col;
                        let is_novel = table.maybe_insert(row_col);
                        assert!(is_novel);
                    }
                }

                sketch.first_interesting_column = all_surprises_ored.trailing_zeros() as u8;
                if sketch.first_interesting_column > offset {
                    sketch.first_interesting_column = offset; // corner case
                }

                // HIP-related fields contain zeros, which is fine because
                // merge_flag is true, so the HIP estimator will not be used.
                sketch.sliding_window = sliding_window;
                sketch.surprising_value_table = Some(table);
                sketch.merge_flag = true;

                sketch
            }
        }
    }

    /// Get the result sketch (alias of [`to_sketch`](Self::to_sketch)).
    pub fn result(&self) -> CpcSketchGeneric<H> {
        self.to_sketch()
    }

    /// Get the cardinality estimate from the union.
    ///
    /// Because the result carries `merge_flag = true`, this uses the ICON
    /// estimator.
    pub fn estimate(&self) -> f64 {
        self.to_sketch().estimate()
    }

    /// Returns the number of coupons in the union.
    ///
    /// Primarily for testing and validation.
    pub fn num_coupons(&self) -> u32 {
        match &self.state {
            UnionState::Accumulator(sketch) => sketch.num_coupons,
            UnionState::BitMatrix(matrix) => count_bits_set_in_matrix(matrix),
        }
    }
}

/// OR a source sliding window into the destination bit matrix, downsampling
/// rows modulo the destination K when the destination lg_k is smaller.
fn or_window_into_matrix(
    dst_matrix: &mut [u64],
    dst_lg_k: u8,
    src_window: &[u8],
    src_offset: u8,
    src_lg_k: u8,
) {
    assert!(dst_lg_k <= src_lg_k);
    let dst_mask = (1usize << dst_lg_k) - 1; // downsamples when dst_lg_k < src_lg_k
    let src_k = 1usize << src_lg_k;
    for (src_row, &window_byte) in src_window.iter().enumerate().take(src_k) {
        dst_matrix[src_row & dst_mask] |= (window_byte as u64) << src_offset;
    }
}

/// OR a source surprising-value table into the destination bit matrix,
/// downsampling rows modulo the destination K when needed.
fn or_table_into_matrix(dst_matrix: &mut [u64], dst_lg_k: u8, src_table: &PairTable) {
    let dst_mask = (1usize << dst_lg_k) - 1; // downsamples when dst_lg_k < src_lg_k
    for &row_col in src_table.slots() {
        if row_col != u32::MAX {
            let src_row = (row_col >> 6) as usize;
            let src_col = (row_col & 63) as usize;
            let dst_row = src_row & dst_mask;
            dst_matrix[dst_row] |= 1u64 << src_col;
        }
    }
}

/// OR a source bit matrix into the destination bit matrix, downsampling rows
/// modulo the destination K when the destination lg_k is smaller.
fn or_matrix_into_matrix(dst_matrix: &mut [u64], dst_lg_k: u8, src_matrix: &[u64], src_lg_k: u8) {
    assert!(dst_lg_k <= src_lg_k);
    let dst_mask = (1usize << dst_lg_k) - 1; // downsamples when dst_lg_k < src_lg_k
    let src_k = 1usize << src_lg_k;
    for (src_row, &src_bits) in src_matrix.iter().enumerate().take(src_k) {
        let dst_row = src_row & dst_mask;
        dst_matrix[dst_row] |= src_bits;
    }
}

/// Walk a source surprising-value table, updating the destination sketch via
/// `row_col_update`. A golden-ratio stride over the table slots avoids the
/// snowplow effect; row indices are masked to the destination lg_k.
fn walk_table_updating_sketch<H: SketchHasher>(
    sketch: &mut CpcSketchGeneric<H>,
    table: &PairTable,
) {
    assert!(sketch.lg_k() <= 26);

    let slots = table.slots();
    let num_slots = slots.len() as u32;

    // downsamples when destination lg_k < source lg_k.
    let dst_mask = (((1u64 << sketch.lg_k()) - 1) << 6) | 63;
    // Using a golden ratio stride fixes the snowplow effect.
    let mut stride = (0.6180339887498949 * (num_slots as f64)) as u32;
    assert!(stride >= 2);
    if stride == ((stride >> 1) << 1) {
        // force the stride to be odd.
        stride += 1;
    }
    assert!((stride >= 3) && (stride < num_slots));

    let mut k = 0u32;
    for _ in 0..num_slots {
        k &= num_slots - 1;
        let row_col = slots[k as usize];
        if row_col != u32::MAX {
            sketch.row_col_update(row_col & (dst_mask as u32));
        }
        k += stride;
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
    fn murmur3_variant_constructs_and_counts() {
        use crate::hash::murmur3::Murmur3Hasher;
        let mut s = CpcSketchGeneric::<Murmur3Hasher>::new(12);
        for i in 0u64..10_000 {
            s.update(&i);
        }
        let est = s.estimate();
        // within 5% of true cardinality 10_000
        assert!((est - 10_000.0).abs() / 10_000.0 < 0.05, "est={est}");
    }

    #[test]
    fn xxh3_alias_unchanged_estimate() {
        let mut s = CpcSketch::new(12);
        for i in 0u64..10_000 {
            s.update(&i);
        }
        assert!((s.estimate() - 10_000.0).abs() / 10_000.0 < 0.05);
    }

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
