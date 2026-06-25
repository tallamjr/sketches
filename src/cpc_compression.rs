// Low-level bit-I/O and entropy coders for CPC compression.
//
// Ported faithfully (logic-identical) from Apache DataSketches
// (lib/datasketches-rust/datasketches/src/cpc/compression.rs). This module
// holds the bit-buffer helpers and the four `low_level_*` pair/byte coders.
// The higher-level flavour compressors that drive these are added in a later
// task of this sub-project; until then a narrowly-scoped `dead_code` allow
// covers the items that task will consume (every item is exercised by the
// round-trip tests below).
#![allow(dead_code)]

use std::cmp::Ordering;

use crate::cpc_compression_data::LENGTH_LIMITED_UNARY_ENCODING_TABLE65;
use crate::cpc_compression_data::length_limited_unary_decoding_table;

// Reference: compression.rs:641
pub(crate) fn determine_pseudo_phase(lg_k: u8, num_coupons: u32) -> u8 {
    let k = 1 << lg_k;
    // This mid-range logic produces pseudo-phases. They are used to select encoding tables.
    // The thresholds were chosen by hand after looking at plots of measured compression.
    if 1000 * num_coupons < 2375 * k {
        if 4 * num_coupons < 3 * k {
            // mid-range table
            16
        } else if 10 * num_coupons < 11 * k {
            // mid-range table
            16 + 1
        } else if 100 * num_coupons < 132 * k {
            // mid-range table
            16 + 2
        } else if 3 * num_coupons < 5 * k {
            // mid-range table
            16 + 3
        } else if 1000 * num_coupons < 1965 * k {
            // mid-range table
            16 + 4
        } else if 1000 * num_coupons < 2275 * k {
            // mid-range table
            16 + 5
        } else {
            // steady-state table employed before its actual phase
            6
        }
    } else {
        // This steady-state logic produces true phases. They are used to select
        // encoding tables, and also column permutations for the "Sliding" flavor.
        debug_assert!(lg_k >= 4);
        let tmp = num_coupons >> (lg_k - 4);
        (tmp & 15) as u8 // phase
    }
}

// Reference: compression.rs:677
fn write_unary(
    compressed_words: &mut [u32],
    next_word_index: &mut usize,
    bitbuf: &mut u64,
    bufbits: &mut u8,
    value: u64,
) {
    assert!(*bufbits <= 31);

    let mut remaining = value;
    while remaining >= 16 {
        remaining -= 16;
        // Here we output 16 zeros, but we don't need to physically write them into bitbuf
        // because it already contains zeros in that region.
        *bufbits += 16; // Record the fact that 16 bits of output have occurred.
        maybe_flush_bitbuf(bitbuf, bufbits, compressed_words, next_word_index);
    }

    let the_unary_code = 1 << remaining;
    *bitbuf |= the_unary_code << *bufbits;
    *bufbits += (remaining + 1) as u8;
    maybe_flush_bitbuf(bitbuf, bufbits, compressed_words, next_word_index);
}

// Reference: compression.rs:701
fn read_unary(
    compressed_words: &[u32],
    next_word_index: &mut usize,
    bitbuf: &mut u64,
    bufbits: &mut u8,
) -> u64 {
    let mut subtotal = 0u64;
    loop {
        // ensure 8 bits in bit buffer
        maybe_fill_bitbuf(bitbuf, bufbits, compressed_words, next_word_index, 8);
        // These 8 bits include either all or part of the Unary codeword
        let peek8 = *bitbuf & 0xff;
        let trailing_zeros = peek8.trailing_zeros() as u8;
        if trailing_zeros < 8 {
            *bufbits -= 1 + trailing_zeros;
            *bitbuf >>= 1 + trailing_zeros;
            return subtotal + trailing_zeros as u64;
        }
        // The codeword was partial, so read some more
        subtotal += 8;
        *bufbits -= 8;
        *bitbuf >>= 8;
    }
}

// Reference: compression.rs:726
fn maybe_flush_bitbuf(
    bitbuf: &mut u64,
    bufbits: &mut u8,
    word: &mut [u32],
    word_index: &mut usize,
) {
    if *bufbits >= 32 {
        word[*word_index] = (*bitbuf & 0xffffffff) as u32;
        *word_index += 1;
        *bitbuf >>= 32;
        *bufbits -= 32;
    }
}

// Reference: compression.rs:740
fn maybe_fill_bitbuf(
    bitbuf: &mut u64,
    bufbits: &mut u8,
    words: &[u32],
    word_index: &mut usize,
    minbits: u8,
) {
    if *bufbits < minbits {
        *bitbuf |= (words[*word_index] as u64) << *bufbits;
        *word_index += 1;
        *bufbits += 32;
    }
}

// Reference: compression.rs:760
//
// Explanation of padding: we write
// 1) xdelta (huffman, provides at least 1 bit, requires 12-bit lookahead)
// 2) ydeltaGolombHi (unary, provides at least 1 bit, requires 8-bit lookahead)
// 3) ydeltaGolombLo (straight B bits).
// So the 12-bit lookahead is the tight constraint, but there are at least (2 + B) bits emitted,
// so we would be safe with max (0, 10 - B) bits of padding at the end of the bitstream.
pub(crate) fn safe_length_for_compressed_window_buf(k: u32) -> usize {
    // 11 bits of padding, due to 12-bit lookahead, with 1 bit certainly present.
    let bits = 12 * k + 11;
    divide_longs_rounding_up(bits as usize, 32)
}

// Reference: compression.rs:766
pub(crate) fn safe_length_for_compressed_pair_buf(
    k: u32,
    num_pairs: u32,
    num_base_bits: u8,
) -> usize {
    // Long ybits = k + numPairs; // simpler and safer UB
    // The following tighter UB on ybits is based on page 198
    // of the textbook "Managing Gigabytes" by Witten, Moffat, and Bell.
    // Notice that if numBaseBits == 0 it coincides with (k + numPairs).

    let k = k as usize;
    let num_pairs = num_pairs as usize;
    let num_base_bits = num_base_bits as usize;

    let ybits = num_pairs * (1 + num_base_bits) + (k >> num_base_bits);
    let xbits = 12 * (num_pairs);
    let padding = 10usize.saturating_sub(num_base_bits);
    divide_longs_rounding_up(xbits + ybits + padding, 32)
}

// Reference: compression.rs:782
pub(crate) fn divide_longs_rounding_up(x: usize, y: usize) -> usize {
    debug_assert_ne!(y, 0);
    let quotient = x / y;
    if quotient * y == x {
        quotient
    } else {
        quotient + 1
    }
}

/// Returns an integer that is between zero and ceil(log_2(k)) - 1, inclusive.
// Reference: compression.rs:793
pub(crate) fn golomb_choose_number_of_base_bits(k: u32, count: u64) -> u8 {
    debug_assert!(k > 0);
    debug_assert!(count > 0);
    let quotient = ((k as u64) - count) / count; // integer division
    if quotient == 0 {
        0
    } else {
        floor_log2_of_long(quotient)
    }
}

// Reference: compression.rs:804
pub(crate) fn floor_log2_of_long(x: u64) -> u8 {
    debug_assert!(x > 0);
    let mut p = 0u8;
    let mut y = 1u64;
    loop {
        match u64::cmp(&y, &x) {
            Ordering::Equal => return p,
            Ordering::Greater => return p - 1,
            Ordering::Less => {
                p += 1;
                y <<= 1;
            }
        }
    }
}

/// Returns the number of compressed words that were actually used.
///
/// It is the caller's responsibility to ensure that `window_data` is long enough.
// Reference: compression.rs:214
pub(crate) fn low_level_compress_bytes(
    byte_array: &[u8],
    num_bytes_to_encode: u32,
    encoding_table: &[u16],
    window_data: &mut [u32],
) -> usize {
    // bits are packed into this first, then are flushed to window_data
    let mut bitbuf: u64 = 0;
    // number of bits currently in bitbuf; must be between 0 and 31
    let mut bufbits: u8 = 0;
    let mut next_word_index = 0;

    for byte_index in 0..num_bytes_to_encode {
        let code_info = encoding_table[byte_array[byte_index as usize] as usize];
        let code_val = (code_info & 0xfff) as u64;
        let code_len = (code_info >> 12) as u8;
        bitbuf |= code_val << bufbits;
        bufbits += code_len;
        maybe_flush_bitbuf(&mut bitbuf, &mut bufbits, window_data, &mut next_word_index);
    }

    // Pad the bitstream with 11 zero-bits so that the decompressor's 12-bit peek can't overrun
    // its input.
    bufbits += 11;
    maybe_flush_bitbuf(&mut bitbuf, &mut bufbits, window_data, &mut next_word_index);

    if bufbits > 0 {
        // We are done encoding now, so we flush the bit buffer.
        debug_assert!(bufbits < 32);
        window_data[next_word_index] = (bitbuf & 0xffffffff) as u32;
        next_word_index += 1;
    }

    next_word_index
}

/// Returns the number of `table_data` words actually used.
///
/// Here "pairs" refers to row/column pairs that specify the positions of surprising values in
/// the bit matrix.
// Reference: compression.rs:268
pub(crate) fn low_level_compress_pairs(
    pairs: &[u32],
    num_base_bits: u8,
    table_data: &mut [u32],
) -> usize {
    let mut bitbuf: u64 = 0;
    let mut bufbits: u8 = 0;
    let mut next_word_index = 0;
    let golomb_lo_mask = (1u64 << num_base_bits) - 1;
    let mut predicted_row_index = 0;
    let mut predicted_col_index = 0;

    for &row_col in pairs {
        let row_index = row_col >> 6;
        let col_index = row_col & 63;

        if row_index != predicted_row_index {
            predicted_col_index = 0;
        }

        assert!(row_index >= predicted_row_index);
        assert!(col_index >= predicted_col_index);

        let y_delta = row_index - predicted_row_index;
        let x_delta = col_index - predicted_col_index;

        predicted_row_index = row_index;
        predicted_col_index = col_index + 1;

        let code_info = LENGTH_LIMITED_UNARY_ENCODING_TABLE65[x_delta as usize];
        let code_val = (code_info & 0xfff) as u64;
        let code_len = (code_info >> 12) as u8;
        bitbuf |= code_val << bufbits;
        bufbits += code_len;

        maybe_flush_bitbuf(&mut bitbuf, &mut bufbits, table_data, &mut next_word_index);

        let golomb_lo = (y_delta as u64) & golomb_lo_mask;
        let golomb_hi = (y_delta as u64) >> num_base_bits;
        write_unary(
            table_data,
            &mut next_word_index,
            &mut bitbuf,
            &mut bufbits,
            golomb_hi,
        );

        bitbuf |= golomb_lo << bufbits;
        bufbits += num_base_bits;
        maybe_flush_bitbuf(&mut bitbuf, &mut bufbits, table_data, &mut next_word_index);
    }

    // Pad the bitstream so that the decompressor's 12-bit peek can't overrun its input.
    let padding = 10u8.saturating_sub(num_base_bits);
    bufbits += padding;
    maybe_flush_bitbuf(&mut bitbuf, &mut bufbits, table_data, &mut next_word_index);

    if bufbits > 0 {
        // We are done encoding now, so we flush the bit buffer
        assert!(bufbits < 32);
        table_data[next_word_index] = (bitbuf & 0xffffffff) as u32;
        next_word_index += 1;
    }

    next_word_index
}

// Reference: compression.rs:537
pub(crate) fn low_level_uncompress_pairs(
    pairs: &mut [u32],
    num_pairs_to_decode: u32,
    num_base_bits: u8,
    compressed_words: &[u32],
    num_compressed_words: usize,
) {
    let mut word_index = 0;
    let mut bitbuf: u64 = 0;
    let mut bufbits: u8 = 0;
    let golomb_lo_mask = (1u64 << num_base_bits) - 1;
    let mut predicted_row_index = 0u32;
    let mut predicted_col_index = 0u8;

    // for each pair we need to read:
    // x_delta (12-bit length-limited unary)
    // y_delta_hi (unary)
    // y_delta_lo (basebits)

    let decoding_table = length_limited_unary_decoding_table();

    for pair_index in 0..num_pairs_to_decode {
        // ensure 12 bits in bit buffer
        maybe_fill_bitbuf(
            &mut bitbuf,
            &mut bufbits,
            compressed_words,
            &mut word_index,
            12,
        );
        let peek12 = bitbuf & 0xfff;
        let lookup = decoding_table[peek12 as usize];
        let code_word_length = (lookup >> 8) as u8;
        let x_delta = (lookup & 0xff) as u8;
        bitbuf >>= code_word_length;
        bufbits -= code_word_length;

        let golomb_hi = read_unary(compressed_words, &mut word_index, &mut bitbuf, &mut bufbits);
        // ensure num_base_bits in the bit buffer
        maybe_fill_bitbuf(
            &mut bitbuf,
            &mut bufbits,
            compressed_words,
            &mut word_index,
            num_base_bits,
        );
        let golomb_lo = bitbuf & golomb_lo_mask;
        bitbuf >>= num_base_bits;
        bufbits -= num_base_bits;
        let y_delta = ((golomb_hi << num_base_bits) | golomb_lo) as u32;

        // Now that we have x_delta and y_delta, we can compute the pair's row and column
        if y_delta > 0 {
            predicted_col_index = 0;
        }
        let row_index = predicted_row_index + y_delta;
        let col_index = predicted_col_index + x_delta;
        let row_col = (row_index << 6) | (col_index as u32);
        pairs[pair_index as usize] = row_col;
        predicted_row_index = row_index;
        predicted_col_index = col_index + 1;
    }

    debug_assert!(
        word_index <= num_compressed_words,
        "word_index: {word_index}, num_compressed_words: {num_compressed_words}",
    );
}

// Reference: compression.rs:604
pub(crate) fn low_level_uncompress_bytes(
    byte_array: &mut [u8],
    num_bytes_to_decode: u32,
    compressed_words: &[u32],
    num_compressed_words: usize,
    decoding_table: &[u16],
) {
    let mut word_index = 0;
    let mut bitbuf: u64 = 0;
    let mut bufbits: u8 = 0;

    for byte_index in 0..num_bytes_to_decode {
        // ensure 12 bits in bit buffer
        maybe_fill_bitbuf(
            &mut bitbuf,
            &mut bufbits,
            compressed_words,
            &mut word_index,
            12,
        );
        // These 12 bits will include an entire Huffman codeword.
        let peek12 = bitbuf & 0xfff;
        let lookup = decoding_table[peek12 as usize];
        let code_word_length = (lookup >> 8) as u8;
        let decoded_byte = (lookup & 0xff) as u8;
        byte_array[byte_index as usize] = decoded_byte;
        bitbuf >>= code_word_length;
        bufbits -= code_word_length;
    }

    // Buffer over-run should be impossible unless there is a bug.
    debug_assert!(
        word_index <= num_compressed_words,
        "word_index: {word_index}, num_compressed_words: {num_compressed_words}",
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpc_compression_data::ENCODING_TABLES_FOR_HIGH_ENTROPY_BYTE;
    use crate::cpc_compression_data::high_entropy_decoding_tables;

    #[test]
    fn pairs_round_trip() {
        // a sorted, distinct set of row_col pairs (as CPC produces)
        let pairs: Vec<u32> = (0..200u32).map(|i| (i * 37 + 5) & 0x3fffff).collect();
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        sorted.dedup();
        let num_base_bits = golomb_choose_number_of_base_bits(4096, sorted.len() as u64);
        let mut buf =
            vec![
                0u32;
                safe_length_for_compressed_pair_buf(4096, sorted.len() as u32, num_base_bits)
            ];
        let words = low_level_compress_pairs(&sorted, num_base_bits, &mut buf);
        let mut out = vec![0u32; sorted.len()];
        low_level_uncompress_pairs(
            &mut out,
            sorted.len() as u32,
            num_base_bits,
            &buf[..words],
            words,
        );
        assert_eq!(out, sorted);
    }

    #[test]
    fn bytes_round_trip() {
        let bytes: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        // The byte coders are parameterised by matching encode/decode tables for a
        // given pseudo-phase; phase 0 of the high-entropy byte tables is used here.
        let phase = 0usize;
        let encoding_table = &ENCODING_TABLES_FOR_HIGH_ENTROPY_BYTE[phase];
        let decoding_table = &high_entropy_decoding_tables()[phase];
        let mut buf = vec![0u32; safe_length_for_compressed_window_buf(4096)];
        let words = low_level_compress_bytes(&bytes, bytes.len() as u32, encoding_table, &mut buf);
        let mut out = vec![0u8; bytes.len()];
        low_level_uncompress_bytes(
            &mut out,
            bytes.len() as u32,
            &buf[..words],
            words,
            decoding_table,
        );
        assert_eq!(out, bytes);
    }
}
