//! MurmurHash3 x64-128, ported from the canonical Apache implementation.
//! Used to isolate the hash effect from implementation speed in benchmarks.

use crate::hash::SketchHasher;

#[derive(Clone, Default, Debug)]
pub struct Murmur3Hasher;

#[inline]
fn rotl64(x: u64, r: i8) -> u64 {
    x.rotate_left(r as u32)
}

#[inline]
fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    k
}

fn murmur3_x64_128(data: &[u8], seed: u64) -> u128 {
    let c1: u64 = 0x87c37b91114253d5;
    let c2: u64 = 0x4cf5ad432745937f;
    let mut h1 = seed;
    let mut h2 = seed;
    let nblocks = data.len() / 16;

    for i in 0..nblocks {
        let base = i * 16;
        let mut k1 = u64::from_le_bytes(data[base..base + 8].try_into().unwrap());
        let mut k2 = u64::from_le_bytes(data[base + 8..base + 16].try_into().unwrap());

        k1 = k1.wrapping_mul(c1);
        k1 = rotl64(k1, 31);
        k1 = k1.wrapping_mul(c2);
        h1 ^= k1;
        h1 = rotl64(h1, 27);
        h1 = h1.wrapping_add(h2);
        h1 = h1.wrapping_mul(5).wrapping_add(0x52dce729);

        k2 = k2.wrapping_mul(c2);
        k2 = rotl64(k2, 33);
        k2 = k2.wrapping_mul(c1);
        h2 ^= k2;
        h2 = rotl64(h2, 31);
        h2 = h2.wrapping_add(h1);
        h2 = h2.wrapping_mul(5).wrapping_add(0x38495ab5);
    }

    let tail = &data[nblocks * 16..];
    let mut k1: u64 = 0;
    let mut k2: u64 = 0;
    let len = data.len();
    if tail.len() >= 15 {
        k2 ^= (tail[14] as u64) << 48;
    }
    if tail.len() >= 14 {
        k2 ^= (tail[13] as u64) << 40;
    }
    if tail.len() >= 13 {
        k2 ^= (tail[12] as u64) << 32;
    }
    if tail.len() >= 12 {
        k2 ^= (tail[11] as u64) << 24;
    }
    if tail.len() >= 11 {
        k2 ^= (tail[10] as u64) << 16;
    }
    if tail.len() >= 10 {
        k2 ^= (tail[9] as u64) << 8;
    }
    if tail.len() >= 9 {
        k2 ^= tail[8] as u64;
        k2 = k2.wrapping_mul(c2);
        k2 = rotl64(k2, 33);
        k2 = k2.wrapping_mul(c1);
        h2 ^= k2;
    }
    if tail.len() >= 8 {
        k1 ^= (tail[7] as u64) << 56;
    }
    if tail.len() >= 7 {
        k1 ^= (tail[6] as u64) << 48;
    }
    if tail.len() >= 6 {
        k1 ^= (tail[5] as u64) << 40;
    }
    if tail.len() >= 5 {
        k1 ^= (tail[4] as u64) << 32;
    }
    if tail.len() >= 4 {
        k1 ^= (tail[3] as u64) << 24;
    }
    if tail.len() >= 3 {
        k1 ^= (tail[2] as u64) << 16;
    }
    if tail.len() >= 2 {
        k1 ^= (tail[1] as u64) << 8;
    }
    if !tail.is_empty() {
        k1 ^= tail[0] as u64;
        k1 = k1.wrapping_mul(c1);
        k1 = rotl64(k1, 31);
        k1 = k1.wrapping_mul(c2);
        h1 ^= k1;
    }

    h1 ^= len as u64;
    h2 ^= len as u64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);

    ((h2 as u128) << 64) | (h1 as u128)
}

impl SketchHasher for Murmur3Hasher {
    fn hash64(&self, data: &[u8], seed: u64) -> u64 {
        murmur3_x64_128(data, seed) as u64
    }
    fn hash128(&self, data: &[u8], seed: u64) -> u128 {
        murmur3_x64_128(data, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::SketchHasher;

    // Known MurmurHash3_x64_128 vectors (seed = 0).
    // Empty input with seed 0 => 0x0000000000000000_0000000000000000.
    #[test]
    fn empty_input_seed_zero_is_zero() {
        let h = Murmur3Hasher;
        assert_eq!(h.hash128(b"", 0), 0u128);
    }

    // Reference 128-bit digest for b"test", seed 0, as produced by the
    // canonical Apache C++ MurmurHash3_x64_128.
    // Cross-checked against the canonical Apache C++ MurmurHash3_x64_128
    // (lib/datasketches-cpp/common/include/MurmurHash3.h): for b"test" seed 0
    // the C++ reference yields h1=0xac7d28cc74bde19d, h2=0x9a128231f9bd4d82,
    // packed as (h2 << 64) | h1.
    const EXPECTED_TEST_SEED0: u128 = 0x9a128231f9bd4d82_ac7d28cc74bde19d;
    #[test]
    fn known_vector_test_seed_zero() {
        let h = Murmur3Hasher;
        assert_eq!(h.hash128(b"test", 0), EXPECTED_TEST_SEED0);
    }

    #[test]
    fn hash64_is_low_64_of_hash128() {
        let h = Murmur3Hasher;
        let d = h.hash128(b"hello world", 9001);
        assert_eq!(h.hash64(b"hello world", 9001), d as u64);
    }
}
