//! xxh3 implementation of `SketchHasher` (the crate default).

use super::SketchHasher;
use xxhash_rust::xxh3::{xxh3_64_with_seed, xxh3_128_with_seed};

#[derive(Clone, Copy, Default)]
pub struct Xxh3Hasher;

impl SketchHasher for Xxh3Hasher {
    #[inline]
    fn hash64(&self, data: &[u8], seed: u64) -> u64 {
        xxh3_64_with_seed(data, seed)
    }
    #[inline]
    fn hash128(&self, data: &[u8], seed: u64) -> u128 {
        xxh3_128_with_seed(data, seed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::DEFAULT_SEED;

    #[test]
    fn avalanche_smoke() {
        // sequential keys should not collide in 64 bits over a small range
        let h = Xxh3Hasher::default();
        let mut seen = std::collections::HashSet::new();
        for i in 0u64..10_000 {
            assert!(seen.insert(h.hash64(&i.to_le_bytes(), DEFAULT_SEED)));
        }
    }
}
