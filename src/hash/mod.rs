//! Pluggable hashing for sketches. xxh3 is the default; the trait allows
//! alternatives without leaking generics to ordinary callers.

pub mod xxh3;

/// Default seed. Arbitrary fixed value for deterministic hashing across runs.
pub const DEFAULT_SEED: u64 = 9001;

/// A hash backend producing 64- and 128-bit digests over byte slices.
pub trait SketchHasher: Clone + Default {
    fn hash64(&self, data: &[u8], seed: u64) -> u64;
    fn hash128(&self, data: &[u8], seed: u64) -> u128;
}

/// Types that can be fed to a sketch hasher as a canonical byte sequence.
/// `with_bytes` avoids allocation by handing a slice to a callback.
pub trait Hashable {
    fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R;
}

impl Hashable for [u8] {
    fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self)
    }
}
impl Hashable for str {
    fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self.as_bytes())
    }
}
impl Hashable for String {
    fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self.as_bytes())
    }
}
impl Hashable for &str {
    fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        f(self.as_bytes())
    }
}

macro_rules! hashable_int {
    ($($t:ty),*) => { $(
        impl Hashable for $t {
            fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R { f(&self.to_le_bytes()) }
        }
    )* };
}
hashable_int!(u32, u64, i32, i64, u128, i128);

impl Hashable for f64 {
    fn with_bytes<R>(&self, f: impl FnOnce(&[u8]) -> R) -> R {
        // canonicalise NaN so all NaNs hash identically
        let bits = if self.is_nan() { 0u64 } else { self.to_bits() };
        f(&bits.to_le_bytes())
    }
}

pub fn hash64_of<H: SketchHasher, K: Hashable + ?Sized>(h: &H, key: &K, seed: u64) -> u64 {
    key.with_bytes(|b| h.hash64(b, seed))
}
pub fn hash128_of<H: SketchHasher, K: Hashable + ?Sized>(h: &H, key: &K, seed: u64) -> u128 {
    key.with_bytes(|b| h.hash128(b, seed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::xxh3::Xxh3Hasher;

    #[test]
    fn hashable_routes_primitives_and_strings() {
        let h = Xxh3Hasher::default();
        // deterministic
        assert_eq!(
            hash64_of(&h, &"hello", DEFAULT_SEED),
            hash64_of(&h, &"hello", DEFAULT_SEED)
        );
        // distinct inputs differ
        assert_ne!(
            hash64_of(&h, &"a", DEFAULT_SEED),
            hash64_of(&h, &"b", DEFAULT_SEED)
        );
        // 128-bit available and stable
        assert_eq!(hash128_of(&h, &42u64, 0), hash128_of(&h, &42u64, 0));
        // seed changes output
        assert_ne!(hash64_of(&h, &42u64, 0), hash64_of(&h, &42u64, 1));
    }
}
