// crates/cpc/src/lib.rs
use std::hash::Hash;
use hll::HllSketch;

/// CPC Sketch for approximate distinct counting, delegated to HyperLogLog.
pub struct CpcSketch {
    inner: HllSketch,
}

impl CpcSketch {
    /// Create a new CPC sketch with log2(k) specified by lg_k.
    pub fn new(lg_k: u8) -> Self {
        CpcSketch { inner: HllSketch::new(lg_k) }
    }

    /// Update the sketch with an item implementing Hash.
    pub fn update<T: Hash>(&mut self, item: &T) {
        self.inner.update(item);
    }

    /// Estimate the cardinality.
    pub fn estimate(&self) -> f64 {
        self.inner.estimate()
    }

    /// Merge another CPC sketch into this one (in-place union).
    pub fn merge(&mut self, other: &CpcSketch) {
        self.inner.merge(&other.inner);
    }

    /// Serialize the sketch to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes()
    }
}
