// crates/cpc/src/lib.rs
use std::hash::{Hash, Hasher};
pub struct CpcSketch {/* … */}
impl CpcSketch {
    pub fn new(lg_k: u8) -> Self { /* … */
    }
    pub fn update<T: Hash>(&mut self, item: &T) { /* … */
    }
    pub fn estimate(&self) -> f64 { /* … */
    }
    pub fn merge(&mut self, other: &CpcSketch) { /* … */
    }
    pub fn to_bytes(&self) -> Vec<u8> { /* … */
    }
}
