//! Counting global allocator: tracks live bytes (allocated minus freed) so the
//! runner can attribute a per-sketch heap delta. Saturating; never panics.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct CountingAlloc;
static LIVE: AtomicUsize = AtomicUsize::new(0);

/// Process-wide global allocator. Declared in the library so that both the
/// `runner-ours` binary and this crate's own test build install the counting
/// allocator. (Declaring it only in `main.rs` would leave `cargo test --lib`
/// running on the default system allocator, so `live_bytes` would stay zero.)
#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            LIVE.fetch_add(layout.size(), Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        // Saturating subtract: clamp to the current counter so a free can never
        // underflow LIVE. A panic in the allocator hot path is undefined
        // behaviour, so saturating is the correct, infallible bookkeeping choice.
        LIVE.fetch_sub(
            layout.size().min(LIVE.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
    }
}

/// Current live heap bytes (cumulative allocated minus freed).
pub fn live_bytes() -> usize {
    LIVE.load(Ordering::Relaxed)
}

/// Build a value and return it with the net heap bytes it added.
pub fn measure_live<T>(build: impl FnOnce() -> T) -> (T, usize) {
    let before = live_bytes();
    let value = build();
    let after = live_bytes();
    (value, after.saturating_sub(before))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delta_is_nonzero_and_monotonic() {
        let before = live_bytes();
        let v: Vec<u64> = (0..10_000).collect();
        let after = live_bytes();
        assert!(after > before, "alloc not counted");
        core::hint::black_box(&v);
        let (built, delta) = measure_live(|| (0u64..1000).collect::<Vec<_>>());
        assert!(delta >= 1000 * std::mem::size_of::<u64>());
        core::hint::black_box(built);
    }
}
