# Changelog

All notable changes to this project are documented here. The format is based on
Keep a Changelog, and this project adheres to Semantic Versioning.

## [Unreleased]

### Changed

- Hashing is now a single canonical, pluggable `SketchHasher` defaulting to xxh3 (64- and 128-bit), used unconditionally in every build. The previous `optimized`-gated `fast_hash`/`DefaultHasher` path was removed. `DefaultHasher` no longer appears anywhere in the crate.
- CPC was rewritten as a port of the Apache algorithm (ICON and HIP estimators, kxp lookup, flavour machinery, union), fixing a 173% estimation error to Apache parity (0.34% on synthetic, 1.17% on a real TPC-H column).
- HLL gained a HIP (Historical Inverse Probability) estimator with a composite fallback for out-of-order (merged) sketches and HIP state persisted through serialisation; multi-trial RMSE improved from 0.0175 to 0.0122 (better than Apache's 0.0129, below the 1/sqrt(k) floor).
- Serialisation is uniform across every sketch via a `Serializable` trait: a shared little-endian codec with a versioned header for the hot path (HLL, Theta, CPC) and postcard for the long tail.
- Documentation rewritten with multi-trial-RMSE-measured accuracy numbers and Tahoma-styled comparison plots; fabricated performance claims removed.

### Added

- A three-way benchmark harness under `benchmarks/` (ours vs Apache Rust vs Apache C++) emitting a shared CSV schema over identical datasets, a Python reporter, a multi-trial RMSE comparison (`make -C benchmarks rmse`), and a CI accuracy gate.
- Accuracy is validated only by multi-trial RMSE; single-run comparisons are treated as statistically meaningless.

### Removed

- The entire `optimized` cargo feature and its modules (`simd_ops`, `compact_memory`), plus the `jemalloc` and `rayon` dependencies: the crate is now pure scalar Rust.
- The inert `use_simd`/`uses_simd` knob (a runtime no-op after the SIMD strip) from the CountMin, Linear, and Bloom APIs and the PyO3 surface.
- Stale optimization-era example files and the unused `AodConfig::seed` field.

### Fixed

- Count-Min now uses per-row independent hash seeds, restoring the pairwise independence its error bound assumes.
- The Python wheel builds cleanly via maturin (validated), with the project version sourced dynamically from Cargo.

### Deferred

- CPC compressed serialisation (ours serialises larger than Apache); a size-only optimisation tracked as a follow-up, since accuracy is already at parity.
