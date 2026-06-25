# Benchmarks

A faithful, multi-plane benchmark harness comparing this crate's sketches
against the Apache DataSketches implementations. Each runner is a standalone
project (a cargo crate, a cmake build, or a Python driver) that emits the same
fixed CSV schema, and the Python reporter joins, tabulates, plots, and gates
the results.

The guiding principle is that a comparison is only meaningful when the two
sides differ in exactly the dimension under test. The harness is built around
that idea: it isolates the hash, the implementation, and the language runtime
into separate planes so no single number conflates them.

## Comparison planes

Throughput and memory are measured across four Rust-and-C++ planes plus a
Python plane. Each row's `implementation` column labels its plane.

- `ours` -- this crate built with its default xxh3 hasher. The headline
  implementation.
- `ours-murmur3` -- this crate with the hasher swapped to MurmurHash3, the hash
  Apache uses. Same data structures and code paths as `ours`, only the hash
  differs. This plane exists to isolate the hash effect from implementation
  speed: comparing `ours` to `ours-murmur3` shows what the hash choice alone
  costs, and comparing `ours-murmur3` to `apache-rust` then compares
  implementations on an equal-hash footing.
- `apache-rust` -- the Apache DataSketches Rust port (the reference impl vendored
  under `lib/datasketches-rust/`).
- `apache-cpp` -- the Apache DataSketches C++ library, built and driven by
  `runner-cpp`.

The Python plane is a separate language-runtime comparison:

- `python-ours` -- our maturin wheel (the PyO3 bindings over this crate).
- `python-apache` -- the pip `datasketches` package (Apache's official Python
  bindings over their C++ library).

This isolates the cost of the Python interop layer on each side, rather than
the underlying sketch implementations.

### Measured observation: the hash carries a per-call cost

Swapping only the hasher (xxh3 to MurmurHash3) while holding everything else
constant changed HLL build throughput substantially, from roughly 34M ops/s on
the xxh3 path to roughly 160M ops/s on the murmur3 path. Because nothing else
changed between `ours` and `ours-murmur3`, this points to the xxh3 path
carrying a per-call cost on the hash-bound sketches. This is a measured
observation to investigate, not a settled SOTA claim: it is recorded here so
the next person knows where to look, and is exactly why the `ours-murmur3` plane
exists.

## Coverage

Six sketch families are benchmarked: HLL, Theta, CPC, Bloom, CountMin, and KLL.

The five distinct-counter and filter families (HLL, Theta, CPC, Bloom,
CountMin) are hash-based, so each gets a `ours-murmur3` row in addition to its
`ours` row. KLL is comparison-based (it has no hasher), so it has no
`ours-murmur3` row.

## Timing protocol

Throughput is measured with a fixed protocol shared by every runner:

- One untimed warmup pass runs the build/update loop first, to settle caches
  and absorb any one-off allocation.
- Then `R` timed repetitions run, default `R = 30`, each individually timed.
- Per-rep throughput is `ops_per_rep / elapsed_secs`.
- The reported `throughput_median_ops_per_s` is the median of the per-rep
  rates, and `throughput_stddev` is their population standard deviation.

The median is preferred over the mean so an occasional slow rep (scheduler
preemption, GC, page fault) does not dominate the figure. The throughput plots
carry the `throughput_stddev` as error bars, so run-to-run variance is visible
rather than hidden.

## Memory: `live_bytes` versus `bytes`

The schema carries two distinct memory columns, and they mean different things:

- `bytes` is the serialised size of the sketch: the length of the byte buffer
  that serialisation produces.
- `live_bytes` is the per-sketch live heap delta: the resident memory
  attributable to the sketch object while it is in use. This is the in-memory
  working footprint, which is generally larger than and unrelated to the
  serialised size.

Each plane measures `live_bytes` with the mechanism native to its runtime:

- Rust (`ours`, `ours-murmur3`) installs a counting `GlobalAlloc` that tracks
  cumulative allocated-minus-freed bytes; `live_bytes` is the delta across the
  sketch's construction.
- C++ (`apache-cpp`) overrides global `operator new`/`operator delete` and sizes
  each block via `malloc_size` (macOS) or `malloc_usable_size` (glibc).
- Python (`python-ours`, `python-apache`) uses `tracemalloc` to read traced
  heap before and after building the sketch.

## What is gated and what is tracked

- Accuracy is hard-gated. `make gate` runs our runner and fails the build if any
  sketch's relative error exceeds its per-sketch threshold in
  `reporter/thresholds.json`. This is the metric that must not regress.
- Throughput and memory are tracked, not gated. They are recorded, plotted, and
  compared against committed baselines, but they do not fail the build. They
  inform tuning (see the hash observation above); they are not a pass/fail bar.

## Running

All targets run from the repo root via `make -C benchmarks <target>`. Two knobs
apply to most targets: `N` (stream size, default 1000000) and `REPS` (timed
reps, default 30).

Run the planes and produce the report and plots:

```
make -C benchmarks ours apache-rust cpp python report N=<n> REPS=<r>
```

Run the accuracy gate (runs `ours`, then checks against the thresholds):

```
make -C benchmarks gate
```

Run the multi-trial RMSE accuracy comparison (does not use the warmup+reps
protocol; controlled by `RMSE_TRIALS` and `RMSE_N`):

```
make -C benchmarks rmse
```

### Python plane prerequisites

The `python` target needs both sides installed into the repo `.venv`:

- Build our wheel with maturin:
  `.venv/bin/maturin develop --release --features extension-module`.
- Install Apache's bindings: `.venv/bin/pip install datasketches`.

macOS strip caveat: `pyproject.toml` sets `strip = true`, which strips the
`PyInit` symbol from the release wheel and leaves an unimportable module. When
building the release wheel on macOS, disable stripping for that build:

```
CARGO_PROFILE_RELEASE_STRIP=false .venv/bin/maturin develop --release --features extension-module
```

## Outputs

- Result CSVs are written to `results/*.csv`, one per plane.
- Committed baselines live alongside them as `results/*.baseline.csv`.
- Committed plots live in `assets/benchmarks/{throughput,memory,accuracy}.png`.
