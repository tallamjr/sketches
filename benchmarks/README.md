# Benchmarks

A multi-plane benchmark harness comparing this crate's sketches
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

- `ours`: this crate built with its default xxh3 hasher. The headline
  implementation.
- `ours-murmur3`: this crate with the hasher swapped to MurmurHash3, the hash
  Apache uses. Same data structures and code paths as `ours`, only the hash
  differs. This plane exists to isolate the hash effect from implementation
  speed: comparing `ours` to `ours-murmur3` shows what the hash choice alone
  costs, and comparing `ours-murmur3` to `apache-rust` then compares
  implementations on an equal-hash footing.
- `apache-rust`: the Apache DataSketches Rust port (the reference impl vendored
  under `lib/datasketches-rust/`).
- `apache-cpp`: the Apache DataSketches C++ library, built and driven by
  `runner-cpp`.

The Python plane is a separate language-runtime comparison:

- `python-ours`: our maturin wheel (the PyO3 bindings over this crate).
- `python-apache`: the pip `datasketches` package (Apache's official Python
  bindings over their C++ library).

This isolates the cost of the Python interop layer on each side, rather than
the underlying sketch implementations.

### Measured observation: the hash carries a per-call cost

Swapping only the hasher (xxh3 to MurmurHash3) while holding everything else
constant changed HLL build throughput substantially, from roughly 34M ops/s on
the xxh3 path to roughly 160M ops/s on the murmur3 path. Because nothing else
changed between `ours` and `ours-murmur3`, this points to the xxh3 path
carrying a per-call cost on the hash-bound sketches. It is recorded here as a
starting point for investigation so the next person knows where to look, and is
why the `ours-murmur3` plane exists. On its own it does not support a SOTA speed
claim.

## Coverage

Six sketch families are benchmarked: HLL, Theta, CPC, Bloom, CountMin, and KLL.

The five distinct-counter and filter families (HLL, Theta, CPC, Bloom,
CountMin) are hash-based, so each gets a `ours-murmur3` row in addition to its
`ours` row. KLL is comparison-based (it has no hasher), so it has no
`ours-murmur3` row.

## Timing protocol

Throughput is measured with a fixed protocol shared by every runner. The unit
of measurement is the round, and `REPS` now controls the number of independent
rounds (default `REPS = 30`):

- Each round runs one untimed warmup pass of the build/update loop (to settle
  caches and absorb one-off allocation), then `REPS_PER_ROUND = 5` timed reps.
- Per-rep throughput is `ops_per_rep / elapsed_secs`; the round's sample is the
  median of its five timed reps.
- This yields `REPS` round-samples. The reported
  `throughput_median_ops_per_s` is the median of those round-samples, and
  `throughput_stddev` is their population standard deviation.
- `throughput_ci_low` and `throughput_ci_high` bound a deterministic 95%
  bootstrap confidence interval over the round-samples: the bootstrap is
  resampled with a fixed seed, so the same inputs always produce the same
  interval and the CI is reproducible across runs.

The median is preferred over the mean so an occasional slow round (scheduler
preemption, GC, page fault) does not dominate the figure. The throughput plots
carry the `throughput_stddev` as error bars, so run-to-run variance is visible
in the plot, and the bootstrap CI is what the separation verdict and the noise
warning (below) are computed from.

### Quiet-machine checklist

The CI is only as trustworthy as the machine it was measured on. Before taking
a measurement seriously, settle the host:

- Run on AC power, not battery (CPU governors throttle aggressively on battery).
- Close other applications, especially browsers, compilers, and anything doing
  background indexing or sync.
- Run a single benchmark at a time; do not run two planes (or a build) in
  parallel with a timing run.
- Let the machine reach a steady thermal state; avoid measuring immediately
  after a heavy build.

### macOS pinning limitation

macOS exposes no `sched_setaffinity` and no `taskset`, so the runners cannot
pin a benchmark to a fixed core. Without core pinning the OS may migrate the
timing thread between cores mid-run, which adds variance. The harness
compensates by measuring many rounds, reporting the bootstrap CI, and emitting
a noise warning: when a measurement's 95% CI half-width exceeds 5% of its
median, the reporter prints a `NOISY .../...: re-run` line to stderr. Treat any
flagged measurement as untrustworthy and re-run it on a quiet machine before
drawing a conclusion from it.

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
  `reporter/thresholds.json`. The gate covers the sketches listed there (hll,
  theta, kll, countmin, cpc); Bloom is a membership filter with no cardinality
  estimate, so it is not accuracy-gated. This is the metric that must not regress.
- Throughput and memory are tracked, not gated. They are recorded, plotted, and
  compared against committed baselines, but they do not fail the build. They
  inform tuning (see the hash observation above); they are not a pass/fail bar.

## Running

All targets run from the repo root via `make -C benchmarks <target>`. Two knobs
apply to most targets: `N` (stream size, default 1000000) and `REPS`
(independent timing rounds, default 30; see the timing protocol below).

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

### Baseline and compare gate

To check whether a change moved throughput beyond measurement noise, snapshot a
baseline before the change and compare against it after:

```
make -C benchmarks baseline   # run ours, snapshot results/ours.csv -> results/baseline-ours.csv
# ... make your change ...
make -C benchmarks compare     # re-run ours, print the per-plane separation verdict vs the baseline
```

`compare` runs `ours` again, then prints a per-plane table giving the
baseline-to-candidate throughput ratio and a separation verdict for each plane.
The verdict reads `separated` only when the two 95% bootstrap CIs are disjoint,
so the change cleared measurement noise; otherwise it reads `within noise` and
the change must not be claimed as a real speedup. `compare` errors if no
baseline has been snapshotted yet (run `make baseline` first). Before trusting
either run, follow the quiet-machine checklist above and re-run any measurement
the reporter flags `NOISY`.

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
