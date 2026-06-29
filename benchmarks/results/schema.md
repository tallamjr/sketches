# Benchmark results schema

All three benchmark runners (ours, Apache-Rust, Apache-C++) emit results as CSV
rows sharing one fixed schema so that throughput and accuracy are directly
comparable across implementations.

## CSV header (exact)

```
implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,throughput_stddev,throughput_ci_low,throughput_ci_high,bytes,live_bytes,estimate,exact,rel_error
```

## Row semantics

Each row records exactly one `(implementation, sketch, dataset, op)`
measurement. The `implementation` column labels the comparison plane: it is one
of `ours`, `apache-rust`, or `apache-cpp`. Rows with the same `sketch`,
`dataset`, and `op` but different `implementation` values are the directly
comparable points.

## Columns

| Column                 | Type    | Units            | Meaning |
|------------------------|---------|------------------|---------|
| `implementation`       | string  | -                | Which library produced the row: `ours`, `apache-rust`, or `apache-cpp`. Identifies the comparison plane. |
| `sketch`               | string  | -                | The sketch under test (for example `hll`, `theta`, `cpc`). |
| `dataset`              | string  | -                | The data source the stream was drawn from (for example `synthetic`, `customer`, `lineitem`). |
| `op`                   | string  | -                | The operation measured (for example `build`, `update`, `estimate`, `merge`, `serialize`). |
| `n`                    | integer | items            | Number of input items processed for this measurement. |
| `reps`                 | integer | -                | Number of independent rounds used to compute the throughput statistics. Each round runs one untimed warmup pass then a fixed number of timed reps (the round-sample is the median of those reps). |
| `throughput_median_ops_per_s` | float | operations / second | Median over the round-samples (each round-sample is the median of that round's per-rep throughput in operations completed per wall-clock second). |
| `throughput_stddev`    | float   | operations / second | Population standard deviation of the round-samples. |
| `throughput_ci_low`    | float   | operations / second | Lower bound of the 95% nonparametric bootstrap confidence interval of the median over the round-samples. Deterministic (fixed-seed resampling). |
| `throughput_ci_high`   | float   | operations / second | Upper bound of the 95% nonparametric bootstrap confidence interval of the median over the round-samples. Deterministic (fixed-seed resampling). |
| `bytes`                | integer | bytes            | Serialised size of the sketch for this measurement (length of the byte buffer produced by serialisation). |
| `live_bytes`           | integer | bytes            | Per-sketch live heap delta: allocations minus frees while building and holding a populated sketch. Distinct from `bytes` (serialised size). Populated on all planes. |
| `estimate`             | float   | items            | The sketch's estimated cardinality (or other estimated quantity for the op). |
| `exact`                | float   | items            | The ground-truth exact value the estimate is compared against. |
| `rel_error`            | float   | dimensionless    | Relative error of the estimate against the exact value: `(estimate - exact) / exact`. |

## Timing protocol

Throughput is measured with a fixed protocol shared by all runners, structured
as independent rounds to capture run-to-run drift rather than only intra-loop
jitter. Each round runs the sketch's build/update loop once as an untimed warmup
pass (to settle caches and any one-off allocation), then runs it a fixed number
of timed reps; the per-rep throughput is `ops_per_rep / elapsed_secs` and the
round-sample is the median of those reps. The `reps` column records the number
of rounds. The reported `throughput_median_ops_per_s` is the median over the
round-samples, `throughput_stddev` is their population standard deviation, and
`throughput_ci_low` / `throughput_ci_high` bound a 95% nonparametric bootstrap
confidence interval of that median (deterministic fixed-seed resampling, so the
interval is reproducible across runs). The median is used in preference to the
mean so that an occasional slow rep (scheduler preemption, GC, page fault) does
not dominate the reported figure. Two measurements whose intervals are disjoint
differ beyond noise; overlapping intervals are within noise. The bootstrap
resampling is identical across the Rust, C++, and Python runners (the same
SplitMix64 seed and resample count), so a given set of round-samples yields the
same interval on every plane.

`bytes` is the serialised size of the sketch (the length of its byte buffer).
`live_bytes` is the per-sketch live heap delta: allocations minus frees while
building and holding a populated sketch. It is the in-memory working footprint,
distinct from `bytes`, and is populated on all planes.

## Empty fields

The `exact` and `rel_error` columns may be empty for operations where a ground
truth is not applicable (for example a pure `serialize` or `merge` throughput
measurement that produces no cardinality estimate). When `exact` is empty,
`rel_error` is also empty. All other columns are always populated.
