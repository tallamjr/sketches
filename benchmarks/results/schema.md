# Benchmark results schema

All three benchmark runners (ours, Apache-Rust, Apache-C++) emit results as CSV
rows sharing one fixed schema so that throughput and accuracy are directly
comparable across implementations.

## CSV header (exact)

```
implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,throughput_stddev,bytes,live_bytes,estimate,exact,rel_error
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
| `reps`                 | integer | -                | Number of timed repetitions used to compute the throughput statistics (excludes the untimed warmup pass). |
| `throughput_median_ops_per_s` | float | operations / second | Median of the per-rep throughput (operations completed per wall-clock second) across the `reps` timed repetitions. |
| `throughput_stddev`    | float   | operations / second | Population standard deviation of the per-rep throughput across the `reps` timed repetitions. |
| `bytes`                | integer | bytes            | Serialised size of the sketch for this measurement (length of the byte buffer produced by serialisation). |
| `live_bytes`           | integer | bytes            | Per-sketch live heap delta (resident memory attributable to the sketch). Populated in a later phase; emitted empty until then. |
| `estimate`             | float   | items            | The sketch's estimated cardinality (or other estimated quantity for the op). |
| `exact`                | float   | items            | The ground-truth exact value the estimate is compared against. |
| `rel_error`            | float   | dimensionless    | Relative error of the estimate against the exact value: `(estimate - exact) / exact`. |

## Timing protocol

Throughput is measured with a fixed protocol shared by all runners: each
sketch's build/update loop is run once as an untimed warmup pass (to settle
caches and any one-off allocation), then run `reps` times with each repetition
individually timed. The per-rep throughput is `ops_per_rep / elapsed_secs`. The
reported `throughput_median_ops_per_s` is the median of those per-rep rates and
`throughput_stddev` is their population standard deviation. The median is used
in preference to the mean so that an occasional slow rep (scheduler preemption,
GC, page fault) does not dominate the reported figure.

`bytes` is the serialised size of the sketch (the length of its byte buffer).
`live_bytes` is the per-sketch live heap delta: the resident memory attributable
to the sketch itself, which is measured and populated in a later phase and is
emitted empty until then.

## Empty fields

The `exact` and `rel_error` columns may be empty for operations where a ground
truth is not applicable (for example a pure `serialize` or `merge` throughput
measurement that produces no cardinality estimate). When `exact` is empty,
`rel_error` is also empty. The `live_bytes` column is emitted empty until the
later phase that wires real per-sketch heap measurement. All other columns are
always populated.
