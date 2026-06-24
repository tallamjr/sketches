# Benchmark results schema

All three benchmark runners (ours, Apache-Rust, Apache-C++) emit results as CSV
rows sharing one fixed schema so that throughput and accuracy are directly
comparable across implementations.

## CSV header (exact)

```
implementation,sketch,dataset,op,n,throughput_ops_per_s,bytes,estimate,exact,rel_error
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
| `throughput_ops_per_s` | float   | operations / second | Measured throughput: operations completed per wall-clock second. |
| `bytes`                | integer | bytes            | Serialised size of the sketch for this measurement. |
| `estimate`             | float   | items            | The sketch's estimated cardinality (or other estimated quantity for the op). |
| `exact`                | float   | items            | The ground-truth exact value the estimate is compared against. |
| `rel_error`            | float   | dimensionless    | Relative error of the estimate against the exact value: `(estimate - exact) / exact`. |

## Empty fields

The `exact` and `rel_error` columns may be empty for operations where a ground
truth is not applicable (for example a pure `serialize` or `merge` throughput
measurement that produces no cardinality estimate). When `exact` is empty,
`rel_error` is also empty. All other columns are always populated.
