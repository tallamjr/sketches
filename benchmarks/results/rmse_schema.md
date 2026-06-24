# RMSE results schema

This schema describes the CSV emitted by the multi-trial RMSE mode of the
benchmark runners (selected with `--trials <T>`). It is separate from, and
additive to, the single-run schema in `schema.md`: each format keeps its own
header and column set, and the two are never mixed in one file.

The RMSE mode runs `T` independent trials per sketch. Trial `t` inserts the
disjoint integer range `[t*n, (t+1)*n)` so that no two trials share keys, then
records the relative error of the sketch's cardinality estimate against the
known true count `n`. One CSV row summarises all `T` trials for a single
`(implementation, sketch)` pair.

## Header

```text
implementation,sketch,lg_k,trials,n_per_trial,rmse,mean_rel_error,max_rel_error
```

## Columns

| Column          | Type   | Unit       | Meaning                                                                 |
| --------------- | ------ | ---------- | ---------------------------------------------------------------------- |
| `implementation`| string | -          | Source of the sketch (`ours` for `runner-ours`).                        |
| `sketch`        | string | -          | Distinct-count sketch under test (`theta`, `hll`, or `cpc`).            |
| `lg_k`          | int    | -          | Size parameter: HLL/CPC `lg_k` (12) or the Theta `k = 2^lg_k` (4096).   |
| `trials`        | int    | count      | Number of independent trials `T` summarised by this row.                |
| `n_per_trial`   | int    | count      | Distinct items inserted per trial; also the true cardinality per trial. |
| `rmse`          | float  | fraction   | Root mean square of the per-trial relative errors (see below).          |
| `mean_rel_error`| float  | fraction   | Arithmetic mean of the per-trial relative errors.                       |
| `max_rel_error` | float  | fraction   | Largest per-trial relative error observed.                              |

## Definitions

For trial `t`, the relative error is:

```text
rel_error[t] = |estimate[t] - n| / n
```

where `n` is `n_per_trial` (the true distinct count for that trial). The three
summary statistics over the `T` per-trial errors are:

```text
rmse           = sqrt( mean( rel_error[t]^2 ) )
mean_rel_error = mean( rel_error[t] )
max_rel_error  = max( rel_error[t] )
```

All three error statistics are dimensionless fractions (for example `0.015`
means a 1.5 % relative error). RMSE is the headline accuracy metric because it
penalises large errors more heavily and is the statistically meaningful measure
for comparing implementations across many trials. For a sketch configured at
`k = 4096`, the theoretical relative error floor is approximately
`1 / sqrt(4096) = 1.56 %`.
