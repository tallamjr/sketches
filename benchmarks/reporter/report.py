#!/usr/bin/env python3
"""Reporter for the shared benchmark CSV schema.

Ingests one or more result CSVs (emitted by runner-ours, runner-apache-rust,
runner-cpp), all sharing the schema:

    implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,
    throughput_stddev,throughput_ci_low,throughput_ci_high,bytes,live_bytes,
    estimate,exact,rel_error

Two modes:

1. Default (table): group rows by the join key (sketch, dataset, op) and render
   a markdown comparison table across implementations, including ours/apache
   ratios for throughput median (higher is better) and live_bytes (lower is
   better, the real heap delta).

2. --check-accuracy <thresholds.json>: for every `ours` row with a non-empty
   rel_error, gate the absolute relative error against the per-sketch threshold.
   Exits 1 on any breach, 0 otherwise.

Stdlib only.
"""

import argparse
import csv
import json
import math
import os
import sys

HEADER = [
    "implementation",
    "sketch",
    "dataset",
    "op",
    "n",
    "reps",
    "throughput_median_ops_per_s",
    "throughput_stddev",
    "throughput_ci_low",
    "throughput_ci_high",
    "bytes",
    "live_bytes",
    "estimate",
    "exact",
    "rel_error",
]

# Throughput-schema columns coerced to numbers when loaded. The remaining
# columns (rel_error, estimate, exact, the join-key strings) stay as strings so
# that check_accuracy's _as_float handling and the empty-string conventions are
# preserved.
NUMERIC_FIELDS = {
    "reps": int,
    "throughput_median_ops_per_s": float,
    "throughput_stddev": float,
    "throughput_ci_low": float,
    "throughput_ci_high": float,
    "bytes": int,
    "live_bytes": int,
}

IMPLEMENTATIONS = ["ours", "apache-rust", "apache-cpp"]

# Native-plane implementation labels rendered by the comparison table, in the
# preferred left-to-right order. `ours-murmur3` sits next to `ours` so the
# xxh3-vs-murmur3 hash effect reads at a glance against the apache references.
# The Python plane emits its own `ours`/`apache` labels that collide with the
# Rust `ours`; those are excluded from this table (the plots and CSVs carry the
# Python plane) so two different `ours` are never conflated.
NATIVE_IMPL_ORDER = ["ours", "ours-murmur3", "apache-rust", "apache-cpp"]


def load_rows(paths):
    """Load all rows from the given CSV paths into a list of dicts.

    Each returned dict has every column of the schema as a string key. Missing
    optional fields remain empty strings. Raises on a missing file or a header
    that does not match the schema.
    """
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"empty CSV (no header): {path}")
            if reader.fieldnames != HEADER:
                raise ValueError(
                    f"unexpected header in {path}: {reader.fieldnames!r}, "
                    f"expected {HEADER!r}"
                )
            for record in reader:
                row = dict(record)
                for field, cast in NUMERIC_FIELDS.items():
                    value = row.get(field)
                    if value is not None and value.strip() != "":
                        row[field] = cast(value)
                rows.append(row)
    return rows


def _join_key(row):
    return (row["sketch"], row["dataset"], row["op"])


def _as_float(value):
    """Parse a cell as a float, returning None when empty or unparseable.

    Accepts already-coerced numbers (load_rows casts the numeric throughput
    columns) as well as raw strings (rel_error and the RMSE schema stay as
    strings).
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    value = value.strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _ratio_cell(ours, other, higher_is_better):
    """Format a ratio cell `ours/other` and label which side is better.

    Returns an empty marker when either value is missing or `other` is zero.
    """
    if ours is None or other is None or other == 0:
        return "-"
    ratio = ours / other
    if higher_is_better:
        better = "ours" if ratio > 1 else ("tie" if ratio == 1 else "other")
    else:
        better = "ours" if ratio < 1 else ("tie" if ratio == 1 else "other")
    return f"{ratio:.3f} ({better})"


def _ordered_native_impls(rows):
    """Native-plane implementations present in `rows`, in preferred display order.

    Only the native-plane labels (NATIVE_IMPL_ORDER) are considered, so the
    Python plane's colliding `ours`/`apache` labels never enter this table. Every
    native label actually present in the data is returned, in NATIVE_IMPL_ORDER,
    so `ours-murmur3` appears alongside `ours` whenever the runner emitted it.
    """
    present = {row["implementation"] for row in rows}
    return [impl for impl in NATIVE_IMPL_ORDER if impl in present]


# Short column labels for the native implementations, used as the per-impl
# throughput/memory column headers.
_IMPL_SHORT = {
    "ours": "ours",
    "ours-murmur3": "ours-m3",
    "apache-rust": "a-rust",
    "apache-cpp": "a-cpp",
}


def render_table(rows):
    """Render a markdown comparison table from loaded rows.

    Rows are grouped by the join key (sketch, dataset, op). The set of
    implementation columns is data-driven: every native-plane implementation
    present in the rows (in NATIVE_IMPL_ORDER, so `ours-murmur3` sits next to
    `ours`) gets a throughput-median (with stddev) and a live_bytes column. The
    Python plane's colliding `ours`/`apache` labels are excluded so two different
    `ours` are never conflated. The table also shows `ours`'s rel_error and the
    ours/apache-rust and ours/apache-cpp ratios for throughput (higher is better)
    and live_bytes (lower is better). The `plane` column labels the join key.
    """
    groups = {}
    for row in rows:
        groups.setdefault(_join_key(row), {})[row["implementation"]] = row

    impls = _ordered_native_impls(rows)

    columns = ["plane (sketch/dataset/op)"]
    for impl in impls:
        columns.append(f"{_IMPL_SHORT[impl]} tput")
    for impl in impls:
        columns.append(f"{_IMPL_SHORT[impl]} mem")
    columns.append("ours rel_err")
    columns += [
        "tput ours/a-rust",
        "tput ours/a-cpp",
        "tput ours-m3/a-rust",
        "tput ours-m3/a-cpp",
        "mem ours/a-rust",
        "mem ours/a-cpp",
    ]

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")

    def tput(impl_rows, impl):
        r = impl_rows.get(impl)
        return _as_float(r["throughput_median_ops_per_s"]) if r else None

    def tput_stddev(impl_rows, impl):
        r = impl_rows.get(impl)
        return _as_float(r.get("throughput_stddev")) if r else None

    def live(impl_rows, impl):
        r = impl_rows.get(impl)
        return _as_float(r["live_bytes"]) if r else None

    def fmt(value):
        return f"{value:.3g}" if value is not None else "-"

    def fmt_tput(median, stddev):
        if median is None:
            return "-"
        if stddev is None:
            return f"{median:.3g}"
        return f"{median:.3g} ± {stddev:.3g}"

    for key in sorted(groups):
        impl_rows = groups[key]
        plane = "/".join(key)

        ours_t = tput(impl_rows, "ours")
        oursm3_t = tput(impl_rows, "ours-murmur3")
        arust_t = tput(impl_rows, "apache-rust")
        acpp_t = tput(impl_rows, "apache-cpp")
        ours_m = live(impl_rows, "ours")
        arust_m = live(impl_rows, "apache-rust")
        acpp_m = live(impl_rows, "apache-cpp")

        ours_row = impl_rows.get("ours")
        ours_re = ours_row["rel_error"] if ours_row else ""
        ours_re = ours_re.strip() if isinstance(ours_re, str) else str(ours_re)
        ours_re_disp = ours_re if ours_re != "" else "-"

        cells = [plane]
        for impl in impls:
            cells.append(
                fmt_tput(tput(impl_rows, impl), tput_stddev(impl_rows, impl))
            )
        for impl in impls:
            cells.append(fmt(live(impl_rows, impl)))
        cells.append(ours_re_disp)
        cells += [
            _ratio_cell(ours_t, arust_t, higher_is_better=True),
            _ratio_cell(ours_t, acpp_t, higher_is_better=True),
            _ratio_cell(oursm3_t, arust_t, higher_is_better=True),
            _ratio_cell(oursm3_t, acpp_t, higher_is_better=True),
            _ratio_cell(ours_m, arust_m, higher_is_better=False),
            _ratio_cell(ours_m, acpp_m, higher_is_better=False),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("**Notes**")
    lines.append("")
    lines.append(
        "> Throughput differences are dominated by hash-function choice: this "
        "crate uses xxh3 while Apache (Rust and C++) use MurmurHash3 x64 128, "
        "and HLL/Theta update is hash-bound. Absolute multiples are "
        "machine-dependent; treat them as directional, not precise. The "
        "TPC-H/string comparison passes borrowed string slices to every "
        "implementation so no runner pays an extra per-item allocation."
    )

    return "\n".join(lines) + "\n"


def separation(median_a, lo_a, hi_a, median_b, lo_b, hi_b):
    """Verdict on whether measurement B differs from A beyond noise.

    'separated' iff the two 95% bootstrap CIs are disjoint; otherwise
    'within noise'. ratio is candidate-over-baseline (median_b / median_a).
    """
    disjoint = hi_a < lo_b or hi_b < lo_a
    verdict = "separated" if disjoint else "within noise"
    ratio = median_b / median_a if median_a else float("nan")
    return verdict, ratio


def render_compare(baseline_rows, candidate_rows):
    """Markdown table: per (sketch,dataset,op,implementation), baseline vs
    candidate throughput median, the speedup ratio, and the separation verdict.
    """
    def index(rows):
        out = {}
        for r in rows:
            key = (r["sketch"], r["dataset"], r["op"], r["implementation"])
            out[key] = r
        return out

    base = index(baseline_rows)
    cand = index(candidate_rows)
    columns = ["plane (sketch/dataset/op) [impl]", "baseline tput",
               "candidate tput", "ratio (cand/base)", "verdict"]
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for key in sorted(set(base) & set(cand)):
        b = base[key]
        c = cand[key]
        bm = _as_float(b["throughput_median_ops_per_s"])
        cm = _as_float(c["throughput_median_ops_per_s"])
        verdict, ratio = separation(
            bm, _as_float(b["throughput_ci_low"]), _as_float(b["throughput_ci_high"]),
            cm, _as_float(c["throughput_ci_low"]), _as_float(c["throughput_ci_high"]),
        )
        plane = "/".join(key[:3]) + f" [{key[3]}]"
        lines.append("| " + " | ".join([
            plane, f"{bm:.3g}", f"{cm:.3g}", f"{ratio:.3f}", verdict,
        ]) + " |")
    lines.append("")
    lines.append("**Notes**")
    lines.append("")
    lines.append("> 'separated' means the 95% bootstrap CIs are disjoint, so the "
                 "throughput change clears measurement noise; 'within noise' means "
                 "it does not and must not be claimed as a real change.")
    return "\n".join(lines) + "\n"


def check_accuracy(rows, thresholds):
    """Gate `ours` relative errors against per-sketch thresholds.

    For every row where implementation == 'ours' and rel_error is non-empty,
    compare abs(rel_error) against thresholds[sketch]. A sketch with no
    threshold entry is not gated (recorded as a note in the returned messages).

    Returns (passed, messages) where messages lists failures and ungated notes.
    """
    failures = []
    notes = []
    ungated_seen = set()
    for row in rows:
        if row["implementation"] != "ours":
            continue
        rel = _as_float(row.get("rel_error"))
        if rel is None:
            continue
        sketch = row["sketch"]
        if sketch not in thresholds:
            if sketch not in ungated_seen:
                ungated_seen.add(sketch)
                notes.append(f"note: sketch '{sketch}' has no threshold, not gated")
            continue
        threshold = thresholds[sketch]
        if abs(rel) > threshold:
            failures.append(
                f"FAIL {sketch}/{row['dataset']}/{row['op']}: "
                f"|rel_error|={abs(rel):.4g} > threshold {threshold:.4g}"
            )
    passed = len(failures) == 0
    return passed, failures + notes


RMSE_HEADER = [
    "implementation",
    "sketch",
    "lg_k",
    "trials",
    "n_per_trial",
    "rmse",
    "mean_rel_error",
    "max_rel_error",
]

RMSE_SKETCHES = ["theta", "hll", "cpc"]


def load_rmse_rows(paths):
    """Load all rows from the given RMSE summary CSV paths into a list of dicts.

    Each row carries the RMSE schema columns as string keys. Raises on a
    missing file or a header that does not match the RMSE schema.
    """
    rows = []
    for path in paths:
        with open(path, newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"empty CSV (no header): {path}")
            if reader.fieldnames != RMSE_HEADER:
                raise ValueError(
                    f"unexpected RMSE header in {path}: {reader.fieldnames!r}, "
                    f"expected {RMSE_HEADER!r}"
                )
            for record in reader:
                rows.append(dict(record))
    return rows


def rmse_parity(ours_rmse, ref_rmse, tol=1.25):
    """Return True iff our RMSE is within a factor `tol` of the reference RMSE.

    Parity holds when ours_rmse <= ref_rmse * tol and ours_rmse >= ref_rmse / tol,
    i.e. ours is neither more than `tol` times worse nor more than `tol` times
    better than the reference.
    """
    return ours_rmse <= ref_rmse * tol and ours_rmse >= ref_rmse / tol


def render_rmse_table(rows, k=4096):
    """Render a markdown RMSE comparison table from loaded RMSE rows.

    Rows are grouped by sketch. For each sketch the table shows, per
    implementation (ours, apache-rust, apache-cpp), the rmse, mean_rel_error and
    max_rel_error, a `theoretical` column equal to 1/sqrt(k) (the expected error
    floor), and a parity verdict comparing ours against apache-rust via
    rmse_parity.
    """
    theoretical = 1.0 / math.sqrt(k)

    groups = {}
    order = []
    for row in rows:
        sketch = row["sketch"]
        if sketch not in groups:
            groups[sketch] = {}
            order.append(sketch)
        groups[sketch][row["implementation"]] = row

    columns = [
        "sketch",
        "ours rmse",
        "ours mean",
        "ours max",
        "a-rust rmse",
        "a-rust mean",
        "a-rust max",
        "a-cpp rmse",
        "a-cpp mean",
        "a-cpp max",
        "theoretical",
        "parity (ours vs a-rust)",
    ]

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")

    def cell(impl_rows, impl, field):
        r = impl_rows.get(impl)
        value = _as_float(r[field]) if r else None
        return f"{value:.4f}" if value is not None else "-"

    def parity_cell(impl_rows):
        ours = impl_rows.get("ours")
        ref = impl_rows.get("apache-rust")
        ours_rmse = _as_float(ours["rmse"]) if ours else None
        ref_rmse = _as_float(ref["rmse"]) if ref else None
        if ours_rmse is None or ref_rmse is None or ref_rmse == 0:
            return "-"
        return "pass" if rmse_parity(ours_rmse, ref_rmse) else "FAIL"

    ordered = [s for s in RMSE_SKETCHES if s in groups]
    ordered += [s for s in order if s not in RMSE_SKETCHES]

    for sketch in ordered:
        impl_rows = groups[sketch]
        cells = [
            sketch,
            cell(impl_rows, "ours", "rmse"),
            cell(impl_rows, "ours", "mean_rel_error"),
            cell(impl_rows, "ours", "max_rel_error"),
            cell(impl_rows, "apache-rust", "rmse"),
            cell(impl_rows, "apache-rust", "mean_rel_error"),
            cell(impl_rows, "apache-rust", "max_rel_error"),
            cell(impl_rows, "apache-cpp", "rmse"),
            cell(impl_rows, "apache-cpp", "mean_rel_error"),
            cell(impl_rows, "apache-cpp", "max_rel_error"),
            f"{theoretical:.4f}",
            parity_cell(impl_rows),
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("**Notes**")
    lines.append("")
    lines.append(
        f"> The `theoretical` column is the 1/sqrt(k) error floor for k={k} "
        f"(~{theoretical:.4f}). Parity holds when our RMSE is within 1.25x of "
        "the apache-rust reference in either direction."
    )

    return "\n".join(lines) + "\n"


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compare benchmark result CSVs and render a markdown table, "
        "or run an accuracy gate over the 'ours' rows."
    )
    parser.add_argument(
        "csv",
        nargs="*",
        help="one or more result CSV paths sharing the benchmark schema",
    )
    parser.add_argument(
        "--rmse",
        nargs="+",
        metavar="CSV",
        help="RMSE summary CSV paths: print the RMSE parity table (and an "
        "rmse.png plot when matplotlib is available)",
    )
    parser.add_argument(
        "--md",
        metavar="PATH",
        help="also write the rendered markdown table to this path",
    )
    parser.add_argument(
        "--check-accuracy",
        metavar="THRESHOLDS_JSON",
        help="run the accuracy gate using per-sketch thresholds from this JSON",
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BASELINE", "CANDIDATE"),
        help="compare two result CSVs and print the per-plane separation verdict",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.compare:
        base_rows = load_rows([args.compare[0]])
        cand_rows = load_rows([args.compare[1]])
        sys.stdout.write(render_compare(base_rows, cand_rows))
        return 0

    if args.rmse:
        rmse_rows = load_rmse_rows(args.rmse)
        table = render_rmse_table(rmse_rows)
        sys.stdout.write(table)
        out_dir = os.path.dirname(os.path.abspath(args.rmse[0]))
        try:
            import plots
        except ImportError:
            print(
                "note: matplotlib not available, skipping rmse.png",
                file=sys.stderr,
            )
            return 0
        path = plots.render_rmse_plot(rmse_rows, out_dir)
        print(f"wrote {path}", file=sys.stderr)
        return 0

    if not args.csv:
        parser.error("at least one result CSV is required (or use --rmse)")

    rows = load_rows(args.csv)

    if args.check_accuracy:
        with open(args.check_accuracy) as handle:
            thresholds = json.load(handle)
        passed, messages = check_accuracy(rows, thresholds)
        for message in messages:
            print(message)
        if passed:
            print("accuracy gate passed")
            return 0
        print("accuracy gate FAILED", file=sys.stderr)
        sys.exit(1)

    table = render_table(rows)
    sys.stdout.write(table)
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)), exist_ok=True)
        with open(args.md, "w") as handle:
            handle.write(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
