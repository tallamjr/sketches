#!/usr/bin/env python3
"""Reporter for the shared benchmark CSV schema.

Ingests one or more result CSVs (emitted by runner-ours, runner-apache-rust,
runner-cpp), all sharing the schema:

    implementation,sketch,dataset,op,n,throughput_ops_per_s,bytes,estimate,exact,rel_error

Two modes:

1. Default (table): group rows by the join key (sketch, dataset, op) and render
   a markdown comparison table across implementations, including ours/apache
   ratios for throughput (higher is better) and bytes (lower is better).

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
    "throughput_ops_per_s",
    "bytes",
    "estimate",
    "exact",
    "rel_error",
]

IMPLEMENTATIONS = ["ours", "apache-rust", "apache-cpp"]


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
                rows.append(dict(record))
    return rows


def _join_key(row):
    return (row["sketch"], row["dataset"], row["op"])


def _as_float(value):
    """Parse a CSV cell as a float, returning None when empty or unparseable."""
    if value is None:
        return None
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


def render_table(rows):
    """Render a markdown comparison table from loaded rows.

    Rows are grouped by the join key (sketch, dataset, op). For each group the
    table shows, per implementation, throughput_ops_per_s, bytes and rel_error,
    plus ours/apache-rust and ours/apache-cpp ratios for throughput and bytes.
    The `plane` column labels the comparison plane (the join key).
    """
    groups = {}
    for row in rows:
        groups.setdefault(_join_key(row), {})[row["implementation"]] = row

    columns = [
        "plane (sketch/dataset/op)",
        "ours tput",
        "a-rust tput",
        "a-cpp tput",
        "ours bytes",
        "a-rust bytes",
        "a-cpp bytes",
        "ours rel_err",
        "tput ours/a-rust",
        "tput ours/a-cpp",
        "bytes ours/a-rust",
        "bytes ours/a-cpp",
    ]

    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")

    def tput(impl_rows, impl):
        r = impl_rows.get(impl)
        return _as_float(r["throughput_ops_per_s"]) if r else None

    def nbytes(impl_rows, impl):
        r = impl_rows.get(impl)
        return _as_float(r["bytes"]) if r else None

    def fmt(value):
        return f"{value:.3g}" if value is not None else "-"

    for key in sorted(groups):
        impl_rows = groups[key]
        plane = "/".join(key)

        ours_t = tput(impl_rows, "ours")
        arust_t = tput(impl_rows, "apache-rust")
        acpp_t = tput(impl_rows, "apache-cpp")
        ours_b = nbytes(impl_rows, "ours")
        arust_b = nbytes(impl_rows, "apache-rust")
        acpp_b = nbytes(impl_rows, "apache-cpp")

        ours_row = impl_rows.get("ours")
        ours_re = ours_row["rel_error"] if ours_row else ""
        ours_re = ours_re.strip() if ours_re else ""
        ours_re_disp = ours_re if ours_re != "" else "-"

        cells = [
            plane,
            fmt(ours_t),
            fmt(arust_t),
            fmt(acpp_t),
            fmt(ours_b),
            fmt(arust_b),
            fmt(acpp_b),
            ours_re_disp,
            _ratio_cell(ours_t, arust_t, higher_is_better=True),
            _ratio_cell(ours_t, acpp_t, higher_is_better=True),
            _ratio_cell(ours_b, arust_b, higher_is_better=False),
            _ratio_cell(ours_b, acpp_b, higher_is_better=False),
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
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

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
