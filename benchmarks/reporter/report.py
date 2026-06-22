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


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Compare benchmark result CSVs and render a markdown table, "
        "or run an accuracy gate over the 'ours' rows."
    )
    parser.add_argument(
        "csv",
        nargs="+",
        help="one or more result CSV paths sharing the benchmark schema",
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
