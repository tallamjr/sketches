#!/usr/bin/env python3
"""Matplotlib plots for the shared benchmark CSV schema.

Renders three grouped-bar charts comparing implementations across sketches:
throughput, memory (bytes) and accuracy (relative error). Headless-safe (Agg
backend) and styled with the user's standard Tahoma font, falling back to
DejaVu Sans when Tahoma is not installed.
"""

import matplotlib

matplotlib.use("Agg")
import os
import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager


def _apply_tahoma():
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Tahoma" in names:
        plt.rcParams["font.family"] = "Tahoma"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
        print(
            "warning: Tahoma not installed, falling back to DejaVu Sans",
            file=sys.stderr,
        )


_apply_tahoma()

IMPLEMENTATIONS = ["ours", "apache-rust", "apache-cpp"]


def _as_float(value):
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


def _label(row):
    """Group label for a row: sketch plus dataset/op when there is variety."""
    return f"{row['sketch']}\n{row['dataset']}/{row['op']}"


def _grouped_bar(rows, field, title, ylabel, out_path, require_field=False):
    """Render one grouped-bar chart (one bar per implementation per group).

    Groups are keyed by (sketch, dataset, op). When require_field is True only
    groups that have at least one non-empty value for `field` are plotted.
    """
    groups = {}
    order = []
    for row in rows:
        key = (row["sketch"], row["dataset"], row["op"])
        if key not in groups:
            groups[key] = {}
            order.append(key)
        groups[key][row["implementation"]] = row

    labels = []
    series = {impl: [] for impl in IMPLEMENTATIONS}
    for key in order:
        impl_rows = groups[key]
        values = {}
        for impl in IMPLEMENTATIONS:
            r = impl_rows.get(impl)
            values[impl] = _as_float(r[field]) if r else None
        if require_field and all(v is None for v in values.values()):
            continue
        sample = next(iter(impl_rows.values()))
        labels.append(_label(sample))
        for impl in IMPLEMENTATIONS:
            series[impl].append(values[impl])

    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(labels) + 2.0), 4.5))

    n_groups = len(labels)
    n_impls = len(IMPLEMENTATIONS)
    bar_width = 0.8 / n_impls
    indices = list(range(n_groups))

    plotted_any = False
    for offset, impl in enumerate(IMPLEMENTATIONS):
        heights = [v if v is not None else 0.0 for v in series[impl]]
        positions = [i + offset * bar_width for i in indices]
        if any(v is not None for v in series[impl]):
            plotted_any = True
        ax.bar(positions, heights, bar_width, label=impl)

    centre = (n_impls - 1) * bar_width / 2.0
    ax.set_xticks([i + centre for i in indices])
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if plotted_any:
        ax.legend()

    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return out_path


RMSE_SKETCHES = ["theta", "hll", "cpc"]


def render_rmse_plot(rows, out_dir):
    """Write rmse.png: grouped bars of rmse per implementation, grouped by sketch.

    One group per sketch, one bar per implementation. Transparent background and
    Tahoma styling. Returns the written path.
    """
    os.makedirs(out_dir, exist_ok=True)

    groups = {}
    order = []
    for row in rows:
        sketch = row["sketch"]
        if sketch not in groups:
            groups[sketch] = {}
            order.append(sketch)
        groups[sketch][row["implementation"]] = row

    sketches = [s for s in RMSE_SKETCHES if s in groups]
    sketches += [s for s in order if s not in RMSE_SKETCHES]

    series = {impl: [] for impl in IMPLEMENTATIONS}
    for sketch in sketches:
        impl_rows = groups[sketch]
        for impl in IMPLEMENTATIONS:
            r = impl_rows.get(impl)
            series[impl].append(_as_float(r["rmse"]) if r else None)

    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(sketches) + 2.0), 4.5))

    n_groups = len(sketches)
    n_impls = len(IMPLEMENTATIONS)
    bar_width = 0.8 / n_impls
    indices = list(range(n_groups))

    for offset, impl in enumerate(IMPLEMENTATIONS):
        heights = [v if v is not None else 0.0 for v in series[impl]]
        positions = [i + offset * bar_width for i in indices]
        ax.bar(positions, heights, bar_width, label=impl)

    centre = (n_impls - 1) * bar_width / 2.0
    ax.set_xticks([i + centre for i in indices])
    ax.set_xticklabels(sketches)
    ax.set_ylabel("RMSE (relative error)")
    ax.set_title("RMSE by sketch and implementation")
    ax.legend()

    out_path = os.path.join(out_dir, "rmse.png")
    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_before_after_rmse_plot(labels_to_rmse, out_path, theoretical):
    """Render a before/after RMSE bar chart styled with Tahoma.

    One bar per named RMSE value (keys are bar labels, e.g. "ours-before",
    "ours-after", "apache-rust"; values are RMSE floats). A dashed horizontal
    line marks the theoretical floor. Transparent background and Tahoma styling.
    Returns out_path.
    """
    _apply_tahoma()

    parent = os.path.dirname(out_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    labels = list(labels_to_rmse.keys())
    values = [labels_to_rmse[label] for label in labels]
    indices = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(max(6.0, 1.6 * len(labels) + 2.0), 4.5))

    ax.bar(indices, values, 0.6)
    ax.axhline(
        theoretical,
        linestyle="--",
        color="black",
        label=f"theoretical ({theoretical:g})",
    )

    ax.set_xticks(indices)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("RMSE (relative error)")
    ax.set_title("HLL accuracy: before vs after HIP")
    ax.legend()

    fig.savefig(out_path, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_plots(rows, out_dir):
    """Write throughput, memory and accuracy PNGs into out_dir.

    Returns the list of written paths in that order.
    """
    os.makedirs(out_dir, exist_ok=True)

    written = []
    written.append(
        _grouped_bar(
            rows,
            field="throughput_median_ops_per_s",
            title="Throughput by sketch and implementation",
            ylabel="throughput median (ops/s)",
            out_path=os.path.join(out_dir, "throughput.png"),
        )
    )
    written.append(
        _grouped_bar(
            rows,
            field="live_bytes",
            title="Memory footprint by sketch and implementation",
            ylabel="live bytes",
            out_path=os.path.join(out_dir, "memory.png"),
        )
    )
    written.append(
        _grouped_bar(
            rows,
            field="rel_error",
            title="Relative error by sketch and implementation",
            ylabel="relative error",
            out_path=os.path.join(out_dir, "accuracy.png"),
            require_field=True,
        )
    )
    return written
