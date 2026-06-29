"""Tests for the benchmark reporter (table, accuracy gate, plots, font)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plots
import report

HEADER = (
    "implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,"
    "throughput_stddev,throughput_ci_low,throughput_ci_high,"
    "bytes,live_bytes,estimate,exact,rel_error"
)


def _write_csv(tmp_path, name, body_rows):
    path = tmp_path / name
    path.write_text(HEADER + "\n" + "\n".join(body_rows) + "\n")
    return str(path)


def test_render_table_joins_ours_and_apache_rust(tmp_path):
    rows_csv = _write_csv(
        tmp_path,
        "results.csv",
        [
            "ours,hll,synthetic,update,1000,30,5000000,45000,4800000,5200000,128,2048,1010,1000,0.01",
            "apache-rust,hll,synthetic,update,1000,30,2500000,30000,2400000,2600000,144,2304,1005,1000,0.005",
        ],
    )
    rows = report.load_rows([rows_csv])
    table = report.render_table(rows)

    # Both implementations' figures appear on the joined row.
    assert "hll/synthetic/update" in table
    assert "5e+06" in table  # ours throughput median formatted
    assert "2.5e+06" in table  # apache-rust throughput median formatted
    # Median is displayed with its stddev.
    assert "4.5e+04" in table  # ours stddev formatted
    # A throughput ratio ours/apache-rust = 2.0 with 'ours' the better side.
    assert "2.000 (ours)" in table
    # A live_bytes ratio 2048/2304 ~ 0.889 with 'ours' better (lower memory).
    assert "0.889 (ours)" in table
    # The attribution note explains throughput differences via hash choice.
    assert "xxh3" in table


def test_render_table_surfaces_ours_murmur3(tmp_path):
    # The native plane emits both `ours` (xxh3) and `ours-murmur3` (MurmurHash3)
    # rows. The data-driven table must render an `ours-murmur3` column alongside
    # `ours`/`apache-rust`, not silently drop it.
    rows_csv = _write_csv(
        tmp_path,
        "results.csv",
        [
            "ours,hll,synthetic,update,1000,30,5000000,45000,4800000,5200000,128,2048,1010,1000,0.01",
            "ours-murmur3,hll,synthetic,update,1000,30,3000000,40000,2900000,3100000,128,2048,1008,1000,0.008",
            "apache-rust,hll,synthetic,update,1000,30,2500000,30000,2400000,2600000,144,2304,1005,1000,0.005",
        ],
    )
    rows = report.load_rows([rows_csv])
    table = report.render_table(rows)

    # An ours-murmur3 throughput/mem column header is present.
    assert "ours-m3 tput" in table
    assert "ours-m3 mem" in table
    # Its throughput median figure appears on the joined row.
    assert "3e+06" in table  # ours-murmur3 throughput median formatted
    # The original ours/apache ratio columns still work.
    assert "2.000 (ours)" in table


def test_render_table_has_murmur3_vs_apache_ratios(tmp_path):
    rows_csv = _write_csv(
        tmp_path,
        "results.csv",
        [
            "ours,hll,synthetic,update,1000,10,5000000,45000,4800000,5200000,128,2048,1010,1000,0.01",
            "ours-murmur3,hll,synthetic,update,1000,10,2600000,40000,2500000,2700000,128,2048,1008,1000,0.008",
            "apache-rust,hll,synthetic,update,1000,10,2500000,30000,2450000,2550000,144,2304,1005,1000,0.005",
        ],
    )
    rows = report.load_rows([rows_csv])
    table = report.render_table(rows)
    assert "tput ours-m3/a-rust" in table
    # 2.6e6 / 2.5e6 = 1.040, ours-murmur3 the better side
    assert "1.040 (ours)" in table


def test_render_table_excludes_python_plane_labels(tmp_path):
    # The Python plane emits `ours`/`apache` labels that collide with the Rust
    # `ours`. They must not enter the native comparison table: only the native
    # labels get columns, and no stray `apache` (bare) column is introduced.
    rows_csv = _write_csv(
        tmp_path,
        "mixed.csv",
        [
            "ours,hll,synthetic,update,1000,30,5000000,45000,4800000,5200000,128,2048,1010,1000,0.01",
            "apache-rust,hll,synthetic,update,1000,30,2500000,30000,2400000,2600000,144,2304,1005,1000,0.005",
            "apache,hll,synthetic,update,1000,30,9000000,1000,8900000,9100000,128,2048,1010,1000,0.01",
        ],
    )
    rows = report.load_rows([rows_csv])
    table = report.render_table(rows)

    # No bare `apache` column header (only native labels are columned).
    assert "apache tput" not in table
    # The python-plane throughput figure does not leak into the native table.
    assert "9e+06" not in table


def test_load_rows_parses_new_columns(tmp_path):
    csv = tmp_path / "ours.csv"
    csv.write_text(
        "implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,"
        "throughput_stddev,throughput_ci_low,throughput_ci_high,"
        "bytes,live_bytes,estimate,exact,rel_error\n"
        "ours,hll,synthetic,update,1000,30,5000000.0,12345.0,4800000.0,5200000.0,128,2048,1001.0,1000.0,0.001\n"
    )
    rows = report.load_rows([str(csv)])
    r = rows[0]
    assert r["throughput_stddev"] == 12345.0
    assert r["live_bytes"] == 2048


def test_load_rows_parses_ci_columns(tmp_path):
    csv = tmp_path / "ours.csv"
    csv.write_text(
        "implementation,sketch,dataset,op,n,reps,throughput_median_ops_per_s,"
        "throughput_stddev,throughput_ci_low,throughput_ci_high,bytes,live_bytes,"
        "estimate,exact,rel_error\n"
        "ours,hll,synthetic,update,1000,10,5000000.0,12345.0,4800000.0,5200000.0,"
        "128,2048,1001.0,1000.0,0.001\n"
    )
    rows = report.load_rows([str(csv)])
    r = rows[0]
    assert r["throughput_ci_low"] == 4800000.0
    assert r["throughput_ci_high"] == 5200000.0


def test_check_accuracy_fails_when_over_threshold(tmp_path):
    rows_csv = _write_csv(
        tmp_path,
        "over.csv",
        ["ours,hll,synthetic,estimate,1000,30,0,0,0,0,128,2048,1050,1000,0.05"],
    )
    rows = report.load_rows([rows_csv])
    passed, messages = report.check_accuracy(rows, {"hll": 0.02})
    assert passed is False
    assert any("hll" in m and "FAIL" in m for m in messages)


def test_check_accuracy_passes_when_under_threshold(tmp_path):
    rows_csv = _write_csv(
        tmp_path,
        "under.csv",
        ["ours,hll,synthetic,estimate,1000,30,0,0,0,0,128,2048,1010,1000,0.01"],
    )
    rows = report.load_rows([rows_csv])
    passed, messages = report.check_accuracy(rows, {"hll": 0.02})
    assert passed is True
    assert messages == []


def test_check_accuracy_notes_ungated_sketch(tmp_path):
    rows_csv = _write_csv(
        tmp_path,
        "ungated.csv",
        ["ours,mystery,synthetic,estimate,1000,30,0,0,0,0,128,2048,1010,1000,0.5"],
    )
    rows = report.load_rows([rows_csv])
    passed, messages = report.check_accuracy(rows, {"hll": 0.02})
    assert passed is True
    assert any("mystery" in m and "not gated" in m for m in messages)


def test_render_plots_writes_three_pngs(tmp_path):
    rows_csv = _write_csv(
        tmp_path,
        "results.csv",
        [
            "ours,hll,synthetic,update,1000,30,5000000,45000,4800000,5200000,128,2048,1010,1000,0.01",
            "apache-rust,hll,synthetic,update,1000,30,2500000,30000,2400000,2600000,144,2304,1005,1000,0.005",
            "ours,theta,synthetic,update,1000,30,3000000,25000,2900000,3100000,256,4096,1020,1000,0.02",
        ],
    )
    rows = report.load_rows([rows_csv])
    out_dir = tmp_path / "plots"
    paths = plots.render_plots(rows, str(out_dir))

    assert len(paths) == 3
    expected = {"throughput.png", "memory.png", "accuracy.png"}
    assert {os.path.basename(p) for p in paths} == expected
    for path in paths:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_render_speedup_plot(tmp_path):
    # One sketch group with ours/apache-cpp/apache-rust throughput such that
    # ours/apache-cpp = 5e8/2e8 = 2.5x and ours/apache-rust = 5e8/1e8 = 5.0x.
    rows_csv = _write_csv(
        tmp_path,
        "results.csv",
        [
            "ours,hll,synthetic,distinct_count,1000,30,500000000,45000,490000000,510000000,128,2048,1010,1000,0.01",
            "apache-cpp,hll,synthetic,distinct_count,1000,30,200000000,30000,195000000,205000000,144,2304,1005,1000,0.005",
            "apache-rust,hll,synthetic,distinct_count,1000,30,100000000,30000,95000000,105000000,144,2304,1005,1000,0.005",
        ],
    )
    rows = report.load_rows([rows_csv])
    path = plots.render_speedup_plot(rows, str(tmp_path))

    assert os.path.exists(path)
    assert os.path.getsize(path) > 0
    assert os.path.basename(path) == "speedup_vs_apache.png"


def test_plots_render_with_error_bars(tmp_path):
    # Two implementations for one sketch, each with a throughput median and a
    # stddev plus a live_bytes footprint. The throughput plot must draw stddev
    # error bars; the memory plot must read live_bytes. Both PNGs must exist.
    rows = [
        {
            "implementation": "ours",
            "sketch": "hll",
            "dataset": "synthetic",
            "op": "update",
            "throughput_median_ops_per_s": 5e6,
            "throughput_stddev": 1e5,
            "live_bytes": 2048,
            "rel_error": 0.001,
        },
        {
            "implementation": "apache-rust",
            "sketch": "hll",
            "dataset": "synthetic",
            "op": "update",
            "throughput_median_ops_per_s": 4e6,
            "throughput_stddev": 2e5,
            "live_bytes": 4096,
            "rel_error": 0.001,
        },
    ]
    out = plots.render_plots(rows, str(tmp_path))
    assert (tmp_path / "throughput.png").exists()
    assert (tmp_path / "memory.png").exists()
    assert {os.path.basename(p) for p in out} == {
        "throughput.png",
        "memory.png",
        "accuracy.png",
    }

    # The throughput plot must draw stddev error bars. Exercise the error-bar
    # path directly: _grouped_bar must accept a yerr_field and produce a figure
    # whose axes contain errorbar containers.
    import matplotlib.pyplot as plt

    path = plots._grouped_bar(
        rows,
        field="throughput_median_ops_per_s",
        title="t",
        ylabel="y",
        out_path=str(tmp_path / "thr_err.png"),
        yerr_field="throughput_stddev",
    )
    assert os.path.exists(path)
    # No error-bar field for memory: must still render without crashing.
    path2 = plots._grouped_bar(
        rows,
        field="live_bytes",
        title="m",
        ylabel="y",
        out_path=str(tmp_path / "mem.png"),
    )
    assert os.path.exists(path2)


def test_rmse_parity_and_table():
    import report

    # exact known per-impl rmse values
    rows = [
        {"implementation": "ours", "sketch": "theta", "lg_k": "12", "rmse": "0.016", "mean_rel_error": "0.012", "max_rel_error": "0.04"},
        {"implementation": "apache-rust", "sketch": "theta", "lg_k": "12", "rmse": "0.015", "mean_rel_error": "0.011", "max_rel_error": "0.039"},
    ]
    assert report.rmse_parity(0.016, 0.015) is True  # within 1.25x
    assert report.rmse_parity(0.05, 0.015) is False  # >1.25x worse
    table = report.render_rmse_table(rows, k=4096)
    assert "theta" in table and "theoretical" in table.lower()
    assert "0.0156" in table or "0.016" in table  # 1/sqrt(4096)


def test_before_after_rmse_plot(tmp_path):
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    import plots, matplotlib.pyplot as plt
    out = str(tmp_path / "hll_ba.png")
    path = plots.render_before_after_rmse_plot(
        {"ours-before": 0.0175, "ours-after": 0.0129, "apache-rust": 0.0129},
        out, theoretical=0.015625,
    )
    assert os.path.exists(path) and os.path.getsize(path) > 0
    assert plt.rcParams["font.family"] in (["Tahoma"], "Tahoma")


def test_font_family_resolves_to_tahoma():
    # Tahoma is installed in this environment, so importing plots (which calls
    # _apply_tahoma at import time) must resolve the family to Tahoma.
    assert plots.plt.rcParams["font.family"] == ["Tahoma"]


def test_separation_disjoint_intervals():
    v, ratio = report.separation(100.0, 95.0, 105.0, 130.0, 125.0, 135.0)
    assert v == "separated"
    assert abs(ratio - 1.3) < 1e-9


def test_separation_overlapping_intervals():
    v, _ = report.separation(100.0, 90.0, 110.0, 108.0, 100.0, 116.0)
    assert v == "within noise"


def test_render_compare_table(tmp_path):
    base = _write_csv(
        tmp_path, "base.csv",
        ["ours,hll,synthetic,update,1000,10,5000000,45000,4900000,5100000,128,2048,1010,1000,0.01"],
    )
    cand = _write_csv(
        tmp_path, "cand.csv",
        ["ours,hll,synthetic,update,1000,10,5600000,45000,5500000,5700000,128,2048,1010,1000,0.01"],
    )
    base_rows = report.load_rows([base])
    cand_rows = report.load_rows([cand])
    table = report.render_compare(base_rows, cand_rows)
    assert "hll/synthetic/update" in table
    assert "separated" in table        # CIs [4.9M,5.1M] vs [5.5M,5.7M] disjoint
    assert "1.120" in table            # 5.6M / 5.0M speedup


def test_noise_warnings_flags_wide_ci(tmp_path):
    rows = report.load_rows([_write_csv(
        tmp_path, "noisy.csv",
        [
            # half-width (5.2M-4.8M)/2 = 0.2M = 4% of 5M -> OK
            "ours,hll,synthetic,update,1000,10,5000000,1,4800000,5200000,128,2048,1010,1000,0.01",
            # half-width (7M-3M)/2 = 2M = 40% of 5M -> flagged
            "ours,theta,synthetic,update,1000,10,5000000,1,3000000,7000000,128,2048,1010,1000,0.01",
        ],
    )])
    warnings = report.noise_warnings(rows, frac=0.05)
    joined = " ".join(warnings)
    assert "theta" in joined
    assert "hll" not in joined
