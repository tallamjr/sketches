"""Tests for the benchmark reporter (table, accuracy gate, plots, font)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import plots
import report

HEADER = (
    "implementation,sketch,dataset,op,n,throughput_ops_per_s,bytes,"
    "estimate,exact,rel_error"
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
            "ours,hll,synthetic,update,1000,5000000,128,1010,1000,0.01",
            "apache-rust,hll,synthetic,update,1000,2500000,144,1005,1000,0.005",
        ],
    )
    rows = report.load_rows([rows_csv])
    table = report.render_table(rows)

    # Both implementations' figures appear on the joined row.
    assert "hll/synthetic/update" in table
    assert "5e+06" in table  # ours throughput formatted
    assert "2.5e+06" in table  # apache-rust throughput formatted
    # A throughput ratio ours/apache-rust = 2.0 with 'ours' the better side.
    assert "2.000 (ours)" in table
    # A bytes ratio 128/144 ~ 0.889 with 'ours' better (lower bytes).
    assert "0.889 (ours)" in table
    # The attribution note explains throughput differences via hash choice.
    assert "xxh3" in table


def test_check_accuracy_fails_when_over_threshold(tmp_path):
    rows_csv = _write_csv(
        tmp_path,
        "over.csv",
        ["ours,hll,synthetic,estimate,1000,0,128,1050,1000,0.05"],
    )
    rows = report.load_rows([rows_csv])
    passed, messages = report.check_accuracy(rows, {"hll": 0.02})
    assert passed is False
    assert any("hll" in m and "FAIL" in m for m in messages)


def test_check_accuracy_passes_when_under_threshold(tmp_path):
    rows_csv = _write_csv(
        tmp_path,
        "under.csv",
        ["ours,hll,synthetic,estimate,1000,0,128,1010,1000,0.01"],
    )
    rows = report.load_rows([rows_csv])
    passed, messages = report.check_accuracy(rows, {"hll": 0.02})
    assert passed is True
    assert messages == []


def test_check_accuracy_notes_ungated_sketch(tmp_path):
    rows_csv = _write_csv(
        tmp_path,
        "ungated.csv",
        ["ours,mystery,synthetic,estimate,1000,0,128,1010,1000,0.5"],
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
            "ours,hll,synthetic,update,1000,5000000,128,1010,1000,0.01",
            "apache-rust,hll,synthetic,update,1000,2500000,144,1005,1000,0.005",
            "ours,theta,synthetic,update,1000,3000000,256,1020,1000,0.02",
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
