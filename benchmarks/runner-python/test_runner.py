import subprocess, sys, os, csv

HERE = os.path.dirname(__file__)
PY = os.path.join(HERE, "..", "..", ".venv", "bin", "python")


def test_ours_emits_rows(tmp_path):
    out = tmp_path / "py_ours.csv"
    subprocess.run(
        [PY, os.path.join(HERE, "runner.py"), "--impl", "ours",
         "--n", "2000", "--reps", "3", "--out", str(out)],
        check=True,
    )
    with open(out) as f:
        rows = list(csv.reader(f))
    assert rows[0] == (
        "implementation,sketch,dataset,op,n,reps,"
        "throughput_median_ops_per_s,throughput_stddev,bytes,live_bytes,"
        "estimate,exact,rel_error"
    ).split(",")
    assert all(r[0] == "ours" for r in rows[1:])
    assert len(rows) > 1
