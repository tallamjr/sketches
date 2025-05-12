import pytest
import logging

import sketches

# Skip if polars is not available
pl = pytest.importorskip("polars")


@pytest.mark.parametrize(
    "sketch_cls,true_rel_err", [
        (sketches.ThetaSketch, 1e-6),
        (sketches.HllSketch, 0.05),
        (sketches.CpcSketch, 0.05),
    ],
)
def test_synthetic_cardinality_vs_polars(sketch_cls, true_rel_err):
    """
    Compare sketch estimates to Polars exact distinct count on synthetic data.
    """
    # Generate synthetic data: 1000 unique string values
    N = 1000
    values = [str(i) for i in range(N)]
    logging.info("Testing %s with N=%d, tolerance=%f", sketch_cls.__name__, N, true_rel_err)

    # Exact count via Polars
    df = pl.DataFrame({"v": values})
    exact = df["v"].n_unique()
    logging.info("Exact count via Polars: %d", exact)

    # Sketch estimate
    sk = sketch_cls()
    for v in values:
        sk.update(v)
    est = sk.estimate()
    rel_err = abs(est - exact) / exact
    logging.info("%s estimate: %s, exact: %s, rel_err: %.6f", sketch_cls.__name__, est, exact, rel_err)

    # Compute relative error
    assert rel_err <= true_rel_err, (
        f"{sketch_cls.__name__} error {rel_err:.3f} exceeds "
        f"tolerance {true_rel_err:.3f}: est={est}, exact={exact}"
    )