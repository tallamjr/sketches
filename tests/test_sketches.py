import csv
from pathlib import Path

import pytest
import logging

# Determine the directory containing the test data
DATA_DIR = Path(__file__).parent / "data"


def load_column(file_name, column):
    """
    Load a column from a CSV file in the test data directory.
    Returns a list of string values.
    """
    path = DATA_DIR / file_name
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        return [row[column] for row in reader]


@pytest.mark.parametrize(
    "sketch_class,column,true_rel_err",
    [
        ("ThetaSketch", "r_regionkey", 1e-6),
        ("HllSketch", "r_regionkey", 0.05),
        ("CpcSketch", "r_regionkey", 0.05),
    ],
)
def test_sketch_on_region(sketch_class, column, true_rel_err):
    """
    Test that each sketch implementation estimates the number
    of distinct values in the region.csv file within tolerance.
    """
    logging.info("Testing %s on column '%s' with tolerance %f", sketch_class, column, true_rel_err)
    # Dynamically import the sketch class from the Python extension module
    import sketches as ds

    sketch_cls = getattr(ds, sketch_class)
    # Load data
    data = load_column("region.csv", column)
    # Compute true distinct count
    true_count = len(set(data))
    logging.info("True distinct count: %d", true_count)
    # Instantiate sketch with default parameters
    sk = sketch_cls()
    # Update sketch
    for v in data:
        sk.update(v)
    # Estimate
    est = sk.estimate()
    # Compute relative error
    rel_err = abs(est - true_count) / true_count
    logging.info("%s estimate: %s, true: %s, rel_err: %.6f", sketch_class, est, true_count, rel_err)
    assert (
        rel_err <= true_rel_err
    ), f"{sketch_class} err >{true_rel_err*100}%: est {est}, true {true_count}"

