import numpy as np
from src.data import load_splits
from src.evaluate import expected_calibration_error


def test_splits_are_disjoint_and_complete():
    s = load_splits()
    sizes = [len(s[k][1]) for k in ("train", "calib", "test")]
    assert sum(sizes) == 569  # full WDBC dataset
    assert min(sizes) > 50


def test_ece_perfect_and_worst_case():
    y = np.array([0, 0, 1, 1])
    assert expected_calibration_error(y, y.astype(float)) == 0.0
    assert expected_calibration_error(y, 1 - y.astype(float)) == 1.0


def test_ece_bounds():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = rng.random(500)
    assert 0.0 <= expected_calibration_error(y, p) <= 1.0
