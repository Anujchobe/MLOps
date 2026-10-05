import pytest

from src.metrics import (
    accuracy,
    confusion_counts,
    evaluate,
    f1_score,
    load_predictions,
    precision,
    recall,
)

DATA_PATH = "data/predictions.csv"


def test_load_predictions():
    y_true, y_pred = load_predictions(DATA_PATH)
    assert len(y_true) == 12
    assert len(y_pred) == 12
    assert set(y_true) <= {0, 1}


def test_confusion_counts():
    y_true, y_pred = load_predictions(DATA_PATH)
    assert confusion_counts(y_true, y_pred) == (5, 1, 2, 4)


def test_precision_recall_f1():
    y_true, y_pred = load_predictions(DATA_PATH)
    assert precision(y_true, y_pred) == pytest.approx(5 / 6)
    assert recall(y_true, y_pred) == pytest.approx(5 / 7)
    assert f1_score(y_true, y_pred) == pytest.approx(10 / 13)


def test_accuracy():
    y_true, y_pred = load_predictions(DATA_PATH)
    assert accuracy(y_true, y_pred) == pytest.approx(0.75)


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        confusion_counts([1, 0, 1], [1, 0])


@pytest.mark.parametrize(
    "y_true, y_pred, expected_precision, expected_recall",
    [
        ([1, 1, 1], [1, 1, 1], 1.0, 1.0),
        ([0, 0, 0], [0, 0, 0], 0.0, 0.0),
        ([1, 0], [0, 1], 0.0, 0.0),
        ([1, 1, 0, 0], [1, 0, 0, 0], 1.0, 0.5),
    ],
)
def test_edge_cases(y_true, y_pred, expected_precision, expected_recall):
    assert precision(y_true, y_pred) == pytest.approx(expected_precision)
    assert recall(y_true, y_pred) == pytest.approx(expected_recall)


def test_evaluate_returns_all_metrics():
    result = evaluate(DATA_PATH)
    assert set(result) == {"precision", "recall", "f1", "accuracy"}
    assert all(0.0 <= v <= 1.0 for v in result.values())
