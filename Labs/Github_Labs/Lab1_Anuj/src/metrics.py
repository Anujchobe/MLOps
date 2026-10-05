"""Binary classification metrics computed from a predictions CSV.

Replaces the upstream calculator.py example with a small, dependency-free
evaluation module of the kind an MLOps pipeline would call after inference.
"""

import csv


def load_predictions(path):
    """Read a CSV with y_true and y_pred columns into two lists of ints."""
    y_true, y_pred = [], []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            y_true.append(int(row["y_true"]))
            y_pred.append(int(row["y_pred"]))
    if not y_true:
        raise ValueError("No rows found in predictions file")
    return y_true, y_pred


def confusion_counts(y_true, y_pred):
    """Return (tp, fp, fn, tn) for binary labels."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    return tp, fp, fn, tn


def precision(y_true, y_pred):
    """TP / (TP + FP). Returns 0.0 when nothing was predicted positive."""
    tp, fp, _, _ = confusion_counts(y_true, y_pred)
    denom = tp + fp
    return tp / denom if denom else 0.0


def recall(y_true, y_pred):
    """TP / (TP + FN). Returns 0.0 when there are no actual positives."""
    tp, _, fn, _ = confusion_counts(y_true, y_pred)
    denom = tp + fn
    return tp / denom if denom else 0.0


def f1_score(y_true, y_pred):
    """Harmonic mean of precision and recall; 0.0 when both are 0."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def accuracy(y_true, y_pred):
    """(TP + TN) / total."""
    tp, fp, fn, tn = confusion_counts(y_true, y_pred)
    total = tp + fp + fn + tn
    return (tp + tn) / total if total else 0.0


def evaluate(path):
    """Load a predictions CSV and return all metrics as a dict."""
    y_true, y_pred = load_predictions(path)
    return {
        "precision": precision(y_true, y_pred),
        "recall": recall(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
    }


if __name__ == "__main__":
    for name, value in evaluate("data/predictions.csv").items():
        print(f"{name:>10}: {value:.4f}")
