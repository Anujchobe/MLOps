"""Step 3: evaluate baseline vs raw vs calibrated on the untouched test split,
apply the quality gate, and write versioned metrics."""
import json, sys
import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             log_loss, precision_score, recall_score, roc_auc_score)
from src.config import WORK_DIR, METRICS_DIR, ECE_BINS, GATE
from src.data import load_splits


def expected_calibration_error(y_true, p, n_bins=ECE_BINS):
    """Weighted mean |observed positive rate - mean predicted prob| over equal-width bins."""
    y_true, p = np.asarray(y_true), np.asarray(p)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y_true[m].mean() - p[m].mean())
    return float(ece)


def score(model, X, y):
    p = model.predict_proba(X)[:, 1]
    yhat = (p >= 0.5).astype(int)
    return {
        "f1": float(f1_score(y, yhat)),
        "precision": float(precision_score(y, yhat)),
        "recall": float(recall_score(y, yhat)),
        "accuracy": float(accuracy_score(y, yhat)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))),
        "ece": expected_calibration_error(y, p),
    }


def main():
    X_te, y_te = load_splits()["test"]
    meta = json.loads((WORK_DIR / "meta.json").read_text())
    results = {name: score(joblib.load(WORK_DIR / f), X_te, y_te) for name, f in [
        ("baseline", "baseline.joblib"),
        ("raw", "raw_model.joblib"),
        ("calibrated", "calibrated_model.joblib")]}

    c, b = results["calibrated"], results["baseline"]
    checks = {
        "f1_min": c["f1"] >= GATE["f1_min"],
        "roc_auc_min": c["roc_auc"] >= GATE["roc_auc_min"],
        "beats_baseline_f1": (c["f1"] >= b["f1"]) or not GATE["must_beat_baseline_f1"],
    }
    gate = {"passed": all(checks.values()), "checks": checks, "thresholds": GATE}

    out = {**meta, "n_test": int(len(y_te)), "results": results, "gate": gate}
    METRICS_DIR.mkdir(exist_ok=True)
    (METRICS_DIR / f"{meta['version']}.json").write_text(json.dumps(out, indent=2))
    (WORK_DIR / "eval.json").write_text(json.dumps(out, indent=2))

    for k, r in results.items():
        print(f"{k:11s} F1={r['f1']:.4f} AUC={r['roc_auc']:.4f} "
              f"Brier={r['brier']:.4f} ECE={r['ece']:.4f}")
    print("GATE:", "PASSED" if gate["passed"] else f"FAILED {checks}")
    sys.exit(0 if gate["passed"] else 1)


if __name__ == "__main__":
    main()
