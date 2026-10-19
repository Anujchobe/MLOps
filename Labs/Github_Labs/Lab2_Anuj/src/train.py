"""Step 1: train the candidate model and a baseline comparator."""
import json, os, subprocess
from datetime import datetime, timezone
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.config import SEED, WORK_DIR
from src.data import load_splits, data_fingerprint


def make_version():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sha = os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                          stderr=subprocess.DEVNULL).strip()
        except Exception:
            sha = "local"
    return f"v{ts}-{sha[:7]}"


def main():
    WORK_DIR.mkdir(exist_ok=True)
    s = load_splits()
    X_tr, y_tr = s["train"]

    model = HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
        l2_regularization=1.0, early_stopping=False, random_state=SEED)
    model.fit(X_tr, y_tr)

    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    baseline.fit(X_tr, y_tr)

    version = make_version()
    joblib.dump(model, WORK_DIR / "raw_model.joblib")
    joblib.dump(baseline, WORK_DIR / "baseline.joblib")
    meta = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": os.environ.get("GITHUB_SHA", "local"),
        "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
        "data_fingerprint": data_fingerprint(),
        "model": "HistGradientBoostingClassifier",
        "baseline": "StandardScaler + LogisticRegression",
        "n_train": int(len(y_tr)),
        "hyperparams": {k: v for k, v in model.get_params().items()
                        if k in ("learning_rate", "max_iter", "max_leaf_nodes", "l2_regularization")},
    }
    (WORK_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Trained {version} on {len(y_tr)} rows")


if __name__ == "__main__":
    main()
