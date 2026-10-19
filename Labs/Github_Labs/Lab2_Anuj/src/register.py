"""Step 4 (only after the gate passes): promote the model into models/,
append to the registry, point 'latest' at it, prune old binaries."""
import json, shutil
from src.config import WORK_DIR, MODELS_DIR, REGISTRY, KEEP_LAST_N_MODELS


def main():
    ev = json.loads((WORK_DIR / "eval.json").read_text())
    v = ev["version"]
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(WORK_DIR / "raw_model.joblib", MODELS_DIR / f"model_{v}.joblib")
    shutil.copy(WORK_DIR / "calibrated_model.joblib", MODELS_DIR / f"model_{v}_calibrated.joblib")

    reg = json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {"latest": None, "versions": []}
    reg["versions"].append({
        "version": v, "trained_at": ev["trained_at"], "git_sha": ev["git_sha"],
        "run_id": ev["run_id"], "data_fingerprint": ev["data_fingerprint"],
        "calibration_method": ev["calibration_method"],
        "metrics": {k: ev["results"]["calibrated"][k] for k in ("f1", "roc_auc", "brier", "ece")},
        "raw_brier": ev["results"]["raw"]["brier"],
        "baseline_f1": ev["results"]["baseline"]["f1"],
        "files": [f"model_{v}.joblib", f"model_{v}_calibrated.joblib"],
    })
    reg["latest"] = v

    # Keep full metadata history, but only the last N model binaries in git
    keep = {f for e in reg["versions"][-KEEP_LAST_N_MODELS:] for f in e["files"]}
    for e in reg["versions"][:-KEEP_LAST_N_MODELS]:
        e["pruned"] = True
    for p in MODELS_DIR.glob("model_*.joblib"):
        if p.name not in keep:
            p.unlink()

    REGISTRY.write_text(json.dumps(reg, indent=2))
    print(f"Registered {v}; {len(reg['versions'])} versions in registry")


if __name__ == "__main__":
    main()
