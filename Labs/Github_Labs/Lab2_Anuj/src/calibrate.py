"""Step 2: calibrate the trained model on the held-out calibration split.
The base model is frozen so calibration only learns the probability mapping."""
import json
import joblib
from sklearn.calibration import CalibratedClassifierCV
from src.config import WORK_DIR, CALIB_METHOD
from src.data import load_splits

try:  # scikit-learn >= 1.6
    from sklearn.frozen import FrozenEstimator
except ImportError:  # pragma: no cover
    FrozenEstimator = None


def main():
    model = joblib.load(WORK_DIR / "raw_model.joblib")
    X_cal, y_cal = load_splits()["calib"]
    if FrozenEstimator is not None:
        cal = CalibratedClassifierCV(FrozenEstimator(model), method=CALIB_METHOD)
    else:
        cal = CalibratedClassifierCV(model, method=CALIB_METHOD, cv="prefit")
    cal.fit(X_cal, y_cal)
    joblib.dump(cal, WORK_DIR / "calibrated_model.joblib")

    meta = json.loads((WORK_DIR / "meta.json").read_text())
    meta.update({"calibration_method": CALIB_METHOD, "n_calib": int(len(y_cal))})
    (WORK_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Calibrated with {CALIB_METHOD} on {len(y_cal)} rows")


if __name__ == "__main__":
    main()
