"""Real tabular dataset (Wisconsin Diagnostic Breast Cancer, bundled with
scikit-learn, so CI needs no network) with a 3-way stratified split:
train / calibration / test. Calibration must use data the model never saw."""
import hashlib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from src.config import SEED, TEST_SIZE, CALIB_SIZE


def load_splits():
    ds = load_breast_cancer()
    X, y = ds.data, ds.target
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_rest, y_rest, test_size=CALIB_SIZE, stratify=y_rest, random_state=SEED)
    return {
        "train": (X_train, y_train),
        "calib": (X_cal, y_cal),
        "test": (X_test, y_test),
        "feature_names": list(ds.feature_names),
    }


def data_fingerprint():
    """Hash of the raw data so each model version records exactly what it saw."""
    ds = load_breast_cancer()
    h = hashlib.sha256(np.ascontiguousarray(ds.data).tobytes())
    h.update(np.ascontiguousarray(ds.target).tobytes())
    return h.hexdigest()[:12]
