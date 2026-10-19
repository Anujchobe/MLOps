# Lab 2 — GitHub Actions: Train, Calibrate, Gate & Version (Anuj)

Covers the Lab 2 modules (automated training, evaluation, calibration, and model
versioning via GitHub Actions) with an independent implementation rather than the
upstream synthetic-data random forest.

## How this differs from the course repo

| Upstream Lab 2 | This lab |
|---|---|
| Synthetic data from `make_classification`, regenerated every run | Real dataset: Wisconsin Diagnostic Breast Cancer (569 rows, 30 features), bundled with scikit-learn so CI needs no network |
| Random forest | `HistGradientBoostingClassifier`, plus a **logistic-regression baseline** trained on the same split |
| Train/test only; calibration fit on whatever data is at hand | 3-way stratified split: train (364) / **calibration (91)** / test (114). Base model is frozen (`FrozenEstimator`) so calibration never sees training data |
| F1 only | F1, precision, recall, accuracy, ROC-AUC, **Brier, log loss, ECE** (custom 10-bin implementation, unit-tested) |
| Two workflows that each commit back (can race each other) | One pipeline workflow with sequential steps and a `concurrency` group; `git pull --rebase` before push |
| Every run is stored regardless of quality | **Quality gate**: run fails and nothing is committed unless F1 ≥ 0.93 and ROC-AUC ≥ 0.97 |
| Timestamp-only version | Version = `v<UTC timestamp>-<commit sha>`; `models/registry.json` records sha, run id, data fingerprint (SHA-256), calibration method, and metrics |
| Every model binary committed forever | Registry keeps full history; only the **last 5 binaries** stay in git (older ones marked `pruned`) |
| No dashboard | `reports/dashboard.html`: baseline/raw/calibrated table, reliability diagram, metric history across versions. Also a Markdown summary on the Actions run page and a downloadable artifact |
| Fixed calibration method | `workflow_dispatch` input to choose `sigmoid` (Platt) or `isotonic` |
| `upload-artifact` v2/v3, Python 3.8 | `checkout@v4`, `setup-python@v5` with pip cache, `upload-artifact@v4`, Python 3.12; `paths` filter scopes runs to this lab |

## Pipeline

```
push to main (paths: Lab2_Anuj/**)
  → pytest (split integrity, ECE correctness)
  → train      src/train.py      HistGB + LogReg baseline
  → calibrate  src/calibrate.py  Platt scaling on held-out calibration split
  → evaluate   src/evaluate.py   test-set metrics + quality gate (exit 1 on fail)
  → register   src/register.py   copy into models/, update registry.json, prune
  → dashboard  src/dashboard.py  reports/dashboard.html + step summary
  → upload artifact → commit back with [skip ci]
```

## Results (test set, seed 42)

| model | F1 | ROC-AUC | Brier | ECE |
|---|---|---|---|---|
| baseline (LogReg) | 0.986 | 0.996 | 0.023 | 0.038 |
| raw HistGB | 0.958 | 0.991 | 0.036 | 0.042 |
| calibrated HistGB (Platt) | 0.966 | 0.991 | 0.034 | 0.063 |

Honest read: on this small, nearly linearly separable dataset the logistic-regression
baseline beats the boosted model, which is exactly why the baseline is in the pipeline.
Calibration lowered the Brier score but ECE went up; with 114 test rows, 10-bin ECE is
noisy (≈11 samples per bin), so Brier/log loss are the more reliable calibration signals
here. The gate therefore does not require beating the baseline (`GATE` in `src/config.py`).

## Layout

    Lab2_Anuj/
    ├── conftest.py
    ├── requirements.txt
    ├── src/  config.py data.py train.py calibrate.py evaluate.py register.py dashboard.py
    ├── tests/test_pipeline.py
    ├── models/    registry.json + last 5 versioned .joblib pairs (written by CI)
    ├── metrics/   one JSON per version (written by CI)
    └── reports/   dashboard.html, latest_summary.md (written by CI)

## Running locally

    python3 -m venv lab_02 && source lab_02/bin/activate
    pip install -r requirements.txt
    pytest tests -q
    python -m src.train && python -m src.calibrate && python -m src.evaluate
    python -m src.register && python -m src.dashboard
    open reports/dashboard.html

## CI

- `.github/workflows/anuj_lab2_train_calibrate.yml`
