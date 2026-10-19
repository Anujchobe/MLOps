"""Central config. Paths are anchored to the lab root, not the CWD,
so scripts work whether run from the repo root or the lab folder."""
import os
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = LAB_ROOT / "models"
METRICS_DIR = LAB_ROOT / "metrics"
REPORTS_DIR = LAB_ROOT / "reports"
REGISTRY = MODELS_DIR / "registry.json"
WORK_DIR = LAB_ROOT / ".work"          # scratch handoff between steps (gitignored)

SEED = 42
TEST_SIZE = 0.20      # held-out evaluation set
CALIB_SIZE = 0.20     # fraction of the remainder reserved for calibration
# Platt scaling by default; isotonic tends to overfit on ~90 calibration rows.
# Override per run via the workflow_dispatch input (sets CALIB_METHOD).
CALIB_METHOD = os.environ.get("CALIB_METHOD") or "sigmoid"
ECE_BINS = 10
KEEP_LAST_N_MODELS = 5  # registry keeps full history; binaries pruned to last N

# Quality gate: the pipeline fails (and nothing is committed) if these are not met
GATE = {"f1_min": 0.93, "roc_auc_min": 0.97, "must_beat_baseline_f1": False}
