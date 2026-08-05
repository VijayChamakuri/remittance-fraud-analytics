"""Central paths and constants for the pipeline."""
from __future__ import annotations
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(ROOT, "data")
REPORTS_DIR = os.path.join(ROOT, "reports")
MODELS_DIR = os.path.join(ROOT, "models")
DASH_DIR = os.path.join(ROOT, "dashboard")

RAW_CSV = os.path.join(DATA_DIR, "synthetic_creditcard.csv")
PROCESSED_CSV = os.path.join(DATA_DIR, "processed_features.parquet")

# Time-based (out-of-time) split boundaries as fractions of the Time axis
TRAIN_FRAC = 0.60
VALID_FRAC = 0.70   # valid = [TRAIN_FRAC, VALID_FRAC); test/OOT = [VALID_FRAC, 1.0]

TARGET = "Class"
STRUCTURING_THRESHOLD = 1000.0
SEED = 42

# Business cost assumptions for threshold selection (documented, editable).
# These are ILLUSTRATIVE remittance-style unit economics, not Remitly figures.
#   * A missed fraud costs ~ the transaction amount (avg ~$161 in this data).
#   * A false positive costs ~$4 of review/step-up friction.
#   -> implied cost ratio ~ 40 : 1 (missed fraud : false positive). At that ratio
#      it is rational to accept many false positives to avoid one missed fraud,
#      which is exactly why the cost-minimizing threshold lands at HIGH RECALL.
COST_FALSE_POSITIVE = 4.0     # avg cost of a false decline / step-up friction ($)
COST_FALSE_NEGATIVE_MULT = 1.0  # missed fraud costs ~ the transaction amount ($)

for _d in (DATA_DIR, REPORTS_DIR, MODELS_DIR, DASH_DIR):
    os.makedirs(_d, exist_ok=True)
for _sub in ("eda", "evaluation", "rca", "ab_test"):
    os.makedirs(os.path.join(REPORTS_DIR, _sub), exist_ok=True)
