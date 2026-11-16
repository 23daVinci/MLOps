"""Central configuration. Every script imports from here so that values such as the
seed or the tracking URI live in exactly one place."""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"

# --- MLflow ---------------------------------------------------------------
# A tracking *server* with a SQL backend (see README). The registry needs this.
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "adult-income"
# One registered model per business problem; versions may be different algorithms.
REGISTERED_MODEL_NAME = "adult-income-classifier"
CHAMPION_ALIAS = "champion"
PREVIOUS_ALIAS = "previous-champion"  # what a rollback restores
CHALLENGER_ALIAS = "challenger"       # a candidate under evaluation (shadow comparison, Stage 3)
# Environment promotion: a second registered model holds only what is approved for production.
PROD_REGISTERED_MODEL_NAME = "adult-income-classifier-prod"

# --- Promotion policy -------------------------------------------------------
# A challenger must beat the champion's val AUC by at least this much to be promoted.
PROMOTION_MARGIN = 0.005
# Registering is not free (a crowded registry stops being a catalog): a new run of a model family that
# already has a registered version is only registered if it improves on the best one by this much.
MIN_REGISTER_GAIN = 0.002
# Operational budgets checked before promotion. Overridable by env var so tests can force a failure.
LATENCY_P95_MS_BUDGET = float(os.environ.get("PROMOTE_LATENCY_P95_MS", 50))   # single-row, in-process
MODEL_SIZE_MB_BUDGET = float(os.environ.get("PROMOTE_MODEL_SIZE_MB", 50))

# --- Data -----------------------------------------------------------------
SEED = 42
TARGET = "income_gt_50k"
# fnlwgt is a census sampling weight, not a property of the person: drop it.
DROP_COLUMNS = ["fnlwgt"]
# 60 / 20 / 20 split. Test is held out until the final evaluation stage.
VAL_SIZE = 0.2
TEST_SIZE = 0.2
