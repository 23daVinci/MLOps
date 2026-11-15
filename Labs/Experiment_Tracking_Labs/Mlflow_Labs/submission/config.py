"""Central configuration. Every script imports from here so that values such as the
seed or the tracking URI live in exactly one place."""
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
# A challenger must beat the champion's val AUC by at least this much to be promoted.
PROMOTION_MARGIN = 0.005

# --- Data -----------------------------------------------------------------
SEED = 42
TARGET = "income_gt_50k"
# fnlwgt is a census sampling weight, not a property of the person: drop it.
DROP_COLUMNS = ["fnlwgt"]
# 60 / 20 / 20 split. Test is held out until the final evaluation stage.
VAL_SIZE = 0.2
TEST_SIZE = 0.2
