"""Metrics and plots shared by every model we train or evaluate."""
import matplotlib

matplotlib.use("Agg")  # no GUI: we only save figures
import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Threshold-free metrics (auc, avg precision, log loss) and thresholded ones.
    Accuracy alone is misleading with ~76/24 classes, so we log several."""
    pred = (proba >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, proba),
        "avg_precision": average_precision_score(y_true, proba),
        "log_loss": log_loss(y_true, proba),
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
    }


def make_plots(y_true, proba, threshold: float = 0.5) -> dict:
    """Return {filename: matplotlib Figure} for mlflow.log_figure."""
    figs = {}
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    figs["roc_curve.png"] = fig

    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (proba >= threshold).astype(int), ax=ax, colorbar=False
    )
    figs["confusion_matrix.png"] = fig
    return figs
