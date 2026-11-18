
import numpy as np

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss
)


def evaluate_binary_predictions(
    y_true,
    probabilities,
    threshold=0.50
):
    """Evaluate binary fraud predictions."""

    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    return {
        "pr_auc": average_precision_score(
            y_true, probabilities
        ),

        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),

        "precision": precision_score(
            y_true, predictions,
            zero_division=0
        ),

        "recall": recall_score(
            y_true, predictions,
            zero_division=0
        ),

        "f1": f1_score(
            y_true, predictions,
            zero_division=0
        ),

        "fpr": fp / (fp + tn)
        if (fp + tn) > 0 else 0,

        "brier": brier_score_loss(
            y_true, probabilities
        ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }
