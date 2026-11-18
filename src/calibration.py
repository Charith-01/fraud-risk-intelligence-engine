
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator

from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    average_precision_score,
    roc_auc_score
)


def fit_calibrator(
    fitted_pipeline,
    X_calibration,
    y_calibration,
    method="sigmoid"
):
    """Calibrate an already-trained model without refitting it."""

    if method not in ["sigmoid", "isotonic"]:
        raise ValueError(
            f"Unsupported calibration method: {method}"
        )

    # Record original predictions
    before = fitted_pipeline.predict_proba(
        X_calibration
    )

    calibrator = CalibratedClassifierCV(
        estimator=FrozenEstimator(fitted_pipeline),
        method=method,
        ensemble=False
    )

    calibrator.fit(
        X_calibration,
        y_calibration
    )

    # Ensure original model predictions remain unchanged
    after = fitted_pipeline.predict_proba(
        X_calibration
    )

    np.testing.assert_allclose(before, after)

    return calibrator


def reliability_table(
    y_true,
    probabilities,
    n_bins=5
):
    """Calculate calibration statistics using quantile bins."""

    data = pd.DataFrame({
        "actual": np.asarray(y_true),
        "probability": np.asarray(probabilities)
    })

    # Quantile bins handle concentrated fraud probabilities
    data["bin"] = pd.qcut(
        data["probability"],
        q=n_bins,
        duplicates="drop"
    )

    # Handle constant probability predictions
    if data["bin"].isna().all():
        data["bin"] = "All observations"

    summary = data.groupby(
        "bin",
        observed=True
    ).agg(
        transactions=("actual", "size"),
        fraud_count=("actual", "sum"),
        mean_probability=("probability", "mean"),
        observed_fraud_rate=("actual", "mean")
    ).reset_index()

    summary["calibration_gap"] = (
        summary["mean_probability"]
        - summary["observed_fraud_rate"]
    ).abs()

    summary["weight"] = (
        summary["transactions"] / len(data)
    )

    return summary


def evaluate_calibration(
    y_true,
    probabilities,
    n_bins=5
):
    """Evaluate probability reliability and discrimination."""

    probabilities = np.asarray(probabilities)

    table = reliability_table(
        y_true,
        probabilities,
        n_bins=n_bins
    )

    ece = (
        table["weight"]
        * table["calibration_gap"]
    ).sum()

    return {
        "brier": brier_score_loss(
            y_true, probabilities
        ),

        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1]
        ),

        "ece": float(ece),

        "pr_auc": average_precision_score(
            y_true, probabilities
        ),

        "roc_auc": roc_auc_score(
            y_true, probabilities
        ),

        "mean_probability": float(
            probabilities.mean()
        ),

        "observed_fraud_rate": float(
            np.mean(y_true)
        ),

        "min_probability": float(
            probabilities.min()
        ),

        "max_probability": float(
            probabilities.max()
        )
    }


def plot_calibration_curves(
    y_true,
    probability_sets,
    title,
    save_path,
    n_bins=5
):
    """Compare probability reliability across methods."""

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect calibration"
    )

    colors = {
        "uncalibrated": "tomato",
        "sigmoid": "steelblue",
        "isotonic": "seagreen"
    }

    all_tables = []

    for method, probabilities in (
        probability_sets.items()
    ):

        table = reliability_table(
            y_true,
            probabilities,
            n_bins=n_bins
        )

        ax.plot(
            table["mean_probability"],
            table["observed_fraud_rate"],
            marker="o",
            linewidth=2,
            color=colors.get(method),
            label=method
        )

        all_tables.append(table)

    max_observed = max(
        max(
            table["mean_probability"].max(),
            table["observed_fraud_rate"].max()
        )
        for table in all_tables
    )

    upper = min(
        1.0,
        max(0.3, max_observed + 0.05)
    )

    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)

    ax.set_title(title)

    ax.set_xlabel(
        "Mean Predicted Fraud Probability"
    )

    ax.set_ylabel(
        "Observed Fraud Rate"
    )

    ax.set_aspect("equal", adjustable="box")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)
