
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    log_loss
)

from src.cost_analysis import (
    policy_metrics,
    decision_codes,
    create_cost_matrix
)

from src.evaluation import evaluate_binary_predictions


def create_stress_scenarios(
    X,
    y,
    random_state=42,
    mix_size=250
):
    """Generate reproducible transaction stress scenarios."""

    if not X.index.is_unique:
        raise ValueError("Expected unique row indexes.")

    if not X.index.equals(y.index):
        raise ValueError("Predictors and labels are misaligned.")

    rng = np.random.default_rng(random_state)

    scenarios = {
        "Clean": (X.copy(), y.copy())
    }

    # 1. Increased transaction amounts
    shifted = X.copy()

    shifted["transaction_amount"] *= 2

    scenarios["Amount x2"] = (
        shifted, y.copy()
    )

    # 2. Higher transaction velocity
    shifted = X.copy()

    shifted["transactions_last_1h"] += 2
    shifted["transactions_last_24h"] += 4

    scenarios["Velocity surge"] = (
        shifted, y.copy()
    )

    # 3. Combined amount and velocity shift
    shifted = X.copy()

    shifted["transaction_amount"] *= 2
    shifted["transactions_last_1h"] += 2
    shifted["transactions_last_24h"] += 4

    scenarios["Combined surge"] = (
        shifted, y.copy()
    )

    # 4. Inject additional missing values
    shifted = X.copy()

    missing_features = [
        "transaction_amount",
        "merchant_risk_score",
        "ip_risk_score",
        "authentication_level",
        "distance_from_usual_location_km"
    ]

    for feature in missing_features:

        indexes = rng.choice(
            shifted.index,
            size=int(0.30 * len(shifted)),
            replace=False
        )

        shifted.loc[indexes, feature] = np.nan

    scenarios["30% missing injection"] = (
        shifted, y.copy()
    )

    # 5. Complete authentication service outage
    shifted = X.copy()

    shifted["authentication_level"] = np.nan

    shifted["authentication_level"] = (
        shifted["authentication_level"].astype(object)
    )

    scenarios["Authentication outage"] = (
        shifted, y.copy()
    )

    # 6. Extreme observations
    shifted = X.copy()

    indexes = rng.choice(
        shifted.index,
        size=max(1, int(0.05 * len(shifted))),
        replace=False
    )

    shifted.loc[
        indexes, "transaction_amount"
    ] *= 10

    shifted.loc[
        indexes, "distance_from_usual_location_km"
    ] *= 10

    scenarios["Extreme 5%"] = (
        shifted, y.copy()
    )

    # 7. Previously unseen categorical values
    shifted = X.copy()

    shifted["device_type"] = "Unseen_Device"

    scenarios["Unseen device"] = (
        shifted, y.copy()
    )

    # 8. Channel-mix distribution shift
    online = X.index[
        X["channel"].eq("E-commerce")
    ]

    other = X.index[
        ~X["channel"].eq("E-commerce")
    ]

    online_n = round(0.70 * mix_size)
    other_n = mix_size - online_n

    if (
        len(online) < online_n
        or len(other) < other_n
    ):
        raise ValueError(
            "Insufficient observations for channel shift."
        )

    indexes = np.concatenate([
        rng.choice(
            online,
            online_n,
            replace=False
        ),

        rng.choice(
            other,
            other_n,
            replace=False
        )
    ])

    scenarios["70% ecommerce"] = (
        X.loc[indexes].copy(),
        y.loc[indexes].copy()
    )

    return scenarios


def evaluate_frozen_policy(y_true, probabilities, policy):
    """Evaluate predictive and operational performance."""

    p = np.asarray(probabilities, dtype=float)

    review = policy["review_threshold"]
    block = policy["block_threshold"]

    metrics = policy_metrics(
        y_true,
        p,
        review,
        block,
        policy["cost_assumptions"],
        policy["max_review_rate"],
        policy["max_block_rate"]
    )

    # Treat review or block as a positive fraud alert
    classification = evaluate_binary_predictions(
        y_true,
        p,
        threshold=review
    )

    return {
        "transactions": len(y_true),

        "fraud_count": int(np.sum(y_true)),

        "fraud_prevalence": float(np.mean(y_true)),

        "pr_auc": classification["pr_auc"],

        "roc_auc": classification["roc_auc"],

        "brier": classification["brier"],

        "log_loss": log_loss(
            y_true, p, labels=[0, 1]
        ),

        "flag_precision": classification["precision"],

        "flag_recall": classification["recall"],

        "flag_f1": classification["f1"],

        "mean_probability": float(p.mean()),

        **metrics
    }


def bootstrap_policy_intervals(
    y_true,
    probabilities,
    policy,
    n_bootstrap=1000,
    random_state=42
):
    """Estimate uncertainty without fitting or tuning the model."""

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)

    if len(y) != len(p):
        raise ValueError("Labels and probabilities are misaligned.")

    d = decision_codes(
        p,
        policy["review_threshold"],
        policy["block_threshold"]
    )

    costs = policy["cost_assumptions"]

    matrix = create_cost_matrix(costs)

    rng = np.random.default_rng(random_state)

    records = []

    for _ in range(n_bootstrap):

        indexes = rng.integers(
            0, len(y), size=len(y)
        )

        actual = y[indexes]
        prob = p[indexes]
        decision = d[indexes]

        if len(np.unique(actual)) < 2:
            continue

        total_cost = matrix[
            decision, actual
        ].sum()

        baseline_cost = (
            costs["fraud_loss"] * actual.sum()
        )

        fraud_reviewed = (
            (actual == 1) & (decision == 1)
        ).sum()

        fraud_blocked = (
            (actual == 1) & (decision == 2)
        ).sum()

        prevented = (
            fraud_blocked
            + costs["review_effectiveness"]
            * fraud_reviewed
        )

        records.append({
            "pr_auc": average_precision_score(
                actual, prob
            ),

            "roc_auc": roc_auc_score(
                actual, prob
            ),

            "brier": brier_score_loss(
                actual, prob
            ),

            "review_rate": (
                decision == 1
            ).mean(),

            "expected_fraud_capture": (
                prevented / actual.sum()
            ),

            "cost_per_transaction": (
                total_cost / len(actual)
            ),

            "saving_pct": (
                100 * (1 - total_cost / baseline_cost)
            )
        })

    results = pd.DataFrame(records)

    intervals = results.quantile(
        [0.025, 0.975]
    ).T

    intervals.columns = [
        "lower_95", "upper_95"
    ]

    return intervals, results
