
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss
)


DEFAULT_CONFIG = {
    "minimum_batch": 200,
    "minimum_fraud": 20,
    "minimum_legitimate": 20,
    "minimum_blocks_for_precision": 20,

    "psi_warning": 0.10,
    "psi_major": 0.25,

    "missing_increase_warning": 0.05,
    "missing_increase_critical": 0.20,

    "authentication_missing_critical": 0.30,

    "unknown_warning": 0.05,
    "unknown_critical": 0.20,

    "ap_relative_drop": 0.20,
    "brier_increase": 0.02,

    "capture_drop": 0.10,
    "legitimate_flag_increase": 0.10,

    "block_precision_target": 0.50,
    "tail_warning": 0.03
}


def _valid_probabilities(values, n):
    """Validate probability values."""

    p = np.asarray(values, dtype=float)

    if (
        p.shape != (n,)
        or not np.isfinite(p).all()
        or not ((p >= 0) & (p <= 1)).all()
    ):
        raise ValueError(
            "Invalid probability vector."
        )

    return p


def _counts(series, spec):
    """Count observations using fixed reference groups."""

    s = pd.Series(series)

    missing = s.isna().to_numpy()

    if spec["kind"] == "numeric":

        values = pd.to_numeric(
            s,
            errors="raise"
        ).to_numpy(
            dtype=float,
            copy=True
        )

        if np.isinf(values).any():
            raise ValueError(
                f"Infinite values in {s.name}"
            )

        cuts = np.asarray(
            spec["cuts"],
            dtype=float
        )

        n_normal = len(cuts) + 1

        codes = np.searchsorted(
            cuts,
            values,
            side="left"
        ).copy()

        codes[missing] = n_normal

        k = n_normal + 1

        unknown = 0

    else:

        categories = spec["categories"]

        mapping = {
            value: i
            for i, value in enumerate(categories)
        }

        n_normal = len(categories)

        # Important correction:
        # copy=True ensures the array is writable.
        codes = (
            s.map(mapping)
            .fillna(n_normal + 1)
            .to_numpy(
                dtype=int,
                copy=True
            )
        )

        codes[missing] = n_normal

        k = n_normal + 2

        unknown = int(
            (codes == n_normal + 1).sum()
        )

    return (
        np.bincount(codes, minlength=k),
        int(missing.sum()),
        unknown
    )


def _make_spec(s, n_bins):
    """Create fixed reference groups for one feature."""

    if pd.api.types.is_numeric_dtype(s):

        observed = pd.to_numeric(
            s.dropna(),
            errors="raise"
        ).to_numpy(
            dtype=float,
            copy=True
        )

        if (
            not len(observed)
            or not np.isfinite(observed).all()
        ):
            raise ValueError(
                f"Invalid numeric reference: {s.name}"
            )

        cuts = np.unique(
            np.quantile(
                observed,
                np.arange(1, n_bins) / n_bins
            )
        )

        spec = {
            "kind": "numeric",
            "cuts": cuts.tolist()
        }

        if s.name in {
            "transaction_amount",
            "distance_from_usual_location_km"
        }:

            spec["upper_tail_cut"] = float(
                np.quantile(observed, 0.99)
            )

    else:

        spec = {
            "kind": "categorical",

            "categories": sorted(
                s.dropna().unique().tolist()
            )
        }

    counts, missing, _ = _counts(s, spec)

    return {
        **spec,

        "counts": counts.tolist(),

        "missing_rate": missing / len(s)
    }


def population_stability_index(
    reference_counts,
    current_counts,
    alpha=0.5
):
    """Calculate PSI with additive smoothing."""

    a = np.asarray(
        reference_counts,
        dtype=float
    )

    b = np.asarray(
        current_counts,
        dtype=float
    )

    if (
        a.shape != b.shape
        or a.ndim != 1
        or not len(a)
        or alpha <= 0
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or (a < 0).any()
        or (b < 0).any()
        or a.sum() == 0
        or b.sum() == 0
    ):
        raise ValueError(
            "Invalid PSI counts."
        )

    a = (
        (a + alpha)
        / (a.sum() + alpha * len(a))
    )

    b = (
        (b + alpha)
        / (b.sum() + alpha * len(b))
    )

    return float(
        np.sum(
            (b - a) * np.log(b / a)
        )
    )


def build_monitoring_reference(
    X_train,
    validation_probabilities,
    validation_metrics,
    policy,
    n_bins=10
):
    """Create fixed feature and prediction references."""

    if (
        X_train.empty
        or not X_train.columns.is_unique
        or n_bins < 2
    ):
        raise ValueError(
            "Invalid reference data."
        )

    if set(X_train.columns) != set(
        policy["feature_columns"]
    ):
        raise ValueError(
            "Reference feature schema differs from policy."
        )

    p = _valid_probabilities(
        validation_probabilities,
        len(validation_probabilities)
    )

    if not len(p):
        raise ValueError(
            "Empty probability reference."
        )

    keys = [
        "pr_auc",
        "roc_auc",
        "brier",
        "expected_fraud_capture",
        "legitimate_flag_rate",
        "cost_per_transaction",
        "saving_pct",
        "review_rate",
        "block_rate"
    ]

    return {
        "version": "1.0",

        "input_reference": "Training",

        "score_reference": "Threshold Validation",

        "reference_n": len(X_train),

        "score_reference_n": len(p),

        "features": {
            column: _make_spec(
                X_train[column],
                n_bins
            )
            for column in X_train.columns
        },

        "score": _make_spec(
            pd.Series(
                p,
                name="fraud_probability"
            ),
            n_bins
        ),

        "baseline_metrics": {
            key: float(validation_metrics[key])
            for key in keys
        },

        "policy": policy,

        "config": DEFAULT_CONFIG.copy()
    }


def _drift_row(
    name,
    s,
    spec,
    config,
    sufficient
):
    """Calculate drift statistics for one feature."""

    counts, missing, unknown = _counts(
        s,
        spec
    )

    psi = population_stability_index(
        spec["counts"],
        counts
    )

    if not sufficient:

        level = "insufficient_data"

    elif psi >= config["psi_major"]:

        level = "major"

    elif psi >= config["psi_warning"]:

        level = "warning"

    else:

        level = "stable"

    tail_rate = (
        float(
            (s > spec["upper_tail_cut"]).sum()
            / len(s)
        )

        if "upper_tail_cut" in spec

        else np.nan
    )

    return {
        "feature": name,

        "psi": psi,

        "drift_level": level,

        "reference_missing_rate": (
            spec["missing_rate"]
        ),

        "current_missing_rate": missing / len(s),

        "missing_increase": (
            missing / len(s)
            - spec["missing_rate"]
        ),

        "unknown_rate": unknown / len(s),

        "upper_tail_cut": (
            spec.get("upper_tail_cut", np.nan)
        ),

        "upper_tail_rate": tail_rate
    }


def monitor_batch(
    X,
    probabilities,
    reference,
    y_true=None
):
    """Monitor data, predictions and operational performance."""

    if (
        X.empty
        or not X.columns.is_unique
        or set(X.columns) != set(reference["features"])
    ):
        raise ValueError(
            "Invalid monitoring batch schema."
        )

    n = len(X)

    p = _valid_probabilities(
        probabilities,
        n
    )

    cfg = reference["config"]
    policy = reference["policy"]

    baseline = reference[
        "baseline_metrics"
    ]

    sufficient = (
        n >= cfg["minimum_batch"]
    )

    rows = [
        _drift_row(
            column,
            X[column],
            spec,
            cfg,
            sufficient
        )
        for column, spec in
        reference["features"].items()
    ]

    rows.append(
        _drift_row(
            "fraud_probability",
            pd.Series(p),
            reference["score"],
            cfg,
            sufficient
        )
    )

    drift = pd.DataFrame(rows).sort_values(
        "psi",
        ascending=False
    ).reset_index(drop=True)

    alerts = []

    def alert(
        level,
        metric,
        value,
        limit,
        action
    ):

        alerts.append({

            "severity": level,

            "metric": metric,

            "value": float(value),

            "limit": float(limit),

            "recommended_action": action
        })

    if not sufficient:

        alert(
            "info",
            "batch_size",
            n,
            cfg["minimum_batch"],
            "Accumulate a larger monitoring window."
        )

    # Feature and probability drift
    for row in rows:

        name = row["feature"]

        if (
            sufficient
            and row["psi"] >= cfg["psi_major"]
        ):

            alert(
                "major",
                f"{name}:PSI",
                row["psi"],
                cfg["psi_major"],
                "Investigate significant distribution drift."
            )

        elif (
            sufficient
            and row["psi"] >= cfg["psi_warning"]
        ):

            alert(
                "warning",
                f"{name}:PSI",
                row["psi"],
                cfg["psi_warning"],
                "Review feature distribution."
            )

        # Extreme value monitoring
        if (
            sufficient
            and row["upper_tail_rate"]
            > cfg["tail_warning"]
        ):

            alert(
                "warning",
                f"{name}:upper_tail_rate",
                row["upper_tail_rate"],
                cfg["tail_warning"],
                "Investigate extreme observations."
            )

        # Missing value monitoring
        if (
            row["missing_increase"]
            >= cfg["missing_increase_critical"]
        ):

            alert(
                "critical",
                f"{name}:missing_increase",
                row["missing_increase"],
                cfg["missing_increase_critical"],
                "Pause automatic blocking and investigate feed."
            )

        elif (
            sufficient
            and row["missing_increase"]
            >= cfg["missing_increase_warning"]
        ):

            alert(
                "warning",
                f"{name}:missing_increase",
                row["missing_increase"],
                cfg["missing_increase_warning"],
                "Investigate missing values."
            )

        # Unknown category monitoring
        if (
            row["unknown_rate"]
            >= cfg["unknown_critical"]
        ):

            alert(
                "critical",
                f"{name}:unknown_rate",
                row["unknown_rate"],
                cfg["unknown_critical"],
                "Pause automatic blocking and investigate schema."
            )

        elif (
            sufficient
            and row["unknown_rate"]
            >= cfg["unknown_warning"]
        ):

            alert(
                "warning",
                f"{name}:unknown_rate",
                row["unknown_rate"],
                cfg["unknown_warning"],
                "Investigate new categories."
            )

    # Critical authentication outage detection
    if "authentication_level" in X:

        auth_missing = X[
            "authentication_level"
        ].isna().mean()

        if (
            auth_missing
            >= cfg["authentication_missing_critical"]
        ):

            alert(
                "critical",
                "authentication_missing_rate",
                auth_missing,
                cfg["authentication_missing_critical"],
                "Suspend automatic blocking and restore feed."
            )

    # Frozen operational policy
    d = np.where(
        p >= policy["block_threshold"],
        2,
        np.where(
            p >= policy["review_threshold"],
            1,
            0
        )
    )

    review_rate = float(
        (d == 1).mean()
    )

    block_rate = float(
        (d == 2).mean()
    )

    kpis = {
        "transactions": n,

        "mean_probability": float(p.mean()),

        "score_psi": rows[-1]["psi"],

        "review_rate": review_rate,

        "block_rate": block_rate,

        "review_count": int((d == 1).sum()),

        "block_count": int((d == 2).sum()),

        "label_status": "not_supplied",

        "pr_auc": np.nan,
        "roc_auc": np.nan,
        "brier": np.nan,

        "expected_fraud_capture": np.nan,

        "legitimate_flag_rate": np.nan,

        "block_precision": np.nan,

        "cost_per_transaction": np.nan,

        "saving_pct": np.nan
    }

    # Operational capacity monitoring
    if review_rate > policy["max_review_rate"]:

        alert(
            "critical",
            "review_rate",
            review_rate,
            policy["max_review_rate"],
            "Escalate queue overload; use approved overflow procedure."
        )

    if block_rate > policy["max_block_rate"]:

        alert(
            "critical",
            "block_rate",
            block_rate,
            policy["max_block_rate"],
            "Suspend automatic blocking and investigate."
        )

    # Performance monitoring when mature labels exist
    if y_true is not None:

        if (
            isinstance(y_true, pd.Series)
            and not X.index.equals(y_true.index)
        ):
            raise ValueError(
                "Labels and features must be aligned."
            )

        y = np.asarray(y_true)

        if (
            y.shape != (n,)
            or not np.isin(y, [0, 1]).all()
        ):
            raise ValueError(
                "Expected complete mature binary labels."
            )

        y = y.astype(int)

        fraud = int(y.sum())

        legit = n - fraud

        kpis["fraud_count"] = fraud
        kpis["fraud_prevalence"] = fraud / n

        if (
            sufficient
            and fraud >= cfg["minimum_fraud"]
            and legit >= cfg["minimum_legitimate"]
        ):

            kpis["label_status"] = "evaluated"

            c = policy["cost_assumptions"]

            q = c["review_effectiveness"]
            loss = c["fraud_loss"]

            matrix = np.array([
                [
                    0,
                    loss
                ],
                [
                    c["review_fee"],
                    c["review_fee"] + (1 - q) * loss
                ],
                [
                    c["block_fee"]
                    + c["customer_friction"],

                    c["block_fee"]
                ]
            ])

            total = float(
                matrix[d, y].sum()
            )

            prevented = (
                ((y == 1) & (d == 2)).sum()
                + q * ((y == 1) & (d == 1)).sum()
            )

            kpis.update({

                "pr_auc": float(
                    average_precision_score(y, p)
                ),

                "roc_auc": float(
                    roc_auc_score(y, p)
                ),

                "brier": float(
                    brier_score_loss(y, p)
                ),

                "expected_fraud_capture": (
                    float(prevented) / fraud
                ),

                "legitimate_flag_rate": float(
                    ((y == 0) & (d != 0)).sum()
                    / legit
                ),

                "block_precision": (
                    float(y[d == 2].mean())
                    if (d == 2).any()
                    else np.nan
                ),

                "cost_per_transaction": total / n,

                "saving_pct": (
                    100 * (
                        1 - total / (loss * fraud)
                    )
                    if loss > 0
                    else np.nan
                )
            })

            ap_limit = (
                baseline["pr_auc"]
                * (1 - cfg["ap_relative_drop"])
            )

            brier_limit = (
                baseline["brier"]
                + cfg["brier_increase"]
            )

            capture_limit = (
                baseline["expected_fraud_capture"]
                - cfg["capture_drop"]
            )

            flag_limit = (
                baseline["legitimate_flag_rate"]
                + cfg["legitimate_flag_increase"]
            )

            checks = [
                (
                    "pr_auc",
                    kpis["pr_auc"] < ap_limit,
                    ap_limit
                ),
                (
                    "brier",
                    kpis["brier"] > brier_limit,
                    brier_limit
                ),
                (
                    "expected_fraud_capture",
                    kpis["expected_fraud_capture"]
                    < capture_limit,
                    capture_limit
                ),
                (
                    "legitimate_flag_rate",
                    kpis["legitimate_flag_rate"]
                    > flag_limit,
                    flag_limit
                )
            ]

            for key, condition, limit in checks:

                if condition:

                    alert(
                        "major",
                        key,
                        kpis[key],
                        limit,
                        "Investigate performance degradation."
                    )

            if kpis["saving_pct"] < 0:

                alert(
                    "critical",
                    "saving_pct",
                    kpis["saving_pct"],
                    0,
                    "Reassess business value and cost assumptions."
                )

            if (
                kpis["block_count"]
                >= cfg["minimum_blocks_for_precision"]
                and kpis["block_precision"]
                < cfg["block_precision_target"]
            ):

                alert(
                    "critical",
                    "block_precision",
                    kpis["block_precision"],
                    cfg["block_precision_target"],
                    "Suspend automatic blocking."
                )

            elif (
                0 < kpis["block_count"]
                < cfg["minimum_blocks_for_precision"]
            ):

                alert(
                    "info",
                    "block_count",
                    kpis["block_count"],
                    cfg["minimum_blocks_for_precision"],
                    "Insufficient support for stable block precision."
                )

        else:

            kpis["label_status"] = (
                "insufficient_support"
            )

            alert(
                "info",
                "mature_label_support",
                fraud,
                cfg["minimum_fraud"],
                "Collect more mature labeled observations."
            )

    alert_columns = [
        "severity",
        "metric",
        "value",
        "limit",
        "recommended_action"
    ]

    alerts_df = pd.DataFrame(
        alerts,
        columns=alert_columns
    )

    severities = set(
        alerts_df["severity"]
    )

    if "critical" in severities:

        kpis["status"] = "PAUSE_OR_ESCALATE"

    elif severities & {"warning", "major"}:

        kpis["status"] = "REVIEW"

    elif not sufficient:

        kpis["status"] = "INSUFFICIENT_DATA"

    else:

        kpis["status"] = "NO_ALERT"

    return drift, kpis, alerts_df
