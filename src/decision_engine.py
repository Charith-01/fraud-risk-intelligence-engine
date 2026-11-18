
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.cost_analysis import (
    decision_codes,
    create_cost_matrix
)


DECISION_NAMES = np.array([
    "Approve",
    "Manual Review",
    "Block"
])


CATEGORICAL_COLUMNS = {
    "card_presence",
    "channel",
    "country",
    "device_type",
    "merchant_category",
    "authentication_level",
    "customer_tier"
}


INTEGER_COLUMNS = {
    "hour",
    "day_of_week",
    "transactions_last_1h",
    "transactions_last_24h",
    "account_age_days",
    "failed_logins_24h",
    "prior_chargebacks",
    "device_age_days",
    "email_age_days",
    "new_device",
    "billing_address_mismatch"
}


class FraudDecisionEngine:
    """Generate fraud probabilities and operational decisions."""

    def __init__(self, model, policy):

        self.model = model
        self.policy = policy

        self.feature_columns = list(
            policy["feature_columns"]
        )

        if (
            len(set(self.feature_columns))
            != len(self.feature_columns)
            or {"fraud", "transaction_id", "dataset"}
            & set(self.feature_columns)
        ):
            raise ValueError("Invalid predictor schema.")

        self.review_threshold = float(
            policy["review_threshold"]
        )

        self.block_threshold = float(
            policy["block_threshold"]
        )

        self.cost_matrix = create_cost_matrix(
            policy["cost_assumptions"]
        )

        # Validate thresholds
        decision_codes(
            [0.0, 1.0],
            self.review_threshold,
            self.block_threshold
        )

        if list(model.classes_) != [0, 1]:
            raise ValueError("Unexpected model classes.")

    @classmethod
    def load(cls, models_dir):
        """Load trusted model and decision configuration."""

        models_dir = Path(models_dir)

        model = joblib.load(
            models_dir / "04_calibrated_model.joblib"
        )

        with open(
            models_dir / "decision_policy.json",
            encoding="utf-8"
        ) as file:

            policy = json.load(file)

        return cls(model, policy)

    def validate_input(self, transactions):
        """Validate raw transaction input before scoring."""

        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        if transactions.empty:
            raise ValueError("No transactions supplied.")

        if not transactions.columns.is_unique:
            raise ValueError("Duplicate input columns.")

        required = set(
            self.feature_columns + ["transaction_id"]
        )

        missing = required - set(transactions.columns)
        unexpected = set(transactions.columns) - required

        if missing:
            raise ValueError(
                f"Missing columns: {sorted(missing)}"
            )

        if unexpected:
            raise ValueError(
                f"Unexpected columns: {sorted(unexpected)}"
            )

        ids = transactions["transaction_id"]

        if (
            ids.isna().any()
            or not ids.is_unique
            or ids.astype(str).str.strip().eq("").any()
        ):
            raise ValueError("Invalid transaction identifiers.")

        X = transactions[
            self.feature_columns
        ].copy()

        for column in X.columns:

            if column in CATEGORICAL_COLUMNS:

                values = X[column].dropna()

                if not values.map(
                    lambda value: isinstance(value, str)
                ).all():
                    raise ValueError(
                        f"Invalid category type: {column}"
                    )

                X[column] = (
                    X[column].astype(object)
                    .where(X[column].notna(), np.nan)
                )

            else:

                X[column] = pd.to_numeric(
                    X[column],
                    errors="raise"
                ).astype(float)

                values = X[column].dropna()

                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Infinite value: {column}"
                    )

                if (values < 0).any():
                    raise ValueError(
                        f"Negative value: {column}"
                    )

                if column in INTEGER_COLUMNS:

                    if (values % 1 != 0).any():
                        raise ValueError(
                            f"Non-integer value: {column}"
                        )

        range_rules = {
            "hour": (0, 23),
            "day_of_week": (0, 6),
            "merchant_risk_score": (0, 1),
            "ip_risk_score": (0, 1),
            "new_device": (0, 1),
            "billing_address_mismatch": (0, 1)
        }

        for column, (lower, upper) in range_rules.items():

            if column in X.columns:

                if not X[column].dropna().between(
                    lower, upper
                ).all():
                    raise ValueError(
                        f"Out-of-range value: {column}"
                    )

        if "transaction_amount" in X.columns:

            if (
                X["transaction_amount"].dropna() <= 0
            ).any():
                raise ValueError(
                    "Transaction amount must be positive."
                )

        if {
            "transactions_last_1h",
            "transactions_last_24h"
        }.issubset(X.columns):

            if (
                X["transactions_last_1h"]
                > X["transactions_last_24h"]
            ).any():
                raise ValueError(
                    "Inconsistent transaction velocity."
                )

        if X.isna().all(axis=1).any():
            raise ValueError(
                "Transaction has no available predictors."
            )

        return X

    def score_transactions(self, transactions):
        """Return calibrated risk, decision and expected cost."""

        X = self.validate_input(transactions)

        probabilities = self.model.predict_proba(
            X
        )[:, 1]

        codes = decision_codes(
            probabilities,
            self.review_threshold,
            self.block_threshold
        )

        expected_cost = (
            (1 - probabilities)
            * self.cost_matrix[codes, 0]
            + probabilities
            * self.cost_matrix[codes, 1]
        )

        reasons = np.select(
            [
                codes == 0,
                codes == 1,
                codes == 2
            ],
            [
                "Below review threshold",
                "Between review and block thresholds",
                "At or above block threshold"
            ],
            default="Unknown"
        )

        return pd.DataFrame({
            "transaction_id": (
                transactions["transaction_id"].to_numpy()
            ),

            "fraud_probability": probabilities,

            "decision": DECISION_NAMES[codes],

            "decision_reason": reasons,

            "expected_cost": expected_cost,

            "missing_feature_count": (
                X.isna().sum(axis=1).to_numpy()
            )
        }, index=transactions.index)

    def create_review_queue(self, scored):
        """Prioritize manual-review cases by fraud probability."""

        queue = scored.loc[
            scored["decision"] == "Manual Review"
        ].copy()

        queue = queue.sort_values(
            ["fraud_probability", "transaction_id"],
            ascending=[False, True]
        ).reset_index(drop=True)

        queue.insert(
            0,
            "review_priority",
            np.arange(1, len(queue) + 1)
        )

        return queue

    def capacity_status(self, scored):
        """Check batch workloads against configured limits."""

        if scored.empty:
            raise ValueError("No scored transactions.")

        review_rate = (
            scored["decision"] == "Manual Review"
        ).mean()

        block_rate = (
            scored["decision"] == "Block"
        ).mean()

        review_limit = self.policy["max_review_rate"]
        block_limit = self.policy["max_block_rate"]

        return {
            "transactions": len(scored),

            "review_rate": float(review_rate),

            "block_rate": float(block_rate),

            "review_capacity_exceeded": bool(
                review_rate > review_limit
            ),

            "block_capacity_exceeded": bool(
                block_rate > block_limit
            )
        }
