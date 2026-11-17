
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


ENGINEERED_FEATURES = [
    "velocity_share",
    "velocity_excess",
    "log_amount",
    "amount_to_merchant_median",
    "log_distance",
    "combined_risk_mean",
    "risk_product",
    "hour_sin",
    "hour_cos",
    "is_night",
    "day_sin",
    "day_cos",
    "is_young_account",
    "new_device_distance",
    "amount_merchant_risk",
    "ecommerce_missing_auth",
    "failed_login_velocity",
    "new_device_billing_mismatch"
]


class FraudFeatureEngineer(BaseEstimator, TransformerMixin):
    """Leakage-aware transaction feature engineering transformer."""

    def fit(self, X, y=None):

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        forbidden = {
            "fraud", "transaction_id", "dataset"
        }

        if forbidden & set(X.columns):
            raise ValueError(
                "Remove identifiers, target and split labels."
            )

        if set(ENGINEERED_FEATURES) & set(X.columns):
            raise ValueError(
                "Expected original predictors only."
            )

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object
        )

        self.n_features_in_ = X.shape[1]

        # Learn amount statistics from training data only
        self.global_amount_median_ = float(
            X["transaction_amount"].median()
        )

        if (
            not np.isfinite(self.global_amount_median_)
            or self.global_amount_median_ <= 0
        ):
            raise ValueError(
                "A positive training amount median is required."
            )

        medians = X.groupby(
            "merchant_category"
        )["transaction_amount"].median()

        self.merchant_amount_medians_ = medians[
            medians.gt(0) & np.isfinite(medians)
        ].to_dict()

        return self

    def transform(self, X):

        check_is_fitted(
            self,
            "merchant_amount_medians_"
        )

        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expected a pandas DataFrame.")

        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError(
                "Input feature names or order changed."
            )

        # Preserve the original input
        out = X.copy()

        h1 = out["transactions_last_1h"]
        h24 = out["transactions_last_24h"]

        amount = out["transaction_amount"]
        distance = out["distance_from_usual_location_km"]

        merchant = out["merchant_risk_score"]
        ip = out["ip_risk_score"]

        hour = out["hour"]
        day = out["day_of_week"]
        age = out["account_age_days"]
        new = out["new_device"]

        # 1. Transaction velocity ratio
        out["velocity_share"] = h1 / (h24 + 1)

        # 2. Behavioral velocity deviation
        out["velocity_excess"] = (
            h1 - (h24 - h1) / 23
        )

        # 3. Log-transformed transaction amount
        out["log_amount"] = np.log1p(amount)

        # 4. Merchant-relative amount normalization
        reference = out["merchant_category"].map(
            self.merchant_amount_medians_
        ).fillna(
            self.global_amount_median_
        )

        out["amount_to_merchant_median"] = (
            amount / reference
        )

        # 5. Log-transformed distance
        out["log_distance"] = np.log1p(distance)

        # 6. Combined average risk
        out["combined_risk_mean"] = (
            merchant + ip
        ) / 2

        # 7. Combined risk interaction
        out["risk_product"] = merchant * ip

        # 8-9. Cyclical hour features
        out["hour_sin"] = np.sin(
            2 * np.pi * hour / 24
        )

        out["hour_cos"] = np.cos(
            2 * np.pi * hour / 24
        )

        # 10. Nighttime indicator: 00:00-05:59
        out["is_night"] = (
            hour.between(0, 5)
            .astype(float)
            .where(hour.notna())
        )

        # 11-12. Cyclical day-of-week features
        out["day_sin"] = np.sin(
            2 * np.pi * day / 7
        )

        out["day_cos"] = np.cos(
            2 * np.pi * day / 7
        )

        # 13. Young-account indicator
        out["is_young_account"] = (
            age.le(90)
            .astype(float)
            .where(age.notna())
        )

        # 14. New device and distance interaction
        out["new_device_distance"] = (
            new * distance
        )

        # 15. Amount and merchant risk interaction
        out["amount_merchant_risk"] = (
            out["log_amount"] * merchant
        )

        # 16. E-commerce and missing authentication
        channel = out["channel"]

        out["ecommerce_missing_auth"] = (
            channel.eq("E-commerce").fillna(False)
            & out["authentication_level"].isna()
        ).astype(float).where(
            channel.notna()
        )

        # 17. Failed login and transaction velocity
        out["failed_login_velocity"] = (
            out["failed_logins_24h"] * h1
        )

        # 18. New device and billing mismatch
        out["new_device_billing_mismatch"] = (
            new * out["billing_address_mismatch"]
        )

        return out

    def get_feature_names_out(self, input_features=None):

        check_is_fitted(
            self,
            "feature_names_in_"
        )

        if (
            input_features is not None
            and list(input_features)
            != list(self.feature_names_in_)
        ):
            raise ValueError(
                "Input feature names changed."
            )

        return np.concatenate([
            self.feature_names_in_,
            np.asarray(
                ENGINEERED_FEATURES,
                dtype=object
            )
        ])
