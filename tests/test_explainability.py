
import numpy as np
import pandas as pd

from src.explainability import (
    FraudSHAPExplainer,
    summarize_local_explanation
)


class MockProbabilityModel:

    def predict_proba(self, X):

        p = (
            0.1
            + 0.3 * X["risk"].fillna(0)
            + 0.2 * X["channel"].eq("Online")
        ).to_numpy()

        return np.column_stack([
            1 - p,
            p
        ])


def test_shap_additivity():

    X = pd.DataFrame({
        "risk": [0.1, 0.2, 0.8, np.nan],
        "channel": ["POS", "Online", "Online", "POS"]
    })

    model = MockProbabilityModel()

    explainer = FraudSHAPExplainer(
        model,
        X,
        background_size=4,
        random_state=42,
        permutations=2
    )

    values, baseline = explainer.explain(X)

    actual = baseline + values.sum(axis=1)

    np.testing.assert_allclose(
        actual,
        model.predict_proba(X)[:, 1],
        atol=1e-7
    )


def test_local_summary():

    values = pd.Series({
        "risk": 0.10,
        "channel": -0.02
    })

    transaction = pd.Series({
        "risk": 0.8,
        "channel": "POS"
    })

    summary = summarize_local_explanation(
        values,
        transaction
    )

    assert "risk=" in summary[
        "risk_increasing_factors"
    ]

    assert "channel=" in summary[
        "risk_decreasing_factors"
    ]
