
import numpy as np
import pandas as pd

from src.robustness import (
    create_stress_scenarios,
    bootstrap_policy_intervals
)

from src.cost_analysis import BASE_COSTS


def create_test_data():

    X = pd.DataFrame({

        "transaction_amount": [100.0] * 10,

        "transactions_last_1h": [2] * 10,

        "transactions_last_24h": [8] * 10,

        "distance_from_usual_location_km": [30.0] * 10,

        "merchant_risk_score": [0.3] * 10,

        "ip_risk_score": [0.2] * 10,

        "authentication_level": ["Standard"] * 10,

        "device_type": ["Windows"] * 10,

        "channel": (
            ["E-commerce"] * 7
            + ["POS"] * 3
        )
    })

    y = pd.Series([0, 1] * 5)

    return X, y


def test_amount_shift():

    X, y = create_test_data()

    scenarios = create_stress_scenarios(
        X, y, mix_size=10
    )

    shifted, labels = scenarios["Amount x2"]

    assert (
        shifted["transaction_amount"] == 200
    ).all()

    pd.testing.assert_series_equal(labels, y)


def test_velocity_consistency():

    X, y = create_test_data()

    scenarios = create_stress_scenarios(
        X, y, mix_size=10
    )

    shifted, _ = scenarios["Velocity surge"]

    assert (
        shifted["transactions_last_1h"] == 4
    ).all()

    assert (
        shifted["transactions_last_24h"] == 12
    ).all()

    assert (
        shifted["transactions_last_1h"]
        <= shifted["transactions_last_24h"]
    ).all()


def test_authentication_outage():

    X, y = create_test_data()

    scenarios = create_stress_scenarios(
        X, y, mix_size=10
    )

    shifted, _ = scenarios[
        "Authentication outage"
    ]

    assert shifted[
        "authentication_level"
    ].isna().all()


def test_original_data_preserved():

    X, y = create_test_data()

    original = X.copy(deep=True)

    create_stress_scenarios(
        X, y, mix_size=10
    )

    pd.testing.assert_frame_equal(
        X, original
    )


def test_channel_mix():

    X, y = create_test_data()

    scenarios = create_stress_scenarios(
        X, y, mix_size=10
    )

    shifted, labels = scenarios[
        "70% ecommerce"
    ]

    assert np.isclose(
        shifted["channel"]
        .eq("E-commerce").mean(),
        0.70
    )

    assert shifted.index.is_unique

    assert shifted.index.equals(labels.index)


def test_bootstrap_intervals():

    policy = {

        "review_threshold": 0.145,

        "block_threshold": 0.265,

        "cost_assumptions": BASE_COSTS
    }

    intervals, results = (
        bootstrap_policy_intervals(
            [0, 1, 0, 1, 0, 1],
            [0.05, 0.2, 0.1, 0.3, 0.15, 0.4],
            policy,
            n_bootstrap=100,
            random_state=42
        )
    )

    assert (
        intervals["lower_95"]
        <= intervals["upper_95"]
    ).all()

    assert np.isfinite(
        intervals.to_numpy()
    ).all()
