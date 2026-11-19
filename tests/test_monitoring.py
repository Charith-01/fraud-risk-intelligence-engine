
import numpy as np
import pandas as pd
import pytest

from src.monitoring import (
    build_monitoring_reference,
    monitor_batch,
    population_stability_index
)

from src.cost_analysis import BASE_COSTS


def create_reference(X, p):

    metrics = {
        "pr_auc": 0.20,
        "roc_auc": 0.65,
        "brier": 0.10,
        "expected_fraud_capture": 0.40,
        "legitimate_flag_rate": 0.20,
        "cost_per_transaction": 8,
        "saving_pct": 20,
        "review_rate": 0.20,
        "block_rate": 0.01
    }

    policy = {
        "feature_columns": list(X.columns),

        "review_threshold": 0.145,

        "block_threshold": 0.265,

        "max_review_rate": 0.25,

        "max_block_rate": 0.05,

        "cost_assumptions": BASE_COSTS.copy()
    }

    return build_monitoring_reference(
        X, p, metrics, policy
    )


@pytest.fixture
def example():

    X = pd.DataFrame({
        "amount": np.tile(
            np.arange(10), 30
        ),

        "authentication_level": [
            "Standard"
        ] * 300
    })

    p = np.full(300, 0.10)

    reference = create_reference(X, p)

    return X, p, reference


def test_identical_psi():

    result = population_stability_index(
        [50, 50],
        [50, 50]
    )

    assert result == pytest.approx(0)


def test_zero_bins():

    result = population_stability_index(
        [0, 100],
        [100, 0]
    )

    assert np.isfinite(result)

    assert result > 0.25


def test_clean(example):

    X, p, reference = example

    drift, kpis, alerts = monitor_batch(
        X, p, reference
    )

    assert (drift["psi"] == 0).all()

    assert alerts.empty

    assert (
        kpis["label_status"]
        == "not_supplied"
    )

    assert np.isnan(kpis["pr_auc"])


def test_outage(example):

    X, p, reference = example

    shifted = X.assign(
        authentication_level=np.nan
    )

    _, kpis, alerts = monitor_batch(
        shifted,
        p,
        reference
    )

    assert (
        kpis["status"]
        == "PAUSE_OR_ESCALATE"
    )

    assert (
        "authentication_missing_rate"
        in alerts["metric"].tolist()
    )


def test_unknown_category(example):

    X, p, reference = example

    shifted = X.assign(
        authentication_level="Unknown"
    )

    drift, kpis, _ = monitor_batch(
        shifted,
        p,
        reference
    )

    unknown_rate = drift.loc[
        drift["feature"]
        == "authentication_level",
        "unknown_rate"
    ].iloc[0]

    assert unknown_rate == 1

    assert (
        kpis["status"]
        == "PAUSE_OR_ESCALATE"
    )


def test_capacity_and_mature_labels(example):

    X, p, reference = example

    y = pd.Series(
        [0] * 240 + [1] * 60
    )

    shifted_probabilities = np.full(
        300, 0.20
    )

    _, kpis, alerts = monitor_batch(
        X,
        shifted_probabilities,
        reference,
        y
    )

    assert kpis["review_rate"] == 1

    assert (
        kpis["label_status"]
        == "evaluated"
    )

    assert (
        kpis["expected_fraud_capture"]
        == pytest.approx(0.90)
    )

    assert (
        "review_rate"
        in alerts["metric"].tolist()
    )


def test_misaligned_labels(example):

    X, p, reference = example

    y = pd.Series(
        [0] * 300,
        index=np.arange(1, 301)
    )

    with pytest.raises(ValueError):

        monitor_batch(
            X, p, reference, y
        )


def test_small_window(example):

    X, p, reference = example

    _, kpis, _ = monitor_batch(
        X.iloc[:10],
        p[:10],
        reference
    )

    assert (
        kpis["status"]
        == "INSUFFICIENT_DATA"
    )


def test_extreme_tail():

    X = pd.DataFrame({

        "transaction_amount": np.arange(
            1, 301,
            dtype=float
        )
    })

    p = np.full(300, 0.10)

    reference = create_reference(X, p)

    shifted = X.copy()

    shifted.loc[
        :20,
        "transaction_amount"
    ] = 10000

    _, _, alerts = monitor_batch(
        shifted,
        p,
        reference
    )

    assert (
        "transaction_amount:upper_tail_rate"
        in alerts["metric"].tolist()
    )
