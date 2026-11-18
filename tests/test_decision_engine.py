
import numpy as np
import pandas as pd
import pytest

from src.decision_engine import FraudDecisionEngine
from src.cost_analysis import BASE_COSTS


class MockFraudModel:

    classes_ = np.array([0, 1])

    def predict_proba(self, X):

        p = X["ip_risk_score"].to_numpy()

        return np.column_stack([
            1 - p,
            p
        ])


@pytest.fixture
def engine():

    policy = {
        "feature_columns": [
            "ip_risk_score",
            "channel"
        ],

        "review_threshold": 0.145,

        "block_threshold": 0.265,

        "max_review_rate": 0.25,

        "max_block_rate": 0.05,

        "cost_assumptions": BASE_COSTS.copy()
    }

    return FraudDecisionEngine(
        MockFraudModel(),
        policy
    )


@pytest.fixture
def transactions():

    return pd.DataFrame({

        "transaction_id": ["T1", "T2", "T3"],

        "ip_risk_score": [0.10, 0.145, 0.265],

        "channel": ["ATM", "POS", "E-commerce"]
    })


def test_decision_boundaries(engine, transactions):

    result = engine.score_transactions(
        transactions
    )

    assert result["decision"].tolist() == [
        "Approve",
        "Manual Review",
        "Block"
    ]


def test_missing_required_column(engine, transactions):

    invalid = transactions.drop(
        columns="channel"
    )

    with pytest.raises(ValueError):

        engine.score_transactions(invalid)


def test_target_rejection(engine, transactions):

    invalid = transactions.assign(
        fraud=[0, 1, 1]
    )

    with pytest.raises(ValueError):

        engine.score_transactions(invalid)


def test_invalid_risk_score(engine, transactions):

    invalid = transactions.copy()

    invalid.loc[0, "ip_risk_score"] = 1.5

    with pytest.raises(ValueError):

        engine.score_transactions(invalid)


def test_expected_cost(engine, transactions):

    result = engine.score_transactions(
        transactions
    )

    expected = [
        10.0,
        5 + 10 * 0.145,
        25 - 23 * 0.265
    ]

    np.testing.assert_allclose(
        result["expected_cost"],
        expected
    )


def test_review_queue(engine, transactions):

    result = engine.score_transactions(
        transactions
    )

    queue = engine.create_review_queue(
        result
    )

    assert len(queue) == 1

    assert queue.iloc[0]["transaction_id"] == "T2"

    assert queue.iloc[0]["review_priority"] == 1


def test_capacity_warning(engine, transactions):

    result = engine.score_transactions(
        transactions
    )

    status = engine.capacity_status(result)

    assert status["review_capacity_exceeded"]

    assert status["block_capacity_exceeded"]


def test_input_not_modified(engine, transactions):

    original = transactions.copy(deep=True)

    engine.score_transactions(transactions)

    pd.testing.assert_frame_equal(
        original,
        transactions
    )


def test_duplicate_transaction_id(engine, transactions):

    invalid = transactions.copy()

    invalid.loc[1, "transaction_id"] = "T1"

    with pytest.raises(ValueError):

        engine.score_transactions(invalid)
