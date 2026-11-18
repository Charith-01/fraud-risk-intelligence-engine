
import numpy as np

from src.cost_analysis import (
    BASE_COSTS,
    create_cost_matrix,
    decision_codes,
    policy_metrics,
    optimize_thresholds
)


def test_base_cost_matrix():

    expected = np.array([
        [0, 100],
        [5, 15],
        [25, 2]
    ])

    np.testing.assert_allclose(
        create_cost_matrix(BASE_COSTS),
        expected
    )


def test_decision_boundaries():

    probabilities = [
        0.0, 0.1, 0.4, 1.0
    ]

    actual = decision_codes(
        probabilities,
        0.1,
        0.4
    )

    np.testing.assert_array_equal(
        actual,
        [0, 1, 2, 2]
    )


def test_approve_all_policy():

    actual = decision_codes(
        [0, 1],
        1.01,
        1.01
    )

    np.testing.assert_array_equal(
        actual,
        [0, 0]
    )


def test_policy_cost_calculation():

    result = policy_metrics(
        y_true=[0, 1, 0, 1, 0, 1],
        probabilities=[
            0.01, 0.02, 0.2, 0.3, 0.8, 0.9
        ],
        review_threshold=0.1,
        block_threshold=0.5,
        costs=BASE_COSTS
    )

    assert np.isclose(
        result["total_cost"],
        147.0
    )


def test_optimized_capacity_limits():

    result = optimize_thresholds(
        y_true=[0, 1, 0, 1, 0, 1],
        probabilities=[
            0.01, 0.02, 0.2, 0.3, 0.8, 0.9
        ],
        costs=BASE_COSTS,
        max_review_rate=0.25,
        max_block_rate=0.05
    )

    assert result["capacity_feasible"]

    assert result["review_rate"] <= 0.25

    assert result["block_rate"] <= 0.05

    assert (
        result["review_threshold"]
        <= result["block_threshold"]
    )
