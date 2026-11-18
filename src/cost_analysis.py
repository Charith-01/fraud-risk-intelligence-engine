
import numpy as np
import pandas as pd


NO_ACTION_THRESHOLD = 1.01

ACTIONS = np.array([
    "Approve", "Review", "Block"
])


BASE_COSTS = {
    "fraud_loss": 100,
    "review_fee": 5,
    "review_effectiveness": 0.90,
    "block_fee": 2,
    "customer_friction": 23
}


SCENARIOS = {
    "Base": BASE_COSTS.copy(),

    "Low fraud loss": {
        **BASE_COSTS,
        "fraud_loss": 60
    },

    "High fraud loss": {
        **BASE_COSTS,
        "fraud_loss": 200
    },

    "High customer friction": {
        **BASE_COSTS,
        "customer_friction": 60
    },

    "Expensive review": {
        **BASE_COSTS,
        "review_fee": 10
    },

    "Lower review effectiveness": {
        **BASE_COSTS,
        "review_effectiveness": 0.65
    }
}


def create_cost_matrix(costs):
    """Create the decision-versus-outcome cost matrix."""

    loss = costs["fraud_loss"]
    fee = costs["review_fee"]
    q = costs["review_effectiveness"]
    block = costs["block_fee"]
    friction = costs["customer_friction"]

    if (
        min(loss, fee, block, friction) < 0
        or not 0 <= q <= 1
    ):
        raise ValueError("Invalid cost assumptions.")

    return np.array([
        [0, loss],
        [fee, fee + (1 - q) * loss],
        [block + friction, block]
    ], dtype=float)


def decision_codes(
    probabilities,
    review_threshold,
    block_threshold
):
    """Convert fraud probabilities into decision codes."""

    p = np.asarray(probabilities, dtype=float)

    if (
        p.ndim != 1
        or not np.isfinite(p).all()
        or not ((0 <= p) & (p <= 1)).all()
    ):
        raise ValueError(
            "Expected finite probabilities in [0, 1]."
        )

    if not (
        0 <= review_threshold
        <= block_threshold
        <= NO_ACTION_THRESHOLD
    ):
        raise ValueError(
            "Invalid or unordered thresholds."
        )

    return np.where(
        p >= block_threshold,
        2,
        np.where(
            p >= review_threshold,
            1,
            0
        )
    )


def policy_metrics(
    y_true,
    probabilities,
    review_threshold,
    block_threshold,
    costs,
    max_review_rate=0.25,
    max_block_rate=0.05
):
    """Evaluate operational decisions and financial costs."""

    y = np.asarray(y_true)
    p = np.asarray(probabilities, dtype=float)

    d = decision_codes(
        p,
        review_threshold,
        block_threshold
    )

    if (
        y.shape != p.shape
        or len(y) == 0
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Invalid binary labels.")

    y = y.astype(int)

    matrix = create_cost_matrix(costs)

    # Rows: Approve, Review, Block
    # Columns: Legitimate, Fraud
    counts = np.zeros((3, 2), dtype=int)

    np.add.at(counts, (d, y), 1)

    n = len(y)

    a, r, b = counts.sum(axis=1)

    total = (counts * matrix).sum()

    # Cost estimated directly from predicted probabilities
    expected = (
        (1 - p) * matrix[d, 0]
        + p * matrix[d, 1]
    ).sum()

    baseline = y.sum() * costs["fraud_loss"]

    legit, fraud = counts.sum(axis=0)

    prevented = (
        counts[2, 1]
        + costs["review_effectiveness"] * counts[1, 1]
    )

    return {
        "review_threshold": float(review_threshold),
        "block_threshold": float(block_threshold),

        "approve_count": int(a),
        "review_count": int(r),
        "block_count": int(b),

        "fraud_approved": int(counts[0, 1]),
        "fraud_reviewed": int(counts[1, 1]),
        "fraud_blocked": int(counts[2, 1]),

        "legitimate_approved": int(counts[0, 0]),
        "legitimate_reviewed": int(counts[1, 0]),
        "legitimate_blocked": int(counts[2, 0]),

        "review_rate": r / n,
        "block_rate": b / n,

        "fraud_flag_rate": (
            (counts[1, 1] + counts[2, 1]) / fraud
            if fraud else 0.0
        ),

        "expected_fraud_prevented": float(prevented),

        "expected_fraud_capture": (
            prevented / fraud
            if fraud else 0.0
        ),

        "legitimate_flag_rate": (
            (counts[1, 0] + counts[2, 0]) / legit
            if legit else 0.0
        ),

        "total_cost": float(total),

        "cost_per_transaction": float(total / n),

        "probability_expected_cost": float(expected),

        "approve_all_cost": float(baseline),

        "cost_saving": float(baseline - total),

        "saving_pct": (
            (baseline - total) / baseline * 100
            if baseline else np.nan
        ),

        "capacity_feasible": bool(
            r / n <= max_review_rate
            and b / n <= max_block_rate
        )
    }


def optimize_thresholds(
    y_true,
    probabilities,
    costs,
    max_review_rate=0.25,
    max_block_rate=0.05,
    binary_only=False
):
    """Find thresholds minimizing validation business cost."""

    p = np.asarray(probabilities, dtype=float)

    decision_codes(p, 0, NO_ACTION_THRESHOLD)

    y = np.asarray(y_true)

    if (
        y.shape != p.shape
        or len(y) == 0
        or not np.isin(y, [0, 1]).all()
    ):
        raise ValueError("Invalid binary labels.")

    if not (
        0 <= max_review_rate <= 1
        and 0 <= max_block_rate <= 1
    ):
        raise ValueError("Invalid capacity rates.")

    order = np.argsort(p, kind="stable")

    y = y[order].astype(int)

    n = len(p)

    # Search thresholds from 0.000 to 1.000
    # with step size 0.005
    grid = np.r_[
        np.arange(201) / 200,
        NO_ACTION_THRESHOLD
    ]

    pos = np.searchsorted(
        p[order],
        grid,
        side="left"
    )

    matrix = create_cost_matrix(costs)

    # Prefix sums reduce repeated cost calculations
    prefix = np.c_[
        np.zeros(3),
        np.cumsum(matrix[:, y], axis=1)
    ]

    a, r, b = prefix[:, pos]

    total = (
        a[:, None]
        + r[None, :]
        - r[:, None]
        + prefix[2, -1]
        - b[None, :]
    )

    review_n = pos[None, :] - pos[:, None]

    block_n = np.broadcast_to(
        n - pos[None, :],
        total.shape
    )

    feasible = (
        (grid[:, None] <= grid[None, :])
        & (review_n / n <= max_review_rate)
        & (block_n / n <= max_block_rate)
    )

    if binary_only:
        feasible &= (
            grid[:, None] == grid[None, :]
        )

    i, j = np.where(feasible)

    # Prioritize lowest cost.
    # Break ties using fewer blocks and reviews.
    ranking = np.lexsort((
        -grid[j],
        -grid[i],
        review_n[i, j],
        block_n[i, j],
        np.round(total[i, j], 8)
    ))

    k = ranking[0]

    return policy_metrics(
        y_true,
        p,
        grid[i[k]],
        grid[j[k]],
        costs,
        max_review_rate,
        max_block_rate
    )
