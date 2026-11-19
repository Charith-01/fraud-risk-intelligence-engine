
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


class FraudSHAPExplainer:
    """Explain calibrated fraud probabilities using permutation SHAP."""

    def __init__(
        self,
        model,
        X_reference,
        background_size=24,
        random_state=42,
        permutations=3
    ):

        self.model = model

        self.columns = list(
            X_reference.columns
        )

        self.categorical = list(
            X_reference.select_dtypes(
                exclude=np.number
            ).columns
        )

        self.background = X_reference.sample(
            n=min(background_size, len(X_reference)),
            random_state=random_state
        ).copy()

        self.random_state = random_state
        self.permutations = permutations

    def explain(self, X):
        """Return SHAP contributions and reference probabilities."""

        X = X[self.columns].copy()

        # Temporary numerical representation for SHAP.
        # Categories are decoded before model prediction.
        levels = {
            column: sorted(
                pd.concat([
                    self.background[column],
                    X[column]
                ]).dropna().unique().tolist()
            )
            for column in self.categorical
        }

        def encode(frame):

            out = frame.copy()

            for column, categories in levels.items():

                mapping = {
                    value: i
                    for i, value in enumerate(categories)
                }

                out[column] = (
                    out[column].map(mapping).fillna(-1)
                )

            return out.to_numpy(dtype=float)

        def predict(matrix):

            frame = pd.DataFrame(
                matrix,
                columns=self.columns
            )

            for column, categories in levels.items():

                frame[column] = frame[column].map(
                    dict(enumerate(categories))
                ).astype(object)

            return self.model.predict_proba(frame)[:, 1]

        encoded_X = encode(X)

        # Ensure temporary encoding preserves predictions
        np.testing.assert_allclose(
            predict(encoded_X),
            self.model.predict_proba(X)[:, 1],
            atol=1e-9
        )

        masker = shap.maskers.Independent(
            encode(self.background)
        )

        explainer = shap.Explainer(
            predict,
            masker,
            algorithm="permutation",
            feature_names=self.columns,
            seed=self.random_state
        )

        result = explainer(
            encoded_X,
            max_evals=(
                self.permutations
                * (2 * len(self.columns) + 1)
            ),
            silent=True
        )

        contributions = pd.DataFrame(
            result.values,
            index=X.index,
            columns=self.columns
        )

        baseline = pd.Series(
            np.asarray(result.base_values).reshape(-1),
            index=X.index,
            name="shap_baseline"
        )

        # SHAP additivity verification
        np.testing.assert_allclose(
            baseline + contributions.sum(axis=1),
            self.model.predict_proba(X)[:, 1],
            atol=1e-7
        )

        return contributions, baseline


def summarize_local_explanation(
    shap_row,
    transaction,
    top_n=3
):
    """Summarize positive and negative risk contributions."""

    increasing = shap_row[
        shap_row > 0
    ].nlargest(top_n)

    decreasing = shap_row[
        shap_row < 0
    ].nsmallest(top_n)

    def format_factors(contributions):

        factors = []

        for feature, value in contributions.items():

            observed = transaction[feature]

            if pd.isna(observed):
                observed = "Missing"

            elif isinstance(observed, (float, np.floating)):
                observed = f"{observed:.3g}"

            factors.append(
                f"{feature}={observed}: "
                f"{value * 100:+.2f} pp"
            )

        return "; ".join(factors) or "None"

    return {
        "risk_increasing_factors": format_factors(
            increasing
        ),

        "risk_decreasing_factors": format_factors(
            decreasing
        )
    }


def plot_local_explanation(
    shap_row,
    baseline,
    probability,
    path,
    top_n=8
):
    """Plot the largest signed contributions for one transaction."""

    important = shap_row.abs().nlargest(
        top_n
    ).index

    selected = shap_row.loc[important].copy()

    remaining = shap_row.drop(important)

    if len(remaining):

        selected.loc["Other features"] = remaining.sum()

    selected = selected.sort_values()

    colors = [
        "tomato" if value > 0 else "steelblue"
        for value in selected
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.barh(
        selected.index,
        selected.values * 100,
        color=colors
    )

    ax.axvline(0, color="black", linewidth=1)

    ax.set_title(
        f"Transaction Risk Explanation\n"
        f"Baseline: {baseline:.2%} | "
        f"Predicted risk: {probability:.2%}"
    )

    ax.set_xlabel(
        "SHAP Contribution (Percentage Points)"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


def pairwise_dependence(
    model,
    X_reference,
    feature_a,
    values_a,
    feature_b,
    values_b
):
    """Calculate two-feature partial dependence."""

    grid = np.zeros((
        len(values_a),
        len(values_b)
    ))

    for i, a in enumerate(values_a):

        for j, b in enumerate(values_b):

            sample = X_reference.copy()

            sample[feature_a] = a
            sample[feature_b] = b

            grid[i, j] = model.predict_proba(
                sample
            )[:, 1].mean()

    # Remove additive marginal effects
    interaction = (
        grid
        - grid.mean(axis=0)
        - grid.mean(axis=1)[:, None]
        + grid.mean()
    )

    labels_a = [
        "Missing" if pd.isna(v) else str(v)
        for v in values_a
    ]

    labels_b = [
        "Missing" if pd.isna(v) else str(v)
        for v in values_b
    ]

    dependence_table = pd.DataFrame(
        grid,
        index=labels_a,
        columns=labels_b
    )

    interaction_table = pd.DataFrame(
        interaction,
        index=labels_a,
        columns=labels_b
    )

    return dependence_table, interaction_table
