
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.ticker import PercentFormatter


def create_bins(series, edges, labels):
    """Create interpretable numerical intervals."""

    return pd.cut(
        series,
        bins=edges,
        labels=labels,
        include_lowest=True
    ).astype(object).fillna("Missing")


def fraud_rate_table(
    df, feature, target="fraud", order=None
):
    """Calculate group-level fraud rates and confidence intervals."""

    groups = df[feature].astype(object).fillna("Missing")

    summary = df.groupby(
        groups, sort=False
    )[target].agg(
        transactions="size",
        fraud_count="sum"
    ).rename_axis("group")

    if order is not None:

        summary = summary.reindex(order).dropna(
            subset=["transactions"]
        )

    n = summary["transactions"]
    p = summary["fraud_count"] / n

    # Wilson 95% confidence interval
    z = 1.96

    denominator = 1 + z**2 / n

    center = (
        p + z**2 / (2 * n)
    ) / denominator

    margin = (
        z * np.sqrt(
            p * (1 - p) / n
            + z**2 / (4 * n**2)
        )
    ) / denominator

    summary["fraud_rate"] = p

    summary["ci_lower"] = (
        center - margin
    ).clip(0, 1)

    summary["ci_upper"] = (
        center + margin
    ).clip(0, 1)

    summary["risk_lift"] = (
        p / df[target].mean()
    )

    summary["low_support"] = n < 30

    if order is None:

        summary = summary.sort_values(
            "fraud_rate",
            ascending=False
        )

    return summary


def plot_fraud_rates(
    summary, title, baseline, path
):
    """Visualize fraud rates with confidence intervals."""

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(summary))

    rates = summary["fraud_rate"].to_numpy()

    errors = np.maximum(
        0,
        np.vstack([
            rates - summary["ci_lower"].to_numpy(),
            summary["ci_upper"].to_numpy() - rates
        ])
    )

    colors = [
        "lightgray" if low else "steelblue"
        for low in summary["low_support"]
    ]

    ax.bar(
        x,
        rates,
        color=colors,
        yerr=errors,
        capsize=3
    )

    ax.axhline(
        baseline,
        color="tomato",
        linestyle="--",
        label="Training baseline"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            f"{group}\n(n={int(n)})"
            for group, n in zip(
                summary.index,
                summary["transactions"]
            )
        ],
        rotation=35,
        ha="right"
    )

    ax.yaxis.set_major_formatter(
        PercentFormatter(1)
    )

    ax.set_title(title)
    ax.set_ylabel("Observed Fraud Rate")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


def plot_interaction(
    df, row, col, title, path,
    target="fraud",
    row_order=None,
    col_order=None
):
    """Analyze joint fraud rates and group sample sizes."""

    a = df[row].astype(object).fillna("Missing")
    b = df[col].astype(object).fillna("Missing")

    counts = pd.crosstab(a, b).reindex(
        index=row_order,
        columns=col_order,
        fill_value=0
    )

    fraud_counts = pd.crosstab(
        a, b,
        values=df[target],
        aggfunc="sum"
    ).reindex(
        index=counts.index,
        columns=counts.columns
    ).fillna(0)

    rates = fraud_counts.div(
        counts.replace(0, np.nan)
    )

    annotations = np.empty(
        rates.shape,
        dtype=object
    )

    for i in range(rates.shape[0]):

        for j in range(rates.shape[1]):

            n = counts.iloc[i, j]

            if n == 0:
                annotations[i, j] = ""
            else:
                warning = "*" if n < 30 else ""

                annotations[i, j] = (
                    f"{rates.iloc[i, j]:.1%}"
                    f"\nn={n}{warning}"
                )

    fig, ax = plt.subplots(figsize=(11, 6))

    sns.heatmap(
        rates,
        annot=annotations,
        fmt="",
        mask=counts.eq(0),
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={
            "format": PercentFormatter(1),
            "label": "Observed Fraud Rate"
        },
        ax=ax
    )

    ax.set_title(
        title + "\n* fewer than 30 transactions"
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    return rates, counts
