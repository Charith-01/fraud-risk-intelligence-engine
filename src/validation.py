
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)


def split_development_data(
    development_df,
    target="fraud",
    random_state=42
):
    """Split development data into training and validation sets."""

    train_df, remaining_df = train_test_split(
        development_df,
        test_size=0.375,
        stratify=development_df[target],
        random_state=random_state
    )

    calibration_df, threshold_df = train_test_split(
        remaining_df,
        test_size=0.50,
        stratify=remaining_df[target],
        random_state=random_state
    )

    return (
        train_df.copy(),
        calibration_df.copy(),
        threshold_df.copy()
    )


def create_split_summary(datasets, target="fraud"):
    """Summarize class distribution across data partitions."""

    records = []

    for name, dataset in datasets.items():

        records.append({
            "dataset": name,
            "total_transactions": len(dataset),
            "legitimate_transactions": int(
                (dataset[target] == 0).sum()
            ),
            "fraudulent_transactions": int(
                (dataset[target] == 1).sum()
            ),
            "fraud_rate_percentage": (
                dataset[target].mean() * 100
            )
        })

    return pd.DataFrame(records)


def create_cv_strategy(random_state=42):
    """Create reproducible stratified cross-validation."""

    return StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )
