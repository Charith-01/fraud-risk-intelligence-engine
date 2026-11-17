
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


def create_schema_report(df):
    """Generate a column-level data quality report."""

    report = pd.DataFrame({
        "data_type": df.dtypes.astype(str),
        "non_null_count": df.notna().sum(),
        "missing_count": df.isna().sum(),
        "missing_percentage": df.isna().mean() * 100,
        "unique_count": df.nunique(),
        "cardinality_percentage": (
            df.nunique() / len(df) * 100
        )
    })

    report.index.name = "feature"

    return report


def analyze_missingness(df, target):
    """Investigate associations between missingness and fraud."""

    results = []

    for column in df.columns:

        if column == target or not df[column].isna().any():
            continue

        missing = df[column].isna()

        contingency = pd.crosstab(
            missing, df[target]
        ).reindex(
            index=[False, True],
            columns=[0, 1],
            fill_value=0
        )

        _, p_value = fisher_exact(contingency)

        results.append({
            "feature": column,
            "missing_count": missing.sum(),
            "missing_percentage": missing.mean() * 100,
            "fraud_rate_missing": (
                df.loc[missing, target].mean()
            ),
            "fraud_rate_present": (
                df.loc[~missing, target].mean()
            ),
            "fisher_p_value": p_value
        })

    return pd.DataFrame(results)


def detect_iqr_outliers(df, columns):
    """Identify extreme observations without removing them."""

    results = []

    for column in columns:

        values = df[column].dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        if iqr == 0:
            count = np.nan
        else:
            count = (
                (values < lower) | (values > upper)
            ).sum()

        results.append({
            "feature": column,
            "minimum": values.min(),
            "maximum": values.max(),
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": count,
            "outlier_percentage": count / len(values) * 100,
            "iqr_applicable": iqr > 0
        })

    return pd.DataFrame(results)
