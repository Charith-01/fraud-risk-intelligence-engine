
import numpy as np

from sklearn.compose import (
    ColumnTransformer,
    make_column_selector
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

from src.feature_engineering import FraudFeatureEngineer


def build_preprocessor(scale_numeric=True):
    """Create numerical and categorical preprocessing."""

    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True
            )
        ),
        (
            "scaler",
            StandardScaler()
            if scale_numeric
            else "passthrough"
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="Missing"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                make_column_selector(
                    dtype_include=np.number
                )
            ),
            (
                "categorical",
                categorical_pipeline,
                make_column_selector(
                    dtype_exclude=np.number
                )
            )
        ],
        remainder="drop",
        sparse_threshold=0.0
    )

    return preprocessor


def build_model_pipeline(
    model,
    use_engineered=False,
    sampling_ratio=None,
    scale_numeric=True,
    random_state=42
):
    """Build a complete leakage-aware ML pipeline."""

    feature_step = (
        FraudFeatureEngineer()
        if use_engineered
        else "passthrough"
    )

    sampler = (
        RandomOverSampler(
            sampling_strategy=sampling_ratio,
            random_state=random_state
        )
        if sampling_ratio is not None
        else "passthrough"
    )

    return ImbPipeline([
        ("features", feature_step),
        (
            "preprocessing",
            build_preprocessor(
                scale_numeric=scale_numeric
            )
        ),
        ("sampler", sampler),
        ("model", model)
    ])
