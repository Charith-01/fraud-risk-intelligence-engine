
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier

from src.preprocessing import build_model_pipeline


def false_positive_rate(y_true, y_pred):
    """Calculate false-positive rate."""

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return fp / (fp + tn) if fp + tn else 0.0


def make_scoring():
    """Define consistent model evaluation metrics."""

    return {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",

        "precision": make_scorer(
            precision_score,
            zero_division=0
        ),

        "recall": make_scorer(
            recall_score,
            zero_division=0
        ),

        "f1": make_scorer(
            f1_score,
            zero_division=0
        ),

        "neg_brier": "neg_brier_score",

        "neg_fpr": make_scorer(
            false_positive_rate,
            greater_is_better=False
        )
    }


def model_search_space(name, random_state=42):
    """Configure model-specific hyperparameter spaces."""

    if name == "Logistic Regression":

        model = LogisticRegression(
            max_iter=2000,
            random_state=random_state
        )

        params = {
            "model__C": [
                0.01, 0.05, 0.1, 0.5,
                1.0, 5.0, 10.0
            ]
        }

        weight_param = "model__class_weight"
        weights = [None, "balanced"]
        neutral = None
        scale = True

    elif name == "Random Forest":

        model = RandomForestClassifier(
            random_state=random_state,
            n_jobs=1
        )

        params = {
            "model__n_estimators": [100, 200],

            "model__max_depth": [
                3, 5, 8, None
            ],

            "model__min_samples_leaf": [
                2, 5, 10, 20
            ],

            "model__max_features": [
                "sqrt", 0.5
            ]
        }

        weight_param = "model__class_weight"
        weights = [None, "balanced"]
        neutral = None
        scale = False

    elif name == "XGBoost":

        model = XGBClassifier(
            random_state=random_state,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
            objective="binary:logistic"
        )

        params = {
            "model__n_estimators": [100, 200],

            "model__max_depth": [2, 3, 4],

            "model__learning_rate": [
                0.02, 0.05, 0.1
            ],

            "model__min_child_weight": [
                1, 5, 10
            ],

            "model__subsample": [0.8, 1.0],

            "model__colsample_bytree": [
                0.8, 1.0
            ],

            "model__reg_lambda": [1, 5, 10]
        }

        weight_param = "model__scale_pos_weight"
        weights = [1.0, 3.0, 7.0]
        neutral = 1.0
        scale = False

    else:

        raise ValueError(
            f"Unknown model: {name}"
        )

    # Compare standard learning, weighting and oversampling.
    # Avoid applying weighting and oversampling together.
    space = [
        {
            **params,

            weight_param: weights,

            "sampler": ["passthrough"]
        },

        {
            **params,

            weight_param: [neutral],

            "sampler": [
                RandomOverSampler(
                    sampling_strategy=0.5,
                    random_state=random_state
                )
            ]
        }
    ]

    return model, space, scale


def create_model_search(
    name,
    use_engineered,
    cv,
    random_state=42,
    n_iter=12,
    n_jobs=2
):
    """Build a leakage-aware randomized model search."""

    model, space, scale = model_search_space(
        name,
        random_state
    )

    pipeline = build_model_pipeline(
        model=model,
        use_engineered=use_engineered,
        scale_numeric=scale,
        random_state=random_state
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=space,
        n_iter=n_iter,
        scoring=make_scoring(),
        refit="pr_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        error_score="raise",
        return_train_score=True
    )

    return search
