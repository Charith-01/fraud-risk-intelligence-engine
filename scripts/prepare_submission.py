
import shutil
import hashlib

from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SUBMISSION_DIR = (
    PROJECT_ROOT / "submission"
)

MODELS_DIR = (
    PROJECT_ROOT / "outputs" / "models"
)

METRICS_DIR = (
    PROJECT_ROOT / "outputs" / "metrics"
)


def prepare_submission():

    required = {

        "01_fraud_risk_analysis.ipynb":
            PROJECT_ROOT / "01_fraud_risk_analysis.ipynb",

        "02_processed_dataset.csv":
            PROJECT_ROOT / "data" / "processed"
            / "02_processed_dataset.csv",

        "03_fraud_model_pipeline.joblib":
            MODELS_DIR / "03_fraud_model_pipeline.joblib",

        "04_calibrated_model.joblib":
            MODELS_DIR / "04_calibrated_model.joblib",

        "05_threshold_cost_analysis.csv":
            METRICS_DIR / "05_threshold_cost_analysis.csv",

        "06_model_card.pdf":
            PROJECT_ROOT / "reports"
            / "06_model_card.pdf",

        "07_business_fraud_strategy.pdf":
            PROJECT_ROOT / "reports"
            / "07_business_fraud_strategy.pdf",

        "08_README.md":
            PROJECT_ROOT / "08_README.md"
    }

    # Validate required deliverables

    for name, path in required.items():

        if (
            not path.is_file()
            or path.stat().st_size == 0
        ):

            raise FileNotFoundError(
                f"Missing or empty deliverable: {name}"
            )

    # Prepare supporting files

    support_files = [

        PROJECT_ROOT / "requirements.txt",

        PROJECT_ROOT / "requirements-lock.txt",

        PROJECT_ROOT / "README.md",

        MODELS_DIR / "decision_policy.json",

        MODELS_DIR / "monitoring_reference.json",

        MODELS_DIR / "frozen_artifact_hashes.json"
    ]

    # Source modules, tests and report generators

    for folder in ["src", "tests", "scripts"]:

        support_files.extend(
            (PROJECT_ROOT / folder).glob("*.py")
        )

    # Supporting Markdown reports

    support_files.extend(
        (PROJECT_ROOT / "reports").glob("*.md")
    )

    # Aggregate experimental evidence

    evidence_files = [
        "data_split_summary.csv",
        "tuned_model_comparison.csv",
        "tuned_model_fold_results.csv",
        "advanced_feature_ablation.csv",
        "imbalance_comparison.csv",
        "imbalance_cost_comparison.csv",
        "calibration_comparison.csv",
        "operational_model_ranking.csv",
        "cost_scenario_winners.csv",
        "block_fee_sensitivity.csv",
        "review_capacity_sensitivity.csv",
        "selected_policy_comparison.csv",
        "leakage_feature_sensitivity.csv",
        "fraud_operations_summary.csv",
        "operational_segment_analysis.csv",
        "transaction_volume_simulation.csv",
        "shap_global_importance.csv",
        "feature_importance_stability.csv",
        "shap_background_stability.csv",
        "model_interaction_summary.csv",
        "robustness_stress_results.csv",
        "cv_robustness_variability.csv",
        "final_test_metrics.csv",
        "final_test_operational_outcomes.csv",
        "final_test_policy_comparison.csv",
        "final_test_calibration.csv",
        "final_test_confidence_intervals.csv",
        "monitoring_stress_summary.csv",
        "monitoring_feature_drift.csv",
        "monitoring_alerts.csv",
        "alternative_model_cost_evidence.csv",
        "final_project_summary.csv"
    ]

    support_files.extend([
        METRICS_DIR / filename
        for filename in evidence_files
    ])

    # Aggregate visualizations only.
    # Individual transaction plots may contain
    # restricted transaction information.

    support_files.extend([
        path
        for path in (
            PROJECT_ROOT / "outputs" / "figures"
        ).glob("*.png")
        if not path.name.startswith("shap_local_")
    ])

    # Validate supporting files before copying

    for path in support_files:

        if not path.is_file():

            raise FileNotFoundError(
                f"Missing supporting file: {path}"
            )

    # Prevent accidentally overwriting an old submission

    if SUBMISSION_DIR.exists():

        raise FileExistsError(
            "The submission folder already exists. "
            "Rename or remove it before rebuilding."
        )

    SUBMISSION_DIR.mkdir()

    # Copy all eight deliverables to submission root

    for name, source in required.items():

        destination = SUBMISSION_DIR / name

        shutil.copy2(
            source,
            destination
        )

        print(f"Copied: {name}")

        # Preserve the project-relative structure
        # required by notebook and scripts.

        relative = source.relative_to(
            PROJECT_ROOT
        )

        nested_destination = (
            SUBMISSION_DIR / relative
        )

        if nested_destination != destination:

            nested_destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(
                source,
                nested_destination
            )

    # Copy supporting files

    for source in support_files:

        relative = source.relative_to(
            PROJECT_ROOT
        )

        destination = (
            SUBMISSION_DIR / relative
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        shutil.copy2(
            source,
            destination
        )

    # Prepare raw-data directory.
    # Raw dataset is not copied automatically.

    (
        SUBMISSION_DIR / "data" / "raw"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    # Verify copied deliverables

    manifest_lines = []

    for name, original in required.items():

        copied = SUBMISSION_DIR / name

        original_hash = hashlib.sha256(
            original.read_bytes()
        ).hexdigest()

        copied_hash = hashlib.sha256(
            copied.read_bytes()
        ).hexdigest()

        assert original_hash == copied_hash

        manifest_lines.append(
            f"{copied_hash}  {name}"
        )

    # Save required-deliverable checksum manifest

    (
        SUBMISSION_DIR / "SHA256SUMS.txt"
    ).write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8"
    )

    print("\nSubmission verification passed.")

    print(
        "Required deliverables:",
        len(required)
    )

    print(
        "Submission folder:",
        SUBMISSION_DIR
    )


if __name__ == "__main__":

    prepare_submission()
