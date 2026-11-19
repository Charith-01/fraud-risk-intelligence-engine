# Fraud Risk Intelligence Engine

## 1. Project Overview

The Fraud Risk Intelligence Engine is a production-oriented machine learning project designed to identify suspicious financial transactions while balancing fraud prevention, legitimate-customer disruption and manual-review costs.

The system integrates data-quality engineering, exploratory analysis, feature engineering, machine learning, probability calibration, cost-sensitive decision thresholds, explainability, robustness testing and model governance.

It generates:

- A calibrated fraud probability.
- An operational decision: Approve, Manual Review or Block.
- Risk-increasing and risk-decreasing explanations for flagged transactions.
- Estimated decision costs and operational monitoring alerts.

The implementation is an experimental decision-support system and does not execute actual payment transactions.

## 2. Dataset

Source file:

`transaction_fraud_risk_advanced_dataset.csv`

The dataset contains 3,000 transactions, 23 original predictors, one transaction identifier and one binary fraud target.

- Legitimate transactions: 2,636
- Fraudulent transactions: 364
- Fraud prevalence: approximately 12.13%

The transaction identifier and fraud target are excluded from model predictors.

## 3. Environment

Recommended Python version: Python 3.11.

Main dependencies:

- NumPy
- Pandas
- SciPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- SHAP
- Joblib
- ReportLab
- pypdf
- Pytest
- Jupyter
- IPykernel

Exact installed dependency versions are recorded in `requirements-lock.txt`.

Scikit-learn 1.6 or newer is required for FrozenEstimator-based probability calibration.

## 4. Project Structure

```text
fraud-risk-intelligence-engine/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_quality.py
│   ├── validation.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── preprocessing.py
│   ├── evaluation.py
│   ├── model_training.py
│   ├── calibration.py
│   ├── cost_analysis.py
│   ├── decision_engine.py
│   ├── explainability.py
│   ├── robustness.py
│   └── monitoring.py
├── tests/
├── scripts/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── reports/
├── 01_fraud_risk_analysis.ipynb
├── requirements.txt
├── requirements-lock.txt
├── README.md
└── 08_README.md
```

## 5. Environment Setup

Open the project root in Visual Studio Code.

Create and activate a Python virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
```

If the lock file is incompatible with the platform, install from `requirements.txt` and record the resulting dependency versions.

Register the project environment as a Jupyter kernel:

```powershell
python -m ipykernel install --user --name fraud-risk-engine --display-name "Python (Fraud Risk Engine)"
```

Select the corresponding interpreter and notebook kernel in VS Code.

## 6. Dataset Placement

Place the authorized source dataset at:

`data/raw/transaction_fraud_risk_advanced_dataset.csv`

Raw data is deliberately excluded from public Git version control.

If the source dataset is not included in the submission package, it must be obtained from the authorized original source.

## 7. Execution Procedure

1. Open `01_fraud_risk_analysis.ipynb` from the project root.
2. Select the fraud-risk virtual environment.
3. Execute the notebook cells sequentially from top to bottom.
4. Complete data auditing, splitting and exploratory analysis.
5. Execute feature engineering and preprocessing experiments.
6. Train and optimize all model configurations.
7. Fit and compare probability calibration methods.
8. Optimize business-cost thresholds using validation data.
9. Execute operational simulation and SHAP explanations.
10. Conduct robustness testing and frozen final evaluation.
11. Execute governance monitoring.
12. Generate and verify the final reports.

The model training and SHAP experiments are computationally intensive.

The PDF reports can also be regenerated using:

```powershell
python scripts/generate_model_card.py
python scripts/generate_business_report.py
```

## 8. Reproducibility and Leakage Prevention

Random seed: 42.

Data partitions:

| Partition | Transactions | Purpose |
|---|---:|---|
| Training | 1,500 | Model fitting and five-fold cross-validation |
| Calibration | 450 | Probability calibration fitting |
| Threshold validation | 450 | Calibration comparison, threshold optimization and model selection |
| Final test | 600 | Frozen independent evaluation |

All learned preprocessing, feature engineering and resampling operations occur inside the appropriate training pipeline.

The final test dataset is excluded from feature selection, model tuning, calibration fitting, resampling and threshold optimization.

The processed dataset is an export for reporting and submission. Models must receive original predictors rather than directly using the exported engineered dataset for cross-validation.

The dataset lacks complete timestamps and customer identifiers; chronological and customer-level independence cannot be established.

## 9. Machine Learning Models

Three model families are evaluated:

- Logistic Regression
- Random Forest
- XGBoost

Both original and engineered feature configurations are investigated.

RandomizedSearchCV uses stratified five-fold cross-validation and Average Precision as the primary model ranking metric.

Final model selection considers calibration quality, simulated business cost and operational constraints.

The selected model and thresholds are recorded in `outputs/models/decision_policy.json`.

## 10. Probability Calibration

The project compares uncalibrated probabilities, Platt scaling and isotonic calibration.

Calibration uses a dedicated partition, with original classifier pipelines frozen.

Calibration quality is evaluated using Brier score, log loss and reliability diagrams.

## 11. Business Cost Assumptions

The base financial scenario assumes:

| Component | Value |
|---|---:|
| Missed fraud loss | $100 |
| Manual-review expense | $5 |
| Review effectiveness | 90% |
| Blocking expense | $2 |
| Additional legitimate-customer blocking cost | $23 |
| Maximum review rate | 25% |
| Maximum block rate | 5% |

The resulting cost matrix is:

| Decision | Legitimate | Fraud |
|---|---:|---:|
| Approve | $0 | $100 |
| Review | $5 | $15 |
| Block | $25 | $2 |

These are hypothetical financial and operational assumptions rather than verified company costs.

Sensitivity analysis investigates alternative fraud losses, customer-friction costs, review expenses, review effectiveness, blocking expenses and review capacity.

## 12. Fraud Decision Policy

The decision engine loads the saved calibrated model and decision configuration.

Transactions receive an approval, review or blocking recommendation according to the stored thresholds.

The exact selected model, thresholds and cost assumptions should always be retrieved from `decision_policy.json`.

Capacity violations generate alerts rather than silently approving excess high-risk transactions.

## 13. Model Explainability

Model-agnostic permutation SHAP explains calibrated fraud probabilities at the original-feature level.

The project includes global importance, dependence analysis, two-feature partial dependence, local explanations and feature-importance stability analysis.

SHAP explanations describe model behavior and do not establish causation or criminal intent.

## 14. Robustness and Monitoring

Robustness experiments investigate:

- Increased transaction amounts
- Higher transaction velocity
- Missing-value injection
- Authentication outages
- Extreme observations
- Unseen categories
- Channel-mix shifts

Monitoring includes Population Stability Index, missingness, unknown-category rates, prediction drift, operational capacity, predictive performance and financial outcomes.

Monitoring alerts are recommendations for operational teams. They do not automatically retrain or disable a payment system.

## 15. Automated Testing

Run the complete test suite:

```powershell
python -m pytest tests/ -v
```

The planned suite contains 31 tests across cost analysis, decision-engine functionality, explainability, robustness and monitoring.

## 16. Required Deliverables

| File | Description |
|---|---|
| 01_fraud_risk_analysis.ipynb | Complete reproducible analysis |
| 02_processed_dataset.csv | Original and engineered feature export |
| 03_fraud_model_pipeline.joblib | Complete fitted classifier pipeline |
| 04_calibrated_model.joblib | Calibrated model candidate |
| 05_threshold_cost_analysis.csv | Financial and threshold sensitivity results |
| 06_model_card.pdf | Model governance documentation |
| 07_business_fraud_strategy.pdf | Management report |
| 08_README.md | Environment and reproducibility instructions |

Additional source modules, tests, report generators and decision configuration files support reproducibility.

## 17. Security and Limitations

Joblib artifacts must only be loaded from trusted sources.

Raw transaction records, transaction identifiers, model explanations and review queues must be handled as restricted data.

Financial outcomes are simulated. The dataset is relatively small and does not provide verified point-in-time feature lineage, complete chronological data or live operational feedback.

The model is recommended for shadow-mode evaluation or an analyst-assisted pilot.

Unrestricted automatic blocking requires additional independent validation and formal operational approval.

## 18. Conclusion

The project demonstrates an end-to-end fraud-risk intelligence workflow integrating machine learning, probability calibration, financial decision optimization, explainability, operational simulation, robustness analysis and model governance.

All reported results should be interpreted within the documented data and financial assumptions.