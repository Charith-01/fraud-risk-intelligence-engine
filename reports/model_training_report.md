# Model Training and Hyperparameter Optimization Report

## 1. Objective

Develop and optimize transparent and nonlinear fraud-risk classification models using reproducible validation and leakage-aware preprocessing.

## 2. Methodology

Three model families were evaluated:

- Logistic Regression
- Random Forest
- XGBoost

Each model was evaluated with original predictors and original plus engineered predictors, resulting in six configurations.

RandomizedSearchCV used 12 parameter configurations per experiment, stratified five-fold cross-validation and random seed 42.

Average Precision was the primary model selection metric.

All feature engineering, preprocessing and resampling operations were performed within the training pipeline.

## 3. Model Comparison

| Model | Feature Set | PR-AUC (AP) | ROC-AUC |
|---|---|---:|---:|
| Random Forest | Engineered | 0.2651 | 0.6649 |
| XGBoost | Engineered | 0.2573 | 0.6387 |
| Logistic Regression | Engineered | 0.2520 | 0.6509 |
| Logistic Regression | Original | 0.2473 | 0.6394 |
| XGBoost | Original | 0.2453 | 0.6352 |
| Random Forest | Original | 0.2414 | 0.6693 |

Random Forest with engineered features achieves the highest mean Average Precision.

Random Forest with original predictors achieves the highest ROC-AUC.

## 4. Feature Engineering Effectiveness

After hyperparameter optimization, engineered features improve mean Average Precision across all three model families.

Random Forest achieves the largest relative improvement of approximately 9.84%.

These differences are developmental estimates and do not establish statistical significance.

## 5. Model Overfitting and Stability

Random Forest demonstrates substantial training-validation performance gaps.

Logistic Regression has smaller generalization gaps. Its engineered configuration also produces the lowest Average Precision variability across validation folds.

The differences in cross-validation performance should therefore be interpreted cautiously.

## 6. Default Classification Performance

The tuned Random Forest and XGBoost configurations fail to identify fraudulent transactions at their default classification thresholds.

Logistic Regression with engineered features achieves approximately 51.67% recall but also produces substantial false positives.

These findings justify probability calibration and operational threshold optimization.

## 7. Potential Leakage Sensitivity

Excluding 14 conditionally available and time-sensitive predictors reduces Logistic Regression Average Precision from 0.2473 to 0.1690 and ROC-AUC from 0.6394 to 0.5496.

This demonstrates predictive dependence on historical and risk-related features but does not establish that actual leakage exists.

Point-in-time feature availability must be verified before deployment.

## 8. Candidate Selection

All six model configurations are retained for probability calibration and business cost analysis.

Random Forest with engineered features is the leading predictive candidate, but the final production candidate has not yet been selected.

## 9. Limitations

The dataset has limited predictive signal and a relatively small number of fraud cases.

The tuning results are developmental cross-validation estimates rather than independent final test results.

Complete temporal information, customer identifiers and feature lineage are unavailable.

The final test dataset remains excluded from model selection, calibration fitting and threshold optimization.