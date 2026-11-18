# Preprocessing and Imbalanced Learning Report

## 1. Objective

Develop leakage-aware preprocessing pipelines and evaluate the effectiveness of different class imbalance treatments for transaction fraud detection.

## 2. Class Distribution

- Training observations: 1,500
- Legitimate transactions: 1,318
- Fraudulent transactions: 182
- Fraud prevalence: 12.13%
- Imbalance ratio: 7.24:1

## 3. Preprocessing Methodology

Numerical features use median imputation, missingness indicators and standardization.

Categorical missing values are represented using a separate Missing category. One-Hot Encoding is used with unknown-category handling enabled.

Feature engineering and preprocessing are integrated into a reusable Imbalanced-learn pipeline.

All learned transformations are fitted exclusively on the corresponding training fold.

## 4. Imbalance Strategies

Three Logistic Regression configurations were evaluated:

- Standard learning without imbalance correction.
- Balanced class weighting.
- Random Oversampling with a minority-to-majority ratio of 0.5.

Original predictors and identical stratified five-fold cross-validation were used to ensure a controlled comparison.

Random Oversampling was selected to avoid generating fractional categorical representations.

## 5. Predictive Performance

| Metric | Standard | Class weighting | Oversampling |
|---|---:|---:|---:|
| PR-AUC (AP) | 0.2372 | 0.2386 | 0.2405 |
| ROC-AUC | 0.6402 | 0.6429 | 0.6337 |
| Precision | 0.4833 | 0.1808 | 0.2392 |
| Recall | 0.0494 | 0.5381 | 0.2910 |
| F1 | 0.0884 | 0.2704 | 0.2620 |
| FPR | 0.0091 | 0.3376 | 0.1297 |
| Brier score | 0.1061 | 0.2224 | 0.1502 |

Class weighting substantially improves recall but increases the false-positive burden.

Controlled oversampling provides an intermediate trade-off between fraud detection and false positives.

Differences in ranking performance are small relative to cross-validation variability.

## 6. Preliminary Business Cost Sensitivity

A false positive is assigned one hypothetical cost unit. False negative costs of 5, 10 and 20 units are evaluated.

| Cost ratio | Standard | Class weighting | Oversampling |
|---|---:|---:|---:|
| 5:1 | 877 | 865 | 816 |
| 10:1 | 1,742 | 1,285 | 1,461 |
| 20:1 | 3,472 | 2,125 | 2,751 |

The preferred imbalance strategy depends on the assumed cost of missed fraud.

These results represent preliminary binary classification costs at a fixed 0.50 threshold. They do not represent the final operational decision policy.

## 7. Leakage Prevention

Resampling is performed inside the training pipeline after preprocessing.

Validation folds retain their original class distribution. Calibration, threshold validation and final test data remain excluded from these experiments.

## 8. Limitations and Recommendations

Class weighting and oversampling can distort probability estimates. Probability calibration is therefore necessary before operational threshold optimization.

The final imbalance strategy should be selected jointly with model architecture, feature configuration, calibration and business cost.

No final production candidate is selected at this stage.