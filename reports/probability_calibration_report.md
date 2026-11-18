# Probability Calibration Report

## 1. Objective

Evaluate and improve the reliability of predicted fraud probabilities for operational risk-based decision-making.

## 2. Methodology

Six previously trained model configurations were evaluated using:

- Uncalibrated probabilities
- Platt scaling
- Isotonic calibration

A total of 18 probability configurations were compared.

The original classifiers were frozen during calibration. Calibrators were fitted using 450 calibration observations, including 55 fraudulent transactions.

Calibration performance was evaluated on the separate threshold validation partition containing 450 transactions, including 54 fraudulent transactions.

The final test dataset was excluded.

## 3. Evaluation Metrics

Probability reliability was evaluated using Brier score, log loss, Expected Calibration Error and calibration curves.

Average Precision and ROC-AUC were retained to investigate changes in discrimination performance.

The constant training-prevalence probability baseline achieved a Brier score of approximately 0.1056.

## 4. Brier Score Results

| Model | Features | Uncalibrated | Platt | Isotonic |
|---|---|---|---|---|
| Logistic Regression | Original | 0.1464 | 0.1024 | 0.1035 |
| Logistic Regression | Engineered | 0.2207 | 0.1019 | 0.1056 |
| Random Forest | Original | 0.1044 | 0.1053 | 0.1042 |
| Random Forest | Engineered | 0.1069 | 0.1065 | 0.1064 |
| XGBoost | Original | 0.1042 | 0.1050 | 0.1048 |
| XGBoost | Engineered | 0.1046 | 0.1058 | 0.1089 |

Logistic Regression with engineered features and Platt scaling achieved the lowest observed Brier score.

## 5. Probability Reliability Findings

The class-weighted engineered Logistic Regression model initially predicted an average fraud probability of 45.29%, compared with an actual fraud prevalence of 12%.

Platt scaling reduced the mean predicted probability to approximately 12.12%.

The Brier score improved from 0.2207 to 0.1019, representing approximately a 53.85% reduction.

The original ranking performance was preserved.

## 6. Isotonic Calibration Risks

Isotonic calibration produced extreme probability estimates for some configurations.

Several configurations showed substantially higher log loss following isotonic calibration, indicating potential calibration overfitting.

The limited calibration sample size reduces confidence in highly flexible probability mappings.

## 7. Operational Importance

Reliable fraud probabilities support expected financial cost estimation, manual-review prioritization, transaction blocking decisions and risk monitoring.

A model with strong ranking performance does not necessarily produce reliable probability estimates.

Calibration does not guarantee improved performance for every candidate.

## 8. Candidate Selection

A preferred calibrated configuration was identified for each of the six candidate models using Brier score.

All calibrated alternatives and uncalibrated reference models are retained for business cost evaluation.

The final production candidate has not yet been selected.

## 9. Limitations

The calibration dataset contains only 55 fraud cases.

Calibration method selection and operational threshold optimization use the same development validation partition. Resulting validation performance may therefore be optimistic.

The earlier development-level data-quality investigation also included observations assigned to these partitions.

The reserved test dataset will provide the final independent transaction-level evaluation, subject to the dataset's temporal and customer-identity limitations.

## 10. Conclusion

Probability calibration substantially improves the weighted Logistic Regression configurations.

Platt scaling demonstrates a useful balance between probability reliability and limited-data calibration complexity.

Final model selection must additionally consider expected business cost, false-positive burden, fraud capture and operational review capacity.