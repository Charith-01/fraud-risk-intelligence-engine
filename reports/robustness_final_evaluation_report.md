# Robustness and Final Model Evaluation Report

## 1. Objective

Evaluate the robustness, generalization capability and operational reliability of the selected calibrated fraud model under simulated distribution shifts and independent final testing.

## 2. Evaluation Methodology

The selected Logistic Regression model uses original predictors, controlled Random Oversampling and Platt scaling.

The review threshold of 0.145 and blocking threshold of 0.265 were frozen before final testing.

Robustness experiments used the 450-transaction threshold validation partition. The final evaluation used the reserved 600-transaction test partition.

No final test information was used for model fitting, feature selection, calibration fitting or threshold optimization.

## 3. Robustness Findings

Eight modified stress scenarios were evaluated against the original validation baseline.

Missing-value injection reduced ROC-AUC from 0.6781 to 0.6513 and increased simulated cost per transaction from $7.99 to $8.69.

Extreme observations reduced Average Precision from 0.2091 to 0.2016.

A complete authentication outage increased manual-review workload from 24.44% to 54.44% and exceeded both operational capacity constraints.

Six of the eight modified stress scenarios violated the manual-review capacity limit.

## 4. Cross-Validation Variability

The selected classifier achieved mean cross-validation Average Precision of 0.2473 and mean ROC-AUC of 0.6394.

The sample standard deviation of Average Precision was approximately 0.0418.

Noticeable fold-level variation indicates uncertainty in model generalization.

## 5. Final Test Performance

| Metric | Result |
|---|---:|
| PR-AUC (Average Precision) | 0.2195 |
| ROC-AUC | 0.6895 |
| Brier score | 0.1028 |
| Fraud alert precision | 21.68% |
| Fraud alert recall | 42.47% |
| F1-score | 0.2870 |
| Expected fraud capture | 38.36% |

## 6. Operational Results

The final test partition contains 600 transactions, including 73 fraudulent observations.

| Decision | Legitimate | Fraud |
|---|---:|---:|
| Approve | 415 | 42 |
| Manual Review | 108 | 30 |
| Block | 4 | 1 |

The manual-review rate was 23.00%, and the blocking rate was 0.83%.

Both operational capacity limits were satisfied on this partition.

However, four of the five automatically blocked transactions were legitimate, creating a significant customer-impact concern.

## 7. Financial Performance

The selected three-way decision policy produced a simulated cost of $5,292 compared with $7,300 for approving all transactions.

The estimated savings were $2,008, representing a 27.51% cost reduction.

The optimized binary policy produced a cost of $7,302.

The three-way policy demonstrated the strongest financial outcome among the evaluated frozen reference policies.

## 8. Calibration

The final calibrated model achieved a Brier score of 0.1028, compared with 0.1439 for the original uncalibrated model.

This provides independent evidence that Platt scaling improved overall probability prediction quality.

## 9. Generalization and Uncertainty

Expected fraud capture decreased from 45.56% on validation data to 38.36% on final test data.

Estimated financial savings decreased from 33.41% to 27.51%.

A 1,000-replicate bootstrap analysis produced an approximate 95% interval of 17.14%–37.19% for cost savings.

The review-rate confidence interval extends above the assumed operational capacity limit.

## 10. Potential Production Degradation

Important risks include concept drift, transaction velocity changes, authentication outages, risk-score degradation, changing channel mix, unseen categories, delayed fraud labels and insufficient investigation capacity.

Production monitoring should include both predictive performance and operational decision metrics.

## 11. Limitations

The dataset does not contain complete chronological information, customer identifiers or independently verified point-in-time feature lineage.

The stress tests use hypothetical predictor changes and do not reproduce actual evolving fraud behavior.

Financial costs, manual-review effectiveness and staffing capacity are assumed rather than measured.

Validation results may be optimistic because the validation partition was used for model and threshold selection.

## 12. Deployment Recommendation

The model demonstrates modest predictive performance and useful simulated financial value.

However, operational capacity violations during stress testing and low automatic-block precision prevent a recommendation for unrestricted automated deployment.

The model should initially be evaluated in shadow mode or an analyst-assisted pilot, with additional chronological validation, feature lineage verification and active monitoring.

Automatic blocking should remain disabled during the initial pilot.

## 13. Conclusion

The selected calibrated fraud model and cost-sensitive decision policy provide measurable simulated financial benefits on the reserved test partition.

Nevertheless, successful deployment depends on data quality, reliable operational capacity, continued monitoring and appropriate human oversight.