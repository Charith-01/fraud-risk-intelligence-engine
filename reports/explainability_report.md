# Explainable AI Report

## 1. Objective

Explain the selected calibrated fraud-risk model at global and individual transaction levels to support model transparency, fraud investigation and governance.

## 2. Methodology

Model-agnostic permutation SHAP was applied directly to calibrated fraud probabilities.

A reference background of 24 training transactions and a randomly selected sample of 150 threshold validation transactions were used for global analysis.

SHAP additivity was verified by reconstructing the original calibrated predictions.

## 3. Global Feature Importance

The five strongest features according to mean absolute SHAP values were:

1. Account age
2. Merchant risk score
3. Transaction channel
4. Merchant category
5. Transactions during the preceding hour

These features demonstrate the greatest average influence on the selected model's predictions within the analyzed sample.

## 4. Feature Dependence

SHAP dependence analysis investigated account age, merchant risk, transaction velocity, transaction amount, IP risk and prior chargebacks.

The results describe model behavior rather than causal relationships.

## 5. Interaction Analysis

Two-dimensional partial dependence was used to investigate three feature combinations.

| Interaction | Non-additive Range |
|---|---:|
| Authentication and channel | 0.665 percentage points |
| Amount and merchant risk | 0.247 percentage points |
| New device and distance | 0.055 percentage points |

Authentication and channel demonstrated the largest measured non-additive relationship among the selected feature grids.

The interaction measurements are not formal SHAP interaction values.

## 6. Transaction-Level Explanations

The decision engine generates explanations for all flagged validation transactions.

A total of 117 transactions were explained, including 110 manual reviews and seven blocking recommendations.

Five detailed transaction-level explanation plots were generated.

Each explanation includes positive and negative risk contributions relative to the reference probability.

## 7. Feature Importance Stability

Permutation importance was evaluated across five cross-validation folds.

Merchant risk, prior chargebacks, transaction velocity, authentication level and IP risk demonstrated positive predictive importance across all five folds.

Country and channel importance was less consistent.

## 8. SHAP Reference Stability

The explanation experiment was repeated using an alternative training background and random seed.

The global importance rankings achieved a Spearman correlation of approximately 0.9674.

All five leading features were preserved across the two experiments.

The reference probability changed from approximately 13.09% to 11.31%.

## 9. Counterintuitive Explanations

Higher geographical distance slightly reduced predicted risk in the partial-dependence experiment.

Strong authentication sometimes received higher predicted risk than standard authentication.

These relationships require further investigation and must not be interpreted as causal effects.

## 10. Operational Integration

The fraud decision engine was extended to generate risk-increasing factors and risk-decreasing factors for every manually reviewed or blocked transaction.

Individual explanations support analyst investigation but do not independently establish whether a transaction is fraudulent.

## 11. Limitations

Permutation SHAP is approximate and sensitive to the reference dataset.

Independent feature masking may create combinations that are uncommon or inconsistent in real transactions.

Partial dependence may similarly evaluate unrealistic feature combinations.

The dataset lacks complete timestamps and feature lineage information.

Importance measures do not establish causality, fairness or guaranteed predictive usefulness.

All analyses use training and threshold validation data. The final test dataset remains reserved.

## 12. Conclusion

The selected fraud model provides interpretable transaction-level risk contributions.

Historical chargebacks, merchant risk and transaction velocity are important predictors requiring verification of point-in-time availability and ongoing monitoring.

Explainability should support human oversight rather than replace fraud investigators or independently justify automatic transaction blocking.