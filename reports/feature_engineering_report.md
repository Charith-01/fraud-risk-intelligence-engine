# Feature Engineering Report

## 1. Objective

Develop interpretable transaction-level features that capture behavioral deviations, transaction concentration, relative transaction amounts, time patterns and combined fraud-risk factors.

## 2. Feature Dictionary

| Feature | Calculation | Prediction-time availability |
|---|---|---|
| velocity_share | T1h / (T24h + 1) | Conditional on historical window accuracy |
| velocity_excess | T1h − (T24h − T1h) / 23 | Conditional on nested historical windows |
| log_amount | ln(1 + amount) | Available at transaction initiation |
| amount_to_merchant_median | Amount / training merchant-category median | Available using stored historical medians |
| log_distance | ln(1 + distance) | Conditional on location-baseline accuracy |
| combined_risk_mean | (Merchant risk + IP risk) / 2 | Conditional on risk-score availability |
| risk_product | Merchant risk × IP risk | Conditional on risk-score availability |
| hour_sin | sin(2π × hour / 24) | Available before the decision |
| hour_cos | cos(2π × hour / 24) | Available before the decision |
| is_night | 1 if hour is between 0 and 5 | Available before the decision |
| day_sin | sin(2π × day / 7) | Available before the decision |
| day_cos | cos(2π × day / 7) | Available before the decision |
| is_young_account | 1 if account age ≤ 90 days | Conditional on point-in-time account age |
| new_device_distance | New device × distance | Conditional on device and location information |
| amount_merchant_risk | Log amount × merchant risk | Conditional on merchant risk availability |
| ecommerce_missing_auth | E-commerce and missing authentication | Conditional on authentication status being known |
| failed_login_velocity | Failed logins in 24h × T1h | Conditional on historical event availability |
| new_device_billing_mismatch | New device × billing mismatch | Conditional on both checks preceding the decision |

## 3. Leakage Prevention

Merchant-category amount medians are fitted exclusively on the corresponding training data. Unknown merchant categories use the training-derived global median.

Feature engineering is implemented as a Scikit-learn-compatible transformer inside the model pipeline.

Neither the fraud outcome nor the transaction identifier is used to generate predictors.

Historical counts, risk scores and authentication information require point-in-time availability verification before deployment.

## 4. Missing Values and Extreme Observations

Missing values are preserved when source information is unavailable. Learned imputation is deferred to the preprocessing pipeline.

Extreme observations are retained. Logarithmic transformations reduce skewness without deleting transaction records.

## 5. Assumptions

- The one-hour transaction window is contained within the 24-hour window.
- Transaction amounts are expressed in comparable monetary units.
- Nighttime is defined as 00:00–05:59.
- Young accounts are defined as accounts aged 90 days or less.
- Merchant and IP risk scores are assumed to range between zero and one.
- The derived features describe observed associations, not causal relationships.

## 6. Preliminary Feature Ablation

Stratified five-fold cross-validation using Logistic Regression produced:

| Feature configuration | Mean PR-AUC (AP) | Mean ROC-AUC |
|---|---:|---:|
| Original predictors | 0.2372 | 0.6402 |
| Original and engineered predictors | 0.2298 | 0.6320 |

Feature engineering did not improve the preliminary Logistic Regression baseline. Further comparisons with tuned nonlinear models are required.

## 7. Conclusion

Eighteen derived features were implemented, resulting in 41 potential predictors before categorical encoding.

All feature transformations are reproducible and compatible with the model pipeline. Their final inclusion will depend on measured predictive and operational performance.