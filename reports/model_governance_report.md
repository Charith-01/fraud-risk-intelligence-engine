# Model Governance and Production Monitoring Report

## 1. Objective

Establish a production monitoring and governance framework for the selected calibrated fraud-risk model.

## 2. Intended Use

The model supports transaction risk scoring, manual-review prioritization and cost-sensitive operational recommendations.

Initial deployment should be restricted to shadow mode or an analyst-assisted pilot. Unrestricted automatic blocking is not recommended.

## 3. Monitoring Reference

Input feature distributions are established using 1,500 training transactions.

Prediction distributions and performance baselines are established using 450 threshold validation transactions.

Reference distributions remain fixed and are stored in monitoring_reference.json.

## 4. Monitoring Methodology

Population Stability Index measures feature and prediction distribution drift.

Numerical features use fixed quantile-based groups. Missing and unseen categorical values are monitored separately.

Additional checks include data missingness, extreme values, operational workloads, discrimination, calibration and simulated financial performance.

## 5. Monitoring Thresholds

- PSI of 0.10 or above: warning.
- PSI of 0.25 or above: major drift.
- Missingness increase of 5 percentage points: warning.
- Missingness increase of 20 percentage points: critical.
- Authentication missingness of 30% or above: critical.
- Unseen category rate of 20% or above: critical.
- Manual-review rate above 25%: critical.
- Blocking rate above 5%: critical.
- Relative Average Precision reduction above 20%: major.
- Brier score increase above 0.02: major.
- Negative simulated cost savings: critical.

These thresholds are proposed governance settings and require production validation.

## 6. Monitoring Experiment Results

The monitoring system was evaluated using nine transaction scenarios.

| Scenario | Monitoring status |
|---|---|
| Clean | No alert |
| Amount increase | Critical escalation |
| Velocity surge | Critical escalation |
| Combined surge | Critical escalation |
| Missing-value injection | Critical escalation |
| Authentication outage | Critical escalation |
| Extreme observations | Review |
| Unseen device | Critical escalation |
| Channel-mix shift | Critical escalation |

All eight modified stress scenarios triggered review or escalation.

Authentication outage produced prediction PSI of approximately 0.7898 and increased manual-review workload to 54.44%.

## 7. Priority Features

Authentication level, merchant risk, IP risk, prior chargebacks, transaction velocity, transaction amount, account age, distance, channel, country and device type should be monitored closely.

These priorities reflect findings from exploratory analysis, explainability, feature stability and robustness testing.

## 8. Model Review

Moderate or significant drift and deteriorating predictive performance require investigation.

Data drift alone does not establish concept drift or require immediate retraining.

## 9. Retraining

Repeated major drift or performance degradation across two consecutive adequately supported monitoring windows should initiate a retraining or recalibration review.

New candidate models require fresh labeled data, point-in-time verification, independent chronological validation and formal approval.

## 10. Temporary Suspension

Critical upstream data outages, model failures, excessive workloads and serious customer-impact problems require suspension of automated actions as appropriate.

An independently approved manual or rules-based fallback should be available.

The monitoring module generates alerts but does not execute real financial actions.

## 11. Limitations

The dataset lacks complete timestamps, customer identifiers and verified point-in-time feature lineage.

Monitoring thresholds have been evaluated through controlled simulations rather than live production data.

Financial costs, staffing assumptions and manual-review effectiveness remain hypothetical.

Pending fraud outcomes must not be interpreted as legitimate transactions.

## 12. Conclusion

The monitoring framework detects important distribution shifts, data-quality failures and operational capacity violations.

The generated model card documents the selected model's intended use, evaluation, limitations and governance requirements.

Additional validation and approval are required before live automated deployment.