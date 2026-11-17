# Advanced Exploratory Data Analysis Report

## 1. Analysis Methodology

EDA was conducted exclusively on the 1,500 training transactions obtained through stratified splitting with random seed 42.

The analysis included categorical fraud rates, numerical risk bands, time-based patterns and feature interactions. Sample sizes, Wilson 95% confidence intervals and risk lift were calculated to support interpretation.

## 2. Class Imbalance

- Legitimate transactions: 1,318
- Fraudulent transactions: 182
- Overall fraud rate: 12.13%
- Class imbalance ratio: 7.24:1

Accuracy alone is insufficient for evaluating fraud detection performance.

## 3. Categorical Findings

- E-commerce has a fraud rate of 14.90%.
- Jewelry has an observed fraud rate of 22.03%.
- Missing authentication has a fraud rate of 20.55%.
- Basic-tier customers have an observed fraud rate of 14.96%.
- Device types show relatively modest differences in marginal fraud rates.

## 4. Numerical Findings

- Transactions above 500 have an observed fraud rate of 32.00%.
- Transactions with six or more transactions in the previous hour have a fraud rate of 31.03%.
- Accounts aged 90 days or less have a fraud rate of 17.72%.
- Merchant risk scores above 0.4 have a fraud rate of 17.58%.
- IP risk scores above 0.4 have a fraud rate of 16.18%.
- Distance does not show a consistent increasing relationship with fraud when examined independently.

## 5. Interaction Findings

New device and distance: Transactions between 50 and 100 km from the usual location have a fraud rate of 20.63% for new devices and 7.92% for existing devices.

Amount and merchant risk: Transactions between 150 and 500 with merchant risk scores above 0.4 have a fraud rate of 20.24%.

Authentication and channel: E-commerce transactions with missing authentication have a fraud rate of 26.79%.

## 6. Limitations

The analysis is observational and does not establish causality. Small groups can produce unstable fraud-rate estimates.

Numerical intervals were selected for exploratory interpretation and do not represent validated operational thresholds.

The dataset lacks complete timestamps, customer identifiers and feature lineage documentation.

Calibration and threshold partitions were included in the earlier development-level quality audit. Advanced EDA is restricted to training observations.

## 7. Recommendations

Develop defensible derived features involving transaction velocity, amount normalization, risk combinations, time patterns, behavioral deviations and feature interactions.

Validate all engineered features through reproducible cross-validation and comparison against models using original features.

A total of 18 exploratory visualizations, including three interaction heatmaps, are generated to support the analysis.