# Data Quality and Forensics Report

## 1. Dataset Overview

- Total transactions: 3,000
- Total columns: 25
- Potential predictors: 23
- Target variable: fraud
- Random seed: 42
- Development observations: 2,400
- Reserved test observations: 600

## 2. Data Quality Findings and Treatment Decisions

| Issue | Finding | Treatment |
|---|---|---|
| Missing values | 455 missing values in development data | Training-pipeline imputation |
| Numerical missingness | Six features have missing values in total, including one categorical feature | Numerical missingness indicators |
| Categorical missingness | Authentication level contains missing values | Separate Missing category |
| Exact duplicates | None | Retain all observations |
| Duplicate IDs | None | Retain all observations |
| Invalid numerical values | No violations of defined rules | Retain observations |
| Extreme observations | Present in several numerical features | Retain and evaluate transformations |
| High cardinality | Transaction ID is unique | Exclude from model predictors |
| Low-cardinality features | Seven categorical predictors | One-Hot Encoding |
| Logical ambiguity | Device age and new-device status | Retain and document assumptions |
| Potential leakage | Historical and risk-related features | Verify point-in-time availability |

## 3. Missingness Investigation

Missing authentication information is associated with an increased observed fraud rate. Missing merchant risk scores also show a possible association with fraud.

Missingness information will therefore be preserved during preprocessing. These relationships are observational and require validation.

## 4. Extreme Observations

Extreme transaction amounts, distances and transaction velocities may represent genuine suspicious behavior.

No observations will be removed merely because they exceed statistical outlier boundaries.

## 5. Leakage and Data Limitations

The dataset lacks complete timestamps, customer identifiers, feature lineage documentation and label maturity dates.

Temporal validation and point-in-time feature correctness cannot be fully verified. Stratified validation will be used as an available alternative, with these limitations explicitly documented.

## 6. Conclusion

The dataset is suitable for experimental fraud-risk modeling, subject to appropriate preprocessing, class imbalance handling and leakage controls.

All learned preprocessing transformations will be fitted exclusively within the appropriate training or cross-validation pipeline. The final test set will remain reserved for final evaluation.