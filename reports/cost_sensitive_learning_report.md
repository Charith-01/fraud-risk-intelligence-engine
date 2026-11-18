# Cost-Sensitive Learning and Threshold Optimization Report

## 1. Objective

Develop a financially optimized transaction decision framework that balances fraud prevention, customer disruption and manual-review expenses.

## 2. Business Cost Assumptions

The base scenario assumes:

- Missed fraudulent transaction loss: $100
- Manual review expense: $5
- Manual-review effectiveness: 90%
- Transaction blocking expense: $2
- Additional legitimate-customer blocking cost: $23
- Maximum manual-review rate: 25%
- Maximum automatic blocking rate: 5%

All financial and operational assumptions are hypothetical.

## 3. Cost Matrix

| Decision | Legitimate | Fraud |
|---|---:|---:|
| Approve | $0 | $100 |
| Review | $5 | $15 |
| Block | $25 | $2 |

The fraudulent-review cost includes expected residual fraud losses resulting from imperfect manual-review effectiveness.

## 4. Methodology

Eighteen model-probability configurations were evaluated under six financial cost scenarios and four operational policies.

A total of 432 policy comparisons were performed.

Thresholds were optimized using 450 threshold validation observations. The search minimized empirical expected business cost subject to operational capacity constraints.

The final test set remained excluded.

## 5. Selected Production Candidate

Logistic Regression using original predictors, controlled Random Oversampling and Platt scaling achieved the lowest constrained validation cost among calibrated candidates.

- Review threshold: 0.145
- Block threshold: 0.265
- Simulated total cost: $3,596
- Approve All baseline: $5,400
- Simulated cost reduction: 33.41%

## 6. Policy Comparison

| Policy | Total Cost |
|---|---:|
| Approve All | $5,400 |
| Default 0.50 | $5,400 |
| Optimized binary | $5,206 |
| Optimized three-way | $3,596 |

The three-way decision framework provides substantial estimated cost savings compared with the simpler reference policies.

## 7. Operational Results

The selected policy approves 333 transactions, reviews 110 transactions and blocks seven transactions.

Twenty-seven fraudulent transactions are flagged for review or blocking.

Expected fraud prevention is 24.6 transactions, representing 45.56% of actual fraudulent observations under the assumed review effectiveness.

Four legitimate transactions are incorrectly blocked.

## 8. Financial Sensitivity

| Scenario | Preferred Model | Optimized Cost |
|---|---|---:|
| Base | LR Original | $3,596 |
| Low fraud loss | LR Original | $2,420 |
| High fraud loss | LR Original | $6,536 |
| High customer friction | LR Engineered | $3,682 |
| Expensive review | LR Original | $4,146 |
| Lower review effectiveness | LR Original | $4,196 |

All scenario winners use Platt scaling.

Higher customer-friction costs change the preferred model and increase the selected blocking threshold.

## 9. Review Capacity Sensitivity

Increasing manual-review capacity from 10% to 50% reduces the simulated optimized cost from $4,573 to $2,776.

However, increased review workload may create operational bottlenecks.

## 10. Key Findings

The model with the highest cross-validation ROC-AUC does not achieve the lowest constrained business cost.

Cost-sensitive decision thresholds substantially improve the selected model's operational effectiveness compared with the default 0.50 threshold.

The preferred model and decision thresholds depend on financial assumptions and operational capacity.

## 11. Limitations

Financial costs and review effectiveness are hypothetical rather than measured company information.

Threshold optimization and model selection use the same validation dataset, potentially producing optimistic cost estimates.

The relatively small validation sample creates threshold instability.

The simulated expected fraud prevention figures are not actual observed fraud prevention outcomes.

Operational capacity limits are applied to the validation batch and require active monitoring in production.

Probability-based expected cost differs from outcome-based validation cost.

Final independent test evaluation is required before recommending deployment.

## 12. Conclusion

A calibrated Logistic Regression model with cost-sensitive operational thresholds is selected as the experimental production candidate.

The selected configuration balances simulated financial losses, manual-review workload and legitimate-customer blocking costs more effectively than the evaluated default classification policies.