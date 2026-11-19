
import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    Image
)


def generate(root):

    root = Path(root)

    metrics = root / "outputs" / "metrics"
    models = root / "outputs" / "models"
    figures = root / "outputs" / "figures"

    output = (
        root / "reports" /
        "07_business_fraud_strategy.pdf"
    )

    # Helper functions

    def read(name):
        return pd.read_csv(metrics / name)

    def one(name):
        return read(name).iloc[0]

    def pct(x):
        return f"{float(x):.2%}"

    def money(x):
        return f"${float(x):,.2f}"

    def score(x):
        return f"{float(x):.4f}"

    # Load model and experiment results

    policy = json.loads(
        (models / "decision_policy.json")
        .read_text(encoding="utf-8")
    )

    name = policy["model"]
    method = policy["calibration_method"]
    c = policy["cost_assumptions"]

    final = one("final_test_metrics.csv")

    split = read("data_split_summary.csv")

    tuning = read("tuned_model_comparison.csv")

    selected = tuning.loc[
        tuning["candidate"] == name
    ].iloc[0]

    top_ap = tuning.sort_values(
        "pr_auc_mean",
        ascending=False
    ).iloc[0]

    top_roc = tuning.sort_values(
        "roc_auc_mean",
        ascending=False
    ).iloc[0]

    base = read("operational_model_ranking.csv")

    winner = base.loc[
        (base["candidate"] == name)
        & (base["method"] == method)
    ].iloc[0]

    roc_policy = base.loc[
        base["candidate"] == top_roc["candidate"]
    ].sort_values("total_cost").iloc[0]

    costs = read("05_threshold_cost_analysis.csv")

    ablation = read("advanced_feature_ablation.csv")

    imbalance = read("imbalance_comparison.csv")

    calibration = read("calibration_comparison.csv")

    probability = calibration.loc[
        calibration["candidate"] == name
    ].set_index("method")

    sensitivity = read("cost_scenario_winners.csv")

    fees = read("block_fee_sensitivity.csv")

    capacity = read(
        "review_capacity_sensitivity.csv"
    ).sort_values("max_review_capacity")

    outcomes = read(
        "final_test_operational_outcomes.csv"
    ).iloc[:, -2:]

    final_policies = read(
        "final_test_policy_comparison.csv"
    )

    final_cal = read("final_test_calibration.csv")

    important = read(
        "shap_global_importance.csv"
    ).head(5)

    stability = read(
        "feature_importance_stability.csv"
    ).head(5)

    stability = stability.rename(
        columns={
            stability.columns[0]: "feature"
        }
    )

    interactions = read(
        "model_interaction_summary.csv"
    ).sort_values(
        "interaction_range_pp",
        ascending=False
    )

    monitor = read("monitoring_stress_summary.csv")

    conf = read(
        "final_test_confidence_intervals.csv"
    )

    conf = conf.rename(
        columns={
            conf.columns[0]: "metric"
        }
    ).set_index("metric")

    cv = read(
        "cv_robustness_variability.csv"
    ).set_index("candidate").loc[name]

    ref = json.loads(
        (models / "monitoring_reference.json")
        .read_text(encoding="utf-8")
    )

    cfg = ref["config"]

    # PDF styling

    styles = getSampleStyleSheet()

    styles["Normal"].fontSize = 10
    styles["Normal"].leading = 14

    styles["Heading1"].textColor = (
        colors.HexColor("#17354B")
    )

    styles["Heading2"].textColor = (
        colors.HexColor("#166B70")
    )

    story = []

    def p(text, style="Normal"):

        story.extend([
            Paragraph(
                escape(str(text)),
                styles[style]
            ),
            Spacer(1, 7)
        ])

    def page(number, title):

        if number > 1:
            story.append(PageBreak())

        p(
            f"{number:02d} / {title}",
            "Heading1"
        )

    def table(headers, rows, widths):

        cells = [
            [
                Paragraph(
                    escape(str(x)),
                    styles["Normal"]
                )
                for x in row
            ]
            for row in [headers] + rows
        ]

        t = Table(
            cells,
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT"
        )

        t.setStyle(TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#D4E7E7")
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#F2F5F5")
                ]
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                4
            )
        ]))

        story.extend([
            t,
            Spacer(1, 9)
        ])

    def figure(filename, max_height=180):

        chart = Image(
            str(figures / filename)
        )

        scale = min(
            500 / chart.imageWidth,
            max_height / chart.imageHeight
        )

        chart.drawWidth = (
            chart.imageWidth * scale
        )

        chart.drawHeight = (
            chart.imageHeight * scale
        )

        story.extend([
            chart,
            Spacer(1, 8)
        ])

    # PAGE 1 - EXECUTIVE SUMMARY

    page(1, "Executive summary")

    p(
        "Fraud Risk Intelligence Engine",
        "Title"
    )

    p(
        "Management strategy report | "
        "Experimental model | Version 1.0"
    )

    p(
        f"Recommended candidate: {name}, "
        f"{method} calibration. "

        f"Review threshold "
        f"{policy['review_threshold']:.3f}; "

        f"block threshold "
        f"{policy['block_threshold']:.3f}."
    )

    table(
        ["Final test indicator", "Result"],
        [
            [
                "Transactions / fraud",
                f"{int(final['transactions'])} / "
                f"{int(final['fraud_count'])}"
            ],

            [
                "Average Precision / ROC-AUC",
                f"{score(final['pr_auc'])} / "
                f"{score(final['roc_auc'])}"
            ],

            [
                "Expected fraud capture",
                pct(final["expected_fraud_capture"])
            ],

            [
                "Manual review rate",
                pct(final["review_rate"])
            ],

            [
                "Selected cost / approve-all cost",
                f"{money(final['total_cost'])} / "
                f"{money(final['approve_all_cost'])}"
            ],

            [
                "Simulated cost savings",
                f"{final['saving_pct']:.2f}%"
            ],

            [
                "Legitimate blocks",
                f"{int(final['legitimate_blocked'])} of "
                f"{int(final['block_count'])} recommendations"
            ]
        ],
        [235, 265]
    )

    p(
        "Recommendation: shadow-mode evaluation "
        "or an analyst-assisted pilot only.",
        "Heading2"
    )

    p(
        f"Financial savings are simulated using a fixed "
        f"loss of {money(c['fraud_loss'])} per missed fraud "
        f"and {pct(c['review_effectiveness'])} review "
        "effectiveness. These are not realized savings. "
        "Automatic blocking requires further independent "
        "validation and formal approval."
    )

    figure(
        "final_test_business_cost.png",
        155
    )

    # PAGE 2 - DATA AND VALIDATION

    page(2, "Data quality and evaluation design")

    table(
        [
            "Partition",
            "Rows",
            "Fraud cases",
            "Fraud rate"
        ],
        [
            [
                r["dataset"],
                int(r["total_transactions"]),
                int(r["fraudulent_transactions"]),
                f"{r['fraud_rate_percentage']:.2f}%"
            ]
            for _, r in split.iterrows()
        ],
        [185, 95, 110, 110]
    )

    p("Data audit and treatment", "Heading2")

    p(
        "The data audit investigates schema, duplicate "
        "IDs and rows, missingness associations, numerical "
        "extremes, cardinality and logical ranges. Difficult "
        "observations are retained. Missing values are "
        "preserved for pipeline imputation, with separate "
        "missing-category representation."
    )

    p("Leakage controls", "Heading2")

    p(
        "Fraud labels, transaction IDs and split metadata "
        "are excluded from predictors. Historical transaction "
        "counts, chargebacks, authentication and external "
        "risk scores require verified availability before "
        "the transaction decision."
    )

    p(
        "Full timestamps, customer identifiers, currency "
        "and feature lineage are unavailable. Random "
        "stratification cannot establish chronological "
        "or customer-level independence."
    )

    p(
        f"Fixed random seed: {policy['random_state']}. "
        "Train and tune using stratified five-fold CV. "
        "Fit preprocessing, learned feature transformations "
        "and resampling inside training folds. Fit calibration "
        "on a separate partition and select thresholds using "
        "validation data. Freeze before final testing."
    )

    p(
        "The earlier development audit included later "
        "calibration and validation observations. Development "
        "estimates may therefore be optimistic."
    )

    p(
        "Evidence: data_quality_report.md, "
        "data_split_summary.csv and "
        "leakage_feature_sensitivity.csv."
    )

    # PAGE 3 - EDA AND FEATURE ENGINEERING

    page(3, "Exploration and feature engineering")

    p(
        "Training-only EDA covers channel, merchant category, "
        "authentication, customer tier, device type and time. "
        "Nonlinear associations and three feature interactions "
        "are investigated. Low-support groups require cautious "
        "interpretation."
    )

    figure(
        "eda_18_authentication_channel.png",
        205
    )

    p(
        "Eighteen derived features capture velocity ratios, "
        "merchant-relative amounts, risk combinations, "
        "cyclical time, behavioral deviations and interactions. "
        "Customer-specific spending history is unavailable."
    )

    table(
        [
            "Model",
            "Original AP",
            "Engineered AP",
            "Relative change"
        ],
        [
            [
                r["model"],
                score(r["Original"]),
                score(r["Engineered"]),
                f"{r['relative_improvement_pct']:+.2f}%"
            ]
            for _, r in ablation.iterrows()
        ],
        [185, 105, 105, 105]
    )

    p(
        "Feature benefits are measured after independent "
        "hyperparameter tuning and are not established as "
        "statistically significant. The processed CSV must "
        "not be used directly for cross-validation because "
        "its normalization statistics were learned from "
        "the full training partition."
    )

    # PAGE 4 - MODEL DEVELOPMENT

    page(4, "Model optimization and imbalance")

    table(
        [
            "Candidate",
            "CV AP",
            "CV ROC-AUC",
            "AP SD"
        ],
        [
            [
                r["candidate"],
                score(r["pr_auc_mean"]),
                score(r["roc_auc_mean"]),
                score(r["pr_auc_std"])
            ]
            for _, r in tuning.iterrows()
        ],
        [248, 84, 84, 84]
    )

    p(
        f"Best CV AP: {top_ap['candidate']} "
        f"({score(top_ap['pr_auc_mean'])}). "

        f"Best CV ROC-AUC: {top_roc['candidate']} "
        f"({score(top_roc['roc_auc_mean'])}). "

        f"Selected candidate: {name}. "
        f"Its CV AP sample standard deviation is "
        f"{score(cv['pr_auc_std'])}."
    )

    table(
        [
            "Imbalance method",
            "AP",
            "Recall",
            "F1"
        ],
        [
            [
                r["strategy"],
                score(r["pr_auc_mean"]),
                pct(r["recall_mean"]),
                score(r["f1_mean"])
            ]
            for _, r in imbalance.iterrows()
        ],
        [200, 100, 100, 100]
    )

    p(
        "RandomizedSearchCV evaluates 12 configurations "
        "for each model-feature pair. Class weighting and "
        "controlled oversampling change the learning objective "
        "without changing natural validation prevalence."
    )

    p(
        "Recall improvements must be evaluated alongside "
        "false positives, ranking performance and "
        "probability calibration."
    )

    # PAGE 5 - CALIBRATION

    page(5, "Probability calibration")

    table(
        [
            "Method",
            "Validation Brier",
            "Log loss",
            "Mean risk"
        ],
        [
            [
                m,
                score(r["brier"]),
                score(r["log_loss"]),
                pct(r["mean_probability"])
            ]
            for m, r in probability.iterrows()
        ],
        [145, 125, 110, 120]
    )

    p(
        "Q3: Operational meaning of probability",
        "Heading2"
    )

    p(
        f"Mean uncalibrated risk for the selected classifier "
        f"was "
        f"{pct(probability.loc['uncalibrated', 'mean_probability'])}. "

        f"Mean selected-calibrated risk was "
        f"{pct(probability.loc[method, 'mean_probability'])}, "

        f"compared with observed prevalence "
        f"{pct(probability.loc[method, 'observed_fraud_rate'])}. "
        "A calibrated score of 0.20 should correspond to "
        "roughly 20% fraud among comparable transactions "
        "over time, not certainty about an individual."
    )

    table(
        [
            "Final test probability configuration",
            "Brier score"
        ],
        [
            [
                r["method"],
                score(r["brier"])
            ]
            for _, r in final_cal.iterrows()
        ],
        [350, 150]
    )

    figure(
        "final_test_calibration_curve.png",
        200
    )

    p(
        "Platt scaling is lower-complexity. Isotonic is "
        "more flexible but can introduce ties and extreme "
        "probabilities. Calibration requires independent "
        "evaluation and ongoing monitoring."
    )

    # PAGE 6 - COST AND THRESHOLDS

    page(6, "Financial cost and threshold sensitivity")

    table(
        [
            "Action",
            "Legitimate cost",
            "Fraud cost"
        ],
        [
            [
                "Approve",
                "$0",
                money(c["fraud_loss"])
            ],
            [
                "Review",
                money(c["review_fee"]),
                money(
                    c["review_fee"]
                    + (1 - c["review_effectiveness"])
                    * c["fraud_loss"]
                )
            ],
            [
                "Block",
                money(
                    c["block_fee"]
                    + c["customer_friction"]
                ),
                money(c["block_fee"])
            ]
        ],
        [170, 165, 165]
    )

    p(
        f"Review effectiveness: "
        f"{pct(c['review_effectiveness'])}. "

        f"Review capacity: {pct(policy['max_review_rate'])}; "
        f"block capacity: {pct(policy['max_block_rate'])}. "

        "Threshold search uses 0.005 increments. "
        "All financial assumptions are hypothetical."
    )

    table(
        [
            "Cost scenario",
            "Winning candidate",
            "Review / block",
            "Cost"
        ],
        [
            [
                r["scenario"],
                r["candidate"],
                f"{r['review_threshold']:.3f} / "
                f"{r['block_threshold']:.3f}",
                money(r["total_cost"])
            ]
            for _, r in sensitivity.iterrows()
        ],
        [125, 182, 98, 95]
    )

    p(
        "Q6: Financial assumptions influence the selected "
        "model and thresholds. Unchanged thresholds in some "
        "scenarios do not prove universal stability."
    )

    table(
        [
            "Block fee",
            "Frozen cost",
            "Hypothetical optimized cost"
        ],
        [
            [
                money(fee),

                money(
                    f.loc[
                        f["policy"] == "Frozen three-way",
                        "total_cost"
                    ].iloc[0]
                ),

                money(
                    f.loc[
                        f["policy"]
                        == "Scenario-only optimized three-way",
                        "total_cost"
                    ].iloc[0]
                )
            ]
            for fee, f in fees.groupby("block_fee")
        ],
        [140, 160, 200]
    )

    p(
        f"Q2: Highest CV ROC-AUC candidate "
        f"{top_roc['candidate']} has optimized calibrated "
        f"base cost {money(roc_policy['total_cost'])}. "
        f"The selected candidate costs "
        f"{money(winner['total_cost'])}. "
        "Ranking metrics alone do not determine business value."
    )

    # Q7: Find a measured lower-ROC, lower-cost alternative
    # under a non-base financial scenario.

    alternatives = costs.loc[
        (costs["policy"] == "Optimized three-way")
        & (costs["method"] != "uncalibrated")
        & (costs["scenario"] != "Base")
        & (costs["capacity_feasible"] == True)
    ].copy()

    alternatives = alternatives.merge(
        tuning[
            ["candidate", "roc_auc_mean"]
        ],
        on="candidate"
    )

    evidence = None

    for scenario, group in alternatives.groupby(
        "scenario"
    ):

        group = group.sort_values(
            "total_cost"
        ).drop_duplicates("candidate")

        high = group.sort_values(
            "roc_auc_mean",
            ascending=False
        ).iloc[0]

        lower = group.loc[
            (
                group["roc_auc_mean"]
                < high["roc_auc_mean"]
            )
            & (
                group["total_cost"]
                < high["total_cost"]
            )
        ]

        if not lower.empty:

            evidence = (
                scenario,
                lower.iloc[0],
                high
            )

            break

    if evidence:

        scenario, low, high = evidence

        p(
            f"Q7: Under {scenario}, "

            f"{low['candidate']} "
            f"(CV ROC {score(low['roc_auc_mean'])}) "
            f"costs {money(low['total_cost'])} versus "

            f"{money(high['total_cost'])} for "
            f"{high['candidate']} "
            f"(CV ROC {score(high['roc_auc_mean'])}). "

            "This is validation evidence, not a new "
            "model selection."
        )

    else:

        p(
            "Q7: No lower-ROC, lower-cost counterexample "
            "was found under the tested alternative cost "
            "assumptions. Do not claim one without evidence."
        )

    # PAGE 7 - FRAUD OPERATIONS

    page(7, "Operational and final test results")

    table(
        [
            "Decision",
            "Legitimate",
            "Fraud"
        ],
        [
            [
                label,
                int(r["Legitimate"]),
                int(r["Fraud"])
            ]
            for label, (_, r) in zip(
                ["Approve", "Review", "Block"],
                outcomes.iterrows()
            )
        ],
        [200, 150, 150]
    )

    table(
        [
            "Frozen test policy",
            "Review",
            "Block",
            "Cost"
        ],
        [
            [
                r["policy"],
                int(r["review_count"]),
                int(r["block_count"]),
                money(r["total_cost"])
            ]
            for _, r in final_policies.iterrows()
        ],
        [230, 80, 80, 110]
    )

    lo = capacity.iloc[0]
    hi = capacity.iloc[-1]

    p(
        "Q4: Manual review workload versus recall",
        "Heading2"
    )

    p(
        f"At {pct(lo['max_review_capacity'])} review "
        f"capacity, {int(lo['review_count'])} reviews "
        f"produce fraud flag recall "
        f"{pct(lo['fraud_flag_rate'])}. "

        f"At {pct(hi['max_review_capacity'])} capacity, "
        f"{int(hi['review_count'])} reviews produce recall "
        f"{pct(hi['fraud_flag_rate'])}. "

        "Expected fraud prevention accounts for imperfect "
        "manual-review effectiveness."
    )

    p(
        f"Final flag precision: "
        f"{pct(final['flag_precision'])}; "

        f"recall: {pct(final['flag_recall'])}; "

        f"F1: {score(final['flag_f1'])}. "

        f"Approximate bootstrap savings interval: "
        f"{conf.loc['saving_pct', 'lower_95']:.2f}% to "
        f"{conf.loc['saving_pct', 'upper_95']:.2f}%."
    )

    p(
        "At 10,000 daily transactions, validation rates "
        f"imply approximately "
        f"{winner['review_rate'] * 10000:,.0f} reviews "
        "per day. Staffing and review completion must be "
        "validated before deployment."
    )

    # PAGE 8 - EXPLAINABILITY

    page(8, "Explainability and predictor stability")

    p(
        "Q1: Strongest predictors and fold stability",
        "Heading2"
    )

    table(
        [
            "Top global SHAP feature",
            "Mean absolute SHAP (pp)"
        ],
        [
            [
                r["feature"],
                f"{r['mean_abs_shap'] * 100:.3f}"
            ]
            for _, r in important.iterrows()
        ],
        [350, 150]
    )

    table(
        [
            "Top permutation feature",
            "Mean AP decrease",
            "Positive folds"
        ],
        [
            [
                r["feature"],
                score(r["mean_importance"]),
                f"{int(r['positive_folds'])}/5"
            ]
            for _, r in stability.iterrows()
        ],
        [270, 130, 100]
    )

    p(
        "SHAP measures prediction influence. Permutation "
        "importance estimates predictive usefulness. "
        "Their rankings do not necessarily agree."
    )

    p(
        "Q5: Strongest feature interactions",
        "Heading2"
    )

    table(
        [
            "Feature pair",
            "Risk range (pp)",
            "Non-additive range (pp)"
        ],
        [
            [
                r["interaction"],
                f"{r['predicted_risk_range_pp']:.3f}",
                f"{r['interaction_range_pp']:.3f}"
            ]
            for _, r in interactions.iterrows()
        ],
        [240, 120, 140]
    )

    p(
        f"Largest non-additive grid effect: "
        f"{interactions.iloc[0]['interaction']}. "

        "These are partial-dependence measurements, not "
        "formal SHAP interaction values. Independent masking "
        "and artificial feature combinations may be unrealistic. "
        "Five detailed transaction explanations supplement "
        "the global analysis. Prediction is not causation."
    )

    # PAGE 9 - ROBUSTNESS AND GOVERNANCE

    page(9, "Robustness and governance")

    table(
        [
            "Scenario",
            "AP",
            "Review rate",
            "Score PSI",
            "Status"
        ],
        [
            [
                r["scenario"],
                score(r["pr_auc"]),
                pct(r["review_rate"]),
                f"{r['score_psi']:.3f}",
                r["status"]
            ]
            for _, r in monitor.iterrows()
        ],
        [140, 70, 85, 80, 125]
    )

    p(
        "Q8: Priority drift-monitoring fields",
        "Heading2"
    )

    p(
        "Monitor authentication, merchant and IP risk, "
        "prior chargebacks, hourly and daily velocity, "
        "amount, account age, distance, channel, device "
        "type and country."
    )

    p(
        "Monitor feature PSI, score PSI, missingness, "
        "unknown categories, extreme values, operational "
        "capacity, AP, Brier score, fraud capture, legitimate "
        "flags and simulated business cost."
    )

    p(
        f"PSI warning: {cfg['psi_warning']}; "
        f"major drift: {cfg['psi_major']}. "

        f"Authentication missingness of at least "
        f"{pct(cfg['authentication_missing_critical'])} "
        "is critical. Major degradation over two supported "
        "windows triggers a retraining review. Critical "
        "failures trigger escalation and an approved fallback."
    )

    p(
        "Stress tests demonstrate sensitivity to simulated "
        "input changes. They do not establish future "
        "fraud prevalence or actual concept drift. Savings "
        "under overloaded review conditions require "
        "particular caution."
    )

    # PAGE 10 - BUSINESS RECOMMENDATIONS

    page(10, "Strategic recommendations and rollout")

    table(
        [
            "Stage",
            "Required action / release gate"
        ],
        [
            [
                "1. Data readiness",

                "Verify feature lineage, complete timestamps, "
                "label maturity, monetary units and upstream "
                "service reliability."
            ],

            [
                "2. Shadow pilot",

                "Generate scores without executing financial "
                "decisions. Verify latency, explanations, "
                "drift and segment outcomes."
            ],

            [
                "3. Assisted review",

                "Validate staffing capacity, investigation "
                "effectiveness and actual costs. Establish "
                "approved overflow and appeal procedures."
            ],

            [
                "4. Release decision",

                "Require independent chronological evaluation, "
                "customer-impact review, business approval "
                "and versioned rollback. Keep automatic "
                "blocking disabled until approved."
            ],

            [
                "5. Ongoing governance",

                "Daily drift monitoring, weekly mature-label "
                "evaluation, monthly financial and fairness "
                "review, and immediate critical incident response."
            ]
        ],
        [140, 360]
    )

    p(
        "Key limitations include small fraud samples, "
        "random transaction partitions, unverified "
        "point-in-time features, hypothetical financial "
        "costs, modest discrimination, rare blocks and "
        "absence of live operational feedback."
    )

    p(
        "The existing final test dataset must not be "
        "reused for future model selection."
    )

    p("Evidence and reproducibility", "Heading2")

    p(
        "Source: DS 02.pdf assignment and "
        "transaction_fraud_risk_advanced_dataset.csv. "
        "Experimental results are generated by "
        "01_fraud_risk_analysis.ipynb and the metrics "
        "files in outputs/metrics. Governance details "
        "are documented in 06_model_card.pdf."
    )

    p(
        "Management decision: use the model as an "
        "experimental decision-support system. "
        "Do not authorize unrestricted automatic blocking "
        "from this evidence alone.",
        "Heading2"
    )

    # Footer

    def footer(canvas, doc):

        canvas.setFont("Helvetica", 8)

        canvas.setFillColor(
            colors.HexColor("#52616B")
        )

        canvas.drawString(
            46,
            28,
            "Fraud Risk Intelligence Engine | Management Report"
        )

        canvas.drawRightString(
            A4[0] - 46,
            28,
            str(doc.page)
        )

    # Generate report

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    doc = SimpleDocTemplate(
        str(output),

        pagesize=A4,

        leftMargin=46,
        rightMargin=46,

        topMargin=42,
        bottomMargin=52,

        title="Business Fraud Strategy"
    )

    doc.build(
        story,
        onFirstPage=footer,
        onLaterPages=footer
    )

    if not 8 <= doc.page <= 12:

        raise RuntimeError(
            f"Report contains {doc.page} pages. "
            "Expected 8-12 pages."
        )

    print(
        f"{output} ({doc.page} pages)"
    )

    return output


if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    generate(project_root)
