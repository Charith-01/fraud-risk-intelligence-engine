
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
    PageBreak
)


def generate(root):

    root = Path(root)

    metrics = root / "outputs" / "metrics"

    models = root / "outputs" / "models"

    output = (
        root / "reports" / "06_model_card.pdf"
    )

    # Load actual model artifacts and evaluation results

    policy = json.loads(
        (models / "decision_policy.json")
        .read_text(encoding="utf-8")
    )

    ref = json.loads(
        (models / "monitoring_reference.json")
        .read_text(encoding="utf-8")
    )

    assert ref["policy"] == policy

    cfg = ref["config"]

    base = ref["baseline_metrics"]

    final = pd.read_csv(
        metrics / "final_test_metrics.csv"
    ).iloc[0]

    split = pd.read_csv(
        metrics / "data_split_summary.csv"
    )

    tuning = pd.read_csv(
        metrics / "tuned_model_comparison.csv"
    )

    selected = tuning.loc[
        tuning["candidate"] == policy["model"]
    ].iloc[0]

    stress = pd.read_csv(
        metrics / "monitoring_stress_summary.csv"
    )

    outage = stress.loc[
        stress["scenario"] == "Authentication outage"
    ].iloc[0]

    c = policy["cost_assumptions"]

    # Document styling

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

    def p(value, style="Normal"):

        story.append(
            Paragraph(
                escape(str(value)),
                styles[style]
            )
        )

        story.append(Spacer(1, 6))

    def table(headers, rows, widths):

        data = [headers] + rows

        cells = [

            [
                Paragraph(
                    escape(str(value)),
                    styles["Normal"]
                )

                for value in row
            ]

            for row in data
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
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            ),

            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ]))

        story.extend([
            t,
            Spacer(1, 10)
        ])

    def pct(value):

        return f"{float(value):.2%}"

    def money(value):

        return f"${float(value):,.2f}"

    # PAGE 1 - MODEL IDENTITY

    p(
        "Fraud Risk Intelligence Engine",
        "Title"
    )

    p(
        "01 / Model identity and intended use",
        "Heading1"
    )

    p(
        "Version 1.0 | Experimental candidate | "
        "Shadow mode or analyst-assisted pilot only"
    )

    table(
        ["Field", "Value"],

        [
            [
                "Selected model",
                policy["model"]
            ],

            [
                "Predictors",
                f"{int(selected['predictors'])} before encoding"
            ],

            [
                "Calibration",
                policy["calibration_method"]
            ],

            [
                "Review / block thresholds",
                f"{policy['review_threshold']:.3f} / "
                f"{policy['block_threshold']:.3f}"
            ],

            [
                "Validation review / block caps",
                f"{pct(policy['max_review_rate'])} / "
                f"{pct(policy['max_block_rate'])}"
            ],

            [
                "Random seed",
                policy["random_state"]
            ],

            [
                "Owner",
                "Fraud risk and model risk teams; "
                "named owner pending"
            ]
        ],

        [155, 345]
    )

    p(
        "Intended use: calibrated transaction risk scoring, "
        "review prioritization and simulated cost-sensitive "
        "decisions. Out of scope: unattended blocking, "
        "criminal-intent determination and unvalidated "
        "new customer populations."
    )

    p(
        "Financial assumptions",
        "Heading2"
    )

    table(
        ["Decision", "Legitimate cost", "Fraud cost"],

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

        [160, 170, 170]
    )

    p(
        f"Review effectiveness: "
        f"{pct(c['review_effectiveness'])}. "
        "All financial assumptions are hypothetical. "
        "Blocking recommendations are simulated; "
        "no live actions are authorized."
    )

    story.append(PageBreak())

    # PAGE 2 - DATA AND EVALUATION

    p(
        "02 / Data and evaluation",
        "Heading1"
    )

    p(
        "Source: transaction_fraud_risk_advanced_dataset.csv. "
        "The fraud target and transaction identifier are "
        "excluded from predictors. Currency, complete "
        "timestamps, customer identifiers and verified "
        "point-in-time feature lineage are unavailable."
    )

    table(
        ["Partition", "Rows", "Fraud"],

        [
            [
                row["dataset"],
                int(row["total_transactions"]),
                int(row["fraudulent_transactions"])
            ]

            for _, row in split.iterrows()
        ],

        [250, 125, 125]
    )

    p(
        "Stratified five-fold CV tunes classifier pipelines. "
        "Feature engineering, imputation, encoding, scaling "
        "and resampling are fitted within training folds. "
        "Calibration uses a separate partition; threshold "
        "validation selects the policy. The final test "
        "evaluates the frozen candidate only."
    )

    p(
        "Selected parameters: "
        + str(selected["best_params"])
    )

    table(
        [
            "Metric",
            "Validation",
            "Final test"
        ],

        [
            [
                "Average Precision (PR-AUC)",
                f"{base['pr_auc']:.4f}",
                f"{final['pr_auc']:.4f}"
            ],

            [
                "ROC-AUC",
                f"{base['roc_auc']:.4f}",
                f"{final['roc_auc']:.4f}"
            ],

            [
                "Brier score",
                f"{base['brier']:.4f}",
                f"{final['brier']:.4f}"
            ],

            [
                "Expected fraud capture",
                pct(base["expected_fraud_capture"]),
                pct(final["expected_fraud_capture"])
            ],

            [
                "Review rate",
                pct(base["review_rate"]),
                pct(final["review_rate"])
            ],

            [
                "Cost savings",
                f"{base['saving_pct']:.2f}%",
                f"{final['saving_pct']:.2f}%"
            ]
        ],

        [220, 140, 140]
    )

    p(
        f"Final simulated cost: "
        f"{money(final['total_cost'])} vs "
        f"{money(final['approve_all_cost'])} "
        "for approving all. "

        f"Of {int(final['block_count'])} block "
        f"recommendations, "
        f"{int(final['legitimate_blocked'])} "
        "were legitimate. "

        f"Test alert recall: "
        f"{pct(final['flag_recall'])}; "

        f"precision: {pct(final['flag_precision'])}."
    )

    story.append(PageBreak())

    # PAGE 3 - MONITORING AND RISKS

    p(
        "03 / Risks and monitoring",
        "Heading1"
    )

    p(
        "Authentication outage raised review workload to "
        f"{pct(outage['review_rate'])} and block workload "
        f"to {pct(outage['block_rate'])}. Fixed thresholds "
        "do not guarantee capacity compliance."
    )

    p(
        "Priority fields: authentication, merchant/IP risk, "
        "prior chargebacks, recent transaction counts, amount, "
        "distance, account age, channel, device type and "
        "country. Fixed training bins for feature PSI; "
        "threshold-validation scores for prediction PSI. "
        "Separate missing/unseen groups and half-count smoothing."
    )

    table(
        ["Metric", "Proposed trigger"],

        [
            [
                "Feature / score PSI",

                f"{cfg['psi_warning']} warning; "
                f"{cfg['psi_major']} major"
            ],

            [
                "Missingness increase",

                f"+{cfg['missing_increase_warning'] * 100:.0f} "
                "pp warning; "

                f"+{cfg['missing_increase_critical'] * 100:.0f} "
                "pp critical"
            ],

            [
                "Authentication missingness",

                "At least "
                f"{pct(cfg['authentication_missing_critical'])}"
            ],

            [
                "Unseen category rate",

                f"{pct(cfg['unknown_warning'])} warning; "
                f"{pct(cfg['unknown_critical'])} critical"
            ],

            [
                "Extreme amount / distance",

                f"Above {pct(cfg['tail_warning'])} beyond "
                "reference 99th percentile"
            ],

            [
                "Review / block rates",

                f"Above {pct(policy['max_review_rate'])} / "
                f"{pct(policy['max_block_rate'])}"
            ],

            [
                "AP / Brier score",

                f"AP drop > {pct(cfg['ap_relative_drop'])}; "
                f"Brier increase > {cfg['brier_increase']}"
            ],

            [
                "Fraud capture / legitimate flags",

                f"Drop / increase > "
                f"{cfg['capture_drop'] * 100:.0f} / "
                f"{cfg['legitimate_flag_increase'] * 100:.0f} pp"
            ],

            [
                "Cost savings / block precision",

                "Savings < 0%; precision < "
                f"{pct(cfg['block_precision_target'])} "
                "with at least "
                f"{cfg['minimum_blocks_for_precision']} blocks"
            ]
        ],

        [185, 315]
    )

    p(
        "These triggers are proposed heuristics, not validated "
        "SLA targets. A no-alert result does not certify safety. "
        "Monitor latency, queue backlog and actual review "
        "effectiveness once production telemetry exists."
    )

    story.append(PageBreak())

    # PAGE 4 - GOVERNANCE

    p(
        "04 / Governance and response",
        "Heading1"
    )

    table(
        [
            "Frequency",
            "Proposed owner and action"
        ],

        [
            [
                "Every batch",

                "Fraud operations: input validation, service "
                "failures, capacity and unsafe actions."
            ],

            [
                "Daily",

                "Data engineering: input drift, score drift "
                "and missingness."
            ],

            [
                "Weekly after label maturity",

                "Data science: discrimination, calibration "
                "and customer impact."
            ],

            [
                "Monthly / incidents",

                "Model risk and business owners: financial "
                "value, fairness, review and release approval."
            ]
        ],

        [155, 345]
    )

    p(
        f"Use at least {cfg['minimum_batch']} transactions "
        "for routine drift and labeled evaluation and at least "
        f"{cfg['minimum_fraud']} fraud and "
        f"{cfg['minimum_legitimate']} legitimate mature labels. "
        "Never treat pending labels as legitimate."
    )

    p(
        "Review: investigate moderate or major drift and "
        "degraded performance. Retraining/recalibration review: "
        "two consecutive adequately supported major-drift or "
        "degradation windows; investigate causes, train on "
        "new point-in-time data and validate on a new "
        "chronological holdout. Do not retrain on PSI alone "
        "or reuse final test data for selection."
    )

    p(
        "Temporary suspension: critical feed failure, invalid "
        "model input, unsafe workload or customer harm. Suspend "
        "automated actions as appropriate, alert the operational "
        "owner and use an approved manual/rules fallback. "
        "Do not silently approve overflow. Resume after recovery "
        "checks, shadow testing and sign-off."
    )

    p(
        "Limitations: small fraud samples, random transaction "
        "splits, unverified feature availability, development "
        "audit overlap, hypothetical costs, imperfect review "
        "and stress tests that do not reproduce concept drift. "
        "SHAP explains predictions, not causality. Monitor "
        "segment outcomes with counts and uncertainty; "
        "fairness is not certified. Protect transaction IDs "
        "and explanations, version artifacts and maintain "
        "an approved rollback plan."
    )

    p(
        "Recommendation: shadow mode or analyst-assisted "
        "pilot only. Automatic blocking disabled pending "
        "additional evidence and approval."
    )

    # Footer and page numbering

    def footer(canvas, doc):

        canvas.setFont(
            "Helvetica", 8
        )

        canvas.setFillColor(
            colors.HexColor("#52616B")
        )

        canvas.drawString(
            46, 28,
            "Fraud Risk Intelligence Engine | Model Card v1.0"
        )

        canvas.drawRightString(
            A4[0] - 46,
            28,
            str(doc.page)
        )

    # Generate PDF

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

        title=(
            "Fraud Risk Intelligence Engine - Model Card"
        )
    )

    doc.build(
        story,

        onFirstPage=footer,

        onLaterPages=footer
    )

    return output


if __name__ == "__main__":

    project_root = (
        Path(__file__).resolve().parents[1]
    )

    print(generate(project_root))
