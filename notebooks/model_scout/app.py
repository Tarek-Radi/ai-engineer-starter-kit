from __future__ import annotations

import html
from pathlib import Path

import gradio as gr
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULTS_FILE = BASE_DIR / "data" / "final_result_of_comparison.csv"

REQUIRED_COLUMNS = [
    "model_name",
    "model_id",
    "accuracy_percent",
    "average_latency_seconds",
    "average_confidence_percent",
    "cost_usd",
    "cost_description",
    "correct_predictions",
    "wrong_predictions",
    "total_examples",
]


def load_results() -> pd.DataFrame:
    """Read, validate, clean, and rank the model-comparison results."""
    if not RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Comparison file was not found:\n{RESULTS_FILE}\n\n"
            "Create data/final_result_of_comparison.csv first."
        )

    df = pd.read_csv(RESULTS_FILE)

    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            "The CSV is missing required columns: "
            + ", ".join(missing_columns)
        )

    numeric_columns = [
        "accuracy_percent",
        "average_latency_seconds",
        "average_confidence_percent",
        "cost_usd",
        "correct_predictions",
        "wrong_predictions",
        "total_examples",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if df["model_name"].isna().any():
        raise ValueError("model_name contains an empty value.")

    if df["accuracy_percent"].isna().any():
        raise ValueError("accuracy_percent contains an invalid or empty value.")

    if df["average_latency_seconds"].isna().any():
        raise ValueError(
            "average_latency_seconds contains an invalid or empty value."
        )

    # Higher accuracy wins; latency breaks an accuracy tie.
    df = df.sort_values(
        by=["accuracy_percent", "average_latency_seconds"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df


def format_number(value: object, decimals: int = 2, suffix: str = "") -> str:
    if pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}{suffix}"


def render_dashboard() -> tuple[str, pd.DataFrame, str]:
    """Build the dashboard HTML and the detailed comparison table."""
    df = load_results()

    accuracy_winner = df.iloc[0]
    fastest_model = df.loc[df["average_latency_seconds"].idxmin()]

    confidence_rows = df.dropna(subset=["average_confidence_percent"])
    confidence_winner = (
        confidence_rows.loc[
            confidence_rows["average_confidence_percent"].idxmax()
        ]
        if not confidence_rows.empty
        else None
    )

    max_accuracy = max(float(df["accuracy_percent"].max()), 1.0)
    max_latency = max(float(df["average_latency_seconds"].max()), 0.0001)
    max_confidence = (
        max(float(confidence_rows["average_confidence_percent"].max()), 1.0)
        if not confidence_rows.empty
        else 100.0
    )

    cards = f"""
    <section class="metric-grid">
        <article class="metric-card-custom">
            <div class="metric-label">Accuracy winner</div>
            <div class="metric-value">{html.escape(str(accuracy_winner["model_name"]))}</div>
            <div class="metric-detail">
                {format_number(accuracy_winner["accuracy_percent"], 2, "%")}
            </div>
        </article>

        <article class="metric-card-custom">
            <div class="metric-label">Fastest model</div>
            <div class="metric-value">{html.escape(str(fastest_model["model_name"]))}</div>
            <div class="metric-detail">
                {format_number(fastest_model["average_latency_seconds"], 4, " s")}
            </div>
        </article>

        <article class="metric-card-custom">
            <div class="metric-label">Models evaluated</div>
            <div class="metric-value">{len(df)}</div>
            <div class="metric-detail">Same evaluation dataset</div>
        </article>
    </section>
    """

    rows_html: list[str] = []

    for rank, row in df.iterrows():
        model_name = html.escape(str(row["model_name"]))
        model_id = html.escape(str(row["model_id"]))
        cost_description = html.escape(str(row["cost_description"]))

        accuracy = float(row["accuracy_percent"])
        latency = float(row["average_latency_seconds"])
        confidence = row["average_confidence_percent"]

        accuracy_width = min((accuracy / max_accuracy) * 100, 100)
        # A shorter latency should still show a visible bar.
        latency_width = max(min((latency / max_latency) * 100, 100), 7)
        confidence_width = (
            min((float(confidence) / max_confidence) * 100, 100)
            if not pd.isna(confidence)
            else 0
        )

        winner_badge = (
            '<span class="winner-badge">Winner</span>'
            if rank == 0
            else f'<span class="rank-badge">#{rank + 1}</span>'
        )

        confidence_content = (
            f"""
            <div class="bar-line">
                <span>{format_number(confidence, 2, "%")}</span>
                <div class="bar-track">
                    <div class="bar-fill confidence-fill"
                         style="width: {confidence_width:.2f}%"></div>
                </div>
            </div>
            """
            if not pd.isna(confidence)
            else '<span class="muted-value">N/A</span>'
        )

        rows_html.append(
            f"""
            <article class="model-row">
                <div class="model-identity">
                    <div class="model-title-line">
                        {winner_badge}
                        <strong>{model_name}</strong>
                    </div>
                    <code>{model_id}</code>
                </div>

                <div class="metric-column">
                    <span class="column-label">Accuracy</span>
                    <div class="bar-line">
                        <span>{format_number(accuracy, 2, "%")}</span>
                        <div class="bar-track">
                            <div class="bar-fill accuracy-fill"
                                 style="width: {accuracy_width:.2f}%"></div>
                        </div>
                    </div>
                </div>

                <div class="metric-column">
                    <span class="column-label">Latency</span>
                    <div class="bar-line">
                        <span>{format_number(latency, 4, " s")}</span>
                        <div class="bar-track">
                            <div class="bar-fill latency-fill"
                                 style="width: {latency_width:.2f}%"></div>
                        </div>
                    </div>
                </div>

                <div class="metric-column">
                    <span class="column-label">Avg. confidence</span>
                    {confidence_content}
                </div>

                <div class="result-details">
                    <span>
                        Correct:
                        <strong>{int(row["correct_predictions"])}</strong>
                    </span>
                    <span>
                        Wrong:
                        <strong>{int(row["wrong_predictions"])}</strong>
                    </span>
                    <span>
                        Total:
                        <strong>{int(row["total_examples"])}</strong>
                    </span>
                    <span>
                        Cost:
                        <strong>${format_number(row["cost_usd"], 2)}</strong>
                    </span>
                    <span class="cost-note">{cost_description}</span>
                </div>
            </article>
            """
        )

    confidence_note = (
        f'{html.escape(str(confidence_winner["model_name"]))} had the '
        f'highest average confidence '
        f'({format_number(confidence_winner["average_confidence_percent"], 2, "%")}).'
        if confidence_winner is not None
        else "Average confidence was not available for the evaluated models."
    )

    recommendation = f"""
    <section class="recommendation">
        <div class="recommendation-kicker">Recommended model</div>
        <h2>{html.escape(str(accuracy_winner["model_name"]))}</h2>
        <p>
            It achieved the highest measured accuracy at
            <strong>{format_number(accuracy_winner["accuracy_percent"], 2, "%")}</strong>.
            The fastest model was
            <strong>{html.escape(str(fastest_model["model_name"]))}</strong>
            at
            <strong>{format_number(fastest_model["average_latency_seconds"], 4, " seconds")}</strong>.
            {confidence_note}
        </p>
    </section>
    """

    dashboard_html = (
        cards
        + '<section class="comparison-list">'
        + "".join(rows_html)
        + "</section>"
        + recommendation
    )

    display_df = df[
        [
            "model_name",
            "model_id",
            "accuracy_percent",
            "average_latency_seconds",
            "average_confidence_percent",
            "cost_usd",
            "correct_predictions",
            "wrong_predictions",
            "total_examples",
        ]
    ].copy()

    display_df.columns = [
        "Model",
        "Model ID",
        "Accuracy (%)",
        "Avg. latency (s)",
        "Avg. confidence (%)",
        "Cost (USD)",
        "Correct",
        "Wrong",
        "Total",
    ]

    status = (
        f"Loaded {len(df)} models from "
        f"`data/final_result_of_comparison.csv`."
    )

    return dashboard_html, display_df, status


CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
}

.hero {
    padding: 24px;
    border: 1px solid var(--border-color-primary);
    border-radius: 20px;
    background:
        radial-gradient(circle at top right,
        rgba(99, 102, 241, 0.16), transparent 38%),
        var(--background-fill-secondary);
    margin-bottom: 16px;
}

.hero h1 {
    margin: 0 0 8px;
    font-size: 32px;
}

.hero p {
    margin: 0;
    opacity: 0.82;
}

.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin: 4px 0 18px;
}

.metric-card-custom {
    padding: 18px;
    border: 1px solid var(--border-color-primary);
    border-radius: 16px;
    background: var(--background-fill-secondary);
}

.metric-label,
.column-label,
.recommendation-kicker {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    opacity: 0.65;
}

.metric-value {
    margin-top: 6px;
    font-size: 24px;
    font-weight: 700;
}

.metric-detail {
    margin-top: 5px;
    opacity: 0.76;
}

.comparison-list {
    display: grid;
    gap: 12px;
}

.model-row {
    display: grid;
    grid-template-columns: 1.35fr 1fr 1fr 1fr;
    gap: 18px;
    padding: 18px;
    border: 1px solid var(--border-color-primary);
    border-radius: 18px;
    background: var(--background-fill-primary);
}

.model-identity {
    min-width: 0;
}

.model-title-line {
    display: flex;
    align-items: center;
    gap: 9px;
    margin-bottom: 9px;
    font-size: 18px;
}

.model-identity code {
    display: block;
    overflow-wrap: anywhere;
    font-size: 12px;
    opacity: 0.72;
}

.winner-badge,
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 34px;
    padding: 4px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
}

.winner-badge {
    background: var(--button-primary-background-fill);
    color: var(--button-primary-text-color);
}

.rank-badge {
    background: var(--background-fill-secondary);
    border: 1px solid var(--border-color-primary);
}

.metric-column {
    min-width: 0;
}

.bar-line {
    margin-top: 10px;
}

.bar-line > span {
    display: block;
    margin-bottom: 7px;
    font-weight: 650;
}

.bar-track {
    width: 100%;
    height: 9px;
    overflow: hidden;
    border-radius: 999px;
    background: var(--background-fill-secondary);
}

.bar-fill {
    height: 100%;
    border-radius: inherit;
    background: var(--button-primary-background-fill);
}

.latency-fill {
    opacity: 0.62;
}

.confidence-fill {
    opacity: 0.8;
}

.muted-value {
    display: inline-block;
    margin-top: 10px;
    opacity: 0.55;
}

.result-details {
    grid-column: 1 / -1;
    display: flex;
    flex-wrap: wrap;
    gap: 8px 18px;
    padding-top: 14px;
    border-top: 1px solid var(--border-color-primary);
    font-size: 13px;
}

.cost-note {
    opacity: 0.64;
}

.recommendation {
    margin-top: 18px;
    padding: 20px;
    border-radius: 18px;
    border: 1px solid var(--border-color-primary);
    background: var(--background-fill-secondary);
}

.recommendation h2 {
    margin: 7px 0 8px;
}

.recommendation p {
    margin: 0;
    line-height: 1.7;
}

@media (max-width: 900px) {
    .model-row {
        grid-template-columns: 1fr 1fr;
    }

    .model-identity {
        grid-column: 1 / -1;
    }
}

@media (max-width: 640px) {
    .metric-grid,
    .model-row {
        grid-template-columns: 1fr;
    }

    .model-identity,
    .result-details {
        grid-column: 1;
    }

    .hero h1 {
        font-size: 25px;
    }
}
"""


with gr.Blocks(css=CSS, title="Model Scout Comparison") as demo:
    gr.HTML(
        """
        <section class="hero">
            <h1>Model Scout Comparison</h1>
            <p>
                A side-by-side benchmark of pretrained zero-shot
                classification models for technical job-post classification.
            </p>
        </section>
        """
    )

    dashboard = gr.HTML()

    with gr.Row():
        refresh_button = gr.Button(
            "Refresh comparison",
            variant="primary",
        )
        download_button = gr.DownloadButton(
            "Download comparison CSV",
            value=str(RESULTS_FILE),
        )

    status = gr.Markdown()

    with gr.Accordion("Detailed results table", open=False):
        details_table = gr.Dataframe(
            interactive=False,
            wrap=True,
        )

    demo.load(
        fn=render_dashboard,
        inputs=None,
        outputs=[dashboard, details_table, status],
    )

    refresh_button.click(
        fn=render_dashboard,
        inputs=None,
        outputs=[dashboard, details_table, status],
    )


if __name__ == "__main__":
    demo.launch(
        share=True,
        inbrowser=True,
    )