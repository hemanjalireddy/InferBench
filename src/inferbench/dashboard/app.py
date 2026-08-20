from __future__ import annotations

from ast import literal_eval
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_RESULTS_PATH = Path("results/prompt_length/results.csv")

REQUIRED_COLUMNS = {
    "experiment_type",
    "ttft_ms",
    "tpot_ms",
    "decode_tokens_per_second",
    "approximate_prefill_tokens_per_second",
}

def configure_page() -> None:
    st.set_page_config(
        page_title="InferBench Dashboard",
        layout="wide",
    )


@st.cache_data(show_spinner=False)
def load_results(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Results file not found: {path}"
        )

    dataframe = pd.read_csv(path)

    dataframe = normalize_prompt_length_columns(dataframe)

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    dataframe = dataframe[
        dataframe["experiment_type"] == "prompt_length"
    ].copy()

    if dataframe.empty:
        raise ValueError(
            "No prompt_length rows found in the results file."
        )

    numeric_columns = [
        "param_actual_prompt_tokens",
        "ttft_ms",
        "tpot_ms",
        "decode_tokens_per_second",
        "approximate_prefill_tokens_per_second",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    dataframe = dataframe.dropna(
        subset=["param_actual_prompt_tokens"]
    )

    return dataframe.sort_values(
        "param_actual_prompt_tokens"
    )


def normalize_prompt_length_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    if "param_actual_prompt_tokens" in dataframe.columns:
        return dataframe

    if "parameters" in dataframe.columns:
        actual_prompt_tokens = dataframe["parameters"].apply(
            extract_actual_prompt_tokens
        )

        if actual_prompt_tokens.notna().any():
            dataframe["param_actual_prompt_tokens"] = (
                actual_prompt_tokens
            )
            return dataframe

    if "prompt_tokens" in dataframe.columns:
        dataframe["param_actual_prompt_tokens"] = dataframe[
            "prompt_tokens"
        ]
        return dataframe

    return dataframe


def extract_actual_prompt_tokens(value: object) -> int | float | None:
    if not isinstance(value, str) or not value.strip():
        return None

    try:
        parameters = literal_eval(value)
    except (SyntaxError, ValueError):
        return None

    if not isinstance(parameters, dict):
        return None

    actual_prompt_tokens = parameters.get("actual_prompt_tokens")

    if isinstance(actual_prompt_tokens, int | float):
        return actual_prompt_tokens

    return None


def summarize_results(dataframe: pd.DataFrame) -> pd.DataFrame:
    return (
        dataframe.groupby(
            "param_actual_prompt_tokens",
            as_index=False,
        )
        .agg(
            median_ttft_ms=("ttft_ms", "median"),
            median_tpot_ms=("tpot_ms", "median"),
            median_decode_tokens_per_second=(
                "decode_tokens_per_second",
                "median",
            ),
            median_prefill_tokens_per_second=(
                "approximate_prefill_tokens_per_second",
                "median",
            ),
            runs=("experiment_type", "count"),
        )
        .sort_values("param_actual_prompt_tokens")
    )


def format_metric(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:,.2f}{suffix}"


def render_header(dataframe: pd.DataFrame, csv_path: str) -> None:
    st.title("InferBench")
    st.caption("Local LLM inference performance lab")

    left, middle, right = st.columns(3)

    left.metric(
        "Experiment",
        "Prompt length",
    )
    middle.metric(
        "Runs",
        f"{len(dataframe):,}",
    )
    right.metric(
        "Results file",
        csv_path,
    )


def render_kpis(dataframe: pd.DataFrame) -> None:
    columns = st.columns(4)

    metrics = [
        (
            "Median TTFT",
            dataframe["ttft_ms"].median(),
            " ms",
        ),
        (
            "Median TPOT",
            dataframe["tpot_ms"].median(),
            " ms/token",
        ),
        (
            "Median decode tok/s",
            dataframe["decode_tokens_per_second"].median(),
            " tok/s",
        ),
        (
            "Median prefill tok/s",
            dataframe[
                "approximate_prefill_tokens_per_second"
            ].median(),
            " tok/s",
        ),
    ]

    for column, (label, value, suffix) in zip(
        columns,
        metrics,
        strict=True,
    ):
        column.metric(
            label,
            format_metric(value, suffix),
        )


def make_line_chart(
    summary: pd.DataFrame,
    y_column: str,
    title: str,
    y_axis_title: str,
):
    fig = px.line(
        summary,
        x="param_actual_prompt_tokens",
        y=y_column,
        markers=True,
        title=title,
        labels={
            "param_actual_prompt_tokens": "Actual prompt tokens",
            y_column: y_axis_title,
        },
        hover_data={
            "param_actual_prompt_tokens": ":,.0f",
            y_column: ":,.2f",
            "runs": True,
        },
    )

    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 55, "b": 20},
        hovermode="x unified",
    )

    return fig


def render_charts(summary: pd.DataFrame) -> None:
    first_row = st.columns(2)

    first_row[0].plotly_chart(
        make_line_chart(
            summary=summary,
            y_column="median_ttft_ms",
            title="TTFT vs Prompt Length",
            y_axis_title="Median TTFT (ms)",
        ),
        use_container_width=True,
    )

    first_row[1].plotly_chart(
        make_line_chart(
            summary=summary,
            y_column="median_tpot_ms",
            title="TPOT vs Prompt Length",
            y_axis_title="Median TPOT (ms/token)",
        ),
        use_container_width=True,
    )

    second_row = st.columns(2)

    second_row[0].plotly_chart(
        make_line_chart(
            summary=summary,
            y_column="median_decode_tokens_per_second",
            title="Decode Throughput vs Prompt Length",
            y_axis_title="Median decode tokens/second",
        ),
        use_container_width=True,
    )

    second_row[1].plotly_chart(
        make_line_chart(
            summary=summary,
            y_column="median_prefill_tokens_per_second",
            title="Approximate Prefill Throughput vs Prompt Length",
            y_axis_title="Median prompt tokens/second",
        ),
        use_container_width=True,
    )


def render_data_tables(
    dataframe: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    with st.expander("Summary data", expanded=False):
        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Raw benchmark rows", expanded=False):
        st.dataframe(
            dataframe,
            use_container_width=True,
            hide_index=True,
        )


def render_sidebar() -> str:
    st.sidebar.header("Results")

    return st.sidebar.text_input(
        "CSV path",
        value=str(DEFAULT_RESULTS_PATH),
        help="Path is resolved from the InferBench project root.",
    )


def main() -> None:
    configure_page()

    csv_path = render_sidebar()

    try:
        dataframe = load_results(csv_path)
    except (FileNotFoundError, ValueError) as error:
        st.title("InferBench")
        st.error(str(error))
        st.info(
            "Run a prompt-length benchmark and save it to "
            "results/prompt_length/results.csv, or enter an existing "
            "CSV path in the sidebar."
        )
        return

    summary = summarize_results(dataframe)

    render_header(
        dataframe=dataframe,
        csv_path=csv_path,
    )
    render_kpis(dataframe)
    st.divider()
    render_charts(summary)
    render_data_tables(
        dataframe=dataframe,
        summary=summary,
    )


if __name__ == "__main__":
    main()
