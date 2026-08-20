from __future__ import annotations

import json
from ast import literal_eval
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


DEFAULT_RESULTS_PATH = Path("results/prompt_length/results.csv")
DEFAULT_NO_KV_CACHE_PATH = Path("results/prompt_length_no_kv_cache/results.csv")
RESULTS_ROOT = Path("results")

REQUIRED_COLUMNS = {
    "experiment_type",
    "ttft_ms",
    "tpot_ms",
    "decode_tokens_per_second",
    "approximate_prefill_tokens_per_second",
}

METRICS = {
    "median_ttft_ms": ("TTFT", "Median TTFT (ms)", "lower"),
    "median_tpot_ms": ("TPOT", "Median TPOT (ms/token)", "lower"),
    "median_decode_tokens_per_second": (
        "Decode throughput",
        "Median decode tokens/second",
        "higher",
    ),
    "median_prefill_tokens_per_second": (
        "Prefill throughput",
        "Median prompt tokens/second",
        "higher",
    ),
}


def configure_page() -> None:
    st.set_page_config(
        page_title="InferBench Dashboard",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        :root {
            --ib-bg: #0f1117;
            --ib-panel: #171d27;
            --ib-panel-strong: #202938;
            --ib-ink: #f5f7fb;
            --ib-muted: #aab6c5;
            --ib-line: #364253;
            --ib-accent: #53c7de;
            --ib-accent-2: #ffb86b;
            --ib-good: #8bd17c;
        }
        .stApp {
            background: var(--ib-bg);
        }
        section[data-testid="stSidebar"] {
            background: #1b202b;
            border-right: 1px solid var(--ib-line);
        }
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1240px;
        }
        h1, h2, h3, p, label {
            color: var(--ib-ink) !important;
            letter-spacing: 0;
        }
        [data-testid="stMetric"] {
            background: var(--ib-panel);
            border: 1px solid var(--ib-line);
            border-radius: 8px;
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
        }
        [data-testid="stMetricLabel"] {
            color: var(--ib-muted) !important;
        }
        [data-testid="stMetricValue"] {
            color: var(--ib-ink) !important;
        }
        .ib-hero {
            border-bottom: 1px solid var(--ib-line);
            padding-bottom: 1.1rem;
            margin-bottom: 1.1rem;
        }
        .ib-eyebrow {
            color: var(--ib-accent);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.3rem;
        }
        .ib-subtitle {
            color: var(--ib-muted);
            font-size: 1rem;
            max-width: 820px;
        }
        .ib-badges {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.85rem;
        }
        .ib-badge {
            background: var(--ib-panel-strong);
            border: 1px solid var(--ib-line);
            border-radius: 999px;
            color: var(--ib-ink);
            display: inline-flex;
            font-size: 0.82rem;
            font-weight: 650;
            padding: 0.28rem 0.7rem;
        }
        .ib-badge strong {
            color: var(--ib-accent);
            margin-right: 0.25rem;
        }
        .ib-note {
            background: var(--ib-panel-strong);
            border: 1px solid var(--ib-line);
            border-left: 4px solid var(--ib-accent-2);
            border-radius: 8px;
            color: var(--ib-muted);
            padding: 0.8rem 1rem;
            margin: 0.5rem 0 1.1rem;
        }
        .stDataFrame {
            border: 1px solid var(--ib-line);
            border-radius: 8px;
        }
        div[data-baseweb="radio"] {
            background: var(--ib-panel);
            border: 1px solid var(--ib-line);
            border-radius: 8px;
            padding: 0.35rem 0.6rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_results(csv_path: str, label: str) -> pd.DataFrame:
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
    dataframe["run_label"] = label
    dataframe["source_path"] = str(path)

    return dataframe.sort_values(
        ["run_label", "param_actual_prompt_tokens"]
    )


def normalize_prompt_length_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    if "param_actual_prompt_tokens" not in dataframe.columns:
        dataframe = add_prompt_token_column(dataframe)

    if "backend_use_kv_cache" not in dataframe.columns:
        dataframe["backend_use_kv_cache"] = dataframe.get(
            "generation_use_kv_cache",
            "unknown",
        )

    if "backend_flash_attention" not in dataframe.columns:
        dataframe["backend_flash_attention"] = dataframe.get(
            "generation_flash_attention",
            False,
        )

    if "backend_attention_backend" not in dataframe.columns:
        dataframe["backend_attention_backend"] = dataframe.get(
            "generation_attention_backend",
            "torch_default",
        )

    dataframe["kv_cache_mode"] = dataframe["backend_use_kv_cache"].apply(
        format_kv_cache_mode
    )

    return dataframe


def format_kv_cache_mode(value: object) -> str:
    text = str(value).strip().lower()

    if text in {"true", "1", "yes"}:
        return "With KV cache"

    if text in {"false", "0", "no"}:
        return "Without KV cache"

    return "KV cache unknown"


def add_prompt_token_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    if "parameters" in dataframe.columns:
        actual_prompt_tokens = dataframe["parameters"].apply(
            extract_actual_prompt_tokens
        )

        if actual_prompt_tokens.notna().any():
            dataframe["param_actual_prompt_tokens"] = actual_prompt_tokens
            return dataframe

    if "prompt_tokens" in dataframe.columns:
        dataframe["param_actual_prompt_tokens"] = dataframe["prompt_tokens"]

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
            [
                "run_label",
                "param_actual_prompt_tokens",
            ],
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
            kv_cache=("backend_use_kv_cache", "first"),
            kv_cache_mode=("kv_cache_mode", "first"),
            attention_backend=("backend_attention_backend", "first"),
            flash_attention=("backend_flash_attention", "first"),
            source_path=("source_path", "first"),
        )
        .sort_values(["run_label", "param_actual_prompt_tokens"])
    )


def format_metric(value: float | int | None, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"

    return f"{value:,.2f}{suffix}"


def discover_result_csvs() -> list[str]:
    if not RESULTS_ROOT.exists():
        return []

    return [
        str(path)
        for path in sorted(RESULTS_ROOT.rglob("results.csv"))
    ]


def metadata_path_for_csv(csv_path: str) -> Path:
    return Path(csv_path).parent / "metadata.json"


def load_metadata_for_csv(csv_path: str) -> dict[str, object]:
    metadata_path = metadata_path_for_csv(csv_path)

    if not metadata_path.exists():
        return {}

    with metadata_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def render_header(dataframe: pd.DataFrame) -> None:
    labels = ", ".join(dataframe["run_label"].drop_duplicates().astype(str))
    min_tokens = dataframe["param_actual_prompt_tokens"].min()
    max_tokens = dataframe["param_actual_prompt_tokens"].max()
    modes = sorted(dataframe["kv_cache_mode"].drop_duplicates())
    attention_modes = sorted(
        dataframe["backend_attention_backend"].astype(str).drop_duplicates()
    )
    mode_badges = "".join(
        f'<span class="ib-badge"><strong>KV</strong>{mode}</span>'
        for mode in modes
    )
    attention_badges = "".join(
        f'<span class="ib-badge"><strong>Attention</strong>{mode}</span>'
        for mode in attention_modes
    )

    st.markdown(
        f"""
        <div class="ib-hero">
          <div class="ib-eyebrow">InferBench</div>
          <h1>Inference Performance Lab</h1>
          <div class="ib-subtitle">
            Compare prompt-length scaling across runtime settings. Loaded runs:
            <strong>{labels}</strong>. Token range: {min_tokens:,.0f} to
            {max_tokens:,.0f}.
          </div>
          <div class="ib-badges">
            {mode_badges}
            {attention_badges}
            <span class="ib-badge"><strong>Flash</strong>Off</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="ib-note">
          This CPU PyTorch backend compares KV-cache behavior and regular
          PyTorch attention implementations such as eager and SDPA. Flash
          Attention is intentionally shown as off; it should be added later
          through a GPU/runtime-specific backend.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(summary: pd.DataFrame) -> None:
    latest_label = summary["run_label"].iloc[-1]
    latest = summary[summary["run_label"] == latest_label]
    columns = st.columns(4)

    metrics = [
        ("Median TTFT", latest["median_ttft_ms"].median(), " ms"),
        ("Median TPOT", latest["median_tpot_ms"].median(), " ms/token"),
        (
            "Median decode",
            latest["median_decode_tokens_per_second"].median(),
            " tok/s",
        ),
        (
            "Median prefill",
            latest["median_prefill_tokens_per_second"].median(),
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


def render_kv_cache_filter(dataframe: pd.DataFrame) -> pd.DataFrame:
    available_modes = list(dataframe["kv_cache_mode"].drop_duplicates())

    if len(available_modes) <= 1:
        st.caption(
            f"KV cache view: {available_modes[0]}"
        )
        return dataframe

    options = [
        "All",
        "With KV cache",
        "Without KV cache",
    ]
    visible_options = [
        option
        for option in options
        if option == "All" or option in available_modes
    ]

    selected_mode = st.radio(
        "KV cache view",
        options=visible_options,
        horizontal=True,
        help="Filter the plots and comparison table by KV-cache mode.",
    )

    if selected_mode == "All":
        return dataframe

    return dataframe[
        dataframe["kv_cache_mode"] == selected_mode
    ].copy()


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
        color="run_label",
        markers=True,
        title=title,
        labels={
            "param_actual_prompt_tokens": "Prompt tokens",
            y_column: y_axis_title,
            "run_label": "Run",
        },
        hover_data={
            "param_actual_prompt_tokens": ":,.0f",
            y_column: ":,.2f",
            "runs": True,
            "kv_cache_mode": True,
            "attention_backend": True,
            "flash_attention": True,
        },
        color_discrete_sequence=[
            "#3fb6cc",
            "#f1a35b",
            "#7fd17f",
            "#b995ff",
        ],
    )

    fig.update_layout(
        margin={"l": 20, "r": 20, "t": 58, "b": 24},
        hovermode="x unified",
        plot_bgcolor="#151b25",
        paper_bgcolor="#151b25",
        legend_title_text="Run",
        font={"family": "Arial", "size": 13, "color": "#f5f7fb"},
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#303b4c",
        zeroline=False,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#303b4c",
        zeroline=False,
    )

    return fig


def render_charts(summary: pd.DataFrame) -> None:
    st.subheader("Prompt-Length Scaling")

    first_row = st.columns(2)
    chart_items = list(METRICS.items())

    for column, (metric_column, (_, axis_title, _)) in zip(
        first_row,
        chart_items[:2],
        strict=True,
    ):
        column.plotly_chart(
            make_line_chart(
                summary=summary,
                y_column=metric_column,
                title=axis_title,
                y_axis_title=axis_title,
            ),
            use_container_width=True,
        )

    second_row = st.columns(2)

    for column, (metric_column, (_, axis_title, _)) in zip(
        second_row,
        chart_items[2:],
        strict=True,
    ):
        column.plotly_chart(
            make_line_chart(
                summary=summary,
                y_column=metric_column,
                title=axis_title,
                y_axis_title=axis_title,
            ),
            use_container_width=True,
        )


def build_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    labels = list(summary["run_label"].drop_duplicates())

    if len(labels) < 2:
        return pd.DataFrame()

    baseline_label = labels[0]
    comparison_label = labels[1]
    baseline = summary[summary["run_label"] == baseline_label]
    comparison = summary[summary["run_label"] == comparison_label]

    merged = baseline.merge(
        comparison,
        on="param_actual_prompt_tokens",
        suffixes=("_baseline", "_comparison"),
    )

    rows = []

    for metric_column, (metric_name, _, direction) in METRICS.items():
        baseline_column = f"{metric_column}_baseline"
        comparison_column = f"{metric_column}_comparison"
        delta = merged[comparison_column] - merged[baseline_column]
        percent = delta / merged[baseline_column] * 100

        if direction == "lower":
            verdict = percent.apply(
                lambda value: "better" if value < 0 else "slower"
            )
        else:
            verdict = percent.apply(
                lambda value: "better" if value > 0 else "lower"
            )

        metric_rows = pd.DataFrame(
            {
                "prompt_tokens": merged["param_actual_prompt_tokens"],
                "metric": metric_name,
                f"{baseline_label}": merged[baseline_column],
                f"{comparison_label}": merged[comparison_column],
                "delta": delta,
                "delta_percent": percent,
                "comparison": verdict,
            }
        )
        rows.append(metric_rows)

    return pd.concat(rows, ignore_index=True)


def build_run_profiles(summary: pd.DataFrame) -> pd.DataFrame:
    return (
        summary.groupby(
            [
                "run_label",
                "kv_cache_mode",
                "attention_backend",
            ],
            as_index=False,
        )
        .agg(
            median_ttft_ms=("median_ttft_ms", "median"),
            median_tpot_ms=("median_tpot_ms", "median"),
            median_decode_tokens_per_second=(
                "median_decode_tokens_per_second",
                "median",
            ),
            median_prefill_tokens_per_second=(
                "median_prefill_tokens_per_second",
                "median",
            ),
            prompt_points=("param_actual_prompt_tokens", "count"),
            total_runs=("runs", "sum"),
        )
    )


def normalize_for_score(
    series: pd.Series,
    direction: str,
) -> pd.Series:
    if series.empty:
        return series

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )
    positive = numeric[numeric > 0]

    if positive.empty:
        return pd.Series(
            [0.0] * len(series),
            index=series.index,
        )

    if direction == "lower":
        best = positive.min()
        return (best / numeric).fillna(0.0)

    best = positive.max()
    return (numeric / best).fillna(0.0)


def score_profiles(
    profiles: pd.DataFrame,
    latency_weight: float,
) -> pd.DataFrame:
    scored = profiles.copy()
    throughput_weight = 1.0 - latency_weight

    ttft_score = normalize_for_score(
        scored["median_ttft_ms"],
        "lower",
    )
    tpot_score = normalize_for_score(
        scored["median_tpot_ms"],
        "lower",
    )
    decode_score = normalize_for_score(
        scored["median_decode_tokens_per_second"],
        "higher",
    )
    prefill_score = normalize_for_score(
        scored["median_prefill_tokens_per_second"],
        "higher",
    )

    scored["latency_score"] = (ttft_score + tpot_score) / 2
    scored["throughput_score"] = (decode_score + prefill_score) / 2
    scored["workload_score"] = (
        scored["latency_score"] * latency_weight
        + scored["throughput_score"] * throughput_weight
    )

    if latency_weight >= 0.75:
        scored["tie_breaker"] = "Lower TTFT, then lower TPOT"
        sort_columns = [
            "workload_score",
            "median_ttft_ms",
            "median_tpot_ms",
        ]
        ascending = [
            False,
            True,
            True,
        ]
    elif latency_weight <= 0.25:
        scored["tie_breaker"] = "Higher decode, then higher prefill"
        sort_columns = [
            "workload_score",
            "median_decode_tokens_per_second",
            "median_prefill_tokens_per_second",
        ]
        ascending = [
            False,
            False,
            False,
        ]
    else:
        scored["tie_breaker"] = "Balanced score, then TTFT"
        sort_columns = [
            "workload_score",
            "latency_score",
            "throughput_score",
            "median_ttft_ms",
        ]
        ascending = [
            False,
            False,
            False,
            True,
        ]

    scored = scored.sort_values(
        sort_columns,
        ascending=ascending,
    )
    scored["rank"] = range(1, len(scored) + 1)

    return scored


def build_metric_winners(profiles: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for metric_column, (metric_name, _, direction) in METRICS.items():
        ascending = direction == "lower"
        winner = profiles.sort_values(
            metric_column,
            ascending=ascending,
        ).iloc[0]

        rows.append(
            {
                "metric": metric_name,
                "best_run": winner["run_label"],
                "kv_cache": winner["kv_cache_mode"],
                "attention": winner["attention_backend"],
                "value": winner[metric_column],
                "why": (
                    "lower is better"
                    if direction == "lower"
                    else "higher is better"
                ),
            }
        )

    return pd.DataFrame(rows)


def render_ranking(summary: pd.DataFrame) -> None:
    profiles = build_run_profiles(summary)

    if profiles.empty:
        return

    st.subheader("Best Configurations")

    winners = build_metric_winners(profiles)
    st.dataframe(
        winners,
        use_container_width=True,
        hide_index=True,
        column_config={
            "metric": "Metric",
            "best_run": "Best run",
            "kv_cache": "KV cache",
            "attention": "Attention",
            "value": st.column_config.NumberColumn(
                "Value",
                format="%.2f",
            ),
            "why": "Rule",
        },
    )

    latency_weight = st.slider(
        "Workload preference",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        format="%d%% latency",
        help=(
            "Higher values favor TTFT and TPOT. Lower values favor decode "
            "and prefill throughput."
        ),
    ) / 100

    scored = score_profiles(
        profiles,
        latency_weight=latency_weight,
    )

    st.dataframe(
        scored[
            [
                "rank",
                "run_label",
                "kv_cache_mode",
                "attention_backend",
                "workload_score",
                "latency_score",
                "throughput_score",
                "tie_breaker",
                "median_ttft_ms",
                "median_tpot_ms",
                "median_decode_tokens_per_second",
                "median_prefill_tokens_per_second",
            ]
        ],
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": st.column_config.NumberColumn(
                "Rank",
                format="%.0f",
            ),
            "run_label": "Run",
            "kv_cache_mode": "KV cache",
            "attention_backend": "Attention",
            "workload_score": st.column_config.NumberColumn(
                "Workload score",
                format="%.3f",
                help=(
                    "Ratio-based score. For latency, best/value; for "
                    "throughput, value/best. Higher is better."
                ),
            ),
            "latency_score": st.column_config.NumberColumn(
                "Latency score",
                format="%.3f",
            ),
            "throughput_score": st.column_config.NumberColumn(
                "Throughput score",
                format="%.3f",
            ),
            "tie_breaker": "Tie breaker",
            "median_ttft_ms": st.column_config.NumberColumn(
                "TTFT ms",
                format="%.2f",
            ),
            "median_tpot_ms": st.column_config.NumberColumn(
                "TPOT ms/token",
                format="%.2f",
            ),
            "median_decode_tokens_per_second": st.column_config.NumberColumn(
                "Decode tok/s",
                format="%.2f",
            ),
            "median_prefill_tokens_per_second": st.column_config.NumberColumn(
                "Prefill tok/s",
                format="%.2f",
            ),
        },
    )


def render_comparison(summary: pd.DataFrame) -> None:
    comparison = build_comparison_table(summary)

    if comparison.empty:
        st.info(
            "Load an uncached CSV in the sidebar to see side-by-side deltas."
        )
        return

    st.subheader("Run Comparison")
    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
        column_config={
            "prompt_tokens": st.column_config.NumberColumn(
                "Prompt tokens",
                format="%.0f",
            ),
            "delta": st.column_config.NumberColumn(
                "Delta",
                format="%.2f",
            ),
            "delta_percent": st.column_config.NumberColumn(
                "Delta %",
                format="%.2f%%",
            ),
        },
    )


def render_data_tables(
    dataframe: pd.DataFrame,
    summary: pd.DataFrame,
    csv_paths: list[str],
) -> None:
    with st.expander("Run metadata", expanded=False):
        for csv_path in csv_paths:
            metadata = load_metadata_for_csv(csv_path)

            if metadata:
                st.markdown(f"**{csv_path}**")
                st.json(metadata)
            else:
                st.caption(f"No metadata found for {csv_path}")

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


def render_sidebar() -> tuple[str, str, str, str | None]:
    st.sidebar.header("Results")
    discovered_csvs = discover_result_csvs()

    primary_options = discovered_csvs or [str(DEFAULT_RESULTS_PATH)]

    primary_label = st.sidebar.text_input(
        "Primary label",
        value="With KV cache",
    )
    primary_path = st.sidebar.selectbox(
        "Primary CSV path",
        options=primary_options,
        index=(
            primary_options.index(str(DEFAULT_RESULTS_PATH))
            if str(DEFAULT_RESULTS_PATH) in primary_options
            else 0
        ),
        help="Discovered from results/**/results.csv.",
    )
    primary_path = st.sidebar.text_input(
        "Edit primary path",
        value=primary_path,
    )
    comparison_enabled = st.sidebar.toggle(
        "Compare without KV cache",
        value=False,
        help=(
            "Load a second CSV created with --no-kv-cache."
        ),
    )

    comparison_label = "Without KV cache"
    comparison_path = None

    if comparison_enabled:
        comparison_label = st.sidebar.text_input(
            "Comparison label",
            value="Without KV cache",
        )
        comparison_path = st.sidebar.text_input(
            "Comparison CSV path",
            value=str(DEFAULT_NO_KV_CACHE_PATH),
        )

    st.sidebar.divider()
    st.sidebar.caption(
        "For KV-cache comparison, run once normally and once with "
        "--no-kv-cache."
    )

    return (
        primary_label,
        primary_path,
        comparison_label,
        comparison_path,
    )


def main() -> None:
    configure_page()

    (
        primary_label,
        primary_path,
        comparison_label,
        comparison_path,
    ) = render_sidebar()

    try:
        csv_paths = [primary_path]
        frames = [
            load_results(
                primary_path,
                primary_label,
            )
        ]

        if comparison_path:
            csv_paths.append(comparison_path)
            frames.append(
                load_results(
                    comparison_path,
                    comparison_label,
                )
            )

        dataframe = pd.concat(
            frames,
            ignore_index=True,
        )
    except (FileNotFoundError, ValueError) as error:
        st.title("InferBench")
        st.error(str(error))
        st.info(
            "Run a prompt-length benchmark and save it to "
            "results/prompt_length/results.csv, or enter an existing "
            "CSV path in the sidebar."
        )
        return

    render_header(dataframe)
    filtered_dataframe = render_kv_cache_filter(dataframe)
    summary = summarize_results(filtered_dataframe)
    render_kpis(summary)
    st.divider()
    render_charts(summary)
    st.divider()
    render_ranking(summary)
    st.divider()
    render_comparison(summary)
    render_data_tables(
        dataframe=dataframe,
        summary=summary,
        csv_paths=csv_paths,
    )


if __name__ == "__main__":
    main()
