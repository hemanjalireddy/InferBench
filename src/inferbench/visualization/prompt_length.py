from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {
    "experiment_type",
    "param_actual_prompt_tokens",
    "ttft_ms",
    "approximate_prefill_tokens_per_second",
    "decode_tokens_per_second",
    "tpot_ms",
}


def load_prompt_length_results(
    csv_path: str | Path,
) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {csv_path}"
        )

    dataframe = pd.read_csv(csv_path)

    missing_columns = (
        REQUIRED_COLUMNS - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    dataframe = dataframe[
        dataframe["experiment_type"] == "prompt_length"
    ].copy()

    if dataframe.empty:
        raise ValueError(
            "No prompt_length experiment rows found."
        )

    dataframe["param_actual_prompt_tokens"] = pd.to_numeric(
        dataframe["param_actual_prompt_tokens"]
    )

    return dataframe


def summarize_prompt_length_results(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    summary = (
        dataframe
        .groupby(
            "param_actual_prompt_tokens",
            as_index=False,
        )
        .agg(
            median_ttft_ms=(
                "ttft_ms",
                "median",
            ),
            median_prefill_tokens_per_second=(
                "approximate_prefill_tokens_per_second",
                "median",
            ),
            median_decode_tokens_per_second=(
                "decode_tokens_per_second",
                "median",
            ),
            median_tpot_ms=(
                "tpot_ms",
                "median",
            ),
            p95_ttft_ms=(
                "ttft_ms",
                lambda values: values.quantile(0.95),
            ),
        )
        .sort_values(
            "param_actual_prompt_tokens"
        )
    )

    return summary


def _plot_line(
    summary: pd.DataFrame,
    y_column: str,
    title: str,
    ylabel: str,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots()

    ax.plot(
        summary["param_actual_prompt_tokens"],
        summary[y_column],
        marker="o",
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Prompt tokens"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(fig)

    return output_path


def plot_ttft_vs_prompt_length(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    return _plot_line(
        summary=summary,
        y_column="median_ttft_ms",
        title="TTFT vs Prompt Length",
        ylabel="Median TTFT (ms)",
        output_path=output_path,
    )


def plot_prefill_throughput_vs_prompt_length(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    return _plot_line(
        summary=summary,
        y_column="median_prefill_tokens_per_second",
        title="Approximate Prefill Throughput vs Prompt Length",
        ylabel="Median prompt tokens / second",
        output_path=output_path,
    )


def plot_decode_throughput_vs_prompt_length(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    return _plot_line(
        summary=summary,
        y_column="median_decode_tokens_per_second",
        title="Decode Throughput vs Prompt Length",
        ylabel="Median decode tokens / second",
        output_path=output_path,
    )


def plot_tpot_vs_prompt_length(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    return _plot_line(
        summary=summary,
        y_column="median_tpot_ms",
        title="TPOT vs Prompt Length",
        ylabel="Median TPOT (ms/token)",
        output_path=output_path,
    )


def generate_prompt_length_plots(
    csv_path: str | Path,
    output_directory: str | Path,
) -> list[Path]:
    dataframe = load_prompt_length_results(
        csv_path
    )

    summary = summarize_prompt_length_results(
        dataframe
    )

    output_directory = Path(
        output_directory
    )

    plots = [
        plot_ttft_vs_prompt_length(
            summary,
            output_directory
            / "ttft_vs_prompt_length.png",
        ),
        plot_prefill_throughput_vs_prompt_length(
            summary,
            output_directory
            / "prefill_throughput_vs_prompt_length.png",
        ),
        plot_decode_throughput_vs_prompt_length(
            summary,
            output_directory
            / "decode_throughput_vs_prompt_length.png",
        ),
        plot_tpot_vs_prompt_length(
            summary,
            output_directory
            / "tpot_vs_prompt_length.png",
        ),
    ]

    return plots