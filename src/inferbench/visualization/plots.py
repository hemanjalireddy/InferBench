from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_results(
    csv_path: str | Path,
) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {csv_path}"
        )

    return pd.read_csv(csv_path)


def summarize_for_plotting(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    required_columns = {
        "case_name",
        "ttft_ms",
        "decode_tokens_per_second",
        "tpot_ms",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    summary = (
        dataframe
        .groupby(
            "case_name",
            as_index=False,
        )
        .agg(
            median_ttft_ms=(
                "ttft_ms",
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
        )
    )

    return summary


def plot_ttft(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots()

    ax.bar(
        summary["case_name"],
        summary["median_ttft_ms"],
    )

    ax.set_title(
        "Median Time to First Token"
    )

    ax.set_xlabel(
        "Benchmark case"
    )

    ax.set_ylabel(
        "TTFT (ms)"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(fig)

    return output_path


def plot_decode_throughput(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots()

    ax.bar(
        summary["case_name"],
        summary[
            "median_decode_tokens_per_second"
        ],
    )

    ax.set_title(
        "Median Decode Throughput"
    )

    ax.set_xlabel(
        "Benchmark case"
    )

    ax.set_ylabel(
        "Tokens / second"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(fig)

    return output_path


def plot_tpot(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots()

    ax.bar(
        summary["case_name"],
        summary["median_tpot_ms"],
    )

    ax.set_title(
        "Median Time Per Output Token"
    )

    ax.set_xlabel(
        "Benchmark case"
    )

    ax.set_ylabel(
        "TPOT (ms/token)"
    )

    ax.tick_params(
        axis="x",
        rotation=20,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=160,
    )

    plt.close(fig)

    return output_path


def generate_standard_plots(
    csv_path: str | Path,
    output_directory: str | Path,
) -> list[Path]:
    dataframe = load_results(
        csv_path
    )

    summary = summarize_for_plotting(
        dataframe
    )

    output_directory = Path(
        output_directory
    )

    created_plots = [
        plot_ttft(
            summary,
            output_directory
            / "ttft.png",
        ),

        plot_decode_throughput(
            summary,
            output_directory
            / "decode_throughput.png",
        ),

        plot_tpot(
            summary,
            output_directory
            / "tpot.png",
        ),
    ]

    return created_plots