from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from inferbench.benchmark.runner import BenchmarkRecord
from inferbench.benchmark.summary import BenchmarkSummary


def save_records_to_csv(
    records: list[BenchmarkRecord],
    output_path: str | Path,
) -> Path:
    if not records:
        raise ValueError("No benchmark records provided.")

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        asdict(record)
        for record in records
    ]

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_summary_to_json(
    summary: BenchmarkSummary,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_dict = asdict(summary)

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary_dict,
            file,
            indent=2,
        )

    return output_path