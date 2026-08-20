from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from inferbench.benchmark.runner import BenchmarkRecord
from inferbench.benchmark.summary import BenchmarkSummary


@dataclass
class ExperimentMetadata:
    name: str
    run_id: str
    backend: dict[str, Any]
    parameters: dict[str, Any] = field(default_factory=dict)
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )


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

    rows = []

    for record in records:
        row = asdict(record)

        parameters = row.pop(
            "parameters",
            {},
        )

        backend_metadata = row.pop(
            "backend_metadata",
            {},
        )

        generation_metadata = row.pop(
            "generation_metadata",
            {},
        )

        for key, value in parameters.items():
            row[f"param_{key}"] = value

        for key, value in backend_metadata.items():
            row[f"backend_{key}"] = value

        for key, value in generation_metadata.items():
            row[f"generation_{key}"] = value

        rows.append(row)

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


def save_experiment_metadata(
    metadata: ExperimentMetadata,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            asdict(metadata),
            file,
            indent=2,
        )

    return output_path
