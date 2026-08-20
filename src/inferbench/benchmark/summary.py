from dataclasses import dataclass
from statistics import mean, median, stdev

from inferbench.benchmark.runner import BenchmarkRecord


@dataclass
class MetricSummary:
    mean: float
    median: float
    std_dev: float
    p50: float
    p95: float


@dataclass
class BenchmarkSummary:
    case_name: str
    runs: int

    ttft_ms: MetricSummary
    total_latency_ms: MetricSummary
    decode_tokens_per_second: MetricSummary | None
    tpot_ms: MetricSummary | None


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    if not values:
        raise ValueError("Cannot calculate percentile of an empty list.")

    sorted_values = sorted(values)

    index = (len(sorted_values) - 1) * percentile_value

    lower_index = int(index)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)

    fraction = index - lower_index

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]

    return lower_value + (
        upper_value - lower_value
    ) * fraction


def summarize_metric(
    values: list[float],
) -> MetricSummary:

    if not values:
        raise ValueError("Metric list cannot be empty.")

    return MetricSummary(
        mean=mean(values),
        median=median(values),
        std_dev=stdev(values) if len(values) > 1 else 0.0,
        p50=percentile(values, 0.50),
        p95=percentile(values, 0.95),
    )


def summarize_records(
    records: list[BenchmarkRecord],
) -> BenchmarkSummary:

    if not records:
        raise ValueError("No benchmark records provided.")

    case_names = {
        record.case_name
        for record in records
    }

    if len(case_names) != 1:
        raise ValueError(
            "summarize_records expects records from exactly one benchmark case."
        )

    case_name = records[0].case_name

    ttft_values = [
        record.ttft_ms
        for record in records
    ]

    total_latency_values = [
        record.total_latency_ms
        for record in records
    ]

    decode_values = [
        record.decode_tokens_per_second
        for record in records
        if record.decode_tokens_per_second is not None
    ]

    tpot_values = [
        record.tpot_ms
        for record in records
        if record.tpot_ms is not None
    ]

    return BenchmarkSummary(
        case_name=case_name,
        runs=len(records),

        ttft_ms=summarize_metric(
            ttft_values
        ),

        total_latency_ms=summarize_metric(
            total_latency_values
        ),

        decode_tokens_per_second=(
            summarize_metric(decode_values)
            if decode_values
            else None
        ),

        tpot_ms=(
            summarize_metric(tpot_values)
            if tpot_values
            else None
        ),
    )


def summarize_records_by_case(
    records: list[BenchmarkRecord],
) -> dict[str, BenchmarkSummary]:
    if not records:
        raise ValueError("No benchmark records provided.")

    grouped_records: dict[str, list[BenchmarkRecord]] = {}

    for record in records:
        grouped_records.setdefault(
            record.case_name,
            [],
        ).append(record)

    return {
        case_name: summarize_records(case_records)
        for case_name, case_records in grouped_records.items()
    }
