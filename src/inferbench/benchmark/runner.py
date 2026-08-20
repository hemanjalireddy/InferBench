from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import uuid4

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

from inferbench.core.backend import InferenceBackend
from inferbench.core.metrics import calculate_metrics


ParameterValue = str | int | float | bool


@dataclass
class BenchmarkCase:
    """
    Defines one benchmark configuration.

    Example:
        prompt length = 512 tokens
        max_new_tokens = 32
    """

    name: str
    prompt: str
    max_new_tokens: int = 32

    experiment_type: str = "generic"

    parameters: dict[str, ParameterValue] = field(
        default_factory=dict
    )


@dataclass
class BenchmarkRecord:
    """
    Stores the measured result from one benchmark repetition.
    """

    case_name: str
    repetition: int

    experiment_type: str

    parameters: dict[str, ParameterValue]

    prompt_tokens: int
    output_tokens: int

    ttft_ms: float
    total_latency_ms: float
    decode_latency_ms: float | None

    tpot_ms: float | None
    decode_tokens_per_second: float | None
    approximate_prefill_tokens_per_second: float | None

    run_id: str
    timestamp_utc: str

    backend_name: str
    backend_metadata: dict[str, ParameterValue]
    generation_metadata: dict[str, ParameterValue]

    process_rss_mb: float | None
    process_memory_delta_mb: float | None
    process_cpu_percent: float | None


class BenchmarkRunner:
    """
    Runs benchmark cases using any InferenceBackend.

    The runner is backend-agnostic, so later we can use:

        PyTorchBackend
        LlamaCppBackend
        OpenVINOBackend

    without changing the benchmark logic.
    """

    def __init__(
        self,
        backend: InferenceBackend,
        warmup_runs: int = 1,
        repetitions: int = 5,
        run_id: str | None = None,
    ):
        if warmup_runs < 0:
            raise ValueError(
                "warmup_runs cannot be negative."
            )

        if repetitions <= 0:
            raise ValueError(
                "repetitions must be greater than zero."
            )

        self.backend = backend
        self.warmup_runs = warmup_runs
        self.repetitions = repetitions
        self.run_id = run_id or str(uuid4())
        self.process = psutil.Process() if psutil is not None else None

    def _warmup(
        self,
        case: BenchmarkCase,
    ) -> None:
        """
        Runs inference without recording metrics.

        Warmups help reduce noise caused by lazy initialization,
        memory allocation, CPU cache effects, and runtime setup.
        """

        for _ in range(self.warmup_runs):
            self.backend.generate(
                prompt=case.prompt,
                max_new_tokens=case.max_new_tokens,
            )

    def run_case(
        self,
        case: BenchmarkCase,
    ) -> list[BenchmarkRecord]:
        """
        Run one benchmark case multiple times.
        """

        print(
            f"\nBenchmarking: {case.name}"
        )

        if self.warmup_runs > 0:
            print(
                f"Warmup runs: {self.warmup_runs}"
            )

            self._warmup(case)

        records: list[BenchmarkRecord] = []

        for repetition in range(
            1,
            self.repetitions + 1,
        ):
            print(
                f"Run {repetition}/{self.repetitions}",
                end="\r",
            )

            if self.process is not None:
                memory_before = self.process.memory_info().rss
                self.process.cpu_percent(interval=None)
            else:
                memory_before = None

            result = self.backend.generate(
                prompt=case.prompt,
                max_new_tokens=case.max_new_tokens,
            )

            if self.process is not None and memory_before is not None:
                memory_after = self.process.memory_info().rss
                cpu_percent = self.process.cpu_percent(interval=None)
                process_rss_mb = memory_after / (1024 * 1024)
                process_memory_delta_mb = (
                    (memory_after - memory_before) / (1024 * 1024)
                )
            else:
                cpu_percent = None
                process_rss_mb = None
                process_memory_delta_mb = None

            metrics = calculate_metrics(
                result
            )

            record = BenchmarkRecord(
                case_name=case.name,
                repetition=repetition,

                experiment_type=case.experiment_type,

                parameters=case.parameters.copy(),

                prompt_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,

                ttft_ms=metrics.ttft_ms,

                total_latency_ms=(
                    metrics.total_latency_ms
                ),

                decode_latency_ms=(
                    metrics.decode_latency_ms
                ),

                tpot_ms=metrics.tpot_ms,

                decode_tokens_per_second=(
                    metrics.decode_tokens_per_second
                ),

                approximate_prefill_tokens_per_second=(
                    metrics.approximate_prefill_tokens_per_second
                ),

                run_id=self.run_id,
                timestamp_utc=datetime.now(UTC).isoformat(),

                backend_name=self.backend.name,
                backend_metadata=self.backend.metadata,
                generation_metadata=result.metadata,

                process_rss_mb=process_rss_mb,
                process_memory_delta_mb=process_memory_delta_mb,
                process_cpu_percent=cpu_percent,
            )

            records.append(
                record
            )

        print()

        return records

    def run_suite(
        self,
        cases: list[BenchmarkCase],
    ) -> list[BenchmarkRecord]:
        """
        Run multiple benchmark cases and combine their results.
        """

        all_records: list[BenchmarkRecord] = []

        for case in cases:
            records = self.run_case(
                case
            )

            all_records.extend(
                records
            )

        return all_records
