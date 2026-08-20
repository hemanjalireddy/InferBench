from dataclasses import dataclass

from inferbench.core.backend import InferenceBackend
from inferbench.core.metrics import calculate_metrics


@dataclass
class BenchmarkCase:
    name: str
    prompt: str
    max_new_tokens: int = 32


@dataclass
class BenchmarkRecord:
    case_name: str
    repetition: int

    prompt_tokens: int
    output_tokens: int

    ttft_ms: float
    total_latency_ms: float
    decode_latency_ms: float | None

    tpot_ms: float | None
    decode_tokens_per_second: float | None
    approximate_prefill_tokens_per_second: float | None


class BenchmarkRunner:
    def __init__(
        self,
        backend: InferenceBackend,
        warmup_runs: int = 1,
        repetitions: int = 5,
    ):
        if warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative.")

        if repetitions <= 0:
            raise ValueError("repetitions must be greater than zero.")

        self.backend = backend
        self.warmup_runs = warmup_runs
        self.repetitions = repetitions

    def _warmup(self, case: BenchmarkCase) -> None:
        for _ in range(self.warmup_runs):
            self.backend.generate(
                prompt=case.prompt,
                max_new_tokens=case.max_new_tokens,
            )

    def run_case(
        self,
        case: BenchmarkCase,
    ) -> list[BenchmarkRecord]:

        print(f"\nBenchmarking: {case.name}")

        if self.warmup_runs > 0:
            print(f"Warmup runs: {self.warmup_runs}")
            self._warmup(case)

        records = []

        for repetition in range(1, self.repetitions + 1):

            print(
                f"Run {repetition}/{self.repetitions}",
                end="\r",
            )

            result = self.backend.generate(
                prompt=case.prompt,
                max_new_tokens=case.max_new_tokens,
            )

            metrics = calculate_metrics(result)

            record = BenchmarkRecord(
                case_name=case.name,
                repetition=repetition,

                prompt_tokens=result.prompt_tokens,
                output_tokens=result.output_tokens,

                ttft_ms=metrics.ttft_ms,
                total_latency_ms=metrics.total_latency_ms,
                decode_latency_ms=metrics.decode_latency_ms,

                tpot_ms=metrics.tpot_ms,
                decode_tokens_per_second=(
                    metrics.decode_tokens_per_second
                ),
                approximate_prefill_tokens_per_second=(
                    metrics.approximate_prefill_tokens_per_second
                ),
            )

            records.append(record)

        print()

        return records

    def run_suite(
        self,
        cases: list[BenchmarkCase],
    ) -> list[BenchmarkRecord]:

        all_records = []

        for case in cases:
            records = self.run_case(case)
            all_records.extend(records)

        return all_records