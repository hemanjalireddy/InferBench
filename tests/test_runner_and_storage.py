import pandas as pd

from inferbench.benchmark.runner import BenchmarkCase, BenchmarkRunner
from inferbench.core.backend import GenerationResult, InferenceBackend
from inferbench.storage.results import save_records_to_csv


class CountingBackend(InferenceBackend):
    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "backend": self.name,
            "model_name": "fake",
        }

    def load_model(self) -> None:
        return None

    def tokenize(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def detokenize(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        return GenerationResult(
            text="x",
            prompt_tokens=len(self.tokenize(prompt)),
            output_tokens=max_new_tokens,
            ttft_seconds=0.1,
            total_latency_seconds=0.3,
            metadata={"eos_reached": False},
        )

    def close(self) -> None:
        return None


def test_runner_records_metrics_and_metadata() -> None:
    runner = BenchmarkRunner(
        backend=CountingBackend(),
        warmup_runs=0,
        repetitions=2,
        run_id="test-run",
    )

    records = runner.run_case(
        BenchmarkCase(
            name="case",
            prompt="one two three",
            max_new_tokens=3,
            parameters={"target_prompt_tokens": 3},
        )
    )

    assert len(records) == 2
    assert records[0].run_id == "test-run"
    assert records[0].backend_name == "CountingBackend"
    assert records[0].generation_metadata["eos_reached"] is False
    assert records[0].decode_tokens_per_second == 10


def test_save_records_to_csv_flattens_parameters_and_metadata(tmp_path) -> None:
    runner = BenchmarkRunner(
        backend=CountingBackend(),
        warmup_runs=0,
        repetitions=1,
        run_id="test-run",
    )
    records = runner.run_case(
        BenchmarkCase(
            name="case",
            prompt="one two three",
            parameters={"target_prompt_tokens": 3},
        )
    )

    csv_path = save_records_to_csv(records, tmp_path / "results.csv")
    dataframe = pd.read_csv(csv_path)

    assert dataframe.loc[0, "param_target_prompt_tokens"] == 3
    assert dataframe.loc[0, "backend_model_name"] == "fake"
    assert not dataframe.loc[0, "generation_eos_reached"]
