from __future__ import annotations

from dataclasses import dataclass

from inferbench.benchmark.runner import (
    BenchmarkCase,
    BenchmarkRecord,
    BenchmarkRunner,
)
from inferbench.core.backend import InferenceBackend
from inferbench.experiments.prompt_length import (
    DEFAULT_SEED_TEXT,
    create_prompt_for_token_target,
)


@dataclass
class OutputLengthExperiment:
    output_token_targets: list[int]
    prompt_tokens: int = 128
    seed_text: str = DEFAULT_SEED_TEXT


def create_output_length_cases(
    backend: InferenceBackend,
    experiment: OutputLengthExperiment,
) -> list[BenchmarkCase]:
    prompt, actual_prompt_tokens = create_prompt_for_token_target(
        backend=backend,
        target_tokens=experiment.prompt_tokens,
        seed_text=experiment.seed_text,
    )

    cases: list[BenchmarkCase] = []

    for output_tokens in experiment.output_token_targets:
        if output_tokens <= 0:
            raise ValueError("output token targets must be positive.")

        cases.append(
            BenchmarkCase(
                name=f"output_length_{output_tokens}",
                prompt=prompt,
                max_new_tokens=output_tokens,
                experiment_type="output_length",
                parameters={
                    "prompt_tokens": actual_prompt_tokens,
                    "target_output_tokens": output_tokens,
                },
            )
        )

    return cases


def run_output_length_experiment(
    backend: InferenceBackend,
    experiment: OutputLengthExperiment,
    warmup_runs: int = 1,
    repetitions: int = 5,
) -> list[BenchmarkRecord]:
    cases = create_output_length_cases(
        backend=backend,
        experiment=experiment,
    )
    runner = BenchmarkRunner(
        backend=backend,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
    )

    return runner.run_suite(cases)
