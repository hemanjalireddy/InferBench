from __future__ import annotations

from dataclasses import dataclass

from inferbench.benchmark.runner import (
    BenchmarkCase,
    BenchmarkRecord,
    BenchmarkRunner,
)
from inferbench.core.backend import InferenceBackend


DEFAULT_SEED_TEXT = """
The Office is a workplace mockumentary about employees at a paper company
dealing with meetings, sales calls, office politics, friendships, pranks,
awkward management decisions, and the everyday chaos of working together.
Michael often tries to turn ordinary workplace moments into major events,
while Jim and Dwight frequently create their own rivalry. Pam, Jim, Dwight,
Michael, and the rest of the office each react differently to the strange
situations that happen during the workday.
""".strip()


@dataclass
class PromptLengthExperiment:
    token_targets: list[int]
    max_new_tokens: int = 32
    seed_text: str = DEFAULT_SEED_TEXT


def create_prompt_for_token_target(
    backend: InferenceBackend,
    target_tokens: int,
    seed_text: str = DEFAULT_SEED_TEXT,
) -> tuple[str, int]:

    if target_tokens <= 0:
        raise ValueError(
            "target_tokens must be greater than zero."
        )

    if not seed_text.strip():
        raise ValueError(
            "seed_text cannot be empty."
        )

    text = seed_text

    # Keep expanding the same text until we have enough tokens.
    while len(backend.tokenize(text)) < target_tokens:
        text = (
            text
            + "\n\n"
            + seed_text
        )

    token_ids = backend.tokenize(text)

    # Trim to the requested token length.
    trimmed_token_ids = token_ids[:target_tokens]

    # Convert the controlled token sequence back into text.
    prompt = backend.detokenize(
        trimmed_token_ids
    )

    # Re-tokenize so we record the actual token count produced
    # after the tokenize -> detokenize -> tokenize round trip.
    actual_token_count = len(
        backend.tokenize(prompt)
    )

    return (
        prompt,
        actual_token_count,
    )


def create_prompt_length_cases(
    backend: InferenceBackend,
    experiment: PromptLengthExperiment,
) -> list[BenchmarkCase]:

    cases: list[BenchmarkCase] = []

    for target_tokens in experiment.token_targets:

        prompt, actual_tokens = (
            create_prompt_for_token_target(
                backend=backend,
                target_tokens=target_tokens,
                seed_text=experiment.seed_text,
            )
        )

        case = BenchmarkCase(
            name=f"prompt_length_{target_tokens}",
            prompt=prompt,
            max_new_tokens=experiment.max_new_tokens,

            experiment_type="prompt_length",

            parameters={
                "target_prompt_tokens": target_tokens,
                "actual_prompt_tokens": actual_tokens,
            },
        )

        cases.append(case)

    return cases


def run_prompt_length_experiment(
    backend: InferenceBackend,
    experiment: PromptLengthExperiment,
    warmup_runs: int = 1,
    repetitions: int = 5,
) -> list[BenchmarkRecord]:

    cases = create_prompt_length_cases(
        backend=backend,
        experiment=experiment,
    )

    runner = BenchmarkRunner(
        backend=backend,
        warmup_runs=warmup_runs,
        repetitions=repetitions,
    )

    return runner.run_suite(cases)