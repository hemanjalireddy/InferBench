from inferbench.core.backend import GenerationResult, InferenceBackend
from inferbench.experiments.prompt_length import (
    PromptLengthExperiment,
    create_prompt_for_token_target,
    create_prompt_length_cases,
)


class WhitespaceBackend(InferenceBackend):
    def load_model(self) -> None:
        return None

    def tokenize(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def detokenize(self, token_ids: list[int]) -> str:
        return " ".join(f"token_{token_id}" for token_id in token_ids)

    def generate(self, prompt: str, max_new_tokens: int) -> GenerationResult:
        return GenerationResult(
            text="done",
            prompt_tokens=len(self.tokenize(prompt)),
            output_tokens=max_new_tokens,
            ttft_seconds=0.1,
            total_latency_seconds=0.2,
        )

    def close(self) -> None:
        return None


def test_create_prompt_for_token_target_returns_controlled_length() -> None:
    backend = WhitespaceBackend()

    prompt, actual_tokens = create_prompt_for_token_target(
        backend=backend,
        target_tokens=7,
        seed_text="one two three",
    )

    assert actual_tokens == 7
    assert len(backend.tokenize(prompt)) == 7


def test_create_prompt_length_cases_records_target_and_actual_tokens() -> None:
    backend = WhitespaceBackend()

    cases = create_prompt_length_cases(
        backend=backend,
        experiment=PromptLengthExperiment(
            token_targets=[4, 8],
            max_new_tokens=3,
            seed_text="one two",
        ),
    )

    assert [case.name for case in cases] == ["prompt_length_4", "prompt_length_8"]
    assert cases[0].experiment_type == "prompt_length"
    assert cases[0].max_new_tokens == 3
    assert cases[1].parameters["actual_prompt_tokens"] == 8
