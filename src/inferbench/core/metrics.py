from dataclasses import dataclass

from inferbench.core.backend import GenerationResult


@dataclass
class InferenceMetrics:
    ttft_ms: float

    total_latency_ms: float

    decode_latency_ms: float | None

    tpot_ms: float | None

    decode_tokens_per_second: float | None

    approximate_prefill_tokens_per_second: float | None


def calculate_metrics(
    result: GenerationResult,
) -> InferenceMetrics:

    ttft_ms = result.ttft_seconds * 1000

    total_latency_ms = (
        result.total_latency_seconds * 1000
    )

    # The first output token is already included in TTFT.
    decode_token_count = result.output_tokens - 1

    if decode_token_count > 0:

        decode_latency_seconds = (
            result.total_latency_seconds
            - result.ttft_seconds
        )

        decode_latency_ms = (
            decode_latency_seconds * 1000
        )

        tpot_ms = (
            decode_latency_ms
            / decode_token_count
        )

        decode_tokens_per_second = (
            decode_token_count
            / decode_latency_seconds
        )

    else:

        decode_latency_ms = None
        tpot_ms = None
        decode_tokens_per_second = None

    if result.ttft_seconds > 0:

        approximate_prefill_tokens_per_second = (
            result.prompt_tokens
            / result.ttft_seconds
        )

    else:

        approximate_prefill_tokens_per_second = None

    return InferenceMetrics(
        ttft_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        decode_latency_ms=decode_latency_ms,
        tpot_ms=tpot_ms,
        decode_tokens_per_second=decode_tokens_per_second,
        approximate_prefill_tokens_per_second=(
            approximate_prefill_tokens_per_second
        ),
    )