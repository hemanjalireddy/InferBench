from inferbench.core.backend import GenerationResult
from inferbench.core.metrics import calculate_metrics


def test_calculate_metrics_splits_prefill_and_decode() -> None:
    result = GenerationResult(
        text="hello",
        prompt_tokens=10,
        output_tokens=5,
        ttft_seconds=0.2,
        total_latency_seconds=1.0,
    )

    metrics = calculate_metrics(result)

    assert metrics.ttft_ms == 200
    assert metrics.total_latency_ms == 1000
    assert metrics.decode_latency_ms == 800
    assert metrics.tpot_ms == 200
    assert metrics.decode_tokens_per_second == 5
    assert metrics.approximate_prefill_tokens_per_second == 50


def test_calculate_metrics_handles_single_output_token() -> None:
    result = GenerationResult(
        text="hello",
        prompt_tokens=10,
        output_tokens=1,
        ttft_seconds=0.2,
        total_latency_seconds=0.2,
    )

    metrics = calculate_metrics(result)

    assert metrics.decode_latency_ms is None
    assert metrics.tpot_ms is None
    assert metrics.decode_tokens_per_second is None
