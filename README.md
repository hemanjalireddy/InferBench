# InferBench

InferBench is an interactive performance lab for LLM inference. It lets developers run
controlled experiments, measure where inference time is being spent, compare runtime
configurations, and visualize how factors such as prompt length, compilation,
KV-cache behavior, threading, and quantization affect latency and throughput.

The first backend targets CPU-friendly PyTorch + Hugging Face Transformers inference.
The framework is backend-agnostic by design, so additional runtimes such as
`llama.cpp` and OpenVINO can plug into the same benchmark runner later.

## What It Measures

InferBench separates generation into the two systems phases that matter most during
serving:

- **Prefill / TTFT**: prompt processing through the first generated token.
- **Decode / TPOT**: repeated token generation after the first token.

Each benchmark run records:

- Time to First Token (TTFT)
- Time per Output Token (TPOT)
- Decode throughput
- Approximate prefill throughput
- Total latency
- Prompt and output token counts
- Backend metadata and experiment parameters

## Project Layout

```text
src/inferbench/
  backends/        Runtime adapters, currently PyTorch + Transformers.
  benchmark/       Backend-agnostic benchmark cases, runner, and summaries.
  core/            Backend protocol and metric calculations.
  dashboard/       Streamlit + Plotly result explorer.
  experiments/     Controlled experiment definitions.
  storage/         CSV/JSON result writers.
  visualization/   Static plotting helpers.
```

## Install

```powershell
poetry install --with pytorch,dashboard,dev
```

For a lighter install without dashboard support:

```powershell
poetry install --with pytorch
```

## Run A Prompt-Length Experiment

```powershell
poetry run inferbench prompt-length `
  --model sshleifer/tiny-gpt2 `
  --tokens 32,64,128,256 `
  --max-new-tokens 16 `
  --warmups 1 `
  --repetitions 3 `
  --threads 4
```

This writes:

```text
results/prompt_length/results.csv
results/prompt_length/metadata.json
results/prompt_length/prompt_length_<tokens>_summary.json
results/prompt_length/plots/*.png
```

## Dashboard

```powershell
poetry run inferbench dashboard
```

The Streamlit dashboard reads `results/prompt_length/results.csv` by default and
shows TTFT, TPOT, decode throughput, approximate prefill throughput, summary tables,
and raw benchmark rows.

## Static Plots

```powershell
poetry run inferbench plots --csv results/prompt_length/results.csv
```

## Python API

```python
from inferbench.backends.pytorch_backend import PyTorchBackend
from inferbench.experiments.prompt_length import (
    PromptLengthExperiment,
    run_prompt_length_experiment,
)
from inferbench.storage.results import save_records_to_csv

backend = PyTorchBackend(
    "sshleifer/tiny-gpt2",
    num_threads=4,
)
backend.load_model()

try:
    records = run_prompt_length_experiment(
        backend=backend,
        experiment=PromptLengthExperiment(
            token_targets=[32, 64, 128],
            max_new_tokens=16,
        ),
        warmup_runs=1,
        repetitions=3,
    )
finally:
    backend.close()

save_records_to_csv(records, "results/prompt_length/results.csv")
```

## Roadmap

- CPU thread scaling and memory utilization
- PyTorch eager execution vs `torch.compile`
- TorchDynamo and TorchInductor analysis
- KV-cache size and behavior
- MHA, GQA, and MQA comparisons
- Quantization
- Flash Attention
- KV-cache allocation strategies
- Concurrency and serving capacity
- Performance regression detection
- Cross-runtime comparisons
