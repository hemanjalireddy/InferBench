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

For a tiny sanity run:

```powershell
poetry run inferbench prompt-length --preset quick
```

To measure uncached decode, run the same experiment with `--no-kv-cache` into a
separate output directory:

```powershell
poetry run inferbench prompt-length `
  --model sshleifer/tiny-gpt2 `
  --tokens 32,64,128,256 `
  --max-new-tokens 16 `
  --warmups 1 `
  --repetitions 3 `
  --threads 4 `
  --no-kv-cache `
  --output-dir results/prompt_length_no_kv_cache
```

The current CPU PyTorch backend uses regular PyTorch attention. Flash Attention is
recorded as disabled in result metadata and is not exposed as a CPU benchmark
mode yet. It should be added later through a GPU/runtime-specific backend rather
than silently pretending the CPU path used Flash Attention.

You can still compare PyTorch attention implementations that are meaningful for
this backend:

```powershell
poetry run inferbench attention `
  --model sshleifer/tiny-gpt2 `
  --tokens 32,64,128,256 `
  --max-new-tokens 16 `
  --warmups 1 `
  --repetitions 3 `
  --backends torch_default,eager,sdpa `
  --output-dir results/attention
```

## More Experiments

Output-length scaling:

```powershell
poetry run inferbench output-length `
  --model sshleifer/tiny-gpt2 `
  --prompt-tokens 128 `
  --outputs 8,16,32,64 `
  --output-dir results/output_length
```

CPU thread scaling:

```powershell
poetry run inferbench thread-scaling `
  --model sshleifer/tiny-gpt2 `
  --thread-counts 1,2,4,8 `
  --tokens 128 `
  --output-dir results/thread_scaling
```

PyTorch eager vs `torch.compile`:

```powershell
poetry run inferbench compile `
  --model sshleifer/tiny-gpt2 `
  --tokens 128 `
  --output-dir results/compile
```

Regression comparison:

```powershell
poetry run inferbench compare `
  --baseline results/prompt_length/results.csv `
  --candidate results/prompt_length_no_kv_cache/results.csv `
  --threshold 10
```

Experimental llama.cpp subprocess backend:

```powershell
poetry run inferbench llama-cpp `
  --executable C:\path\to\llama-cli.exe `
  --model-path C:\path\to\model.gguf `
  --tokens 32,64 `
  --max-new-tokens 16 `
  --output-dir results/llama_cpp
```

The llama.cpp adapter currently treats TTFT as approximate unless richer timing
output is parsed from the runtime.

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

The Streamlit dashboard reads `results/prompt_length/results.csv` by default. Use
the sidebar to enable comparison mode and point it at another CSV, such as
`results/prompt_length_no_kv_cache/results.csv`. Above the plots, use the KV-cache
view control to switch between all rows, only cached rows, or only uncached rows.
The dashboard shows overlaid TTFT, TPOT, decode throughput, prefill throughput,
best-by-metric winners, workload-weighted scoring, side-by-side deltas, summary
tables, run metadata, discovered `results/**/results.csv` files, and raw
benchmark rows.

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
    use_kv_cache=True,
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
