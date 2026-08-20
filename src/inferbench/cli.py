from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from inferbench.benchmark.summary import summarize_records_by_case
from inferbench.benchmark.runner import BenchmarkRecord
from inferbench.experiments.output_length import (
    OutputLengthExperiment,
    run_output_length_experiment,
)
from inferbench.experiments.prompt_length import (
    PromptLengthExperiment,
    run_prompt_length_experiment,
)
from inferbench.storage.results import (
    ExperimentMetadata,
    save_experiment_metadata,
    save_records_to_csv,
    save_summary_to_json,
)

try:
    from rich.console import Console
except ModuleNotFoundError:
    Console = None


PRESETS = {
    "quick": {
        "tokens": "32,64",
        "max_new_tokens": 8,
        "warmups": 0,
        "repetitions": 1,
    },
    "decode-heavy": {
        "tokens": "128",
        "max_new_tokens": 64,
        "warmups": 1,
        "repetitions": 3,
    },
    "long-context": {
        "tokens": "128,256,512,1024",
        "max_new_tokens": 16,
        "warmups": 1,
        "repetitions": 3,
    },
}


def print_success(message: str) -> None:
    if Console is None:
        print(message)
        return

    Console().print(f"[green]{message}[/green]")


def parse_token_targets(value: str) -> list[int]:
    try:
        targets = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "Prompt token targets must be comma-separated integers."
        ) from error

    if not targets:
        raise argparse.ArgumentTypeError("Provide at least one prompt token target.")

    if any(target <= 0 for target in targets):
        raise argparse.ArgumentTypeError(
            "Prompt token targets must be positive integers."
        )

    return targets


def save_run_outputs(
    records: list[BenchmarkRecord],
    output_dir: Path,
    metadata: ExperimentMetadata,
    plots: bool = False,
) -> None:
    results_path = save_records_to_csv(
        records=records,
        output_path=output_dir / "results.csv",
    )
    summary_paths = [
        save_summary_to_json(
            summary=summary,
            output_path=output_dir / f"{case_name}_summary.json",
        )
        for case_name, summary in summarize_records_by_case(records).items()
    ]
    metadata_path = save_experiment_metadata(
        metadata=metadata,
        output_path=output_dir / "metadata.json",
    )

    print_success(f"Saved results: {results_path}")

    for summary_path in summary_paths:
        print_success(f"Saved summary: {summary_path}")

    print_success(f"Saved metadata: {metadata_path}")

    if plots:
        from inferbench.visualization.prompt_length import generate_prompt_length_plots

        plot_paths = generate_prompt_length_plots(
            csv_path=results_path,
            output_directory=output_dir / "plots",
        )

        for plot_path in plot_paths:
            print_success(f"Saved plot: {plot_path}")


def metadata_parameters(args: argparse.Namespace) -> dict[str, object]:
    parameters = {}

    for key, value in vars(args).items():
        if key == "func":
            continue

        if isinstance(value, Path):
            parameters[key] = str(value)
        else:
            parameters[key] = value

    return parameters


def add_common_pytorch_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        "-m",
        default="sshleifer/tiny-gpt2",
        help="Hugging Face causal language model name or local model path.",
    )
    parser.add_argument(
        "--warmups",
        type=int,
        default=1,
        help="Warmup runs per benchmark case.",
    )
    parser.add_argument(
        "--repetitions",
        "-r",
        type=int,
        default=3,
        help="Measured repetitions per case.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="CPU threads to use for PyTorch.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile_model",
        help="Use torch.compile before running.",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--kv-cache",
        action="store_true",
        dest="use_kv_cache",
        default=True,
        help="Use KV cache during decode.",
    )
    cache_group.add_argument(
        "--no-kv-cache",
        action="store_false",
        dest="use_kv_cache",
        help="Disable KV cache and recompute the sequence every decode step.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=[
            "torch_default",
            "eager",
            "sdpa",
        ],
        default="torch_default",
        help=(
            "Attention implementation requested from Transformers. SDPA is "
            "PyTorch scaled dot-product attention; Flash Attention is not "
            "enabled in this CPU backend."
        ),
    )


def add_prompt_length_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prompt-length",
        help="Run the controlled prompt-length scaling experiment.",
    )
    add_common_pytorch_options(parser)
    parser.add_argument(
        "--tokens",
        "-t",
        type=parse_token_targets,
        default=parse_token_targets("32,64,128"),
        help="Comma-separated target prompt lengths.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Generated tokens per run.",
    )
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS),
        default=None,
        help="Apply a benchmark preset before explicit CLI options.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/prompt_length"),
        help="Directory for CSV/JSON/plot outputs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_false",
        dest="plots",
        help="Skip static PNG plot generation.",
    )
    parser.set_defaults(func=run_prompt_length_command)


def add_output_length_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "output-length",
        help="Vary generated output length with a fixed prompt.",
    )
    add_common_pytorch_options(parser)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=128,
        help="Fixed prompt token target.",
    )
    parser.add_argument(
        "--outputs",
        type=parse_token_targets,
        default=parse_token_targets("8,16,32,64"),
        help="Comma-separated output token targets.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/output_length"),
        help="Directory for CSV/JSON outputs.",
    )
    parser.set_defaults(func=run_output_length_command)


def add_thread_scaling_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "thread-scaling",
        help="Run prompt-length cases across multiple CPU thread counts.",
    )
    add_common_pytorch_options(parser)
    parser.add_argument(
        "--thread-counts",
        type=parse_token_targets,
        default=parse_token_targets("1,2,4,8"),
        help="Comma-separated CPU thread counts.",
    )
    parser.add_argument(
        "--tokens",
        type=parse_token_targets,
        default=parse_token_targets("128"),
        help="Prompt token targets.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Generated tokens per run.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/thread_scaling"),
    )
    parser.set_defaults(func=run_thread_scaling_command)


def add_attention_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "attention",
        help="Compare torch_default, eager, and SDPA attention backends.",
    )
    add_common_pytorch_options(parser)
    parser.add_argument(
        "--backends",
        default="torch_default,eager,sdpa",
        help="Comma-separated attention backends.",
    )
    parser.add_argument(
        "--tokens",
        type=parse_token_targets,
        default=parse_token_targets("128"),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/attention"),
    )
    parser.set_defaults(func=run_attention_command)


def add_compile_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compile",
        help="Compare PyTorch eager execution with torch.compile.",
    )
    add_common_pytorch_options(parser)
    parser.add_argument(
        "--tokens",
        type=parse_token_targets,
        default=parse_token_targets("128"),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/compile"),
    )
    parser.set_defaults(func=run_compile_command)


def add_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compare",
        help="Compare two saved CSV runs and flag regressions.",
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold as a percent.",
    )
    parser.set_defaults(func=run_compare_command)


def add_llama_cpp_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "llama-cpp",
        help="Run prompt-length cases through a llama.cpp subprocess backend.",
    )
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--extra-arg",
        action="append",
        default=[],
        help="Extra llama.cpp CLI argument. Repeat for multiple arguments.",
    )
    parser.add_argument(
        "--tokens",
        type=parse_token_targets,
        default=parse_token_targets("32,64"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=0)
    parser.add_argument("--repetitions", "-r", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/llama_cpp"),
    )
    parser.set_defaults(func=run_llama_cpp_command)


def add_plots_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "plots",
        help="Generate static prompt-length plots from a saved CSV.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("results/prompt_length/results.csv"),
        help="Prompt-length benchmark results CSV.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/prompt_length/plots"),
        help="Directory for generated PNG plots.",
    )
    parser.set_defaults(func=run_plots_command)


def add_dashboard_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "dashboard",
        help="Launch the Streamlit dashboard.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port.",
    )
    parser.set_defaults(func=run_dashboard_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferbench",
        description="Run and inspect CPU-friendly LLM inference benchmarks.",
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    add_prompt_length_parser(subparsers)
    add_output_length_parser(subparsers)
    add_thread_scaling_parser(subparsers)
    add_attention_parser(subparsers)
    add_compile_parser(subparsers)
    add_compare_parser(subparsers)
    add_llama_cpp_parser(subparsers)
    add_plots_parser(subparsers)
    add_dashboard_parser(subparsers)

    return parser


def run_prompt_length_command(args: argparse.Namespace) -> None:
    from inferbench.backends.pytorch_backend import PyTorchBackend

    if args.preset:
        preset = PRESETS[args.preset]
        args.tokens = parse_token_targets(preset["tokens"])
        args.max_new_tokens = preset["max_new_tokens"]
        args.warmups = preset["warmups"]
        args.repetitions = preset["repetitions"]

    backend = PyTorchBackend(
        model_name=args.model,
        num_threads=args.threads,
        compile_model=args.compile_model,
        use_kv_cache=args.use_kv_cache,
        attention_backend=args.attention_backend,
    )
    backend.load_model()

    try:
        experiment = PromptLengthExperiment(
            token_targets=args.tokens,
            max_new_tokens=args.max_new_tokens,
        )
        records = run_prompt_length_experiment(
            backend=backend,
            experiment=experiment,
            warmup_runs=args.warmups,
            repetitions=args.repetitions,
        )
    finally:
        backend.close()

    save_run_outputs(
        records=records,
        output_dir=args.output_dir,
        metadata=ExperimentMetadata(
            name="prompt_length",
            run_id=records[0].run_id,
            backend=records[0].backend_metadata,
            parameters={
                "token_targets": args.tokens,
                "max_new_tokens": args.max_new_tokens,
                "warmups": args.warmups,
                "repetitions": args.repetitions,
                "use_kv_cache": args.use_kv_cache,
                "attention_backend": args.attention_backend,
                "flash_attention": False,
                "preset": args.preset,
            },
        ),
        plots=args.plots,
    )

def run_output_length_command(args: argparse.Namespace) -> None:
    from inferbench.backends.pytorch_backend import PyTorchBackend

    backend = PyTorchBackend(
        model_name=args.model,
        num_threads=args.threads,
        compile_model=args.compile_model,
        use_kv_cache=args.use_kv_cache,
        attention_backend=args.attention_backend,
    )
    backend.load_model()

    try:
        records = run_output_length_experiment(
            backend=backend,
            experiment=OutputLengthExperiment(
                output_token_targets=args.outputs,
                prompt_tokens=args.prompt_tokens,
            ),
            warmup_runs=args.warmups,
            repetitions=args.repetitions,
        )
    finally:
        backend.close()

    save_run_outputs(
        records=records,
        output_dir=args.output_dir,
        metadata=ExperimentMetadata(
            name="output_length",
            run_id=records[0].run_id,
            backend=records[0].backend_metadata,
            parameters=metadata_parameters(args),
        ),
    )


def run_prompt_suite_for_backend(args: argparse.Namespace, backend) -> list[BenchmarkRecord]:
    records = run_prompt_length_experiment(
        backend=backend,
        experiment=PromptLengthExperiment(
            token_targets=args.tokens,
            max_new_tokens=args.max_new_tokens,
        ),
        warmup_runs=args.warmups,
        repetitions=args.repetitions,
    )

    return records


def run_thread_scaling_command(args: argparse.Namespace) -> None:
    from inferbench.backends.pytorch_backend import PyTorchBackend

    records: list[BenchmarkRecord] = []

    for thread_count in args.thread_counts:
        backend = PyTorchBackend(
            model_name=args.model,
            num_threads=thread_count,
            compile_model=args.compile_model,
            use_kv_cache=args.use_kv_cache,
            attention_backend=args.attention_backend,
        )
        backend.load_model()

        try:
            records.extend(run_prompt_suite_for_backend(args, backend))
        finally:
            backend.close()

    save_run_outputs(
        records=records,
        output_dir=args.output_dir,
        metadata=ExperimentMetadata(
            name="thread_scaling",
            run_id=records[0].run_id,
            backend=records[0].backend_metadata,
            parameters=metadata_parameters(args),
        ),
    )


def run_attention_command(args: argparse.Namespace) -> None:
    from inferbench.backends.pytorch_backend import PyTorchBackend

    records: list[BenchmarkRecord] = []
    backends = [part.strip() for part in args.backends.split(",") if part.strip()]

    for attention_backend in backends:
        backend = PyTorchBackend(
            model_name=args.model,
            num_threads=args.threads,
            compile_model=args.compile_model,
            use_kv_cache=args.use_kv_cache,
            attention_backend=attention_backend,
        )
        backend.load_model()

        try:
            records.extend(run_prompt_suite_for_backend(args, backend))
        finally:
            backend.close()

    save_run_outputs(
        records=records,
        output_dir=args.output_dir,
        metadata=ExperimentMetadata(
            name="attention",
            run_id=records[0].run_id,
            backend=records[0].backend_metadata,
            parameters=metadata_parameters(args),
        ),
    )


def run_compile_command(args: argparse.Namespace) -> None:
    from inferbench.backends.pytorch_backend import PyTorchBackend

    records: list[BenchmarkRecord] = []

    for compile_model in [False, True]:
        backend = PyTorchBackend(
            model_name=args.model,
            num_threads=args.threads,
            compile_model=compile_model,
            use_kv_cache=args.use_kv_cache,
            attention_backend=args.attention_backend,
        )
        backend.load_model()

        try:
            records.extend(run_prompt_suite_for_backend(args, backend))
        finally:
            backend.close()

    save_run_outputs(
        records=records,
        output_dir=args.output_dir,
        metadata=ExperimentMetadata(
            name="compile",
            run_id=records[0].run_id,
            backend=records[0].backend_metadata,
            parameters=metadata_parameters(args),
        ),
    )


def summarize_csv(path: Path) -> pd.Series:
    dataframe = pd.read_csv(path)
    return dataframe.agg(
        {
            "ttft_ms": "median",
            "tpot_ms": "median",
            "decode_tokens_per_second": "median",
            "approximate_prefill_tokens_per_second": "median",
            "total_latency_ms": "median",
        }
    )


def run_compare_command(args: argparse.Namespace) -> None:
    baseline = summarize_csv(args.baseline)
    candidate = summarize_csv(args.candidate)

    rows = []
    lower_is_better = {
        "ttft_ms",
        "tpot_ms",
        "total_latency_ms",
    }

    for metric, baseline_value in baseline.items():
        candidate_value = candidate[metric]
        delta_percent = (candidate_value - baseline_value) / baseline_value * 100

        regressed = (
            delta_percent > args.threshold
            if metric in lower_is_better
            else delta_percent < -args.threshold
        )
        rows.append(
            {
                "metric": metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "delta_percent": delta_percent,
                "regressed": regressed,
            }
        )

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))


def run_llama_cpp_command(args: argparse.Namespace) -> None:
    from inferbench.backends.llama_cpp_backend import LlamaCppBackend

    backend = LlamaCppBackend(
        executable=args.executable,
        model_path=args.model_path,
        extra_args=args.extra_arg,
    )
    backend.load_model()

    try:
        records = run_prompt_length_experiment(
            backend=backend,
            experiment=PromptLengthExperiment(
                token_targets=args.tokens,
                max_new_tokens=args.max_new_tokens,
            ),
            warmup_runs=args.warmups,
            repetitions=args.repetitions,
        )
    finally:
        backend.close()

    save_run_outputs(
        records=records,
        output_dir=args.output_dir,
        metadata=ExperimentMetadata(
            name="llama_cpp_prompt_length",
            run_id=records[0].run_id,
            backend=records[0].backend_metadata,
            parameters=metadata_parameters(args),
        ),
    )


def run_plots_command(args: argparse.Namespace) -> None:
    from inferbench.visualization.prompt_length import generate_prompt_length_plots

    plot_paths = generate_prompt_length_plots(
        csv_path=args.csv,
        output_directory=args.output_dir,
    )

    for plot_path in plot_paths:
        print_success(f"Saved plot: {plot_path}")


def run_dashboard_command(args: argparse.Namespace) -> None:
    dashboard_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(dashboard_path),
            "--server.port",
            str(args.port),
        ],
        check=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
