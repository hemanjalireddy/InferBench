from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from inferbench.benchmark.summary import summarize_records_by_case
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


def add_prompt_length_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "prompt-length",
        help="Run the controlled prompt-length scaling experiment.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="sshleifer/tiny-gpt2",
        help="Hugging Face causal language model name or local model path.",
    )
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
    add_plots_parser(subparsers)
    add_dashboard_parser(subparsers)

    return parser


def run_prompt_length_command(args: argparse.Namespace) -> None:
    from inferbench.backends.pytorch_backend import PyTorchBackend

    backend = PyTorchBackend(
        model_name=args.model,
        num_threads=args.threads,
        compile_model=args.compile_model,
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

    results_path = save_records_to_csv(
        records=records,
        output_path=args.output_dir / "results.csv",
    )
    summary_paths = [
        save_summary_to_json(
            summary=summary,
            output_path=args.output_dir / f"{case_name}_summary.json",
        )
        for case_name, summary in summarize_records_by_case(records).items()
    ]
    metadata_path = save_experiment_metadata(
        metadata=ExperimentMetadata(
            name="prompt_length",
            run_id=records[0].run_id,
            backend=records[0].backend_metadata,
            parameters={
                "token_targets": args.tokens,
                "max_new_tokens": args.max_new_tokens,
                "warmups": args.warmups,
                "repetitions": args.repetitions,
            },
        ),
        output_path=args.output_dir / "metadata.json",
    )

    print_success(f"Saved results: {results_path}")
    for summary_path in summary_paths:
        print_success(f"Saved summary: {summary_path}")
    print_success(f"Saved metadata: {metadata_path}")

    if args.plots:
        from inferbench.visualization.prompt_length import generate_prompt_length_plots

        plot_paths = generate_prompt_length_plots(
            csv_path=results_path,
            output_directory=args.output_dir / "plots",
        )
        for plot_path in plot_paths:
            print_success(f"Saved plot: {plot_path}")


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
