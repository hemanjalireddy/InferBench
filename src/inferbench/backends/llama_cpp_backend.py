from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any

from inferbench.core.backend import GenerationResult, InferenceBackend


class LlamaCppBackend(InferenceBackend):
    """
    Minimal subprocess adapter for llama.cpp-style CLIs.

    This is intentionally conservative: it lets InferBench compare a local
    llama.cpp executable through the common backend interface, but TTFT is
    approximated from total latency unless the CLI output is parsed later.
    """

    def __init__(
        self,
        executable: str | Path,
        model_path: str | Path,
        extra_args: list[str] | None = None,
    ):
        self.executable = str(executable)
        self.model_path = str(model_path)
        self.extra_args = extra_args or []

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "runtime": "llama.cpp",
            "executable": self.executable,
            "model_path": self.model_path,
            "ttft_mode": "approximate_total_latency",
        }

    def load_model(self) -> None:
        if not Path(self.executable).exists():
            raise FileNotFoundError(f"llama.cpp executable not found: {self.executable}")

        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"llama.cpp model not found: {self.model_path}")

    def tokenize(self, text: str) -> list[int]:
        return list(range(len(text.split())))

    def detokenize(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> GenerationResult:
        command = [
            self.executable,
            "-m",
            self.model_path,
            "-p",
            prompt,
            "-n",
            str(max_new_tokens),
            *self.extra_args,
        ]

        start_time = time.perf_counter()
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        end_time = time.perf_counter()

        text = completed.stdout.strip()
        output_tokens = max(1, len(text.split()))
        total_latency = end_time - start_time

        return GenerationResult(
            text=text,
            prompt_tokens=len(self.tokenize(prompt)),
            output_tokens=output_tokens,
            ttft_seconds=total_latency,
            total_latency_seconds=total_latency,
            metadata={
                "ttft_approximate": True,
                "stderr": completed.stderr.strip()[:500],
            },
        )

    def close(self) -> None:
        return None
