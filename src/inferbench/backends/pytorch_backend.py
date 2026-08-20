from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferbench.core.backend import GenerationResult, InferenceBackend


class PyTorchBackend(InferenceBackend):
    def __init__(
        self,
        model_name: str,
        *,
        num_threads: int | None = None,
        compile_model: bool = False,
    ):
        self.model_name = model_name
        self.num_threads = num_threads
        self.compile_model = compile_model

        self.device = torch.device("cpu")

        self.model = None
        self.tokenizer = None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "runtime": "pytorch-transformers",
            "model_name": self.model_name,
            "device": str(self.device),
            "num_threads": self.num_threads or torch.get_num_threads(),
            "compile_model": self.compile_model,
            "torch_version": torch.__version__,
        }

    def load_model(self) -> None:
        print(f"Loading model: {self.model_name}")

        if self.num_threads is not None:
            if self.num_threads <= 0:
                raise ValueError("num_threads must be greater than zero.")

            torch.set_num_threads(self.num_threads)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        )

        self.model = self.model.to(self.device)

        self.model.eval()

        if self.compile_model:
            self.model = torch.compile(self.model)

        print("Model loaded successfully.")

    def tokenize(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded.")

        tokens = self.tokenizer.encode(text)

        return tokens

    def detokenize(
        self,
        token_ids: list[int],
    ) -> str:

        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer has not been loaded."
            )

        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> GenerationResult:

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Model has not been loaded. Call load_model() first."
            )

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)

        attention_mask = encoded["attention_mask"].to(self.device)

        prompt_tokens = input_ids.shape[1]

        generated_tokens = []

        past_key_values = None

        current_input_ids = input_ids

        current_attention_mask = attention_mask

        start_time = time.perf_counter()

        first_token_time = None

        with torch.inference_mode():

            for step in range(max_new_tokens):

                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                logits = outputs.logits[:, -1, :]

                next_token = torch.argmax(
                    logits,
                    dim=-1,
                    keepdim=True,
                )

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                generated_tokens.append(next_token)

                past_key_values = outputs.past_key_values

                if (
                    self.tokenizer.eos_token_id is not None
                    and next_token.item()
                    == self.tokenizer.eos_token_id
                ):
                    break

                current_input_ids = next_token

                new_mask_value = torch.ones(
                    (current_attention_mask.shape[0], 1),
                    dtype=current_attention_mask.dtype,
                    device=self.device,
                )

                current_attention_mask = torch.cat(
                    [
                        current_attention_mask,
                        new_mask_value,
                    ],
                    dim=1,
                )

        end_time = time.perf_counter()

        if generated_tokens:
            output_ids = torch.cat(
                generated_tokens,
                dim=1,
            )

            generated_text = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
            )

            output_token_count = output_ids.shape[1]

        else:
            generated_text = ""
            output_token_count = 0

        if first_token_time is None:
            first_token_time = end_time

        return GenerationResult(
            text=generated_text,
            prompt_tokens=prompt_tokens,
            output_tokens=output_token_count,
            ttft_seconds=first_token_time - start_time,
            total_latency_seconds=end_time - start_time,
            metadata={
                "eos_reached": (
                    bool(generated_tokens)
                    and self.tokenizer.eos_token_id is not None
                    and output_ids[0, -1].item() == self.tokenizer.eos_token_id
                ),
            },
        )

    def close(self) -> None:
        self.model = None
        self.tokenizer = None
