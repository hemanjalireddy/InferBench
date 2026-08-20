import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from inferbench.core.backend import GenerationResult, InferenceBackend


class PyTorchBackend(InferenceBackend):
    def __init__(self, model_name: str):
        self.model_name = model_name

        self.device = torch.device("cpu")

        self.model = None
        self.tokenizer = None

    def load_model(self) -> None:
        print(f"Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        )

        self.model = self.model.to(self.device)

        self.model.eval()

        print("Model loaded successfully.")

    def tokenize(self, text: str) -> list[int]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer has not been loaded.")

        tokens = self.tokenizer.encode(text)

        return tokens

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
        )

    def close(self) -> None:
        self.model = None
        self.tokenizer = None