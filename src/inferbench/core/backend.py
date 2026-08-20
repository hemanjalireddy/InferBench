from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str

    prompt_tokens: int
    output_tokens: int

    ttft_seconds: float
    total_latency_seconds: float


class InferenceBackend(ABC):

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Release model resources."""
        pass