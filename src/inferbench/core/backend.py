from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenerationResult:
    text: str

    prompt_tokens: int
    output_tokens: int

    ttft_seconds: float
    total_latency_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class InferenceBackend(ABC):

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def metadata(self) -> dict[str, Any]:
        return {"backend": self.name}

    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory."""
        pass

    @abstractmethod
    def tokenize(self, text: str) -> list[int]:
        """Convert text into token IDs."""
        pass

    @abstractmethod
    def detokenize(self, token_ids: list[int]) -> str:
        """Convert token IDs back into text."""
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
