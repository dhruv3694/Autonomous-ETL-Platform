from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMModelConfig:
    provider: str = "ollama"
    model: Optional[str] = None
    model_path: Optional[str] = None
    cache_dir: Optional[str] = None
    token: Optional[str] = None
    device: Optional[str] = None
    max_tokens: int = 512
    use_4bit: bool = False


class LLMProvider:
    name = "base"

    def __init__(self, config: Optional[LLMModelConfig] = None):
        self.config = config or LLMModelConfig()

    def generate(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def status(self) -> Dict[str, Any]:
        return {"provider": self.name, "available": False, "status": "unavailable"}


class OllamaProvider(LLMProvider):
    name = "ollama"

    def generate(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


class HuggingFaceProvider(LLMProvider):
    name = "huggingface"

    def generate(self, prompt: str) -> Dict[str, Any]:
        raise NotImplementedError


def get_provider(config: Optional[LLMModelConfig] = None) -> LLMProvider:
    provider_name = (config.provider if config else "ollama").lower()
    if provider_name in {"hf", "huggingface", "transformers"}:
        return HuggingFaceProvider(config)
    return OllamaProvider(config)
