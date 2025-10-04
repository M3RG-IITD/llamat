import requests
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    model_endpoint: str = "http://localhost:8000/v1/completions"
    model_name: str = \
        "model_path"
    temperature: float = 0.1
    max_tokens: int = 1024
    timeout_s: int = 60


class LocalCompletionClient:
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.config.model_name,
            "prompt": prompt,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        response = requests.post(
            self.config.model_endpoint,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.config.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        text = data.get("choices", [{}])[0].get("text", "").strip()
        return text


