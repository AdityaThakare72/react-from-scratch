"""OllamaLLM: runs a local open-source model (llama3, qwen, mistral...) through Ollama.

Install Ollama, run `ollama pull llama3.1`, and it serves an HTTP API on localhost:11434.
We call that API directly with `requests`, no extra SDK needed.
"""

import requests

from react.llm import BaseLLM


class OllamaLLM(BaseLLM):
    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        temperature: float = 0.0,
    ):
        self.model = model
        self.url = host.rstrip("/") + "/api/generate"
        self.temperature = temperature

    def generate(self, prompt: str, stop: list[str] | None = None) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            # stream=False -> get the whole answer in one JSON reply instead of chunks
            "stream": False,
            # raw=True -> send our prompt as-is; the few-shot prompt already has the format
            "raw": True,
            "options": {"temperature": self.temperature, "stop": stop or []},
        }
        response = requests.post(self.url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()["response"]
