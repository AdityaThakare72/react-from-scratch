"""GeminiLLM: talks to Google's Gemini models through the official google-genai SDK.

Needs the env var GEMINI_API_KEY (free key from Google AI Studio).
"""

import os

from react.llm import BaseLLM


class GeminiLLM(BaseLLM):
    def __init__(self, model: str = "gemini-2.5-flash", temperature: float = 0.0):
        # Import here (not at the top) so the rest of the project, and the tests,
        # work even if someone never uses Gemini.
        from google import genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set the GEMINI_API_KEY environment variable first")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def generate(self, prompt: str, stop: list[str] | None = None) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            # temperature 0 = as deterministic as possible, which we want for evaluation
            temperature=self.temperature,
            stop_sequences=stop or None,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        # .text can be None if the model returned nothing (e.g. it was blocked)
        return response.text or ""
