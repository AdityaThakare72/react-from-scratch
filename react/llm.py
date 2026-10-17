"""The LLM interface every model backend must follow, plus a FakeLLM for tests.

The agent only ever talks to `BaseLLM.generate()`. That means we can swap Gemini,
Ollama or a scripted fake without touching the agent code at all.
"""

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    """Anything that can turn a prompt into text.

    `stop` is a list of strings where the model should stop writing. ReAct needs this:
    the model must stop right before "Observation", because *we* fill in the observation
    by running the tool. Without it the model would happily invent its own results.
    """

    @abstractmethod
    def generate(self, prompt: str, stop: list[str] | None = None) -> str:
        """Return the model's continuation of `prompt`."""


class FakeLLM(BaseLLM):
    """Returns pre-written replies in order. Used in tests so they run offline."""

    def __init__(self, responses: list[str]):
        # Copy the list so we don't mutate the caller's list when we pop from it.
        self.responses = list(responses)
        # Remember every prompt we were given, so tests can check what the agent sent.
        self.prompts: list[str] = []

    def generate(self, prompt: str, stop: list[str] | None = None) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise RuntimeError("FakeLLM ran out of scripted responses")
        text = self.responses.pop(0)
        # Behave like a real model: cut the text at the first stop sequence.
        for s in stop or []:
            if s in text:
                text = text.split(s)[0]
        return text
