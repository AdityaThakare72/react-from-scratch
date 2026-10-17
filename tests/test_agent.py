"""Tests for the agent loop, using FakeLLM and fake tools (no network, no API keys)."""

from react.agent import run_act, run_cot, run_react, run_tool
from react.llm import FakeLLM

# Tools that return fixed text, so we control exactly what the agent observes.
FAKE_TOOLS = {
    "search": lambda q: f"{q} is a city in France.",
    "lookup": lambda k: "(Result 1 / 1) It has 2 million people.",
    "calculate": lambda e: "4",
}


def test_react_finishes_with_answer():
    llm = FakeLLM([
        " I need to search Paris.\nAction 1: Search[Paris]",
        " Paris is in France, so the answer is France.\nAction 2: Finish[France]",
    ])
    result = run_react("Which country is Paris in?", llm, tools=FAKE_TOOLS)

    assert result.finished
    assert result.answer == "France"
    assert result.steps == 2
    assert "Observation 1: Paris is a city in France." in result.trace


def test_observation_is_fed_back_to_model():
    llm = FakeLLM([" go\nAction 1: Search[Paris]", " done\nAction 2: Finish[France]"])
    run_react("Q?", llm, tools=FAKE_TOOLS)

    second_prompt = llm.prompts[1]
    assert second_prompt.endswith("Observation 1: Paris is a city in France.\nThought 2:")


def test_made_up_observation_is_replaced_by_real_one():
    llm = FakeLLM([
        " go\nAction 1: Search[Paris]\nObservation 1: Paris is on Mars.",
        " done\nAction 2: Finish[France]",
    ])
    result = run_react("Q?", llm, tools=FAKE_TOOLS)
    assert "Mars" not in result.trace
    assert "Paris is a city in France." in result.trace


def test_max_steps_limit():
    # The model never calls Finish, so the loop must stop by itself.
    llm = FakeLLM([" keep going\nAction: Search[Paris]"] * 3)
    result = run_react("Q?", llm, max_steps=3, tools=FAKE_TOOLS)

    assert not result.finished
    assert result.answer is None
    assert result.steps == 3
    assert len(llm.prompts) == 3


def test_missing_action_gets_one_retry():
    llm = FakeLLM([
        " I think I should search Paris.",  # no action at all
        " Search[Paris]",  # the nudge asks for the action only
        " done\nAction 2: Finish[France]",
    ])
    result = run_react("Q?", llm, tools=FAKE_TOOLS)
    assert result.answer == "France"
    assert "Action 1: Search[Paris]" in result.trace
    assert llm.prompts[1].endswith("Action 1:")


def test_invalid_action_becomes_observation():
    llm = FakeLLM([" hmm\nAction 1: Fly[Paris]", " ok\nAction 2: Finish[France]"])
    result = run_react("Q?", llm, tools=FAKE_TOOLS)
    assert "Observation 1: Invalid action: fly." in result.trace
    assert result.answer == "France"


def test_broken_tool_does_not_crash():
    def broken(_):
        raise RuntimeError("boom")

    observation = run_tool("search", "x", {"search": broken})
    assert "boom" in observation


def test_long_observation_is_truncated():
    observation = run_tool("search", "x", {"search": lambda q: "a" * 5000})
    assert len(observation) < 1000
    assert observation.endswith("...")


def test_act_only_has_no_thoughts():
    llm = FakeLLM([" Search[Paris]", " Finish[France]"])
    result = run_act("Which country is Paris in?", llm, tools=FAKE_TOOLS)

    assert result.answer == "France"
    assert "Thought" not in result.trace
    assert llm.prompts[0].endswith("Action 1:")


def test_cot_single_call_no_tools():
    llm = FakeLLM([" Paris is the capital of France.\nAnswer: France"])
    result = run_cot("Which country is Paris in?", llm)

    assert result.answer == "France"
    assert result.steps == 1
    assert len(llm.prompts) == 1


def test_cot_without_answer_line():
    llm = FakeLLM([" I really don't know."])
    result = run_cot("Q?", llm)
    assert result.answer is None
    assert not result.finished
