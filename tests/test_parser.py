"""Tests for react/parser.py: turning messy model text into a Step."""

from react.parser import parse_action, parse_cot_answer, parse_step


def test_clean_step():
    step = parse_step("Thought 1: I should search Paris.\nAction 1: Search[Paris]")
    assert step.thought == "I should search Paris."
    assert step.action == "search"
    assert step.arg == "Paris"


def test_continuation_without_thought_label():
    # The prompt already ends with "Thought 2:", so the reply starts mid-line.
    step = parse_step(" I need the birth year.\nAction 2: Lookup[born]")
    assert step.thought == "I need the birth year."
    assert step.action == "lookup"
    assert step.arg == "born"


def test_unnumbered_labels_and_lowercase():
    step = parse_step("thought: ok\naction: finish[Richard Nixon]")
    assert step.action == "finish"
    assert step.arg == "Richard Nixon"


def test_missing_closing_bracket():
    step = parse_step("Thought: done\nAction: Finish[Paris")
    assert step.action == "finish"
    assert step.arg == "Paris"


def test_hallucinated_observation_is_dropped():
    text = "Thought 1: a\nAction 1: Search[X]\nObservation 1: made up stuff\nThought 2: more"
    step = parse_step(text)
    assert step.action == "search"
    assert step.arg == "X"
    assert "made up" not in step.thought


def test_bare_action_without_label():
    # Act-only replies are just "Search[...]" because the prompt ends with "Action i:".
    step = parse_step(" Search[High Plains (United States)]")
    assert step.action == "search"
    assert step.arg == "High Plains (United States)"


def test_no_action_found():
    step = parse_step("I am not sure what to do here.")
    assert step.action is None
    assert step.thought == "I am not sure what to do here."


def test_quotes_are_stripped():
    assert parse_action('Search["Milhouse"]') == ("search", "Milhouse")


def test_brackets_inside_argument():
    assert parse_action("Calculate[(2 + 3) * 4]") == ("calculate", "(2 + 3) * 4")


def test_parse_action_rejects_plain_text():
    assert parse_action("just some words") is None


def test_cot_answer():
    assert parse_cot_answer("X was born in 1990.\nAnswer: Paris.") == "Paris"
    assert parse_cot_answer("no answer here") is None
