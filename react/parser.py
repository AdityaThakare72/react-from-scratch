"""Turn the model's raw text into a structured Step (thought, action, argument).

A well-behaved model writes something like:

    Thought 2: I need to find out where X was born.
    Action 2: Search[X]

Real models are messier: they drop the step numbers, forget the "Thought:" label,
use lowercase action names, leave out the closing bracket, or keep writing and
invent their own "Observation". This parser tries to cope with all of that.
"""

import re
from dataclasses import dataclass

KNOWN_ACTIONS = ["search", "lookup", "calculate", "finish"]

# "Thought:", "Thought 3:", "thought 3 :" ...
THOUGHT_RE = re.compile(r"thought\s*\d*\s*:", re.IGNORECASE)
# "Action:", "Action 3:" ...
ACTION_RE = re.compile(r"action\s*\d*\s*:", re.IGNORECASE)
# "Observation" in any form -> everything after it was made up by the model
OBSERVATION_RE = re.compile(r"\n?\s*observation\s*\d*\s*:", re.IGNORECASE)
# A bare call like "Search[Paris]" somewhere in the text (used as a fallback)
BARE_CALL_RE = re.compile(r"\b(" + "|".join(KNOWN_ACTIONS) + r")\s*\[", re.IGNORECASE)


@dataclass
class Step:
    """One parsed model turn. `action` is None when no action could be found."""

    thought: str
    action: str | None
    arg: str


def parse_action(text: str) -> tuple[str, str] | None:
    """Parse 'Search[Paris]' into ('search', 'Paris'). Returns None if it doesn't fit.

    Only the first line is used, and a missing closing bracket is forgiven.
    """
    line = text.strip().split("\n")[0].strip()
    match = re.match(r"(\w+)\s*\[(.*)$", line)
    if not match:
        return None
    name = match.group(1).lower()
    arg = match.group(2).strip()
    # Drop the closing bracket (if the model remembered it) and anything after it.
    if "]" in arg:
        arg = arg[: arg.rfind("]")]
    # Models sometimes wrap the argument in quotes: Search["Paris"]
    arg = arg.strip().strip("\"'").strip()
    return name, arg


def parse_step(text: str) -> Step:
    """Split one model reply into thought + action."""
    # 1. Throw away anything the model "hallucinated" after the action.
    text = OBSERVATION_RE.split(text)[0]

    # 2. Find the action. Prefer an explicit "Action:" label...
    action_match = ACTION_RE.search(text)
    if action_match:
        before = text[: action_match.start()]
        parsed = parse_action(text[action_match.end():])
    else:
        # ...otherwise look for a bare "Search[...]" anywhere in the text.
        bare = BARE_CALL_RE.search(text)
        before = text[: bare.start()] if bare else text
        parsed = parse_action(text[bare.start():]) if bare else None

    # 3. The thought is whatever came before the action, minus its "Thought:" label.
    thought = THOUGHT_RE.sub("", before, count=1).strip()

    if parsed is None:
        return Step(thought=thought, action=None, arg="")
    name, arg = parsed
    return Step(thought=thought, action=name, arg=arg)


def parse_cot_answer(text: str) -> str | None:
    """Pull the final answer out of a chain-of-thought reply ("... Answer: Paris")."""
    match = re.search(r"answer\s*:\s*(.+)", text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip().split("\n")[0].strip().rstrip(".")
