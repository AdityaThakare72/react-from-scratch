"""The ReAct loop: Thought -> Action -> Observation, repeated until Finish or max steps.

How one step works:
  1. Send the prompt so far, ending in "Thought i:", to the model.
  2. The model writes a thought and an action, and we stop it at "Observation i:".
  3. We parse the action, run the matching tool ourselves, and append the real result
     as "Observation i: ...".
  4. Repeat with the longer prompt. The model sees everything it did so far.
"""

from dataclasses import dataclass

from react.llm import BaseLLM
from react.parser import parse_step
from react.prompts import build_react_prompt
from react.tools import TOOLS, reset_wiki

# Long Wikipedia paragraphs would blow up the prompt, so cut observations here.
MAX_OBSERVATION_CHARS = 800


@dataclass
class AgentResult:
    """What one agent run produced."""

    question: str
    answer: str | None  # None if the agent never called Finish
    steps: int  # how many model calls (Thought/Action rounds) it took
    finished: bool  # True if it ended with Finish[...], False if it hit max_steps
    trace: str  # the full Thought/Action/Observation text, for reading and debugging


def run_tool(action: str, arg: str, tools: dict) -> str:
    """Run one tool and always return a string, even when something goes wrong.

    Errors become observations instead of crashes, so the model can read the error
    and try something else. That is part of what makes ReAct robust.
    """
    if action not in tools:
        valid = ", ".join(name.capitalize() + "[...]" for name in tools) + ", Finish[...]"
        return f"Invalid action: {action}. Valid actions are {valid}."
    try:
        observation = tools[action](arg)
    except Exception as e:  # a broken tool should never kill the whole run
        return f"Error while running {action}: {e}"
    observation = str(observation).strip()
    if len(observation) > MAX_OBSERVATION_CHARS:
        observation = observation[:MAX_OBSERVATION_CHARS] + " ..."
    return observation


def run_loop(
    question: str,
    llm: BaseLLM,
    prompt: str,
    use_thoughts: bool = True,
    max_steps: int = 7,
    tools: dict | None = None,
) -> AgentResult:
    """The shared loop used by both ReAct (with thoughts) and Act-only (without)."""
    tools = TOOLS if tools is None else tools
    reset_wiki()  # forget the page from the previous question
    trace = ""  # everything after the question: the agent's "scratchpad"

    for i in range(1, max_steps + 1):
        # The prompt ends with the label we want the model to fill in next.
        label = f"Thought {i}:" if use_thoughts else f"Action {i}:"
        reply = llm.generate(prompt + trace + label, stop=[f"\nObservation {i}:"])
        step = parse_step(reply)

        # The model thought but never wrote an action. Like the paper's code, give
        # it one more chance by asking for just the action line.
        if step.action is None and use_thoughts:
            nudge = f"Thought {i}: {step.thought}\nAction {i}:"
            retry = llm.generate(prompt + trace + nudge, stop=["\n"])
            retry_step = parse_step(retry)
            step.action, step.arg = retry_step.action, retry_step.arg

        # Write this step back into the trace in a clean, standard format.
        if use_thoughts:
            trace += f"Thought {i}: {step.thought}\n"

        if step.action is None:
            trace += f"Action {i}: (none)\n"
            trace += f"Observation {i}: Invalid action. Use Search[...], Lookup[...], Calculate[...] or Finish[...].\n"
            continue

        trace += f"Action {i}: {step.action.capitalize()}[{step.arg}]\n"

        if step.action == "finish":
            return AgentResult(question, step.arg, steps=i, finished=True, trace=trace)

        observation = run_tool(step.action, step.arg, tools)
        trace += f"Observation {i}: {observation}\n"

    # Ran out of steps without an answer.
    return AgentResult(question, None, steps=max_steps, finished=False, trace=trace)


def run_react(question: str, llm: BaseLLM, max_steps: int = 7, tools: dict | None = None) -> AgentResult:
    """ReAct: reason (Thought) and act (Action) in turns."""
    prompt = build_react_prompt(question)
    return run_loop(question, llm, prompt, use_thoughts=True, max_steps=max_steps, tools=tools)
