# Notes

What each file does, and the Python ideas used along the way, explained from scratch.

---

## Part 1: the big idea in one paragraph

An LLM is a function: text in, text out. It can't browse Wikipedia. But if we
**stop** it right after it writes `Action: Search[Paris]`, run the search ourselves,
paste the result back as `Observation: ...`, and ask it to continue, it *looks* like
the model used a tool. Do that in a loop and you have an agent. ReAct adds one more
trick: before every action the model writes a `Thought`, which helps it plan and
fix its own mistakes.

---

## Part 2: file by file

### `react/llm.py`: the LLM interface

- `BaseLLM` is an **abstract base class**: it says "every LLM must have a
  `generate(prompt, stop)` method" without saying how.
- `FakeLLM` returns replies from a list you give it. Tests use it so they never
  need the internet or an API key. It also keeps every prompt it received in
  `self.prompts`, so a test can check what the agent actually sent.
- The `stop` list matters: we stop the model at `"\nObservation i:"` so it can't
  make up its own observation. `FakeLLM` copies that behaviour.

### `react/gemini_llm.py` and `react/ollama_llm.py`: real models

- Both subclass `BaseLLM`, so the agent can't tell them apart.
- Gemini uses the official `google-genai` SDK (`from google import genai`). The key
  comes from the `GEMINI_API_KEY` environment variable, never from the code.
- Ollama runs models on your own machine and serves an HTTP API. We just POST JSON
  to `localhost:11434/api/generate` with `requests`.
- `temperature=0` makes answers (nearly) repeatable, which is what you want when
  you compare modes.

### `react/tools.py`: the tools

- `TOOLS` is a plain dict: `{"search": search, "lookup": lookup, "calculate": calculate}`.
- The `@tool("search")` **decorator** fills that dict. Adding a tool is just writing
  a function and putting `@tool("name")` on top.
- `calculate` parses the expression with `ast` and only allows numbers and
  `+ - * / // % **`. Never call `eval()` on model output: the model could write
  `__import__('os').system(...)`.
- `search` and `lookup` copy the paper's Wikipedia API:
  - `search[X]`: first 5 sentences of page X, or "Could not find X. Similar: [...]".
  - `lookup[word]`: next sentence containing `word` in the current page. Call it
    again for the next match: `(Result 2 / 3) ...`.
- `lookup` needs to remember the current page between calls. That memory lives in
  a small **dataclass**, `WikiState`. The agent resets it for each new question.
- All network access goes through `_fetch_page` and `_search_titles`. Tests swap
  those two functions for fakes (`monkeypatch`), so tool tests run offline.
- Wikipedia sometimes answers "429 Too Many Requests"; `_get_json` waits and
  retries a few times.

### `react/parser.py`: reading the model's reply

Models don't always follow the format. The parser handles:

| model wrote                                   | parser gets                 |
|-----------------------------------------------|-----------------------------|
| `Thought 2: ...\nAction 2: Search[X]`         | search, "X"                 |
| ` ...\nAction: search[X]` (no Thought label)  | search, "X"                 |
| `Action: Finish[Paris` (no closing bracket)   | finish, "Paris"             |
| `Search["X"]` (quotes)                        | search, "X"                 |
| `...Action: Search[X]\nObservation: made up`  | made-up part is thrown away |
| `I'm not sure` (no action)                    | action = None               |

It returns a `Step(thought, action, arg)` **dataclass**. `parse_cot_answer` pulls the
text after `Answer:` for the CoT mode.

### `react/prompts.py`: the few-shot prompt

- Few-shot = show the model a few solved examples, then the new question. The
  model copies the pattern. No training needed.
- The examples are stored **once** as data (`EXAMPLES`), then formatted three ways:
  - `build_react_prompt`: Thought / Action / Observation lines
  - `build_act_prompt`: same, minus the Thoughts
  - `build_cot_prompt`: "Let's think step by step. ... Answer: X"
- Three examples come from the paper's HotpotQA prompt (Colorado orogeny, Milhouse,
  Adam Clayton Powell). A fourth one shows the `Calculate` tool.

### `react/agent.py`: the loop

```
for i in 1..max_steps:
    reply = llm.generate(prompt + trace + "Thought i:", stop=["\nObservation i:"])
    step  = parse_step(reply)
    if no action:  ask once more for just the action line
    if Finish:     return the answer
    observation = run the tool
    trace += Thought / Action / Observation lines
return "no answer" (hit max_steps)
```

- `trace` is the agent's scratchpad. It grows every step, so the model always sees
  everything it did before. That is the agent's only "memory".
- `run_tool` never crashes: an unknown action, or a tool that raises an error,
  becomes an observation text. The model reads it and can try something else.
- `run_act` is the same loop with `use_thoughts=False`.
- `run_cot` is one model call, no tools.
- Results come back as an `AgentResult` dataclass: answer, steps, finished, trace.
- `MODES = {"react": run_react, ...}` lets `eval.py` pick a mode by name.

### `eval.py`: the comparison

- Loads `data/questions.json`, runs each mode on each question, prints a table.
- **Exact match (EM)**: lowercase, remove punctuation and "a/an/the", then compare.
  This is how HotpotQA is scored. Each question can also list `aliases`
  (e.g. "Danube" and "Danube River").
- **Avg steps**: how many model calls each mode needed (CoT is always 1).
- A failed API call counts as a wrong answer instead of stopping the whole run.

### `tests/`

- `test_parser.py`: messy outputs from the table above.
- `test_tools.py`: calculator, decorator registry, search/lookup with a fake Wikipedia.
- `test_agent.py`: the full loop with `FakeLLM`: finishing, max steps, invalid
  actions, made-up observations, Act-only and CoT.
- `test_eval.py`: answer normalisation and exact match.

---

## Part 3: Python concepts used

### Abstract base classes (`abc`)

A class that is only a **promise**: "subclasses will have these methods".

```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self) -> str: ...

class Dog(Animal):
    def speak(self):
        return "woof"

Dog().speak()   # "woof"
Animal()        # TypeError: can't instantiate abstract class
```

If `Dog` forgot `speak`, `Dog()` would fail right away instead of much later.
Here, `BaseLLM` is the promise and Gemini / Ollama / Fake keep it.

### Decorators

A decorator is a function that takes a function and gives back a function.
`@something` above a `def` is just shorthand:

```python
@tool("search")
def search(entity): ...

# is exactly the same as:
def search(entity): ...
search = tool("search")(search)
```

`tool("search")` returns a function `register`. `register(search)` stores `search` in
the `TOOLS` dict and returns it unchanged. So the decorator's only job is to
**register** the function. A "decorator factory" is a decorator that takes arguments
(here the name) and builds the real decorator.

A minimal registry:

```python
REGISTRY = {}

def register(name):
    def wrap(func):
        REGISTRY[name] = func
        return func
    return wrap

@register("hello")
def hi():
    return "hi!"

REGISTRY["hello"]()   # "hi!"
```

### Dataclasses

`@dataclass` writes `__init__`, `__repr__` and `__eq__` for you from the type hints:

```python
from dataclasses import dataclass, field

@dataclass
class Step:
    thought: str
    action: str | None
    arg: str

Step("hmm", "search", "Paris")
# Step(thought='hmm', action='search', arg='Paris')
```

For a list default, use `field(default_factory=list)`, never `= []`. A plain `[]`
default would be **one** list shared by every object.

### Type hints like `str | None`

`str | None` means "a string, or None". Python doesn't enforce hints at runtime, but
they document the code and editors use them. `list[str]` = a list of strings.

### Closures

In `tool(name)`, the inner `register` function still "remembers" `name` after
`tool` has returned. A function that keeps variables from the scope that created
it is a **closure**. This is what makes decorator factories work.

### Regular expressions (`re`)

Small patterns for finding text:

```python
import re
re.search(r"action\s*\d*\s*:", "Action 2: Search[X]", re.IGNORECASE)
# \s* = any spaces, \d* = any digits -> matches "Action 2:", "action:", "Action 12 :"
```

### `ast`: parsing code without running it

`ast.parse("2 * (3 + 4)", mode="eval")` turns the text into a tree:
`BinOp(2, Mult, BinOp(3, Add, 4))`. We walk that tree and only allow number nodes
and math operators, so nothing else can run.

### `global` and module-level state

`wiki_state` is a variable in the module. `reset_wiki()` uses `global wiki_state`
to replace it with a fresh `WikiState()`. Without `global`, the assignment would
create a new local variable and the module's state would never change.

### Lazy imports

`GeminiLLM` does `from google import genai` inside `__init__`, not at the top of
the file. Code that never uses Gemini (like the tests) never pays for that import.

### pytest: fixtures and monkeypatch

```python
@pytest.fixture(autouse=True)
def fake_wikipedia(monkeypatch):
    monkeypatch.setattr(tools, "_fetch_page", lambda title: "fake page text")
```

- A **fixture** is setup code pytest runs for you. `autouse=True` = for every test
  in the file.
- `monkeypatch.setattr` swaps out a function for the duration of one test and puts
  the real one back afterwards. That's how tool tests avoid the network.

### Dependency injection (in plain words)

`run_react(question, llm, tools=...)` receives the LLM and the tools as
arguments instead of creating them itself. So a test can pass in `FakeLLM` and
fake tools, and `eval.py` can pass in Gemini. Same loop, different parts.
