# react-from-scratch

A small, from-scratch implementation of **ReAct** ([Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models", ICLR 2023](https://arxiv.org/abs/2210.03629)) in plain Python. No LangChain, no LangGraph: just an LLM, a loop, a parser and a few tools.

## What is ReAct?

A language model can **reason** ("I should first find out where X was born") and it can **act** (call a search tool). ReAct makes it do both, taking turns (shortened example):

```
Question: Which river flows through the capital city of Hungary?
Thought 1: I need to find the capital of Hungary, then which river flows through it.
Action 1: Search[Hungary]
Observation 1: Hungary is a landlocked country in Central Europe ... Its capital is Budapest.
Thought 2: The capital is Budapest. Now I need the river that flows through Budapest.
Action 2: Search[Budapest]
Observation 2: Budapest is the capital and most populous city of Hungary ... on the Danube.
Thought 3: Budapest lies on the Danube, so the answer is Danube.
Action 3: Finish[Danube]
```

The model writes the **Thought** and the **Action**. Our code runs the tool and writes the **Observation**. Then the model continues with the next step. The thoughts help it plan and recover from bad searches, and the tools stop it from inventing facts.

The paper compares ReAct with two simpler setups, and this repo has all three:

| mode    | reasoning | tools | how it works                                  |
|---------|-----------|-------|-----------------------------------------------|
| `react` | yes       | yes   | Thought -> Action -> Observation loop         |
| `act`   | no        | yes   | same loop, but no Thought lines               |
| `cot`   | yes       | no    | "Let's think step by step", answer from memory |

## Architecture

```
            question
               |
               v
   +-----------------------+      prompt + "Thought i:"      +-----------+
   |  agent.py (the loop)  | ------------------------------> |   LLM     |
   |                       | <------------------------------ | (Gemini / |
   |  parser.py            |   "...\nAction i: Search[X]"    |  Ollama / |
   |  -> Step(thought,     |                                 |  Fake)    |
   |     action, arg)      |                                 +-----------+
   |                       |   Search[X]   +-------------+
   |                       | ------------> |  tools.py   |  calculate / search / lookup
   |                       | <------------ |  (registry) |  (Wikipedia API)
   +-----------------------+  observation  +-------------+
               |
               v
      Finish[answer] or max steps
```

- `react/llm.py`: `BaseLLM` abstract class + `FakeLLM` (scripted replies for tests)
- `react/gemini_llm.py`: Gemini through the official `google-genai` SDK (`gemini-2.5-flash`)
- `react/ollama_llm.py`: any local model through Ollama's HTTP API
- `react/tools.py`: tools registered with a `@tool("name")` decorator
- `react/parser.py`: turns messy model output into `Step(thought, action, arg)`
- `react/prompts.py`: few-shot examples (based on the paper's HotpotQA prompt) for all three modes
- `react/agent.py`: `run_react`, `run_act`, `run_cot`
- `eval.py`: runs the modes on `data/questions.json` and prints a results table

## Project structure

```
react-from-scratch/
├── react/
│   ├── llm.py  gemini_llm.py  ollama_llm.py
│   ├── tools.py  parser.py  prompts.py  agent.py
├── tests/            # pytest, fully offline
├── data/questions.json
├── eval.py
├── NOTES.md          # file-by-file explanation + Python concepts
└── pyproject.toml
```

## How to run

You need Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install dependencies
uv run pytest                # run the tests (no internet or API key needed)
```

**With Gemini** (free API key from [Google AI Studio](https://aistudio.google.com/)):

```bash
export GEMINI_API_KEY=your-key
uv run python eval.py --llm gemini
```

**With a local model via [Ollama](https://ollama.com/):**

```bash
ollama pull llama3.1
uv run python eval.py --llm ollama --model llama3.1
```

Useful flags: `--modes react cot`, `--limit 5`, `--max-steps 7`, `--verbose` (print every trace), `--save results.json`, `--sleep 4` (slow down for free-tier rate limits).

Use the agent from Python:

```python
from react.agent import run_react
from react.gemini_llm import GeminiLLM

result = run_react("Which river flows through the capital city of Hungary?", GeminiLLM())
print(result.answer)
print(result.trace)
```

## Results

20 HotpotQA-style questions (`data/questions.json`), exact match after HotpotQA answer normalisation, max 7 steps.

| mode    | model            | EM   | avg steps |
|---------|------------------|------|-----------|
| react   | gemini-2.5-flash | TODO | TODO      |
| act     | gemini-2.5-flash | TODO | TODO      |
| cot     | gemini-2.5-flash | TODO | TODO      |

TODO: fill in after running `eval.py`, and add a few notes on what went right and wrong.

## Differences from the paper

- The paper used PaLM-540B and the full HotpotQA dev set. Here: smaller models and 20 questions, so numbers are only a rough comparison.
- An extra `Calculate[...]` tool, with one few-shot example showing it.
- Wikipedia is read through the live MediaWiki API, not a fixed dump, so pages may have changed since the paper.
