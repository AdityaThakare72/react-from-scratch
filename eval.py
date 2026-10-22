"""Compare ReAct, Act-only and CoT on a small HotpotQA-style question set.

For every question and every mode we run the agent, check the answer with
exact match (after normalising text the same way the HotpotQA scorer does), and
finally print a table with accuracy and average number of steps per mode.

Examples:
    uv run python eval.py --llm gemini
    uv run python eval.py --llm ollama --model qwen2.5:7b --limit 5
    uv run python eval.py --llm gemini --modes react cot --save results.json
"""

import argparse
import json
import re
import string
import time

from react.agent import MODES


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and the words a/an/the, squash spaces.

    So "The Danube." and "danube" count as the same answer.
    """
    text = text.lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str | None, item: dict) -> bool:
    """True if the prediction matches the gold answer or one of its aliases."""
    if prediction is None:
        return False
    golds = [item["answer"]] + item.get("aliases", [])
    return any(normalize_answer(prediction) == normalize_answer(g) for g in golds)


def make_llm(name: str, model: str | None):
    """Build the chosen LLM. Imports happen here so you only need what you use."""
    if name == "gemini":
        from react.gemini_llm import GeminiLLM

        return GeminiLLM(model=model) if model else GeminiLLM()
    if name == "ollama":
        from react.ollama_llm import OllamaLLM

        return OllamaLLM(model=model) if model else OllamaLLM()
    raise ValueError(f"unknown llm: {name}")


def print_table(summary: list[dict]):
    """Print a small plain-text table, e.g.

    mode    | EM     | avg steps | finished | n
    --------+--------+-----------+----------+---
    react   | 45.0%  | 3.4       | 19/20    | 20
    """
    print()
    print(f"{'mode':<8}| {'EM':<7}| {'avg steps':<10}| {'finished':<9}| n")
    print(f"{'-' * 8}+{'-' * 8}+{'-' * 11}+{'-' * 10}+{'-' * 4}")
    for row in summary:
        print(
            f"{row['mode']:<8}| {row['em']:<7.1%}| {row['avg_steps']:<10.1f}"
            f"| {row['finished']:>2}/{row['n']:<6}| {row['n']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReAct vs Act-only vs CoT")
    parser.add_argument("--llm", choices=["gemini", "ollama"], default="gemini")
    parser.add_argument("--model", default=None, help="override the default model name")
    parser.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    parser.add_argument("--data", default="data/questions.json")
    parser.add_argument("--limit", type=int, default=None, help="only use the first N questions")
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--sleep", type=float, default=0.0, help="seconds to wait between questions (rate limits)")
    parser.add_argument("--save", default=None, help="write every answer and trace to this JSON file")
    parser.add_argument("--verbose", action="store_true", help="print the full trace of every run")
    args = parser.parse_args()

    with open(args.data) as f:
        questions = json.load(f)
    if args.limit:
        questions = questions[: args.limit]

    llm = make_llm(args.llm, args.model)
    summary = []
    records = []

    for mode in args.modes:
        run = MODES[mode]
        correct = 0
        total_steps = 0
        finished = 0
        print(f"\n=== {mode} ===")

        for i, item in enumerate(questions, start=1):
            try:
                result = run(item["question"], llm, max_steps=args.max_steps)
                answer, steps, done, trace = result.answer, result.steps, result.finished, result.trace
            except Exception as e:
                # One API hiccup shouldn't throw away the whole run; count it as wrong.
                answer, steps, done, trace = None, 0, False, f"ERROR: {e}"

            ok = exact_match(answer, item)
            correct += ok
            total_steps += steps
            finished += done
            mark = "OK " if ok else "-- "
            print(f"{mark}[{i:>2}] pred={answer!r} gold={item['answer']!r} steps={steps}")
            if args.verbose:
                print(trace)

            records.append({
                "mode": mode,
                "question": item["question"],
                "gold": item["answer"],
                "prediction": answer,
                "correct": ok,
                "steps": steps,
                "finished": done,
                "trace": trace,
            })
            if args.sleep:
                time.sleep(args.sleep)

        n = len(questions)
        summary.append({
            "mode": mode,
            "em": correct / n,
            "avg_steps": total_steps / n,
            "finished": finished,
            "n": n,
        })

    print_table(summary)

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"summary": summary, "records": records}, f, indent=2)
        print(f"\nsaved details to {args.save}")


if __name__ == "__main__":
    main()
