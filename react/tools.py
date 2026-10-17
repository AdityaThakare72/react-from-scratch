"""The tools the agent can call: Calculate, Search and Lookup.

Each tool is a plain function that takes one string and returns one string (the
"observation"). The @tool decorator puts it into the TOOLS dict, so the agent can
find a tool by name: TOOLS["search"]("Paris").

Search and Lookup copy the Wikipedia API from the ReAct paper:
  - search[entity]  -> first sentences of the Wikipedia page, or similar titles if not found
  - lookup[keyword] -> next sentence containing keyword in the page we last searched
"""

import ast
import operator
import re
import time
from dataclasses import dataclass, field

import requests

# name (lowercase) -> function. Filled in by the @tool decorator below.
TOOLS = {}


def tool(name: str):
    """Decorator that registers a function as a tool under `name`.

    `tool("search")` returns the real decorator. The decorator receives the function,
    stores it in TOOLS, and gives the same function back unchanged.
    """

    def register(func):
        TOOLS[name.lower()] = func
        return func

    return register


# ---------------------------------------------------------------- calculator

# Only these operators are allowed. Anything else (function calls, names, imports...)
# is rejected, so the model can't run arbitrary Python through us.
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    """Recursively evaluate a parsed arithmetic expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        # Guard against things like 9**9**9 that would hang the program.
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPERATORS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
        return _OPERATORS[type(node.op)](_eval_node(node.operand))
    raise ValueError("only numbers and + - * / // % ** are allowed")


@tool("calculate")
def calculate(expression: str) -> str:
    """Safely evaluate an arithmetic expression like '2 * (3 + 4)'."""
    # Models sometimes write 1,000 or use 'x' for multiply; clean that up first.
    cleaned = expression.replace(",", "").replace("^", "**").strip()
    try:
        tree = ast.parse(cleaned, mode="eval")
        result = _eval_node(tree.body)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (SyntaxError, ValueError) as e:
        return f"Error: could not calculate '{expression}' ({e})"
    # Show 4.0 as 4, but keep real decimals.
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


# ---------------------------------------------------------------- wikipedia

WIKI_API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks API users to send a descriptive User-Agent.
HEADERS = {"User-Agent": "react-from-scratch/0.1 (learning project)"}


@dataclass
class WikiState:
    """What Lookup needs to remember between calls: the current page and position."""

    page_text: str = ""
    keyword: str = ""
    matches: list[str] = field(default_factory=list)
    index: int = 0


# One shared state object. The agent calls reset_wiki() before every new question.
wiki_state = WikiState()


def reset_wiki():
    global wiki_state
    wiki_state = WikiState()


def _get_json(params: dict) -> dict:
    """GET the Wikipedia API and return the JSON reply.

    Wikipedia answers 429 "too many requests" if we go too fast, which happens during
    an eval run. In that case wait a little and try again (1s, 2s, 4s).
    """
    for wait in [1, 2, 4]:
        response = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=20)
        if response.status_code != 429:
            break
        time.sleep(wait)
    response.raise_for_status()
    return response.json()


def _fetch_page(title: str) -> str | None:
    """Return the plain text of a Wikipedia page, or None if it doesn't exist.

    This is the only place that downloads a page, so tests can replace it with a fake.
    """
    params = {
        "action": "query",
        "format": "json",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
    }
    data = _get_json(params)
    for page in data["query"]["pages"].values():
        if "missing" in page or not page.get("extract"):
            return None
        return page["extract"]
    return None


def _search_titles(query: str) -> list[str]:
    """Return up to 5 Wikipedia titles that match `query` (used when the page is missing)."""
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": 5,
    }
    data = _get_json(params)
    return [hit["title"] for hit in data["query"]["search"]]


def split_sentences(text: str) -> list[str]:
    """Very simple sentence splitter: break after . ! or ? followed by whitespace."""
    text = re.sub(r"\s+", " ", text).strip()
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


@tool("search")
def search(entity: str) -> str:
    """Look up a Wikipedia page and return its first few sentences."""
    entity = entity.strip()
    try:
        text = _fetch_page(entity)
        if text is None:
            similar = _search_titles(entity)
            return f"Could not find [{entity}]. Similar: {similar}."
    except requests.RequestException as e:
        return f"Error: Wikipedia request failed ({e})"

    # Remember the page so Lookup can search inside it later.
    reset_wiki()
    wiki_state.page_text = text
    # Like the paper, only show the start of the page to keep the prompt short.
    return " ".join(split_sentences(text)[:5])


@tool("lookup")
def lookup(keyword: str) -> str:
    """Return the next sentence containing `keyword` in the last searched page."""
    keyword = keyword.strip()
    if not wiki_state.page_text:
        return "No page loaded yet. Use Search first."

    # A new keyword -> collect all matching sentences and start from the first one.
    if keyword.lower() != wiki_state.keyword.lower():
        wiki_state.keyword = keyword
        wiki_state.matches = [
            s for s in split_sentences(wiki_state.page_text) if keyword.lower() in s.lower()
        ]
        wiki_state.index = 0

    if wiki_state.index >= len(wiki_state.matches):
        return "No more results."

    sentence = wiki_state.matches[wiki_state.index]
    wiki_state.index += 1
    return f"(Result {wiki_state.index} / {len(wiki_state.matches)}) {sentence}"
