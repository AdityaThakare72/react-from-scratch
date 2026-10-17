"""Tests for react/tools.py. Wikipedia is faked, so these run offline."""

import pytest

from react import tools
from react.tools import TOOLS, calculate, lookup, search, tool

FAKE_PAGE = (
    "The Colorado orogeny was an episode of mountain building in Colorado. "
    "The eastern sector extends into the High Plains. "
    "It happened about 1700 million years ago. "
    "The eastern sector is also called the Central Plains orogeny."
)


@pytest.fixture(autouse=True)
def fake_wikipedia(monkeypatch):
    """Replace the two network functions with fakes for every test in this file."""

    def fake_fetch_page(title):
        return FAKE_PAGE if title == "Colorado orogeny" else None

    def fake_search_titles(query):
        return ["Colorado orogeny", "Colorado"]

    monkeypatch.setattr(tools, "_fetch_page", fake_fetch_page)
    monkeypatch.setattr(tools, "_search_titles", fake_search_titles)
    tools.reset_wiki()


def test_registry_has_all_tools():
    assert set(TOOLS) >= {"calculate", "search", "lookup"}


def test_decorator_registers_and_returns_function():
    @tool("Echo")
    def echo(text):
        return text

    assert TOOLS["echo"] is echo  # stored under the lowercase name
    assert echo("hi") == "hi"  # the function itself still works normally
    del TOOLS["echo"]  # clean up so other tests see the normal registry


def test_calculate_basic():
    assert calculate("2 * (3 + 4)") == "14"
    assert calculate("1701 - 1636") == "65"
    assert calculate("7 / 2") == "3.5"
    assert calculate("8 / 2") == "4"
    assert calculate("1,000 + 1") == "1001"


def test_calculate_errors_do_not_crash():
    assert "division by zero" in calculate("1 / 0")
    assert calculate("hello").startswith("Error")
    # Only arithmetic is allowed, never real Python code.
    assert calculate("__import__('os').system('ls')").startswith("Error")


def test_search_found():
    result = search("Colorado orogeny")
    assert result.startswith("The Colorado orogeny was an episode")


def test_search_not_found_lists_similar():
    result = search("Colorado orogen")
    assert result.startswith("Could not find [Colorado orogen]")
    assert "Colorado orogeny" in result


def test_lookup_steps_through_results():
    search("Colorado orogeny")
    first = lookup("eastern sector")
    second = lookup("eastern sector")
    third = lookup("eastern sector")
    assert first == "(Result 1 / 2) The eastern sector extends into the High Plains."
    assert second.startswith("(Result 2 / 2) The eastern sector is also called")
    assert third == "No more results."


def test_lookup_before_search():
    assert "Search first" in lookup("anything")
