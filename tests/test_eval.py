"""Tests for the answer scoring in eval.py."""

from eval import exact_match, normalize_answer


def test_normalize_answer():
    assert normalize_answer("The Danube.") == "danube"
    assert normalize_answer("  Richard   Nixon ") == "richard nixon"


def test_exact_match_with_aliases():
    item = {"answer": "Danube", "aliases": ["Danube River"]}
    assert exact_match("the danube", item)
    assert exact_match("Danube River", item)
    assert not exact_match("Rhine", item)
    assert not exact_match(None, item)
