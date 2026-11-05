import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langchain_community")

from cv_agent.tools import addition_tool, division_tool, multiplication_tool  # noqa: E402


def test_multiplication():
    assert multiplication_tool.invoke("25, 4") == "100.0"


def test_stray_observation_word_is_ignored():
    assert addition_tool.invoke("2, 3\nObservation") == "5.0"


def test_division_by_zero_is_reported_not_raised():
    assert division_tool.invoke("1, 0") == "Error: cannot divide by zero."


def test_bad_input_returns_the_usage_message():
    assert addition_tool.invoke("two plus two").startswith("Error:")
