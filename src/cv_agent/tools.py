"""Tools the specialist agents call: CV search, arithmetic, web search."""
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

from cv_agent.rag import search_cvs

_ARITHMETIC_ERROR = "Error: provide two numbers separated by a comma."


def _parse_pair(numbers: str) -> tuple[float, float]:
    """'25, 4' -> (25.0, 4.0). The ReAct loop sometimes appends the word 'Observation'."""
    numbers = numbers.replace("Observation", "").strip()
    a, b = map(float, numbers.split(","))
    return a, b


@tool
def addition_tool(numbers: str) -> str:
    """Add two numbers provided as comma-separated values."""
    try:
        a, b = _parse_pair(numbers)
        return str(a + b)
    except Exception:
        return _ARITHMETIC_ERROR


@tool
def subtraction_tool(numbers: str) -> str:
    """Subtract the second number from the first number using two comma-separated numbers."""
    try:
        a, b = _parse_pair(numbers)
        return str(a - b)
    except Exception:
        return _ARITHMETIC_ERROR


@tool
def multiplication_tool(numbers: str) -> str:
    """Multiply two numbers provided as comma-separated values."""
    try:
        a, b = _parse_pair(numbers)
        return str(a * b)
    except Exception:
        return _ARITHMETIC_ERROR


@tool
def division_tool(numbers: str) -> str:
    """Divide the first number by the second number using two comma-separated numbers."""
    try:
        a, b = _parse_pair(numbers)

        if b == 0:
            return "Error: cannot divide by zero."

        return str(a / b)

    except Exception:
        return _ARITHMETIC_ERROR


CALCULATOR_TOOLS = [addition_tool, subtraction_tool, multiplication_tool, division_tool]


def make_cv_search_tool(vectorstore):
    """Bind the CV search tool to a FAISS index."""

    @tool
    def search_cvs_tool(query: str) -> str:
        """Search the CV database for candidates, skills, education, experience, and qualifications."""
        return search_cvs(vectorstore, query)

    return search_cvs_tool


def make_web_search_tool():
    web_search_tool = DuckDuckGoSearchRun()
    web_search_tool.description = (
        "Search the internet for general, current, recent, or external information "
        "that is not related to the CV database."
    )
    return web_search_tool
