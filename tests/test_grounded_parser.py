import pytest

pytest.importorskip("langchain_classic")

from langchain_core.agents import AgentAction, AgentFinish  # noqa: E402

from cv_agent.grounded_parser import GroundedReActOutputParser  # noqa: E402


def test_action_wins_over_a_guessed_final_answer():
    text = (
        "Thought: I should search.\n"
        "Action: search_cvs_tool\n"
        "Action Input: Python\n"
        "Final Answer: the candidates are [names]"
    )

    result = GroundedReActOutputParser().parse(text)

    assert isinstance(result, AgentAction)
    assert result.tool == "search_cvs_tool"


def test_plain_final_answer_still_finishes():
    result = GroundedReActOutputParser().parse("Thought: done\nFinal Answer: 100")

    assert isinstance(result, AgentFinish)
    assert result.return_values["output"] == "100"
