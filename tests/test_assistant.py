import pytest

pytest.importorskip("mlflow")

from conftest import FakeEmbedder  # noqa: E402
from cv_agent.assistant import CVSearchAssistant  # noqa: E402
from cv_agent.cache import SemanticHistoryCache  # noqa: E402
from cv_agent.memory import ConversationSummaryBufferMemory  # noqa: E402


class FakeCoordinator:
    """Stands in for the ReAct coordinator and returns the kind of tail the small model leaks."""

    def __init__(self):
        self.calls = 0

    def invoke(self, inputs):
        self.calls += 1
        return {"output": "100\n\nQuestion:"}


def make_assistant(tmp_path):
    memory = ConversationSummaryBufferMemory(lambda existing, new: "summary")
    cache = SemanticHistoryCache(FakeEmbedder({"What is 25 times 4?": [1, 0, 0]}))
    coordinator = FakeCoordinator()
    assistant = CVSearchAssistant(coordinator, memory, cache, str(tmp_path / "history.json"))
    return assistant, coordinator


def test_miss_runs_the_agent_and_cleans_the_answer(tmp_path):
    assistant, coordinator = make_assistant(tmp_path)

    assert assistant.ask("What is 25 times 4?", show_trace=False) == "100"
    assert coordinator.calls == 1


def test_repeat_question_is_served_from_cache(tmp_path):
    assistant, coordinator = make_assistant(tmp_path)
    assistant.ask("What is 25 times 4?", show_trace=False)

    assert assistant.ask("what is 25 times 4", show_trace=False) == "100"
    assert coordinator.calls == 1  # the second call never reached the agent


def test_answers_are_persisted(tmp_path):
    assistant, _ = make_assistant(tmp_path)
    assistant.ask("What is 25 times 4?", show_trace=False)

    assert (tmp_path / "history.json").exists()
