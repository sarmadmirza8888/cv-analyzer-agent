import glob
import json

from conftest import FakeEmbedder
from cv_agent.cache import SemanticHistoryCache
from cv_agent.memory import ConversationSummaryBufferMemory
from cv_agent.persistence import load_history, save_history

TABLE = {"q1": [1, 0, 0], "q2": [0, 1, 0]}


def fresh():
    memory = ConversationSummaryBufferMemory(lambda existing, new: "summary")
    cache = SemanticHistoryCache(FakeEmbedder(TABLE))
    return memory, cache


def test_save_then_load_restores_memory_and_cache(tmp_path):
    path = str(tmp_path / "history.json")
    memory, cache = fresh()
    memory.add_turn("q1", "a1")
    cache.add("q1", "a1")
    save_history(memory, cache, path, quiet=True)

    memory2, cache2 = fresh()
    assert load_history(memory2, cache2, path, verbose=False) is True

    assert memory2.buffer == [("q1", "a1")]
    assert cache2.lookup("q1")["answer"] == "a1"


def test_missing_file_is_not_an_error(tmp_path):
    memory, cache = fresh()
    assert load_history(memory, cache, str(tmp_path / "nope.json"), verbose=False) is False


def test_corrupt_file_is_moved_aside(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{not json")
    memory, cache = fresh()

    assert load_history(memory, cache, str(path), verbose=False) is False

    assert not path.exists()
    assert len(glob.glob(str(path) + ".corrupt-*")) == 1


def test_changed_embedding_model_triggers_re_embedding(tmp_path):
    path = tmp_path / "history.json"
    memory, cache = fresh()
    cache.add("q1", "a1")
    save_history(memory, cache, str(path), quiet=True)

    data = json.loads(path.read_text())
    data["embedding_model"] = "some-other-model"
    data["vector_dim"] = 99  # would be skipped if vectors were reused
    path.write_text(json.dumps(data))

    memory2, cache2 = fresh()
    load_history(memory2, cache2, str(path), verbose=False)

    assert cache2.questions == ["q1"]


def test_save_leaves_no_temp_file_behind(tmp_path):
    path = str(tmp_path / "history.json")
    memory, cache = fresh()
    save_history(memory, cache, path, quiet=True)

    assert not glob.glob(path + ".tmp")
