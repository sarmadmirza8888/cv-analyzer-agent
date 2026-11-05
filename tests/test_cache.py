import time

from conftest import FakeEmbedder
from cv_agent.cache import SemanticHistoryCache, normalize_question

Q = "What is 25 times 4?"
Q_PARAPHRASE = "Calculate 25 times 4"
Q_UNRELATED = "Who knows Kubernetes?"

TABLE = {
    Q: [1, 0, 0],
    Q_PARAPHRASE: [0.99, 0.1, 0],  # cosine ~0.995 with Q
    Q_UNRELATED: [0, 1, 0],        # cosine 0 with Q
}


def make(**kwargs):
    return SemanticHistoryCache(FakeEmbedder(TABLE), threshold=0.90, **kwargs)


def test_normalize_unifies_case_and_punctuation():
    assert normalize_question("What is 2 + 5?") == normalize_question("what is 2+5")


def test_exact_hit_does_not_call_the_embedder():
    cache = make()
    cache.add(Q, "100")
    calls_after_add = cache.embedder.calls

    hit = cache.lookup("what is 25 times 4")

    assert hit["method"] == "exact" and hit["answer"] == "100"
    assert cache.embedder.calls == calls_after_add


def test_semantic_hit_above_threshold():
    cache = make()
    cache.add(Q, "100")

    hit = cache.lookup(Q_PARAPHRASE)

    assert hit["method"] == "semantic" and hit["score"] >= 0.90
    assert cache.hits == 1


def test_unrelated_question_misses():
    cache = make()
    cache.add(Q, "100")

    assert cache.lookup(Q_UNRELATED) is None
    assert cache.misses == 1


def test_time_sensitive_questions_are_never_cached_or_looked_up():
    cache = make()
    cache.add(Q, "100")

    assert cache.add("What is the latest version of Python?", "3.x") is False
    # Not in the embedder table: reaching the embedder would raise KeyError.
    assert cache.lookup("What is the latest version of Python?") is None


def test_expired_entries_are_not_served():
    cache = make(ttl_seconds=60)
    cache.add(Q, "100", timestamp=time.time() - 3600)

    assert cache.lookup(Q) is None


def test_export_import_round_trip():
    source = make()
    source.add(Q, "100")
    source.add(Q_UNRELATED, "Ali")

    target = make()
    loaded, skipped = target.import_entries(source.export_entries(), expected_dim=3)

    assert (loaded, skipped) == (2, 0)
    assert target.lookup(Q)["answer"] == "100"


def test_import_skips_vectors_with_the_wrong_dimension():
    target = make()
    entries = [{"question": Q, "answer": "100", "vector": [1.0, 0.0], "timestamp": time.time()}]

    assert target.import_entries(entries, expected_dim=3) == (0, 1)


def test_import_can_re_embed_from_question_text():
    target = make()
    entries = [{"question": Q, "answer": "100", "vector": None, "timestamp": time.time()}]

    assert target.import_entries(entries, re_embed=True) == (1, 0)
