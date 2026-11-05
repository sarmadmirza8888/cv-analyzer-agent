"""Two-layer semantic cache over past (question, answer) pairs.

Layer 1 is an exact match on the normalised question. Layer 2 is cosine similarity
over sentence embeddings. Questions whose answer changes over time are never cached.
"""
import re
import time
from datetime import datetime

import numpy as np


def normalize_question(q: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    This is what makes "What is 2 + 5?" and "what is 2+5" hash to the
    same key in the exact-match layer.
    """
    q = q.lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = re.sub(r"\s+", " ", q)
    return q.strip()   # strip LAST - punctuation becomes a space, so "2+5?" would
                       # otherwise leave a trailing space and never match "2 + 5"


# Questions whose correct answer changes over time must NOT be cached forever.
TIME_SENSITIVE_WORDS = {
    "latest", "current", "currently", "today", "now", "recent", "recently",
    "news", "price", "weather", "version", "newest", "update", "updated",
    "yesterday", "tomorrow", "live",
}


class SemanticHistoryCache:
    """Two-layer question lookup over past (question, answer) pairs.

    Layer 1 - exact match on the normalised string: O(1) dict lookup.
    Layer 2 - cosine similarity over sentence embeddings: O(n) dot product.

    Both vectors are L2-normalised, so the dot product IS the cosine
    similarity (cos = a.b / (|a||b|), and |a| = |b| = 1).

    Each entry holds: question, answer, vector, timestamp.
    """

    def __init__(self, embedder, threshold=0.90, ttl_seconds=None):
        self.embedder = embedder
        self.threshold = threshold      # tune this - see the test cell below
        self.ttl_seconds = ttl_seconds  # None = entries never expire

        self.questions = []
        self.answers = []
        self.vectors = []               # list of normalised np.float32 arrays
        self.timestamps = []
        self.exact_index = {}           # normalised question -> position

        self._matrix = None             # cached np.vstack of self.vectors
        self.hits = 0
        self.misses = 0
        self.loaded_from_disk = 0

    # -- internal helpers ------------------------------------------------

    def _embed(self, text):
        v = np.asarray(self.embedder.embed_query(text), dtype="float32")
        return self._unit(v)

    @staticmethod
    def _unit(v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def _get_matrix(self):
        """Stack vectors once and reuse. Rebuilt only when an entry is added."""
        if self._matrix is None and self.vectors:
            self._matrix = np.vstack(self.vectors)
        return self._matrix

    def _is_fresh(self, i):
        if self.ttl_seconds is None:
            return True
        return (time.time() - self.timestamps[i]) < self.ttl_seconds

    def is_cacheable(self, question: str) -> bool:
        tokens = set(normalize_question(question).split())
        return not (tokens & TIME_SENSITIVE_WORDS)

    # -- public API ------------------------------------------------------

    def add(self, question, answer, vector=None, timestamp=None):
        if not self.is_cacheable(question):
            return False  # e.g. "what is the latest version of Python"

        key = normalize_question(question)

        if key in self.exact_index:          # already seen: refresh in place
            i = self.exact_index[key]
            self.answers[i] = answer
            self.timestamps[i] = timestamp or time.time()
            return False

        self.exact_index[key] = len(self.questions)
        self.questions.append(question)
        self.answers.append(answer)
        self.vectors.append(self._embed(question) if vector is None else self._unit(vector))
        self.timestamps.append(timestamp or time.time())
        self._matrix = None               # invalidate the cached stack
        return True

    def lookup(self, question):
        """Return a hit dict, or None if the agent has to do real work."""
        if not self.questions or not self.is_cacheable(question):
            self.misses += 1
            return None


        key = normalize_question(question)
        if key in self.exact_index:
            i = self.exact_index[key]
            if self._is_fresh(i):
                self.hits += 1
                return {
                    "answer": self.answers[i],
                    "score": 1.0,
                    "matched_question": self.questions[i],
                    "method": "exact",
                    "age_seconds": time.time() - self.timestamps[i],
                }


        query_vec = self._embed(question)
        sims = self._get_matrix() @ query_vec   # cosine sim for every entry

        best = int(np.argmax(sims))
        best_score = float(sims[best])

        if best_score >= self.threshold and self._is_fresh(best):
            self.hits += 1
            return {
                "answer": self.answers[best],
                "score": best_score,
                "matched_question": self.questions[best],
                "method": "semantic",
                "age_seconds": time.time() - self.timestamps[best],
            }

        self.misses += 1
        return None

    def rank(self, question, top_k=3):
        """Debug helper: show the closest stored questions and their scores."""
        if not self.questions:
            print("(cache is empty)")
            return

        sims = self._get_matrix() @ self._embed(question)
        order = np.argsort(sims)[::-1][:top_k]

        print(f"Query: {question}")
        for i in order:
            mark = "HIT " if sims[i] >= self.threshold else "miss"
            print(f"  [{mark}] {sims[i]:.3f}  <- {self.questions[i]}")

    def stats(self):
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0
        print(f"Cached questions: {len(self.questions)} "
              f"({self.loaded_from_disk} loaded from disk) | "
              f"hits: {self.hits} | misses: {self.misses} | hit rate: {rate:.1f}%")

    def clear(self):
        self.questions, self.answers = [], []
        self.vectors, self.timestamps = [], []
        self.exact_index = {}
        self._matrix = None
        self.hits = self.misses = self.loaded_from_disk = 0

    # -- serialisation ---------------------------------------------------

    def export_entries(self):
        """question / answer / vector / timestamp, ready for JSON."""
        out = []
        for q, a, v, ts in zip(self.questions, self.answers,
                               self.vectors, self.timestamps):
            out.append({
                "question": q,
                "normalized": normalize_question(q),
                "answer": a,
                # 6 decimals is far below what cosine similarity can resolve
                # and roughly halves the file size vs full float repr
                "vector": [round(float(x), 6) for x in v],
                "timestamp": ts,
                "saved_at": datetime.fromtimestamp(ts).isoformat(timespec="seconds"),
            })
        return out

    def import_entries(self, entries, expected_dim=None, re_embed=False):
        """Load entries back in. Returns (loaded, skipped).

        re_embed=True recomputes vectors from the stored question text, which
        is what we do if the embedding model changed since the file was written.
        """
        loaded = skipped = 0

        for e in entries:
            try:
                q = e["question"]
                a = e["answer"]
                ts = float(e.get("timestamp") or time.time())
                vec = None

                if not re_embed:
                    raw = e.get("vector")
                    if not raw:
                        skipped += 1
                        continue
                    vec = np.asarray(raw, dtype="float32")
                    if expected_dim and vec.shape[0] != expected_dim:
                        skipped += 1
                        continue
                    # rounding perturbed the norm slightly - renormalise so the
                    # dot product is still exactly the cosine similarity
                    vec = self._unit(vec)

                if self.add(q, a, vector=vec, timestamp=ts):
                    loaded += 1
                else:
                    skipped += 1

            except (KeyError, TypeError, ValueError):
                skipped += 1

        self.loaded_from_disk += loaded
        return loaded, skipped
