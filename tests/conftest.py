import numpy as np


class FakeEmbedder:
    """Maps known texts to fixed vectors. Raises KeyError for anything else, which lets a
    test prove that a code path (e.g. an exact-match hit) never calls the embedder."""

    def __init__(self, table):
        self.table = {k: np.asarray(v, dtype="float32") for k, v in table.items()}
        self.calls = 0

    def embed_query(self, text):
        self.calls += 1
        return self.table[text].tolist()
