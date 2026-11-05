"""Summary-buffer conversation memory: recent turns verbatim, older turns summarised."""


class ConversationSummaryBufferMemory:
    """Recent turns kept verbatim, older turns compressed into a summary.

    Parameters
    ----------
    summarizer : callable(existing_summary, new_lines) -> str. Injected so this
        class has no dependency on the model and can be tested with a stub.
    keep_last_n : how many recent turns stay word-for-word in the window.
    summarize_every : how many turns must overflow the window before we pay
        for a summarisation call (batching keeps it cheap).
    max_answer_chars : long answers get clipped inside the window so one huge
        CV dump cannot eat the whole context.
    """

    def __init__(self, summarizer, keep_last_n=4, summarize_every=2, max_answer_chars=400):
        self.summarizer = summarizer
        self.keep_last_n = keep_last_n
        self.summarize_every = summarize_every
        self.max_answer_chars = max_answer_chars

        self.summary = ""        # compressed older history
        self.buffer = []         # [(question, answer), ...] recent, verbatim
        self.pending = []        # turns that fell out, awaiting summarisation
        self.all_turns = []      # full log, never trimmed (for inspection only)

    # -- internal helpers ------------------------------------------------

    def _clip(self, text):
        text = (text or "").strip()
        if len(text) > self.max_answer_chars:
            return text[: self.max_answer_chars].rstrip() + " ..."
        return text

    def _flush_pending(self):
        """Fold the overflowed turns into the running summary."""
        if not self.pending:
            return

        new_lines = "\n".join(
            f"Human: {q}\nAI: {self._clip(a)}" for q, a in self.pending
        )
        self.summary = self.summarizer(self.summary, new_lines)
        self.pending = []

    def add_turn(self, question, answer):
        self.all_turns.append((question, answer))
        self.buffer.append((question, answer))

        # Anything past the window falls into `pending`.
        while len(self.buffer) > self.keep_last_n:
            self.pending.append(self.buffer.pop(0))

        if len(self.pending) >= self.summarize_every:
            self._flush_pending()

    def get_context(self) -> str:
        """The string that gets injected into {chat_history} in the prompt."""
        parts = []

        if self.summary:
            parts.append(f"Summary of earlier conversation:\n{self.summary}")

        if self.buffer:
            recent = "\n".join(
                f"Human: {q}\nAI: {self._clip(a)}" for q, a in self.buffer
            )
            parts.append(f"Most recent turns:\n{recent}")

        return "\n\n".join(parts)

    def clear(self):
        self.summary = ""
        self.buffer = []
        self.pending = []
        self.all_turns = []

    def show(self):
        print("=" * 60)
        print(f"Turns stored : {len(self.all_turns)}")
        print(f"In window    : {len(self.buffer)} (verbatim)")
        print(f"Awaiting sum.: {len(self.pending)}")
        print("-" * 60)
        print("SUMMARY:", self.summary or "(empty)")
        print("=" * 60)

    # -- serialisation ---------------------------------------------------
    #  Only the summary + the current window are worth saving. `all_turns`
    #  is a debug log; persisting it would grow the file forever for no gain.

    def state_dict(self):
        return {
            "summary": self.summary,
            "buffer": [list(t) for t in self.buffer],
            "pending": [list(t) for t in self.pending],
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.summary = state.get("summary", "") or ""
        self.buffer = [tuple(t) for t in state.get("buffer", []) if len(t) == 2]
        self.pending = [tuple(t) for t in state.get("pending", []) if len(t) == 2]
        self.all_turns = list(self.pending) + list(self.buffer)
