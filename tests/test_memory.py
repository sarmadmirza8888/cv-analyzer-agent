from cv_agent.memory import ConversationSummaryBufferMemory


class RecordingSummarizer:
    def __init__(self):
        self.calls = []

    def __call__(self, existing_summary, new_lines):
        self.calls.append((existing_summary, new_lines))
        return f"summary-{len(self.calls)}"


def test_old_turns_are_summarised_and_recent_turns_stay_verbatim():
    summarizer = RecordingSummarizer()
    memory = ConversationSummaryBufferMemory(summarizer, keep_last_n=2, summarize_every=1)

    for i in range(3):
        memory.add_turn(f"q{i}", f"a{i}")

    assert memory.buffer == [("q1", "a1"), ("q2", "a2")]
    assert memory.summary == "summary-1"
    assert "Human: q0" in summarizer.calls[0][1]


def test_summarisation_is_batched():
    summarizer = RecordingSummarizer()
    memory = ConversationSummaryBufferMemory(summarizer, keep_last_n=1, summarize_every=2)

    memory.add_turn("q0", "a0")
    memory.add_turn("q1", "a1")  # one turn overflowed, below the batch size
    assert summarizer.calls == []

    memory.add_turn("q2", "a2")  # two overflowed: now pay for one call
    assert len(summarizer.calls) == 1


def test_long_answers_are_clipped_in_the_context_window():
    memory = ConversationSummaryBufferMemory(RecordingSummarizer(), max_answer_chars=10)
    memory.add_turn("q", "x" * 500)

    context = memory.get_context()

    assert "x" * 11 not in context and context.endswith("...")


def test_state_round_trip():
    memory = ConversationSummaryBufferMemory(RecordingSummarizer(), keep_last_n=1, summarize_every=1)
    memory.add_turn("q0", "a0")
    memory.add_turn("q1", "a1")

    restored = ConversationSummaryBufferMemory(RecordingSummarizer())
    restored.load_state_dict(memory.state_dict())

    assert restored.summary == memory.summary
    assert restored.buffer == memory.buffer
