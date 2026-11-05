"""The assistant: one traced entry point that ties cache, memory, coordinator and persistence together.

This module is deliberately light (no torch, no LangChain imports) so it can be unit-tested
with a fake coordinator. Model loading lives in cv_agent.bootstrap.
"""
from dataclasses import dataclass
from typing import Any

import mlflow

from cv_agent.output_cleaning import clean_react_output
from cv_agent.persistence import resolve_history_path, save_history


@dataclass
class Components:
    """Heavy objects, exposed so scripts can run sanity checks without reloading the model."""
    tokenizer: Any
    model: Any
    pipeline: Any
    llm: Any
    index: Any
    specialists: Any


class CVSearchAssistant:
    def __init__(self, coordinator, memory, cache, history_path=None, components=None):
        self.coordinator = coordinator
        self.memory = memory
        self.cache = cache
        self.history_path = history_path or resolve_history_path()
        self.components = components

    @mlflow.trace(name="cv_agent_query")
    def ask(self, user_query, use_cache=True, show_trace=True, persist=True):
        """Answer one question.

          1. Ask the history cache first - if we already answered this, return it.
          2. Otherwise build the context window (summary + recent turns).
          3. Run the coordinator agent.
          4. Record the turn in memory + cache, then write history.json.
        """
        # ---- Step 1: can history answer this already? ----
        if use_cache:
            hit = self.cache.lookup(user_query)
            if hit:
                if show_trace:
                    age_h = hit["age_seconds"] / 3600
                    print(f"[history] answered from memory "
                          f"({hit['method']}, similarity={hit['score']:.3f}, "
                          f"stored {age_h:.1f}h ago)")
                    print(f"[history] matched earlier question: {hit['matched_question']}")
                self.memory.add_turn(user_query, hit["answer"])
                if persist:
                    self.save(quiet=True)
                mlflow.update_current_trace(
                    tags={"answer_source": "cache", "cache_method": hit["method"]}
                )
                return hit["answer"]

        # ---- Step 2: build the bounded context window ----
        context = self.memory.get_context()

        # ---- Step 3: real agent run ----
        result = self.coordinator.invoke({
            "input": user_query,
            "chat_history": context,
        })
        answer = clean_react_output(result["output"])

        # ---- Step 4: remember it, on disk as well as in RAM ----
        self.memory.add_turn(user_query, answer)
        self.cache.add(user_query, answer)
        if persist:
            self.save(quiet=True)

        mlflow.update_current_trace(tags={"answer_source": "agent"})
        return answer

    def save(self, quiet=False):
        return save_history(self.memory, self.cache, self.history_path, quiet=quiet)
