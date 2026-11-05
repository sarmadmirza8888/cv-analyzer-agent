"""Composition root: load the model, build the index and agents, restore saved history."""
from cv_agent.agents import build_coordinator, build_specialists
from cv_agent.assistant import Components, CVSearchAssistant
from cv_agent.cache import SemanticHistoryCache
from cv_agent.config import (
    CACHE_THRESHOLD, MEMORY_KEEP_LAST_N, MEMORY_MAX_ANSWER_CHARS, MEMORY_SUMMARIZE_EVERY,
)
from cv_agent.llm import build_llm, find_model_path, load_model
from cv_agent.memory import ConversationSummaryBufferMemory
from cv_agent.persistence import load_history, resolve_history_path
from cv_agent.rag import build_cv_index
from cv_agent.summarizer import build_summarizer
from cv_agent.tracing import configure_tracing


def build_assistant(model_path=None, history_path=None, verbose=True, trace=True):
    """Load everything and return a ready-to-use CVSearchAssistant.

    model_path   : folder with the Mistral weights (auto-detected on Kaggle).
    history_path : where history.json lives (defaults to /kaggle/working or ./).
    verbose      : print each Thought/Action/Observation of the ReAct agents.
    trace        : configure MLflow tracing on Databricks if credentials exist.
    """
    if trace:
        configure_tracing()

    model_path = model_path or find_model_path()
    print("Using model folder:", model_path)

    tokenizer, model = load_model(model_path)
    pipe, llm = build_llm(model, tokenizer)
    index = build_cv_index(model_path)

    specialists = build_specialists(llm, index.vectorstore, verbose)
    coordinator = build_coordinator(llm, specialists, verbose)

    memory = ConversationSummaryBufferMemory(
        build_summarizer(tokenizer, model),
        keep_last_n=MEMORY_KEEP_LAST_N,
        summarize_every=MEMORY_SUMMARIZE_EVERY,
        max_answer_chars=MEMORY_MAX_ANSWER_CHARS,
    )
    cache = SemanticHistoryCache(index.embeddings, threshold=CACHE_THRESHOLD)

    # Creates history.json on the first run, reuses it on every run after that.
    history_path = history_path or resolve_history_path()
    load_history(memory, cache, history_path)

    return CVSearchAssistant(
        coordinator, memory, cache, history_path,
        components=Components(tokenizer, model, pipe, llm, index, specialists),
    )
