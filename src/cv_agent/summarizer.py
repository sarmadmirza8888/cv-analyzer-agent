"""Conversation summariser that calls the local model directly."""
from cv_agent.llm import chat_generate
from cv_agent.output_cleaning import clean_react_output
from cv_agent.prompts import SUMMARY_PROMPT


def summarize_text(tokenizer, model, existing_summary: str, new_lines: str,
                   max_new_tokens: int = 180) -> str:
    """Call the local Mistral model directly (NOT through the ReAct pipeline).

    We bypass `text_generation_pipeline` on purpose: that pipeline is configured
    with ReAct stop strings and a 200-token cap for agent reasoning. Summarising
    is a plain completion task, so we want deterministic decoding instead.
    """
    prompt = SUMMARY_PROMPT.format(
        existing_summary=existing_summary.strip() or "(nothing yet)",
        new_lines=new_lines.strip(),
    )

    messages = [
        {"role": "system", "content": "You write short, factual conversation summaries."},
        {"role": "user", "content": prompt},
    ]

    text = chat_generate(
        tokenizer, model, messages, max_new_tokens,
        do_sample=False,      # a summary should be faithful, not creative
    )

    # Reuse the same guard as the agents: a small model sometimes tries to
    # open a brand-new turn after finishing the summary.
    return clean_react_output(text)


def build_summarizer(tokenizer, model, max_new_tokens: int = 180):
    """Return the callable(existing_summary, new_lines) that the memory expects."""
    def summarizer(existing_summary: str, new_lines: str) -> str:
        return summarize_text(tokenizer, model, existing_summary, new_lines, max_new_tokens)
    return summarizer
