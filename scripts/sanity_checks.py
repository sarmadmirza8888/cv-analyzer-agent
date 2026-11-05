"""Component checks against the real model: raw generation, stop strings, retrieval.

Run:  python scripts/sanity_checks.py
"""
from cv_agent.bootstrap import build_assistant
from cv_agent.llm import ask_mistral
from cv_agent.rag import search_cvs


def main():
    c = build_assistant(verbose=False, trace=False).components

    print("--- raw generation ---")
    print(ask_mistral(c.tokenizer, c.model, "Explain what a vector database is in simple words."))

    print("\n--- stop strings (generation should halt at the first new-turn marker) ---")
    test_output = c.pipeline(
        "Question: Who knows Python?\nThought: I should search.\n"
        "Action: search_cvs_tool\nAction Input: Python\nObservation:"
    )
    print(test_output[0]["generated_text"])

    print("\n--- retrieval ---")
    print(search_cvs(c.index.vectorstore, "Who has experience with Python and Docker?"))


if __name__ == "__main__":
    main()
