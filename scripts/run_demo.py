"""Non-interactive demo: a fixed list of queries, with timing and cache statistics.

Run:  python scripts/run_demo.py
"""
import time

from cv_agent.bootstrap import build_assistant

DEMO_QUERIES = [
    # CV agent (RAG)
    "Who has experience with Python and Docker?",
    # Web agent
    "What is the latest version of Python?",
    # Calculator agent, then the same question reworded, then repeated exactly
    "What is 25 multiplied by 4?",
    "Calculate 25 times 4",
    "What is 25 multiplied by 4?",
]


def main():
    assistant = build_assistant(verbose=False)

    for query in DEMO_QUERIES:
        print(f"\nYou: {query}")
        start = time.perf_counter()
        answer = assistant.ask(query)
        elapsed = time.perf_counter() - start
        print(f"Agent: {answer}\n({elapsed:.2f}s)")

    assistant.cache.stats()


if __name__ == "__main__":
    main()
