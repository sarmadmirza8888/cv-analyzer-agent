# CV Analyzer Agent

A multi-agent system that runs entirely on a locally hosted, 4-bit quantized **Mistral 7B Instruct**. A coordinator agent delegates each request to a specialist sub-agent for **CV search (FAISS)**, **calculations**, or **web search**. It also has conversation memory, a persistent question cache, and **MLflow tracing on Databricks**.

> This project builds on [`cv-search-rag-agent`](../cv-search-rag-agent), which covers the retrieval pipeline and native tool calling. This repo covers the multi-agent layer on top.

## Features

- **Coordinator + specialists.** A ReAct coordinator routes each request to one of three ReAct sub-agents: CV search, calculator, or web search.
- **RAG over CVs.** About 500 CVs are split into 1000-character chunks (200 overlap), embedded with `all-MiniLM-L6-v2`, and indexed in FAISS. The CV agent retrieves the top 3 chunks and answers only from them.
- **Hardened for a small local LLM.**
  - A custom `GroundedReActOutputParser` keeps the real tool call when the model also guesses a "Final Answer" in the same turn.
  - A response cleaner and generation `stop_strings` stop the model from inventing new questions or tool calls.
  - `max_iterations` and `max_new_tokens` are capped.
- **Summary-buffer memory.** The last 4 turns are kept word for word. Older turns are compressed into a short LLM-written summary. Long answers are clipped in the window.
- **Persistent question cache.** Each question is checked first by exact match on its normalized text, then by cosine similarity between sentence embeddings (default threshold 0.90). Time-sensitive questions ("latest", "price", ...) are never cached.
- **Safe persistence.** Memory and cache are saved to `history.json` with atomic writes (temp file + `os.replace`), corrupt-file recovery, schema versioning, and automatic re-embedding if the embedding model changes.
- **MLflow tracing (optional).** LangChain autologging plus `@mlflow.trace` capture every agent, tool and LLM call in Databricks. Traces are tagged with whether the answer came from the agent or the cache.

## Architecture

```
                    User question
                          |
                 +--------v--------+
                 |  History cache  |-- hit --> cached answer
                 +--------+--------+
                          | miss
                 +--------v--------+
                 | Memory context  |  summary + last 4 turns
                 +--------+--------+
