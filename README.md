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
                          |
                 +--------v--------+
                 |   Coordinator   |  ReAct, Mistral 7B (4-bit)
                 +---+----+----+---+
                     |    |    |
          +----------+    |    +-----------+
          v               v                v
   CV search agent   Calculator agent   Web search agent
   (FAISS, k=3)      (+ - x / tools)    (DuckDuckGo)
```

## Tech stack

Python, PyTorch, Hugging Face Transformers, bitsandbytes (4-bit), LangChain, FAISS, sentence-transformers, DuckDuckGo search, MLflow on Databricks.

## Repository contents

| File | Description |
|---|---|
| `search-cv-agent-databricks.ipynb` | The full agent: model loading, RAG, sub-agents, memory, cache, persistence, tracing |
| `requirements.txt` | Python dependencies |

## Setup

The notebook was developed and run on **Kaggle with a GPU runtime**. The 4-bit model needs a CUDA GPU.

1. **Install dependencies**
```bash
   pip install -r requirements.txt
```
2. **Model.** Provide a 4-bit quantized Mistral 7B Instruct checkpoint. On Kaggle, attach it as a dataset. The notebook looks under `/kaggle/input` for a model folder with "mistral" in its path. Elsewhere, point `MODEL_PATH` at your local folder.
3. **Data.** The CV dataset is not included, because it contains personal information. Each file is a JSON with a `"text"` field holding the CV text. The notebook loads every JSON under `/kaggle/input`.
   Dataset source: **[add the source and link here]**
4. **MLflow / Databricks (optional).** Add `DATABRICKS_HOST` and `DATABRICKS_TOKEN` as Kaggle secrets, and change the experiment path in the second cell to your own. To run without tracing, skip that cell and remove the `@mlflow.trace` and `mlflow.update_current_trace` lines.
5. **Run** the notebook top to bottom.

Outside Kaggle, replace `kaggle_secrets` with environment variables and change the `/kaggle/...` paths.

## Usage

The last cell starts an interactive chat:

```
You: Who has experience with Python and Docker?
You: What is 25 multiplied by 4?
You: What is the latest version of Python?
```

Commands: `memory` (show the summary and window), `cache` (cache statistics), `save`, `reset`, `exit`.

## Tracing

[Add a screenshot of an MLflow trace here. Use a calculator question, not a CV question, so no personal data is shown.]

## Known limitations

- The model is small, so it can still misroute or fail to follow the ReAct format. The parser, cleaner, stop strings and iteration caps reduce this but do not remove it.
- At the default 0.90 threshold, the cache only fires on near-identical questions. In my probes, genuine paraphrases scored about 0.6 to 0.8 and missed. Lowering the threshold catches more but risks returning an answer to a different question, so tune it with `qa_cache.rank(...)`.
- Retrieval is plain top-3 similarity search. Questions about many candidates at once ("how many candidates know X?") are not handled well.
- There is no quantitative evaluation of routing or answer quality yet.

## Privacy

The CV data comes from real people. Do not commit the dataset or `history.json`, and clear notebook outputs that print retrieved CV text before sharing.

## Author
Muhammad Sarmad Mirza: [GitHub](https://github.com/sarmadmirza8888) | [LinkedIn](https://linkedin.com/in/muhammad-sarmad-mirza-1a1a66381)
