# Design decisions

Each row is a problem hit while building this with a 7B local model, and the fix in the code.

## Reliability of a small local model

| Problem | Fix | Where |
|---|---|---|
| The model keeps generating after it answers and opens a fake `Question:` / `Thought:` / `Human:` turn. The fake turn leaks into a tool's output and the calling agent follows it onto an invented task. | `stop_strings` halt generation at the start of a new turn; `max_new_tokens` was cut from 400 to 200 so a runaway cannot fabricate several fake turns; `clean_react_output` trims whatever still leaks. The matched stop string stays at the end of the text, so cleaning is a needed second layer. | `llm.py`, `output_cleaning.py` |
| The model emits an `Action` and a guessed `Final Answer` in the same turn. The stock parser raises an error, the error text is fed back, and the model invents a tool that does not exist. | `GroundedReActOutputParser` keeps the Action and drops the guess, so the answer rests on a real Observation. | `grounded_parser.py` |
| The model writes `Observation` into a tool's input (for example `25, 4\nObservation`). | Arithmetic tools strip the word before parsing. | `tools.py` |
| A confused coordinator can loop or wander. | Prompt forbids doing the work itself; `max_iterations=2`; `handle_parsing_errors=True`; `early_stopping_method="force"` (`"generate"` is not valid in this langchain-classic version). | `agents.py`, `prompts.py` |
| `apply_chat_template` return type differs across transformers versions (caused a `shape` KeyError). | `return_dict=True` and explicit `**encoded` unpacking. | `llm.py` |

## Memory and cache

| Problem | Fix | Where |
|---|---|---|
| Summarising through the ReAct pipeline would inherit its stop strings and 200-token cap. | The summariser calls `model.generate` directly with greedy decoding. | `summarizer.py` |
| One long CV dump can fill the context window. | Answers are clipped to 400 characters inside the window; older turns are folded into a summary in batches of two. | `memory.py` |
| A cache would serve stale answers to questions whose answer changes. | Questions containing `TIME_SENSITIVE_WORDS` are never cached or looked up. Optional TTL. | `cache.py` |
| "2+5?" and "2 + 5" should hit the same entry. | `normalize_question` lowercases, turns punctuation into spaces, strips last. | `cache.py` |
| Paraphrases need to hit without a model call, exact repeats should cost nothing. | Two layers: dictionary lookup on the normalised question, then cosine similarity over L2-normalised embeddings (threshold 0.90). | `cache.py` |

## Persistence and observability

| Problem | Fix | Where |
|---|---|---|
| A kernel crash mid-write could truncate saved history. | Write to a temp file, then `os.replace`. | `persistence.py` |
| A corrupt or outdated history file should not crash startup. | Corrupt file is moved aside with a timestamp; a changed embedding model triggers re-embedding; vectors with the wrong dimension are skipped. | `persistence.py`, `cache.py` |
| Need to see which path produced an answer. | LangChain autologging traces every call; `@mlflow.trace` on `CVSearchAssistant.ask` adds a root span, tagged `answer_source` = `cache` or `agent`. | `tracing.py`, `assistant.py` |

## Known failure modes

- The grounded parser reduces unfilled-placeholder answers but does not eliminate them. In one captured run the Java query returned `[Candidate 1, ...]` instead of names (see `notebooks/cv_agent_walkthrough.ipynb`, Section 11.5).
- Retrieval returns the top 3 chunks, so aggregate questions ("who has X and Y", counts) are not reliable.
- The 0.90 cache threshold is untuned: in the captured probe two paraphrase pairs scored 0.773 and 0.827 and did not match.
