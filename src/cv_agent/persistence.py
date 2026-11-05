"""Persist memory and the semantic cache to history.json.

Schema
------
{
  "schema_version": 1,
  "embedding_model": "...",      # so we can detect a model change
  "vector_dim": 384,
  "created_at": <unix ts>,
  "updated_at": <unix ts>,
  "memory":  {"summary": "...", "buffer": [...], "pending": [...]},
  "entries": [ {question, normalized, answer, vector, timestamp, saved_at} ]
}
"""
import glob
import json
import os
import shutil
import time

from cv_agent.config import EMBEDDING_MODEL_NAME, HISTORY_PATH_ENV, SCHEMA_VERSION


def resolve_history_path():
    """CV_AGENT_HISTORY_PATH if set, else /kaggle/working on Kaggle, else the current folder."""
    override = os.environ.get(HISTORY_PATH_ENV)
    if override:
        return override
    if os.path.isdir("/kaggle/working"):
        return "/kaggle/working/history.json"
    return os.path.abspath("history.json")


def find_seed_history(path):
    """First run on a fresh session: look for a history.json we saved earlier.

    If you attached a previous notebook's output as a dataset, it shows up
    under /kaggle/input (read-only), so we copy it into the working dir.
    """
    if os.path.exists(path):
        return None

    candidates = glob.glob("/kaggle/input/**/history.json", recursive=True)
    return candidates[0] if candidates else None


def save_history(memory, cache, path=None, quiet=False):
    """Write the whole history atomically.

    Atomic = write to a temp file, then os.replace(). If the kernel dies
    mid-write you keep the last good file instead of a truncated one.
    """
    path = path or resolve_history_path()
    created = time.time()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                created = json.load(f).get("created_at", created)
        except Exception:
            pass

    payload = {
        "schema_version": SCHEMA_VERSION,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_dim": int(cache.vectors[0].shape[0]) if cache.vectors else None,
        "created_at": created,
        "updated_at": time.time(),
        "memory": memory.state_dict(),
        "entries": cache.export_entries(),
    }

    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, path)

    if not quiet:
        size_kb = os.path.getsize(path) / 1024
        print(f"[history] saved {len(payload['entries'])} entries "
              f"-> {path} ({size_kb:.1f} KB)")

    return path


def load_history(memory, cache, path=None, verbose=True):
    """Restore memory + cache from disk. Safe to call when no file exists."""

    path = path or resolve_history_path()
    seed = find_seed_history(path)
    if seed:
        shutil.copy(seed, path)
        if verbose:
            print(f"[history] seeded from attached dataset: {seed}")

    if not os.path.exists(path):
        if verbose:
            print(f"[history] no file yet - one will be created at {path}")
        return False

    # --- read, tolerating a corrupt file ---
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        backup = f"{path}.corrupt-{int(time.time())}"
        shutil.move(path, backup)
        print(f"[history] file unreadable ({e}); moved to {backup}. Starting fresh.")
        return False

    # --- does the saved model match the one currently loaded? ---
    saved_model = data.get("embedding_model")
    re_embed = bool(saved_model and saved_model != EMBEDDING_MODEL_NAME)
    if re_embed:
        print(f"[history] embedding model changed "
              f"({saved_model} -> {EMBEDDING_MODEL_NAME}); re-embedding questions.")

    memory.load_state_dict(data.get("memory"))
    loaded, skipped = cache.import_entries(
        data.get("entries", []),
        expected_dim=data.get("vector_dim"),
        re_embed=re_embed,
    )

    if verbose:
        age_days = (time.time() - data.get("created_at", time.time())) / 86400
        print(f"[history] loaded {loaded} cached answers from {path} "
              f"(skipped {skipped}, file age {age_days:.1f} days)")
        if memory.summary:
            print(f"[history] restored summary: {memory.summary[:120]}...")

    return True
