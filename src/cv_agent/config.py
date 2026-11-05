"""Central configuration: tunable constants and environment-variable names.

Nothing here imports heavy dependencies, so every module can read it cheaply.
"""

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Locations (override with environment variables to run outside Kaggle) ---
KAGGLE_INPUT_DIR = "/kaggle/input"
MODEL_PATH_ENV = "CV_AGENT_MODEL_PATH"      # folder containing the Mistral weights
DATA_DIR_ENV = "CV_AGENT_DATA_DIR"          # folder containing the CV .json files
HISTORY_PATH_ENV = "CV_AGENT_HISTORY_PATH"  # where history.json is written
EXPERIMENT_ENV = "MLFLOW_EXPERIMENT"        # e.g. /Users/<you>/my-ai-agent

# --- RAG ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

# --- Local LLM pipeline used by the agents ---
# was 400 - a shorter cap so a runaway generation can't fabricate multiple fake Q&A turns
MAX_NEW_TOKENS = 200
# stop the instant the model tries to open a NEW turn (fabricated Question/Thought/Human),
# not just the two cases we originally guarded against
STOP_STRINGS = ["\nObservation", "\nQuestion:", "\nThought:", "\nHuman:"]

# --- Agents ---
SPECIALIST_MAX_ITERATIONS = 4
COORDINATOR_MAX_ITERATIONS = 2  # cap so a confused small model can't loop forever

# --- Conversation memory ---
MEMORY_KEEP_LAST_N = 4
MEMORY_SUMMARIZE_EVERY = 2
MEMORY_MAX_ANSWER_CHARS = 400

# --- Semantic cache ---
CACHE_THRESHOLD = 0.90

# --- Persistence ---
SCHEMA_VERSION = 1
