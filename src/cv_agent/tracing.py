"""MLflow tracing on Databricks.

Credentials are read from environment variables, or from Kaggle secrets when
running on Kaggle. Nothing sensitive is stored in the repository.
"""
import os

import mlflow

from cv_agent.config import EXPERIMENT_ENV


def _load_databricks_credentials() -> bool:
    if os.environ.get("DATABRICKS_HOST") and os.environ.get("DATABRICKS_TOKEN"):
        return True
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        return False

    secrets = UserSecretsClient()
    os.environ["DATABRICKS_HOST"] = secrets.get_secret("DATABRICKS_HOST")
    os.environ["DATABRICKS_TOKEN"] = secrets.get_secret("DATABRICKS_TOKEN")
    return True


def configure_tracing(experiment: str | None = None) -> bool:
    """Point MLflow at Databricks and enable LangChain autologging.

    Returns False (and leaves MLflow untouched) if no credentials are found.
    """
    if not _load_databricks_credentials():
        print("[tracing] DATABRICKS_HOST / DATABRICKS_TOKEN not found; Databricks tracing disabled.")
        return False

    experiment = experiment or os.environ.get(EXPERIMENT_ENV)
    if not experiment or "<your-email>" in experiment:
        raise ValueError(
            f"Set {EXPERIMENT_ENV} to your Databricks experiment path, "
            "for example /Users/you@example.com/my-ai-agent"
        )

    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment(experiment)

    # Automatically trace every LangChain call (agent runs, tool calls, retriever calls, LLM calls)
    mlflow.langchain.autolog()

    print("Connected! MLflow autologging enabled for LangChain.")
    return True
