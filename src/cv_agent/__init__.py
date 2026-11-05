"""CV search multi-agent system: coordinator + specialist ReAct agents on a local Mistral model."""

__version__ = "0.1.0"
__all__ = ["build_assistant", "CVSearchAssistant"]


def __getattr__(name):
    # Lazy import so `import cv_agent.cache` does not pull in torch and LangChain.
    if name in __all__:
        from cv_agent import assistant, bootstrap
        return getattr(bootstrap if name == "build_assistant" else assistant, name)
    raise AttributeError(f"module 'cv_agent' has no attribute {name!r}")
