"""LangSmith wiring. Honours CHECKPOINT_OFFLINE for the no-network claim."""
import os

from app.config import get_settings


def configure_tracing() -> bool:
    s = get_settings()
    if s.checkpoint_offline or not s.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = s.langsmith_project
    return True