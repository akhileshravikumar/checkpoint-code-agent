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
    # Without this, a key that lives only in .env never reaches the client and
    # traces are dropped with a warning. A key already exported by the shell wins.
    if s.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY", s.langsmith_api_key)
    return True
