"""Ollama client factory, tuned for CPU inference."""
from langchain_ollama import ChatOllama

from app.config import get_settings


def get_llm(*, num_predict: int | None = None, streaming: bool = False) -> ChatOllama:
    s = get_settings()
    kwargs = dict(
        model=s.ollama_model,
        base_url=s.ollama_base_url,
        temperature=s.ollama_temperature,
        num_ctx=s.ollama_num_ctx,
        num_predict=num_predict or s.ollama_num_predict,
        # Keeps the model resident across the human approval pause. Without this,
        # every cycle pays a ~15s reload from disk. See week-0-setup.md §0.3.
        keep_alive="30m",
        disable_streaming=not streaming,
    )
    if s.ollama_num_thread > 0:
        kwargs["num_thread"] = s.ollama_num_thread
    return ChatOllama(**kwargs)