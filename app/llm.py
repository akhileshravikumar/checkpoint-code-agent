"""Ollama client factory, tuned for CPU inference."""
from langchain_ollama import ChatOllama

from app.config import get_settings


def retry_model() -> str:
    """The model to use once a first attempt has already failed.

    ADR-002 picked a 3B model for latency. It holds for a first pass, but a
    retry is the case where the extra capability is worth 2-3x the wall clock:
    the loop only gets MAX_RETRIES of them, and a wrong retry costs a whole CI
    round trip. OLLAMA_MODEL_RETRY empty means "same model as always".
    """
    s = get_settings()
    return s.ollama_model_retry or s.ollama_model


def get_llm(
    *,
    num_predict: int | None = None,
    temperature: float | None = None,
    streaming: bool = False,
    model: str | None = None,
) -> ChatOllama:
    s = get_settings()
    kwargs = dict(
        model=model or s.ollama_model,
        base_url=s.ollama_base_url,
        temperature=s.ollama_temperature if temperature is None else temperature,
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