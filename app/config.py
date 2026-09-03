from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Local inference
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:3b"

    # A full-file rewrite (ADR-001) carries the target file TWICE through the
    # context: once as source in the prompt, once as `new_content` in the reply.
    # At MAX_FILE_LINES=200 that is ~4.4k tokens each way, so 8192 overflows and
    # Ollama silently drops the head of the prompt — including the system message.
    ollama_num_ctx: int = 16384
    # Generation budget for short, bounded replies (the plan node).
    ollama_num_predict: int = 1536
    # Generation budget for a whole-file rewrite. JSON escaping inflates the
    # payload ~8%, so a 200-line file needs ~5.4k tokens. 1536 truncated every
    # rewrite past ~130 lines mid-string, which surfaced as an
    # OutputParserException about the field that happens to serialise last.
    ollama_num_predict_rewrite: int = 6144
    ollama_temperature: float = 0.1
    ollama_num_thread: int = 0

    # Target repository
    github_token: str = ""
    github_owner: str = ""
    github_repo: str = ""
    github_base_branch: str = "main"
    workspace_dir: Path = Path("./.workspace")

    # Observability
    langsmith_tracing: bool = True
    langsmith_project: str = "checkpoint"

    # Behaviour
    checkpoint_db: Path = Path("./checkpoint.sqlite")
    max_file_lines: int = 200
    max_retries: int = 2
    ci_poll_interval: int = 10
    ci_poll_timeout: int = 600
    checkpoint_offline: bool = False

    @property
    def repo_slug(self) -> str:
        return f"{self.github_owner}/{self.github_repo}"


@lru_cache
def get_settings() -> Settings:
    return Settings()