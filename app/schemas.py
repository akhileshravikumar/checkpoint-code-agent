"""Structured outputs. Every LLM call is constrained to one of these."""
from pydantic import BaseModel, Field


class ChangePlan(BaseModel):
    """A file-level plan for one change. One file only — see ADR-003."""

    target_file: str = Field(
        description="Repo-relative path of the single file to modify, e.g. 'sandbox/search.py'"
    )
    summary: str = Field(description="One sentence describing the change.")
    steps: list[str] = Field(
        description="2-5 concrete, ordered edits to make.", min_length=1, max_length=5
    )
    rationale: str = Field(description="Why this change addresses the task.")


class FileRewrite(BaseModel):
    """The complete new contents of the target file (ADR-001)."""

    path: str = Field(description="Repo-relative path being rewritten.")
    new_content: str = Field(
        description="The ENTIRE new file content. Not a diff, not a fragment."
    )
    commit_message: str = Field(
        description="Conventional-commit message, e.g. 'fix(search): validate empty query'"
    )