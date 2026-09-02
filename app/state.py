"""The graph's state. Persisted by the SQLite checkpointer at every node boundary."""
from typing import Annotated, Literal, TypedDict

from app.schemas import ChangePlan


def _keep_last(_old, new):
    return new


class AgentState(TypedDict, total=False):
    task: str
    repo_path: str

    plan: ChangePlan | None
    diff: str
    new_content: str
    commit_message: str

    approval_status: Literal["pending", "approved", "rejected", "edit_requested"]
    edit_note: str

    branch: str
    pr_url: str | None
    ci_status: Literal["pending", "passed", "failed", "timeout"] | None
    ci_run_url: str | None
    ci_failure_log: str

    retry_count: Annotated[int, _keep_last]
    error: str