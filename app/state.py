"""The graph's state. Persisted by the SQLite checkpointer at every node boundary.

Everything in here must survive a msgpack round-trip through the checkpointer.
Pydantic models do not: LangGraph warns on deserialising an unregistered type and
will refuse outright in a future release. So `plan` is stored as a plain dict
(`ChangePlan.model_dump()`) and re-validated at the point of use.
"""
from typing import Annotated, Any, Literal, TypedDict


def _keep_last(_old, new):
    return new


class AgentState(TypedDict, total=False):
    task: str
    repo_path: str

    # ChangePlan.model_dump() — see module docstring. Re-validate with
    # ChangePlan.model_validate(state["plan"]) before touching its fields.
    plan: dict[str, Any] | None

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
