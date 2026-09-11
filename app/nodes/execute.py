"""execute node: apply the approved diff, push a branch, open a PR.

Nothing in this module runs before approval_status == "approved". That
invariant is enforced by the graph's routing AND re-checked here, and it is the
whole point of the project.

IDEMPOTENCE. This is the first node with side effects that outlive the process,
so it is the first node where being run twice costs something real: a second
branch and a second pull request for one human approval. It can be re-entered —
a resumed thread, a crash between push and create_pull, a retried approval — so
the branch name is derived from the thread_id rather than the clock, and both
the push and the PR are checked for "already done" before being done again.

ATTEMPTS. "Already done" is per attempt, not per branch. After a CI failure the
same thread comes back through the gate with a new diff, and that approval must
add a commit to the same branch and PR. Keying idempotence on branch existence
alone made every retry return the attempt-1 PR without pushing anything, so
watch_ci re-read the same red run. Each commit now carries a
`Checkpoint-Attempt: N` trailer; the remote tip's trailer says which attempts
have already been pushed.
"""
import re
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.runnables import RunnableConfig

from app.config import get_settings
from app.diffing import apply_patch, diff_stats
from app.github_client import GitHubClient, _redact
from app.schemas import ChangePlan
from app.state import AgentState

PR_BODY = """### Task

{task}

### Plan

{steps}

### Rationale

{rationale}

---
Proposed by **Checkpoint** using `{model}` running locally.
Approved by a human at `{ts}` before this branch was pushed.
Diff: +{additions} / -{deletions} · attempt {attempt}
"""

ATTEMPT_TRAILER = "Checkpoint-Attempt"
_TRAILER_RE = re.compile(rf"^{ATTEMPT_TRAILER}:\s*(\d+)\s*$", re.M)


def with_attempt_trailer(message: str, attempt: int) -> str:
    return f"{message.rstrip()}\n\n{ATTEMPT_TRAILER}: {attempt}"


def pushed_attempt(message: str) -> int:
    """Highest attempt already on the branch. Tips without a trailer predate
    it and can only have come from attempt 1."""
    found = [int(n) for n in _TRAILER_RE.findall(message or "")]
    return max(found) if found else 1


def execute_node(state: AgentState, config: RunnableConfig | None = None) -> dict:
    if state.get("approval_status") != "approved":
        return {"error": "execute reached without approval — refusing to act."}

    s = get_settings()
    thread_id = (config or {}).get("configurable", {}).get("thread_id", "")

    client = GitHubClient()
    try:
        # Re-validate: state["plan"] is a dict (app/state.py), not a model.
        plan = ChangePlan.model_validate(state["plan"])

        # Start from clean origin/main. The human pause before this node is
        # unbounded, so the workspace may have been reset by another task in
        # the meantime — or left on a branch by an earlier execute.
        repo_path = Path(client.ensure_workspace())

        # Same thread -> same branch, every time. Same attempt -> same commit.
        branch = client.branch_name(state["task"], seed=thread_id)
        attempt = state.get("retry_count", 0) + 1
        message = with_attempt_trailer(state["commit_message"], attempt)

        if not client.branch_exists(branch):
            client.create_branch(branch)
            apply_patch(repo_path, state["diff"])
            head_sha = client.commit_and_push(branch, message, [plan.target_file])
        else:
            head_sha, tip_message = client.remote_head(branch)
            if pushed_attempt(tip_message) < attempt:
                # A new approval on an existing branch: a retry after a CI
                # failure. Commit on top of the previous attempt.
                client.checkout_remote_branch(branch)
                apply_patch(repo_path, state["diff"])
                head_sha = client.commit_and_push(branch, message, [plan.target_file])
            # else: this attempt is already on the remote. Re-entered after a
            # crash or a restart; do not apply the patch a second time.

        stats = diff_stats(state["diff"])
        pr = client.open_pr(
            branch=branch,
            title=state["commit_message"].splitlines()[0][:72],
            body=PR_BODY.format(
                task=state["task"],
                steps="\n".join(
                    f"{i}. {st}" for i, st in enumerate(plan.steps, 1)
                ),
                rationale=plan.rationale,
                model=s.ollama_model,
                ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                additions=stats["additions"], deletions=stats["deletions"],
                attempt=state.get("retry_count", 0) + 1,
            ),
        )
        return {
            "branch": branch, "pr_url": pr.url,
            "head_sha": head_sha, "ci_status": "pending", "error": "",
        }
    except Exception as exc:
        # This string is checkpointed and traced. Redact before it escapes.
        return {
            "error": _redact(
                f"execute failed: {type(exc).__name__}: {exc}", s.github_token
            )
        }
    finally:
        client.close()