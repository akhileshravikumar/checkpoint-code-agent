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
"""
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
 
        # Same approval -> same branch, every time.
        branch = client.branch_name(state["task"], seed=thread_id)
 
        if client.branch_exists(branch):
            # Already pushed on an earlier pass. Do not re-apply the patch.
            pr = client.find_pr_for_branch(branch)
            if pr:
                return {
                    "branch": branch, "pr_url": pr.url, "head_sha": pr.head_sha,
                    "ci_status": "pending", "error": "",
                }
            head_sha = client.remote_sha(branch)
        else:
            client.create_branch(branch)
            apply_patch(repo_path, state["diff"])
            head_sha = client.commit_and_push(
                branch, state["commit_message"], [plan.target_file]
            )
 
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