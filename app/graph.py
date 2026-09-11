"""The LangGraph state machine (ARCHITECTURE.md §2)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.config import get_settings
from app.diffing import PatchError, apply_patch, diff_stats
from app.github_client import checkout_local_branch
from app.nodes.execute import execute_node
from app.nodes.plan import plan_node
from app.nodes.propose_diff import propose_diff_node
from app.nodes.watch_ci import watch_ci_node
from app.state import AgentState


def await_approval_node(state: AgentState) -> dict:
    """Suspend the graph. State is checkpointed; the process may exit here.

    interrupt() raises internally, so nothing after it runs on the first pass.
    When resumed with Command(resume=payload), the call *returns* that payload.
    """
    decision = interrupt({
        "type": "diff_proposed",
        "diff": state["diff"],
        "plan": state["plan"],          # already a dict — see app/state.py
        "commit_message": state["commit_message"],
        "stats": diff_stats(state["diff"]),
        "retry_count": state.get("retry_count", 0),
        "retry_reason": ("ci" if state.get("ci_failure_log")
                 else "edit" if state.get("edit_note") else None),
    })
    if isinstance(decision, str):
        decision = {"decision": decision}
    return {
        "approval_status": decision.get("decision", "rejected"),
        "edit_note": decision.get("note", ""),
    }


def route_after_approval(state: AgentState) -> str:
    status = state.get("approval_status")
    if status == "approved":
        # ADR-004: offline, an approval applies locally and never reaches GitHub.
        return "execute_local" if get_settings().checkpoint_offline else "execute"
    if status == "edit_requested":
        if state.get("retry_count", 0) >= get_settings().max_retries:
            return "give_up"
        return "replan"
    return END                      # rejected: a clean, silent stop is correct


def execute_local_node(state: AgentState) -> dict:
    """CHECKPOINT_OFFLINE=1: apply the approved patch to the workspace. No push, no PR.

    Same approval guard as execute. The change stays uncommitted in the agent's
    workspace; the next online run hard-resets it away, by design.
    """
    if state.get("approval_status") != "approved":
        return {"error": "execute reached without approval — refusing to act."}
    try:
        apply_patch(Path(state["repo_path"]), state["diff"])
    except PatchError as exc:
        return {"error": f"local apply failed: {exc}"}
    return {"pr_url": None, "error": ""}


def bump_retry_node(state: AgentState) -> dict:
    """Prepare a retry: clear per-attempt fields, keep branch and failure log.

    If an earlier attempt was pushed, put the workspace back on the agent
    branch so the retry is planned (and its diff computed) on top of that
    attempt rather than on main. Normally the workspace is already there, but
    a server restart resets it to main at startup.
    """
    update = {
        "retry_count": state.get("retry_count", 0) + 1,
        "diff": "", "new_content": "",
        "pr_url": None, "head_sha": "", "ci_status": None, "ci_run_url": None,
        "approval_status": "pending",
        "error": "", "no_change_reason": "",
    }
    if (branch := state.get("branch")) and (repo := state.get("repo_path")):
        try:
            checkout_local_branch(repo, branch)
        except RuntimeError as exc:
            update["error"] = f"replan failed: {exc}"
    return update


def route_after_ci(state: AgentState) -> str:
    if state.get("ci_status") == "passed":
        return END
    if state.get("ci_status") in {"failed", "timeout"}:
        if state.get("retry_count", 0) >= get_settings().max_retries:
            return "give_up"
        return "replan"
    return END


def give_up_node(state: AgentState) -> dict:
    """Retries exhausted. Say so: ending quietly makes a red PR look shipped.

    MAX_RETRIES=2 means up to three gates (the original plus two retries), and
    edit requests share the counter with CI failures.
    """
    attempts = state.get("retry_count", 0) + 1
    if state.get("ci_status") in {"failed", "timeout"}:
        return {"error": f"CI still {state['ci_status']} after {attempts} attempts; stopping. "
                         f"The PR stays open for a human: {state.get('pr_url')}"}
    return {"error": f"Edit limit reached after {attempts} attempts (MAX_RETRIES)."}


def build_graph(checkpointer):
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("propose_diff", propose_diff_node)
    g.add_node("await_approval", await_approval_node)
    g.add_node("execute", execute_node)
    g.add_node("execute_local", execute_local_node)
    g.add_node("watch_ci", watch_ci_node)
    g.add_node("replan", bump_retry_node)
    g.add_node("give_up", give_up_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges(
        "plan",
        lambda s: END if s.get("error") else "propose_diff",
        {"propose_diff": "propose_diff", END: END},
    )
    g.add_conditional_edges(
        "propose_diff",
        # no_change_reason: the task is already done; there is nothing to approve.
        lambda s: END if s.get("error") or s.get("no_change_reason") else "await_approval",
        {"await_approval": "await_approval", END: END},
    )
    g.add_conditional_edges(
        "await_approval",
        route_after_approval,
        {"execute": "execute", "execute_local": "execute_local",
         "replan": "replan", "give_up": "give_up", END: END},
    )
    g.add_edge("execute_local", END)       # offline stops here; watch_ci needs GitHub
    g.add_edge("execute", "watch_ci")
    g.add_conditional_edges(
        "watch_ci",
        route_after_ci,
        {"replan": "replan", "give_up": "give_up", END: END},
    )
    # bump_retry_node can fail to put the workspace back on the agent branch;
    # planning against main after that would build the retry on the wrong base.
    g.add_conditional_edges(
        "replan",
        lambda s: END if s.get("error") else "plan",
        {"plan": "plan", END: END},
    )
    g.add_edge("give_up", END)
    return g.compile(checkpointer=checkpointer)


def make_checkpointer() -> SqliteSaver:
    """Durable checkpointer. check_same_thread=False because FastAPI is threaded."""
    db = get_settings().checkpoint_db
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
