"""The LangGraph state machine (ARCHITECTURE.md §2)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from app.config import get_settings
from app.diffing import apply_patch, diff_stats
from app.nodes.plan import plan_node
from app.nodes.propose_diff import propose_diff_node
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
    })
    if isinstance(decision, str):
        decision = {"decision": decision}
    return {
        "approval_status": decision.get("decision", "rejected"),
        "edit_note": decision.get("note", ""),
    }


def execute_local_node(state: AgentState) -> dict:
    """Week 1: apply to the local clone only. Week 2 replaces this with a real PR."""
    apply_patch(Path(state["repo_path"]), state["diff"])
    return {"error": ""}


def route_after_approval(state: AgentState) -> str:
    status = state.get("approval_status")
    if status == "approved":
        return "execute"
    if status == "edit_requested":
        if state.get("retry_count", 0) >= get_settings().max_retries:
            return END
        return "replan"
    return END


def bump_retry_node(state: AgentState) -> dict:
    return {"retry_count": state.get("retry_count", 0) + 1}


def build_graph(checkpointer):
    g = StateGraph(AgentState)
    g.add_node("plan", plan_node)
    g.add_node("propose_diff", propose_diff_node)
    g.add_node("await_approval", await_approval_node)
    g.add_node("execute", execute_local_node)
    g.add_node("replan", bump_retry_node)

    g.add_edge(START, "plan")
    g.add_conditional_edges(
        "plan",
        lambda s: END if s.get("error") else "propose_diff",
        {"propose_diff": "propose_diff", END: END},
    )
    g.add_conditional_edges(
        "propose_diff",
        lambda s: END if s.get("error") else "await_approval",
        {"await_approval": "await_approval", END: END},
    )
    g.add_conditional_edges(
        "await_approval",
        route_after_approval,
        {"execute": "execute", "replan": "replan", END: END},
    )
    g.add_edge("replan", "plan")
    g.add_edge("execute", END)
    return g.compile(checkpointer=checkpointer)


def make_checkpointer() -> SqliteSaver:
    """Durable checkpointer. check_same_thread=False because FastAPI is threaded."""
    db = get_settings().checkpoint_db
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver