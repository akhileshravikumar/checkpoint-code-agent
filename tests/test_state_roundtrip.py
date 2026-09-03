"""The checkpointer must round-trip state without an unregistered-type warning."""
import sqlite3
import warnings
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from app.schemas import ChangePlan
from app.state import AgentState

PLAN = ChangePlan(
    target_file="search.py",
    summary="validate empty query",
    steps=["raise ValueError on empty input"],
    rationale="crashes downstream otherwise",
)


def _graph(db: Path):
    def plan(s):
        return {"plan": PLAN.model_dump(), "diff": "d"}

    def approve(s):
        interrupt({"plan": s["plan"]})
        return {}

    g = StateGraph(AgentState)
    g.add_node("plan", plan)
    g.add_node("await_approval", approve)
    g.add_edge(START, "plan")
    g.add_edge("plan", "await_approval")
    g.add_edge("await_approval", END)
    saver = SqliteSaver(sqlite3.connect(str(db), check_same_thread=False))
    saver.setup()
    return g.compile(checkpointer=saver)


def test_plan_survives_checkpoint_without_warnings(tmp_path):
    cfg = {"configurable": {"thread_id": "t"}}
    db = tmp_path / "cp.sqlite"
    _graph(db).invoke({"task": "t", "repo_path": "."}, config=cfg)

    # Fresh process would rebuild the graph; a fresh saver is the same thing.
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # any serde warning fails the test
        snap = _graph(db).get_state(cfg)

    assert snap.values["plan"] == PLAN.model_dump()
    assert isinstance(snap.values["plan"], dict)
    # and it re-validates cleanly at the point of use
    assert ChangePlan.model_validate(snap.values["plan"]).target_file == "search.py"
